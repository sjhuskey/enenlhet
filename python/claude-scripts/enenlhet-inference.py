import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import jiwer
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset, load_metric


class EnenlhetASRInference:
    """Class for Enenlhet ASR inference and evaluation"""
    
    def __init__(self, model_path, device=None):
        """
        Initialize the ASR model.
        
        Args:
            model_path: Path to the fine-tuned model
            device: Device to run inference on (None for auto-detection)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading model from {model_path} on {self.device}")
        
        # Load processor and model
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def transcribe_file(self, audio_path, vad_filter=False, return_confidence=False):
        """
        Transcribe a single audio file.
        
        Args:
            audio_path: Path to the audio file
            vad_filter: Whether to apply voice activity detection
            return_confidence: Whether to return token-level confidence scores
            
        Returns:
            Transcription text and optionally confidence scores
        """
        try:
            # Load and preprocess audio
            audio_input, sample_rate = self._load_audio(audio_path)
            
            if vad_filter:
                # Apply VAD if requested (especially useful for noisier recordings)
                audio_input = self._apply_vad(audio_input, sample_rate)
            
            # Process audio
            inputs = self.processor(
                audio_input, 
                sampling_rate=16000,
                return_tensors="pt",
                padding="longest"
            ).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(inputs.input_values)
                
            # Get predictions
            predicted_ids = torch.argmax(outputs.logits, dim=-1)
            transcription = self.processor.decode(predicted_ids[0])
            
            if not return_confidence:
                return transcription
            else:
                # Calculate confidence scores
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                confidence = torch.max(probs, dim=-1)[0]
                mean_confidence = confidence.mean().item()
                
                # Get token-level confidence
                token_confidence = {}
                for i, token_id in enumerate(predicted_ids[0]):
                    token = self.processor.tokenizer.convert_ids_to_tokens(token_id.item())
                    if token not in ["<pad>", "<s>", "</s>"]:
                        token_confidence[token] = confidence[0, i].item()
                
                return transcription, mean_confidence, token_confidence
                
        except Exception as e:
            print(f"Error transcribing {audio_path}: {str(e)}")
            return "[TRANSCRIPTION FAILED]"
    
    def _load_audio(self, audio_path, target_sr=16000):
        """Load and preprocess audio file"""
        try:
            # Use soundfile for faster loading
            audio, sr = sf.read(audio_path)
        except:
            # Fallback to librosa
            audio, sr = librosa.load(audio_path, sr=None)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Resample if needed
        if sr != target_sr:
            audio = librosa.resample(audio.astype(np.float64), orig_sr=sr, target_sr=target_sr)
        
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-10)
        
        return audio, target_sr
    
    def _apply_vad(self, audio, sample_rate):
        """Apply basic voice activity detection"""
        # Simple energy-based VAD
        frame_length = int(sample_rate * 0.025)  # 25ms frames
        hop_length = int(sample_rate * 0.010)    # 10ms hop
        
        # Compute frame energies
        energies = librosa.feature.rms(
            y=audio, 
            frame_length=frame_length, 
            hop_length=hop_length
        )[0]
        
        # Determine energy threshold (adapt to your data)
        threshold = 0.05 * np.max(energies)
        
        # Create a mask for speech frames
        mask = energies > threshold
        
        # Convert frame-level mask to sample-level mask
        sample_mask = np.zeros_like(audio, dtype=bool)
        for i, m in enumerate(mask):
            if m:
                start = i * hop_length
                end = min(len(audio), start + frame_length)
                sample_mask[start:end] = True
        
        # Apply mask
        filtered_audio = audio.copy()
        filtered_audio[~sample_mask] = 0
        
        return filtered_audio
    
    def batch_transcribe(self, audio_dir, output_file=None, vad_filter=False):
        """
        Transcribe all audio files in a directory.
        
        Args:
            audio_dir: Directory containing audio files
            output_file: Path to save transcriptions (if None, just returns results)
            vad_filter: Whether to apply voice activity detection
            
        Returns:
            DataFrame with transcriptions
        """
        # List audio files
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac', '.ogg']:
            audio_files.extend(list(Path(audio_dir).glob(f"*{ext}")))
        
        results = []
        
        # Process each file
        for audio_file in tqdm(audio_files, desc="Transcribing"):
            transcription, confidence, _ = self.transcribe_file(
                str(audio_file), 
                vad_filter=vad_filter,
                return_confidence=True
            )
            
            results.append({
                'file': str(audio_file),
                'transcription': transcription,
                'confidence': confidence
            })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save to file if requested
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Saved transcriptions to {output_file}")
        
        return df
    
    def evaluate(self, test_csv, audio_column="audio_path", text_column="text", vad_filter=False):
        """
        Evaluate the model on a test set.
        
        Args:
            test_csv: Path to CSV file with test data
            audio_column: Name of the column containing audio paths
            text_column: Name of the column containing reference transcriptions
            vad_filter: Whether to apply voice activity detection
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Load test data
        test_df = pd.read_csv(test_csv)
        
        # Initialize metrics
        predictions = []
        references = []
        confidence_scores = []
        
        # Process each test item
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
            audio_path = row[audio_column]
            reference = row[text_column]
            
            # Skip if audio file doesn't exist
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file {audio_path} does not exist, skipping")
                continue
                
            # Get prediction
            try:
                prediction, confidence, _ = self.transcribe_file(
                    audio_path, 
                    vad_filter=vad_filter,
                    return_confidence=True
                )
                
                predictions.append(prediction)
                references.append(reference)
                confidence_scores.append(confidence)
            except Exception as e:
                print(f"Error processing {audio_path}: {str(e)}")
        
        # Calculate metrics
        wer = jiwer.wer(references, predictions)
        cer = jiwer.cer(references, predictions)
        
        # Calculate metrics by confidence
        confidences = np.array(confidence_scores)
        bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        wer_by_confidence = []
        
        for i in range(len(bins)-1):
            lower = bins[i]
            upper = bins[i+1]
            
            mask = (confidences >= lower) & (confidences < upper)
            if mask.sum() > 0:
                bin_wer = jiwer.wer(
                    [r for r, m in zip(references, mask) if m],
                    [p for p, m in zip(predictions, mask) if m]
                )
            else:
                bin_wer = None
                
            wer_by_confidence.append({
                'bin': f"{lower:.1f}-{upper:.1f}",
                'wer': bin_wer,
                'count': mask.sum()
            })
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'reference': references,
            'prediction': predictions,
            'confidence': confidence_scores
        })
        
        # Plot confidence vs WER
        self._plot_confidence_vs_wer(wer_by_confidence)
        
        # Calculate confusion matrix for common errors
        error_analysis = self._analyze_errors(references, predictions)
        
        metrics = {
            'wer': wer,
            'cer': cer,
            'mean_confidence': np.mean(confidence_scores),
            'wer_by_confidence': wer_by_confidence,
            'error_analysis': error_analysis,
            'results_df': results_df
        }
        
        return metrics
    
    def _plot_confidence_vs_wer(self, wer_by_confidence):
        """Plot WER by confidence bin"""
        plt.figure(figsize=(10, 6))
        
        bins = [item['bin'] for item in wer_by_confidence if item['wer'] is not None]
        wers = [item['wer'] for item in wer_by_confidence if item['wer'] is not None]
        counts = [item['count'] for item in wer_by_confidence if item['wer'] is not None]
        
        if not bins:
            return
            
        plt.bar(bins, wers)
        plt.xlabel('Confidence Range')
        plt.ylabel('Word Error Rate')
        plt.title('WER by Confidence Level')
        
        # Add count labels
        for i, (count, wer) in enumerate(zip(counts, wers)):
            plt.text(i, wer + 0.01, f"n={count}", ha='center')
        
        plt.tight_layout()
        plt.savefig('wer_by_confidence.png')
        plt.close()
    
    def _analyze_errors(self, references, predictions, max_errors=20):
        """Analyze common errors in predictions"""
        error_counts = {}
        
        for ref, pred in zip(references, predictions):
            # Convert to lowercase for comparison
            ref_lower = ref.lower()
            pred_lower = pred.lower()
            
            # Simple word-level analysis
            ref_words = ref_lower.split()
            pred_words = pred_lower.split()
            
            # Use alignment algorithm to match words
            import difflib
            matcher = difflib.SequenceMatcher(None, ref_words, pred_words)
            
            for op, i1, i2, j1, j2 in matcher.get_opcodes():
                if op in ['replace', 'delete']:
                    # Reference words that were incorrectly predicted
                    for i in range(i1, i2):
                        if i < len(ref_words):
                            ref_word = ref_words[i]
                            pred_word = pred_words[j1] if j1 < len(pred_words) else "[DELETED]"
                            
                            error_key = f"{ref_word} → {pred_word}"
                            error_counts[error_key] = error_counts.get(error_key, 0) + 1
                
                elif op == 'insert':
                    # Words inserted in prediction that weren't in reference
                    for j in range(j1, j2):
                        if j < len(pred_words):
                            error_key = f"[NONE] → {pred_words[j]}"
                            error_counts[error_key] = error_counts.get(error_key, 0) + 1
        
        # Sort by frequency
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_errors[:max_errors]


def main():
    parser = argparse.ArgumentParser(description="Enenlhet ASR Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--mode", type=str, choices=["single", "batch", "evaluate"], required=True, 
                       help="Inference mode: single file, batch directory, or evaluate on test set")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to audio file, directory, or test CSV depending on mode")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to save output (optional)")
    parser.add_argument("--vad", action="store_true", help="Apply voice activity detection")
    parser.add_argument("--device", type=str, default=None, 
                       help="Device to run inference on (cuda or cpu, default: auto-detect)")
    
    args = parser.parse_args()
    
    # Initialize ASR system
    asr = EnenlhetASRInference(args.model_path, device=args.device)
    
    # Run inference based on mode
    if args.mode == "single":
        result, confidence, token_confidence = asr.transcribe_file(
            args.input, 
            vad_filter=args.vad,
            return_confidence=True
        )
        
        print(f"Transcription: {result}")
        print(f"Confidence: {confidence:.2f}")
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(result)
    
    elif args.mode == "batch":
        results = asr.batch_transcribe(args.input, args.output, vad_filter=args.vad)
        print(f"Transcribed {len(results)} files")
        print(results.head())
    
    elif args.mode == "evaluate":
        metrics = asr.evaluate(args.input, vad_filter=args.vad)
        
        print(f"Evaluation Results:")
        print(f"WER: {metrics['wer']:.4f}")
        print(f"CER: {metrics['cer']:.4f}")
        print(f"Mean Confidence: {metrics['mean_confidence']:.4f}")
        
        print("\nTop Error Patterns:")
        for error, count in metrics['error_analysis'][:10]:
            print(f"  {error}: {count}")
            
        if args.output:
            metrics['results_df'].to_csv(args.output, index=False)
            print(f"Saved detailed results to {args.output}")


if __name__ == "__main__":
    main()
