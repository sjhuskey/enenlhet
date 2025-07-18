import os
import re
import json
import pympi
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from pydub import AudioSegment
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

class EnenlhetDataProcessor:
    """
    A specialized processor for Enenlhet ELAN files and audio recordings.
    Handles various preprocessing tasks specific to Enenlhet speech data.
    """
    def __init__(self, eaf_dir, wav_dir, output_dir, sampling_rate=16000):
        """
        Initialize the processor.
        
        Args:
            eaf_dir: Directory containing ELAN (.eaf) files
            wav_dir: Directory containing WAV audio files
            output_dir: Directory to save processed data
            sampling_rate: Target sampling rate for audio processing
        """
        self.eaf_dir = eaf_dir
        self.wav_dir = wav_dir
        self.output_dir = output_dir
        self.sampling_rate = sampling_rate
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "audio_segments"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)
        
        # Dictionary to track session metadata
        self.session_metadata = {}
        
        # Speaker info for normalization 
        self.speakers = {}
        
    def scan_corpus(self):
        """Scan the corpus and collect basic statistics"""
        eaf_files = [f for f in os.listdir(self.eaf_dir) if f.endswith('.eaf')]
        print(f"Found {len(eaf_files)} ELAN files")
        
        wav_files = [f for f in os.listdir(self.wav_dir) if f.endswith('.wav')]
        print(f"Found {len(wav_files)} WAV files")
        
        # Check for mismatches
        eaf_basenames = [os.path.splitext(f)[0] for f in eaf_files]
        wav_basenames = [os.path.splitext(f)[0] for f in wav_files]
        
        missing_audio = [f for f in eaf_basenames if f not in wav_basenames]
        missing_eaf = [f for f in wav_basenames if f not in eaf_basenames]
        
        if missing_audio:
            print(f"Warning: {len(missing_audio)} ELAN files have no matching audio")
            for f in missing_audio[:5]:
                print(f"  - {f}")
            if len(missing_audio) > 5:
                print(f"  ... and {len(missing_audio)-5} more")
                
        if missing_eaf:
            print(f"Warning: {len(missing_eaf)} WAV files have no matching ELAN file")
            for f in missing_eaf[:5]:
                print(f"  - {f}")
            if len(missing_eaf) > 5:
                print(f"  ... and {len(missing_eaf)-5} more")
        
        return eaf_files
        
    def process_elan_file(self, eaf_file):
        """
        Process a single ELAN file and its matching audio.
        
        Args:
            eaf_file: Filename of the ELAN file to process
            
        Returns:
            DataFrame containing segment information
        """
        eaf_path = os.path.join(self.eaf_dir, eaf_file)
        wav_file = eaf_file.replace('.eaf', '.wav')
        wav_path = os.path.join(self.wav_dir, wav_file)
        
        if not os.path.exists(wav_path):
            print(f"Warning: No matching WAV file found for {eaf_file}")
            return None
            
        print(f"Processing {eaf_file} with audio {wav_file}")
        
        # Load the ELAN file
        eaf = pympi.Elan.Eaf(eaf_path)
        
        # Extract metadata
        self._extract_metadata(eaf, eaf_file)
        
        # Get all the tiers (annotation tracks)
        tiers = eaf.get_tier_names()
        
        # Find the main transcription tier
        transcript_tier = self._find_transcript_tier(eaf, tiers)
        
        if transcript_tier is None:
            print(f"No usable transcript tier found in {eaf_file}, skipping")
            return None
        
        # Load audio file
        audio = AudioSegment.from_wav(wav_path)
        
        # Get all annotations from the transcript tier
        annotations = eaf.get_annotation_data_for_tier(transcript_tier)
        
        # Prepare dataframe to store segment information
        segments_data = []
        
        # Process each annotation
        for i, (start, end, text) in enumerate(annotations):
            # Skip if the annotation is too short
            duration_ms = end - start
            if duration_ms < 500:  # Skip segments shorter than 500ms
                continue
                
            # Skip if the text is empty or too short
            text = text.strip()
            if not text or len(text) < 2:
                continue
                
            # Clean and normalize the text
            text = self._normalize_text(text)
            
            # Create an ID for this segment
            segment_id = f"{os.path.basename(eaf_file).replace('.eaf', '')}_{i:04d}"
            
            # Extract the audio segment
            segment_audio = audio[start:end]
            
            # Save the audio segment
            segment_filename = f"{segment_id}.wav"
            segment_path = os.path.join(self.output_dir, "audio_segments", segment_filename)
            segment_audio.export(segment_path, format="wav")
            
            # Look for speaker information
            speaker = self._get_speaker_for_segment(eaf, start, end, tiers)
            
            # Add segment info to our list
            segments_data.append({
                'id': segment_id,
                'start_ms': start,
                'end_ms': end,
                'duration_ms': duration_ms,
                'text': text,
                'audio_path': segment_path,
                'source_file': os.path.basename(eaf_file),
                'speaker': speaker
            })
        
        # Create a DataFrame with all segment information
        df = pd.DataFrame(segments_data)
        
        # Save segment info for this file
        csv_path = os.path.join(
            self.output_dir, 
            "metadata", 
            f"{os.path.basename(eaf_file).replace('.eaf', '')}_segments.csv"
        )
        df.to_csv(csv_path, index=False)
        
        print(f"Processed {len(df)} segments from {eaf_file}")
        return df
    
    def _extract_metadata(self, eaf, eaf_file):
        """Extract and store metadata from the ELAN file"""
        metadata = {}
        
        # Get file properties
        metadata['filename'] = eaf_file
        
        # Get header info
        header = eaf.get_header()
        if header:
            metadata['media_file'] = header.get('MEDIA_FILE', '')
            metadata['time_units'] = header.get('TIME_UNITS', '')
        
        # Get properties
        properties = eaf.get_properties()
        if properties:
            for key, value in properties.items():
                metadata[f'property_{key}'] = value
        
        # Store in session metadata dictionary
        self.session_metadata[eaf_file] = metadata
    
    def _find_transcript_tier(self, eaf, tiers):
        """Find the main transcription tier in the ELAN file"""
        # Check for common tier names in Enenlhet transcriptions
        tier_preference = [
            'transcription', 'transcript', 'orthographic', 
            'enenlhet', 'enlhet', 'tmf',  # Language-specific tiers
            'tx', 'txt', 'text',  # Common abbreviations
        ]
        
        # First try exact matches
        for preferred in tier_preference:
            if preferred in tiers:
                return preferred
        
        # Then try substring matches
        for tier in tiers:
            tier_lower = tier.lower()
            for preferred in tier_preference:
                if preferred in tier_lower:
                    return tier
        
        # If we still don't have a match, use the first tier
        if tiers:
            print(f"No standard transcript tier found, using '{tiers[0]}'")
            return tiers[0]
        
        return None
    
    def _get_speaker_for_segment(self, eaf, start, end, tiers):
        """Try to identify the speaker for a given segment"""
        # Look for speaker tiers
        speaker_tiers = []
        for tier in tiers:
            tier_lower = tier.lower()
            if 'speaker' in tier_lower or 'spk' in tier_lower:
                speaker_tiers.append(tier)
        
        if not speaker_tiers:
            return "unknown"
            
        # Check each speaker tier for overlapping annotations
        for tier in speaker_tiers:
            annotations = eaf.get_annotation_data_for_tier(tier)
            for ann_start, ann_end, ann_value in annotations:
                # Check if this annotation overlaps with our segment
                if ann_end > start and ann_start < end:
                    # Found a speaker annotation
                    return ann_value.strip()
        
        return "unknown"
    
    def _normalize_text(self, text):
        """Normalize Enenlhet text to handle common transcription variations"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # TODO: Handle common Enenlhet transcription variations
        # (Update these rules based on specific needs for Enenlhet)
        
        # Standardize quotation marks
        text = text.replace('"', '"').replace('"', '"')
        
        # Remove common annotations if needed
        text = re.sub(r'\([^)]*\)', '', text)  # Remove parenthetical comments
        
        return text
    
    def process_corpus(self):
        """Process the entire corpus"""
        eaf_files = self.scan_corpus()
        
        all_segments = []
        processed_files = 0
        total_segments = 0
        
        for eaf_file in tqdm(eaf_files, desc="Processing files"):
            try:
                segments = self.process_elan_file(eaf_file)
                if segments is not None and not segments.empty:
                    all_segments.append(segments)
                    processed_files += 1
                    total_segments += len(segments)
            except Exception as e:
                print(f"Error processing {eaf_file}: {str(e)}")
                continue
        
        print(f"Successfully processed {processed_files} files with {total_segments} total segments")
        
        # Combine all segments into a single DataFrame
        if all_segments:
            combined_df = pd.concat(all_segments)
            combined_df.to_csv(os.path.join(self.output_dir, "all_segments.csv"), index=False)
            
            # Save metadata
            with open(os.path.join(self.output_dir, "corpus_metadata.json"), 'w') as f:
                json.dump(self.session_metadata, f, indent=2)
            
            # Create train/validation/test split
            self._create_dataset_splits(combined_df)
            
            return combined_df
        else:
            print("No segments were processed.")
            return None
    
    def _create_dataset_splits(self, df, test_size=0.1, val_size=0.1, random_state=42):
        """Create train/validation/test splits of the data"""
        # First split off the test set
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df['speaker'] if 'unknown' not in df['speaker'].unique() else None
        )
        
        # Then split the remaining data into train and validation
        adjusted_val_size = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=adjusted_val_size, random_state=random_state, 
            stratify=train_val_df['speaker'] if 'unknown' not in train_val_df['speaker'].unique() else None
        )
        
        # Save the splits
        train_df.to_csv(os.path.join(self.output_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(self.output_dir, "validation.csv"), index=False)
        test_df.to_csv(os.path.join(self.output_dir, "test.csv"), index=False)
        
        print(f"Dataset split: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test")
        
        # Create manifest files for Wav2Vec 2.0 training
        self._create_wav2vec_manifests(train_df, val_df, test_df)
    
    def _create_wav2vec_manifests(self, train_df, val_df, test_df):
        """Create manifest files for Wav2Vec 2.0 training"""
        manifest_dir = os.path.join(self.output_dir, "manifests")
        os.makedirs(manifest_dir, exist_ok=True)
        
        for name, df in [("train", train_df), ("valid", val_df), ("test", test_df)]:
            manifest_path = os.path.join(manifest_dir, f"{name}.tsv")
            
            with open(manifest_path, 'w') as f:
                # Write header
                f.write(f"{name}\n")
                
                # Write data rows
                for _, row in df.iterrows():
                    # Get absolute path to audio file
                    audio_path = os.path.abspath(row['audio_path'])
                    
                    # Calculate audio length in samples
                    audio_info = librosa.get_duration(path=audio_path)
                    sample_length = int(audio_info * self.sampling_rate)
                    
                    # Write to TSV file
                    f.write(f"{audio_path}\t{sample_length}\n")
            
            # Create labels file
            labels_path = os.path.join(manifest_dir, f"{name}.ltr")
            with open(labels_path, 'w') as f:
                for _, row in df.iterrows():
                    # Individual characters as labels
                    chars = " ".join(list(row['text'].strip()))
                    f.write(f"{chars}\n")
            
            # Create dict file from training data
            if name == "train":
                dict_path = os.path.join(manifest_dir, "dict.ltr.txt")
                characters = set()
                for _, row in df.iterrows():
                    characters.update(list(row['text'].strip()))
                
                with open(dict_path, 'w') as f:
                    # Add space and special tokens
                    f.write("| 0\n")  # blank token for CTC
                    f.write(" 1\n")  # space
                    for i, char in enumerate(sorted(characters)):
                        f.write(f"{char} {i+2}\n")  # +2 because 0,1 are special

        print(f"Created manifests in {manifest_dir}")


if __name__ == "__main__":
    # Example usage
    processor = EnenlhetDataProcessor(
        eaf_dir="/users/sjhuskey/Downloads/Enenlhet/have_transcripts",
        wav_dir="/users/sjhuskey/Downloads/Enenlhet/have_transcripts",
        output_dir="/users/sjhuskey/Python/heaton/python/enenlhet_data_processed"
    )
    
    # Process the entire corpus
    df = processor.process_corpus()
