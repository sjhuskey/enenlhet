import os
import torch
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any

import transformers
from transformers import (
    HfArgumentParser,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset, load_metric, Dataset, Audio

import librosa
import soundfile as sf
from jiwer import wer as calculate_wer
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    model_name_or_path: str = "facebook/wav2vec2-xls-r-300m"
    cache_dir: Optional[str] = None
    freeze_feature_encoder: bool = True
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    feat_proj_dropout: float = 0.1
    mask_time_prob: float = 0.05
    layerdrop: float = 0.1
    ctc_loss_reduction: str = "mean"
    apply_spec_augment: bool = True

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_dir: str = None
    dataset_config_name: Optional[str] = None
    overwrite_cache: bool = False
    preprocessing_num_workers: Optional[int] = None
    max_duration_in_seconds: Optional[float] = 15.0
    min_duration_in_seconds: Optional[float] = 0.5
    preprocessing_only: bool = False
    train_split_name: str = "train"
    eval_split_name: str = "validation"
    audio_column_name: str = "audio_path"
    text_column_name: str = "text"
    char_sep: str = ""
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    preprocessing_strategy: str = "char_level"
    eval_metrics: List[str] = None

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be padded differently
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Replace padding with -100 to ignore in CTC loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch


class EnenlhetASRDataset(Dataset):
    """Custom dataset for Enenlhet ASR"""
    
    def __init__(
        self,
        csv_path,
        processor,
        max_duration=None,
        min_duration=None,
        audio_column="audio_path",
        text_column="text",
        sampling_rate=16000
    ):
        """
        Initialize the dataset.
        
        Args:
            csv_path: Path to the CSV file with audio paths and transcriptions
            processor: Wav2Vec2 processor
            max_duration: Maximum audio duration in seconds
            min_duration: Minimum audio duration in seconds
            audio_column: Name of the column containing audio paths
            text_column: Name of the column containing transcriptions
            sampling_rate: Target sampling rate
        """
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.audio_column = audio_column
        self.text_column = text_column
        self.sampling_rate = sampling_rate
        self.max_duration = max_duration
        self.min_duration = min_duration
        
        # Filter by duration if needed
        if max_duration or min_duration:
            self._filter_by_duration()
            
        logger.info(f"Loaded {len(self.df)} examples from {csv_path}")
    
    def _filter_by_duration(self):
        """Filter out examples that are too long or too short"""
        initial_len = len(self.df)
        durations = []
        
        for path in self.df[self.audio_column]:
            try:
                duration = librosa.get_duration(filename=path)
                durations.append(duration)
            except Exception:
                durations.append(-1)  # Mark invalid files
        
        self.df['duration'] = durations
        
        # Filter out invalid files
        self.df = self.df[self.df['duration'] > 0]
        
        # Apply duration filters
        if self.max_duration:
            self.df = self.df[self.df['duration'] <= self.max_duration]
        
        if self.min_duration:
            self.df = self.df[self.df['duration'] >= self.min_duration]
        
        logger.info(f"Filtered dataset from {initial_len} to {len(self.df)} examples based on duration")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """Get an example from the dataset"""
        row = self.df.iloc[idx]
        
        # Load and resample audio
        audio_path = row[self.audio_column]
        try:
            speech_array, sr = sf.read(audio_path)
        except Exception as e:
            logger.warning(f"Could not load audio file {audio_path}: {str(e)}")
            # Return a dummy item that will be filtered out
            return {"input_values": np.zeros(16000), "labels": []}
            
        # Resample if needed
        if sr != self.sampling_rate:
            speech_array = librosa.resample(
                speech_array.astype(np.float64), 
                orig_sr=sr, 
                target_sr=self.sampling_rate
            )
        
        # Normalize audio
        speech_array = speech_array / (np.max(np.abs(speech_array)) + 1e-10)
        
        # Process audio
        input_values = self.processor(
            speech_array, 
            sampling_rate=self.sampling_rate,
            return_tensors="pt"
        ).input_values.squeeze()
        
        # Process text
        text = row[self.text_column]
        if not isinstance(text, str):
            text = str(text)
            
        with self.processor.as_target_processor():
            labels = self.processor(text).input_ids
            
        return {"input_values": input_values, "labels": labels}


def main():
    """Run the training pipeline"""
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Log args
    logger.info(f"Training/evaluation parameters {training_args}")
    
    # Detecting last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Checkpoint detected, resuming from {last_checkpoint}")
    
    # Prepare for training
    
    # Load processor
    if os.path.isdir(model_args.model_name_or_path):
        # If model_name_or_path is a local directory
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_args.model_name_or_path)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_args.model_name_or_path)
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    else:
        # For first-time training, create a new tokenizer from the dataset
        if not os.path.exists(os.path.join(data_args.dataset_dir, "manifests", "dict.ltr.txt")):
            logger.error("Dictionary file not found. Please run the data preparation script first.")
            return
            
        # Create vocabulary from the dictionary file
        vocab_dict = {}
        with open(os.path.join(data_args.dataset_dir, "manifests", "dict.ltr.txt"), "r") as f:
            for line in f:
                token, index = line.strip().split()
                vocab_dict[token] = int(index)
                
        # Create tokenizer config
        tokenizer_config = {
            "vocab": vocab_dict,
            "pad_token": "|",
            "word_delimiter_token": " ",
            "bos_token": None,
            "eos_token": None,
            "unk_token": "[UNK]",
            "do_lower_case": True,
        }
        
        # Save tokenizer files
        os.makedirs(os.path.join(training_args.output_dir, "tokenizer"), exist_ok=True)
        with open(os.path.join(training_args.output_dir, "tokenizer", "vocab.json"), "w") as f:
            import json
            json.dump(vocab_dict, f)
            
        with open(os.path.join(training_args.output_dir, "tokenizer", "config.json"), "w") as f:
            json.dump(tokenizer_config, f)
            
        # Create tokenizer and processor
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(os.path.join(training_args.output_dir, "tokenizer"))
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True
        )
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    # Load model
    if os.path.isdir(model_args.model_name_or_path) and os.path.exists(os.path.join(model_args.model_name_or_path, "pytorch_model.bin")):
        logger.info(f"Loading model from {model_args.model_name_or_path}")
        model = Wav2Vec2ForCTC.from_pretrained(model_args.model_name_or_path)
    else:
        logger.info(f"Loading pretrained model {model_args.model_name_or_path}")
        model = Wav2Vec2ForCTC.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            ctc_loss_reduction=model_args.ctc_loss_reduction,
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer),
            hidden_dropout=model_args.hidden_dropout,
            attention_dropout=model_args.attention_dropout,
            feat_proj_dropout=model_args.feat_proj_dropout,
            mask_time_prob=model_args.mask_time_prob if model_args.apply_spec_augment else 0.0,
            layerdrop=model_args.layerdrop,
        )
    
    # Freeze feature encoder if specified
    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()
    
    # Data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    
    # Metrics
    wer_metric = load_metric("wer")
    cer_metric = load_metric("cer")
    
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        
        return {"wer": wer, "cer": cer}
    
    # Load datasets
    train_csv = os.path.join(data_args.dataset_dir, "train.csv")
    eval_csv = os.path.join(data_args.dataset_dir, "validation.csv")
    
    # Create custom datasets
    train_dataset = EnenlhetASRDataset(
        csv_path=train_csv,
        processor=processor,
        max_duration=data_args.max_duration_in_seconds,
        min_duration=data_args.min_duration_in_seconds
    )
    
    eval_dataset = EnenlhetASRDataset(
        csv_path=eval_csv,
        processor=processor,
        max_duration=data_args.max_duration_in_seconds,
        min_duration=data_args.min_duration_in_seconds
    )
    
    # Initialize trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Start training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
            
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluating ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    # Save processor for inference
    processor.save_pretrained(training_args.output_dir)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()