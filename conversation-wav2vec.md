---
title: "Building an ASR System for Enenlhet Language"
author: Claude 3.7 Sonnet
date: 2025-04-28
---

## Introduction

This document outlines a comprehensive approach to developing an Automatic Speech Recognition (ASR) system for Enenlhet, a low-resource indigenous language spoken in Paraguay, using ELAN (.eaf) transcriptions and WAV audio files.

## Enenlhet Language Context

Enenlhet is an indigenous language from the Enlhet-Enenlhet family (formerly known as Maskoyan or Lengua-Maskoy) spoken in Paraguay. There are approximately 2,000 speakers in the central Paraguayan Chaco region, but the language is endangered due to sociopolitical forces that have forced speakers to relocate from their native lands and abandon traditional means of subsistence.

Dr. Raina Heaton from the University of Oklahoma has collected 120 hours of Enenlhet audio and video data, along with 300+ pages of field notes. This documentation is accessible online via the Archive of the Indigenous Languages of Latin America (AILLA).

## System Components

### 1. Data Preparation

The first step involves processing ELAN (.eaf) files into a format suitable for training ASR models:

```python
class EnenlhetDataProcessor:
    """
    A specialized processor for Enenlhet ELAN files and audio recordings.
    Handles various preprocessing tasks specific to Enenlhet speech data.
    """
    def __init__(self, eaf_dir, wav_dir, output_dir, sampling_rate=16000):
        # Initialize directories and settings
        # ...
    
    def scan_corpus(self):
        # Scan and validate corpus files
        # ...
    
    def process_elan_file(self, eaf_file):
        # Process a single ELAN file and extract segments
        # ...
    
    def process_corpus(self):
        # Process the entire corpus
        # ...
    
    def _create_dataset_splits(self, df):
        # Create train/validation/test splits
        # ...
    
    def _create_wav2vec_manifests(self, train_df, val_df, test_df):
        # Create manifest files for Wav2Vec 2.0 training
        # ...
```

This processor:
- Extracts audio segments from ELAN annotations
- Creates train/validation/test splits
- Generates manifests for model training
- Handles special considerations for Enenlhet transcriptions
- Preserves speaker information where available

### 2. Model Training

For low-resource languages like Enenlhet, Wav2Vec 2.0 is an excellent choice due to its self-supervised learning approach:

```python
def main():
    """Run the training pipeline"""
    # Parse arguments
    
    # Prepare tokenizer and processor
    
    # Load model (Wav2Vec 2.0 XLS-R 300M)
    
    # Training with custom optimizations for low-resource settings
    
    # Evaluation
    
    # Save model and processor
```

Key optimization techniques include:
- Starting with XLS-R (cross-lingual speech representations) pretrained on 53 languages
- Aggressive data augmentation
- Freezing the feature encoder
- Using a specialized learning rate schedule
- Character-level tokenization

### 3. Inference and Evaluation

The inference component handles transcribing new audio files and evaluating model performance:

```python
class EnenlhetASRInference:
    """Class for Enenlhet ASR inference and evaluation"""
    
    def __init__(self, model_path, device=None):
        # Initialize model and processor
        # ...
    
    def transcribe_file(self, audio_path, vad_filter=False, return_confidence=False):
        # Transcribe a single audio file
        # ...
    
    def batch_transcribe(self, audio_dir, output_file=None, vad_filter=False):
        # Transcribe multiple files
        # ...
    
    def evaluate(self, test_csv, audio_column="audio_path", text_column="text", vad_filter=False):
        # Evaluate model on test set
        # ...
    
    def _analyze_errors(self, references, predictions, max_errors=20):
        # Analyze common error patterns
        # ...
```

This component provides:
- Single-file and batch transcription
- Voice activity detection for noisy recordings
- Confidence scoring
- Detailed error analysis
- Performance metrics (WER and CER)

## Full Workflow

### 1. Data Preparation

```bash
python enenlhet-data-pipeline.py \
  --eaf_dir /path/to/elan/files \
  --wav_dir /path/to/wav/files \
  --output_dir ./enenlhet_data_processed
```

### 2. Model Training

```bash
python enenlhet-wav2vec-finetuning.py \
  --model_name_or_path facebook/wav2vec2-xls-r-300m \
  --dataset_dir ./enenlhet_data_processed \
  --output_dir ./enenlhet_asr_model \
  --do_train \
  --do_eval \
  --num_train_epochs 30 \
  --per_device_train_batch_size 8 \
  --learning_rate 3e-4 \
  --warmup_steps 500 \
  --evaluation_strategy steps \
  --save_steps 1000 \
  --eval_steps 1000 \
  --logging_steps 100 \
  --fp16 True \
  --freeze_feature_encoder True
```

### 3. Evaluation and Inference

```bash
# Evaluate on test set
python enenlhet-inference.py \
  --model_path ./enenlhet_asr_model \
  --mode evaluate \
  --input ./enenlhet_data_processed/test.csv \
  --output ./evaluation_results.csv

# Transcribe a single file
python enenlhet-inference.py \
  --model_path ./enenlhet_asr_model \
  --mode single \
  --input /path/to/new_audio.wav
```

## Specific Considerations for Enenlhet

1. **Language-specific preprocessing**:
   - Specialized handling for Enenlhet transcription conventions
   - Speaker identification and normalization

2. **Low-resource optimizations**:
   - Starting with XLS-R model
   - Aggressive data augmentation 
   - Feature encoder freezing

3. **Evaluation methods**:
   - Confidence-based analysis
   - Detailed error analysis
   - Character Error Rate (CER) as primary metric

## Next Steps

1. **Expand vocabulary**: Update the model with new vocabulary as documentation continues
   
2. **Dialect variations**: Consider training separate models for different dialect regions

3. **Language-specific post-processing**: Develop rules based on Enenlhet morphology

4. **Knowledge transfer**: Extend approach to other Enlhet-Enenlhet family languages