---
title: "Building an ASR System for Enenlhet Language"
author: Claude 4 Sonnet
date: 2025-06-13
---

# Fine-tuning Whisper for Enenlhet Language Transcription

## Project Overview

This document describes the development of a fine-tuning pipeline for OpenAI's Whisper automatic speech recognition (ASR) model to transcribe the endangered language Enenlhet. The project utilizes existing gold standard transcriptions in ELAN format along with associated audio files.

## Background

**Enenlhet** is an endangered language that benefits from improved ASR capabilities to support language documentation and preservation efforts. The project leverages:

- 5 hours of gold standard transcriptions in ELAN format
- Associated audio files
- OpenAI's Whisper model as the base for fine-tuning

## Technical Approach

### Key Components

1. **ELAN File Parser**: Extracts time-aligned transcription annotations from ELAN (.eaf) files
2. **Dataset Preparation**: Segments audio files based on ELAN time annotations
3. **Whisper Integration**: Uses HuggingFace Transformers library for model fine-tuning
4. **Training Pipeline**: Complete end-to-end training setup with proper data handling

### Architecture

The solution implements a speech-to-text fine-tuning pipeline with the following workflow:

```
ELAN Files + Audio Files → Data Processing → Dataset Creation → Model Fine-tuning → Trained Model
```

## Implementation Details

### ELAN Parser (`ELANParser` class)

- Parses XML structure of ELAN files
- Extracts time slots and converts milliseconds to seconds
- Identifies transcription tiers (configurable based on tier naming)
- Returns structured annotations with start/end times and transcription text

### Dataset Preparation

- Loads audio files at 16kHz sampling rate (Whisper requirement)
- Segments audio based on ELAN time annotations
- Filters segments by duration (0.5-30 seconds)
- Saves individual audio segments as WAV files
- Creates structured dataset with audio paths and transcriptions

### Training Configuration

**Model Selection**: 
- Starts with `whisper-small` (244M parameters) for initial experiments
- Scalable to larger models (`whisper-medium`, `whisper-large`)

**Training Parameters**:
- Batch size: 4-8 (adjustable based on GPU memory)
- Learning rate: 1e-5
- Training steps: ~1000 (adjustable based on dataset size)
- Gradient accumulation: 2 steps
- Mixed precision: FP16 enabled
- Evaluation strategy: Every 100 steps

**Optimization Features**:
- Gradient checkpointing for memory efficiency
- Automatic mixed precision training
- Best model selection based on evaluation loss
- TensorBoard logging for monitoring

## Usage Instructions

### Prerequisites

```bash
pip install torch transformers datasets librosa pandas xml torch-audio accelerate tensorboard
```

### Setup and Configuration

1. **Prepare File Paths**: Update the `elan_files` and `audio_files` lists with your data paths
2. **Configure Tiers**: Adjust tier filtering logic in `extract_annotations()` if needed
3. **Set Training Parameters**: Modify batch size, learning rate, and training steps as needed

### Running the Pipeline

```bash
python whisper_finetuning.py
```

### Training Process

1. **Data Processing**: Extracts and segments audio from ELAN annotations
2. **Dataset Creation**: Builds HuggingFace dataset with audio features
3. **Train/Eval Split**: 80/20 split with random seed for reproducibility
4. **Model Training**: Fine-tunes Whisper with custom data collator
5. **Model Saving**: Saves best model and processor to output directory

## Expected Results

With 5 hours of gold standard Enenlhet data:

- **High-quality fine-tuning**: Sufficient data for effective adaptation
- **Language-specific improvements**: Better handling of Enenlhet phonetics and vocabulary
- **Robust performance**: Training on diverse speakers and contexts

## Model Output

The fine-tuned model is saved to `whisper-enenlhet-finetuned/` and includes:

- Model weights adapted for Enenlhet
- Tokenizer configured for the target language
- Processor with appropriate feature extraction settings
- Full compatibility with HuggingFace inference pipeline

## Testing and Evaluation

The pipeline includes a `test_model()` function for evaluating the fine-tuned model:

```python
transcription = test_model("whisper-enenlhet-finetuned", "path/to/test_audio.wav")
```

## Optimization Recommendations

### For Limited Resources

- Start with `whisper-small` model
- Use smaller batch sizes (2-4)
- Enable gradient checkpointing
- Use FP16 mixed precision training

### For Better Accuracy

- Scale to `whisper-medium` after initial success
- Increase training steps if data permits
- Fine-tune learning rate based on validation loss
- Consider data augmentation for robustness

### Monitoring Training

- Use TensorBoard for loss visualization
- Monitor evaluation metrics every 100 steps
- Watch for overfitting with small datasets
- Adjust learning rate if loss plateaus

## Technical Considerations

### Data Quality

- Filters out very short (<0.5s) and very long (>30s) segments
- Handles empty or malformed transcriptions
- Preserves original audio quality at 16kHz

### Memory Management

- Gradient checkpointing reduces memory usage
- Adjustable batch sizes for different GPU configurations
- Efficient data loading with HuggingFace datasets

### Language Support

- Configures tokenizer for Enenlhet language
- Maintains compatibility with Whisper's multilingual capabilities
- Preserves special tokens and formatting

## Future Enhancements

### Potential Improvements

1. **Data Augmentation**: Speed/pitch perturbation for robustness
2. **Multi-speaker Adaptation**: Speaker-specific fine-tuning
3. **Active Learning**: Iterative improvement with new annotations
4. **Evaluation Metrics**: WER/CER calculation for quantitative assessment

### Integration Options

1. **Real-time Transcription**: Streaming inference pipeline
2. **Batch Processing**: Large-scale transcription workflows
3. **Web Interface**: User-friendly transcription tool
4. **API Deployment**: Service for external applications

## Conclusion

This fine-tuning pipeline provides a robust foundation for improving Whisper's performance on Enenlhet language transcription. The modular design allows for easy adaptation to other endangered languages and integration into larger language documentation workflows.

The combination of ELAN format support, efficient training procedures, and HuggingFace integration makes this solution both practical for immediate use and extensible for future research and development efforts in computational linguistics and language preservation.

---

**Project Type**: Endangered Language ASR  
**Base Model**: OpenAI Whisper  
**Framework**: HuggingFace Transformers  
**Data Format**: ELAN + Audio Files  
**Training Data**: 5 hours gold standard transcriptions  
**Target Language**: Enenlhet