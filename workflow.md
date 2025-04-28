---
name: workflow
author: Claude
date: 2025-04-28
---
Let me now provide you with a script for inference and evaluation, which will allow you to transcribe new Enenlhet audio files using your fine-tuned model:

## Full Workflow for Enenlhet ASR Development

Now that I've provided the key components, let me explain how you would use these scripts to develop an ASR system for Enenlhet, building on Dr. Heaton's work:

### 1. Data Preparation

First, use the `EnenlhetDataProcessor` from my first artifact to process the ELAN files:

```bash
python enenlhet-data-pipeline.py \
  --eaf_dir /path/to/elan/files \
  --wav_dir /path/to/wav/files \
  --output_dir ./enenlhet_data_processed
```

This will:
- Convert ELAN files to a format suitable for ASR training
- Split audio into manageable segments
- Create train/validation/test splits
- Generate necessary manifest files

### 2. Model Training

Next, train the Wav2Vec 2.0 model using the second artifact:

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

Finally, use the inference script to transcribe new audio files:

```bash
python enenlhet-inference.py \
  --model_path ./enenlhet_asr_model \
  --mode evaluate \
  --input ./enenlhet_data_processed/test.csv \
  --output ./evaluation_results.csv
```

For transcribing a single file:
```bash
python enenlhet-inference.py \
  --model_path ./enenlhet_asr_model \
  --mode single \
  --input /path/to/new_audio.wav
```

## Specific Considerations for Enenlhet

Based on what we know about Enenlhet, I've incorporated several specific optimizations:

1. **Language-specific preprocessing**:
   - The data pipeline includes specialized handling for Enenlhet transcription conventions
   - Speaker identification and normalization for multi-speaker data

2. **Low-resource optimizations**:
   - Starting with XLS-R (cross-lingual speech representations) model that was pretrained on 53 languages
   - Aggressive data augmentation 
   - Using a freezing strategy for the feature encoder

3. **Evaluation methods for extremely low-resource languages**:
   - Confidence-based analysis to identify uncertain predictions
   - Detailed error analysis to understand common patterns
   - Character Error Rate (CER) as the primary metric (more reliable for low-resource scenarios)

## Next Steps

1. **Expand vocabulary**: As Dr. Heaton continues to document Enenlhet, you can update the model with new vocabulary
   
2. **Dialect variations**: Consider training separate models for different dialect regions if needed

3. **Language-specific post-processing**: Develop rules to fix common errors based on knowledge of Enenlhet morphology

4. **Knowledge transfer**: The approach can be extended to other languages in the Enlhet-Enenlhet family

Would you like me to elaborate on any particular aspect of this workflow? Or would you prefer suggestions on how to improve transcription quality specifically for Enenlhet characteristics?