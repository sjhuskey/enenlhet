import os
import json
from glob import glob
from pydub import AudioSegment
import xml.etree.ElementTree as ET
from datasets import load_dataset, Audio, ClassLabel, Dataset, DatasetDict
from datasets import load_from_disk
from collections import Counter
from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor

# --- Configuration ---
data_dir = "/Users/sjhuskey/enenlhet-raw-data"  # replace with your folder path
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
valid_tiers = [
    'transcript_MR', 'Transcript', 'transcript_ER', 'transcript_LF',
    'transcript_LM', 'transcript_MM', 'transcript_TF', 'transcript_SSA',
    'transcripcion_CA', 'transcript_PA', 'Transcripcion', 'AR_transcripcion',
    'transcripcion', 'FF_Transcripcion', 'Transcripcion_FF'
]
min_duration = 2.0  # seconds
max_duration = 20.0  # seconds

# --- Helper Function ---
def parse_eaf(eaf_path):
    tree = ET.parse(eaf_path)
    root = tree.getroot()
    ns = {'elan': 'http://www.w3.org/2001/XMLSchema-instance'}

    time_slots = {}
    for ts in root.findall(".//TIME_SLOT"):
        time_id = ts.attrib['TIME_SLOT_ID']
        time_value = int(ts.attrib.get('TIME_VALUE', 0))
        time_slots[time_id] = time_value / 1000  # convert to seconds

    annotations = []
    for tier in root.findall(".//TIER"):
        tier_name = tier.attrib.get("TIER_ID", "")
        if tier_name not in valid_tiers:
            continue
        for annotation in tier.findall(".//ANNOTATION/ALIGNABLE_ANNOTATION"):
            start = time_slots.get(annotation.attrib['TIME_SLOT_REF1'])
            end = time_slots.get(annotation.attrib['TIME_SLOT_REF2'])
            if not start or not end:
                continue
            duration = end - start
            if duration > max_duration:
                print(f"Skipping long annotation in {os.path.basename(eaf_path)}: {duration:.2f}s")
                continue
            if duration < min_duration or duration > max_duration:
                continue
            text_elem = annotation.find("ANNOTATION_VALUE")
            if text_elem is None or not text_elem.text or not text_elem.text.strip():
                continue
            annotations.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text_elem.text.strip()
            })
    return annotations

# --- Main Extraction ---
entries = []
all_eafs = sorted(glob(os.path.join(data_dir, "*.eaf")))

for eaf_path in all_eafs:
    base = os.path.splitext(os.path.basename(eaf_path))[0]
    wav_path = os.path.join(data_dir, base + ".wav")
    if not os.path.exists(wav_path):
        print(f"Missing audio for {base}")
        continue

    try:
        audio = AudioSegment.from_wav(wav_path)
        audio_duration = len(audio) / 1000
    except Exception as e:
        print(f"Error reading {wav_path}: {e}")
        continue

    for ann in parse_eaf(eaf_path):
        if ann['end'] > audio_duration:
            continue
        entries.append({
            "audio": {
                "path": os.path.abspath(wav_path),
                "start": ann['start'],
                "end": ann['end']
            },
            "text": ann['text']
        })

print(f"Extracted {len(entries)} segments from {len(all_eafs)} files.")

# --- Extract Physical Audio Segments (Like Your Colleague) ---
import os
from pydub import AudioSegment

# Create output directory for segmented audio
segment_dir = "/Users/sjhuskey/enenlhet-segmented-audio-whisper"
os.makedirs(segment_dir, exist_ok=True)

# Convert entries to actual segmented files
segmented_entries = []
processed_files = {}  # Cache loaded audio files

print("Extracting physical audio segments...")
for i, entry in enumerate(entries):
    if i % 100 == 0:
        print(f"Processing segment {i}/{len(entries)}")
    
    wav_path = entry["audio"]["path"]
    start_sec = entry["audio"]["start"]
    end_sec = entry["audio"]["end"]
    text = entry["text"]
    
    # Load audio file (cache it)
    if wav_path not in processed_files:
        try:
            processed_files[wav_path] = AudioSegment.from_wav(wav_path)
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            continue
    
    audio = processed_files[wav_path]
    
    # Extract the specific segment
    start_ms = int(start_sec * 1000)
    end_ms = int(end_sec * 1000)
    segment = audio[start_ms:end_ms]
    
    # Create a unique filename for this segment
    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    segment_filename = f"{base_name}_{start_sec}_{end_sec}.wav"
    segment_path = os.path.join(segment_dir, segment_filename)
    
    # Export the segment
    segment.export(segment_path, format="wav")
    
    # Create new entry pointing to the segmented file
    segmented_entries.append({
        "audio": {
            "path": os.path.abspath(segment_path)
            # No start/end needed - the file IS the segment
        },
        "text": text
    })

print(f"Extracted {len(segmented_entries)} physical audio segments")

# Replace the original entries
entries = segmented_entries

output_jsonl = "/Users/sjhuskey/enenlhet-raw-data/enenlhet-whisper-segmented.jsonl"

with open(output_jsonl, "w", encoding="utf-8") as f:
    for entry in entries:
        json.dump({
            "audio": entry["audio"]["path"],
            "text": entry["text"]
        }, f, ensure_ascii=False)
        f.write("\n")

print(f"Saved Whisper-style segmented JSONL to {output_jsonl}")

# Check the size difference
import subprocess
result = subprocess.run(['du', '-sh', segment_dir], capture_output=True, text=True)
if result.returncode == 0:
    print(f"Total segmented audio size: {result.stdout.strip().split()[0]}")
    
# Show a sample entry
print(f"\nSample segmented entry:")
print(f"Audio path: {entries[0]['audio']['path']}")
print(f"Text: {entries[0]['text']}")

# Check if a sample file exists
sample_path = entries[0]['audio']['path']
if os.path.exists(sample_path):
    segment = AudioSegment.from_wav(sample_path)
    duration = len(segment) / 1000
    print(f"Sample segment duration: {duration:.2f} seconds")

print("Loading dataset...")
dataset = load_dataset("json", data_files="/Users/sjhuskey/enenlhet-raw-data/enenlhet-whisper-segmented.jsonl", split="train")
# Convert audio paths to Audio objects
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Split the dataset into train, test, and validation sets
print("Splitting dataset into train, test, and validation sets...")
train_test_split = dataset.train_test_split(test_size=0.1)
val_test = train_test_split['test'].train_test_split(test_size=0.5)

# Make a dataset dictionary
dataset = DatasetDict({
    "train": train_test_split["train"],
    "test": val_test["test"],
    "validation": val_test["train"]
})

print(f"Dataset sizes: train={len(dataset['train'])}, test={len(dataset['test'])}, validation={len(dataset['validation'])}")

# Load processor, which includes both the feature extractor and tokenizer
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
processor.tokenizer.set_prefix_tokens(language=None, task=None)

def prepare_dataset(batch):
    audio = batch["audio"]

    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # Disable special/timestamp tokens
    batch["labels"] = processor.tokenizer(
        batch["text"],
        truncation=True,
        # This is important to avoid adding special tokens like <|startoftranscript|>
        add_special_tokens=False,
        return_tensors="pt"
    ).input_ids[0]

    return batch

print("Preparing dataset for training...")
dataset["train"] = dataset["train"].map(prepare_dataset, remove_columns=["audio", "text"], num_proc=4)
print("Dataset prepared for training.")
print("Preparing test dataset...")
dataset["test"] = dataset["test"].map(prepare_dataset, remove_columns=["audio", "text"], num_proc=4)
print("Test dataset prepared.")
print("Preparing validation dataset...")
dataset["validation"] = dataset["validation"].map(prepare_dataset, remove_columns=["audio", "text"], num_proc=4)
print("Validation dataset prepared.")
print("Dataset preparation complete.")

dataset.save_to_disk("/Users/sjhuskey/enenlhet-whisper-dataset")
print("Dataset saved to /Users/sjhuskey/enenlhet-whisper-dataset")