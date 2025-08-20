"""
Checks audio files for stereo channels and converts them to mono if necessary.

Usage: Set the audio directory variable and then run `python stereo-check.py`
"""

from pydub import AudioSegment
import os
from glob import glob

# Set this to the path where the .wav files are located
audio_dir = "/Users/sjhuskey/enenlhet-raw-data"

# Loop through all .wav files in the directory
for wav_path in glob(os.path.join(audio_dir, "*.wav")):
    audio = AudioSegment.from_wav(wav_path)

    if audio.channels > 1:
        print(f"⚠️ Stereo detected: {os.path.basename(wav_path)} (channels = {audio.channels})")
        # Convert to mono
        mono_audio = audio.set_channels(1)
        # Overwrite the file with the mono version
        mono_audio.export(wav_path, format="wav")
        print(f"✅ Converted to mono: {os.path.basename(wav_path)}")
    else:
        print(f"✔️ Already mono: {os.path.basename(wav_path)}")
