import torch
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa

# --- Load Model and Processor ---
MODEL_PATH = "/Users/saya0001/consultations/huskey/enelhet/wav2vec_enelhet"
processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)
model.eval()


# --- Transcription Function ---
def transcribe_audio(audio_file_path):
    """
    Transcribes an audio file using the fine-tuned Wav2Vec2 model.

    Args:
        audio_file_path (str): The path to the audio file to be transcribed.

    Returns:
        str: The transcribed text.
    """
    # Load the audio file and resample it to 16kHz
    speech_array, sampling_rate = sf.read(audio_file_path)
    if sampling_rate != 16000:
        speech_array = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=16000)

    # Process the audio file
    inputs = processor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True)

    # Move inputs to the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.to(device)

    # Get the model's prediction
    with torch.no_grad():
        logits = model(**inputs).logits

    # Decode the prediction
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    return transcription[0]


if __name__ == "__main__":
    # Replace with the path to your test audio file
    audio_path = "/Users/saya0001/consultations/huskey/enelhet/DEMO.wav"

    # Check if the file exists
    import os

    if not os.path.exists(audio_path):
        print(f"Error: The file '{audio_path}' was not found.")
        print("Please replace 'path/to/your/test_audio.wav' with the actual path to your audio file.")
    else:
        # Transcribe the audio file and print the result
        transcribed_text = transcribe_audio(audio_path)
        print("Transcription:", transcribed_text)