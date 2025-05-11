import noisereduce as nr
from jiwer import wer
import torchaudio
import torchaudio.transforms as T
import torch


bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model()
labels = bundle.get_labels()
sample_rate = bundle.sample_rate

def load_audio(path):
    waveform, sr = torchaudio.load(path)
    return waveform, sr

def reduce_noise(audio, sr):
    print("Reducing noise...")
    y = audio.numpy()[0]
    y_denoised = nr.reduce_noise(y=y, sr=sr)
    return torch.tensor([y_denoised])

def transcribe(audio, sr):
    print("Transcribing...")
    if sr != sample_rate:
        resample = T.Resample(sr, sample_rate)
        audio = resample(audio)

    with torch.inference_mode():
        emissions, _ = model(audio)
        emission = torch.argmax(emissions[0], dim=-1)


    transcript = []
    prev_token = None
    for token in emission:
        if token != prev_token and labels[token] != "|":
            transcript.append(labels[token])
        prev_token = token
    return "".join(transcript).lower()

def evaluate(prediction, reference):
    print("Evaluating...")
    error = wer(reference.lower(), prediction)
    return round(error, 3)

def main():
    ref_text = "this is an example of speech recognition pipeline"
    audio_path = "sample.wav"

    waveform, sr = load_audio(audio_path)
    clean_waveform = reduce_noise(waveform, sr)
    prediction = transcribe(clean_waveform, sr)
    score = evaluate(prediction, ref_text)

    print("\nTranscription Result:\n", prediction)
    print("Reference Text:\n", ref_text)
    print(f"Word Error Rate: {score}")

if __name__ == "__main__":
    main()
