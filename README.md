# Speech Recognition Pipeline using torchaudio

This project implements a basic Automatic Speech Recognition (ASR) pipeline using PyTorch and torchaudio. It uses a pretrained Wav2Vec2 model to transcribe audio files and evaluate accuracy using the Word Error Rate (WER) metric.

---

## 🔧 Features

- Pretrained Wav2Vec2 ASR model (`torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H`)
- Noise reduction using `noisereduce`
- Resampling for compatible input
- Word Error Rate (WER) evaluation
- Command-line runnable script

---

## 📦 Dependencies

Install the required Python packages using pip:

```bash
pip install torch torchaudio noisereduce jiwer scipy numpy
