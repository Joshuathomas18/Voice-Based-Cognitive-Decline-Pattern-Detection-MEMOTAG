# 🧠 Cognitive Impairment Detection API

This API uses acoustic and NLP features from speech to detect signs of cognitive impairment.

## 🚀 How it works:
- Accepts `.wav` audio files via a POST request to `/predict`
- Extracts pitch, jitter, shimmer, MFCCs, lexical richness, hesitations, and semantic coherence.
- Predicts impairment using a trained Random Forest model.

## 📦 Features used:
- Acoustic: pitch, jitter, shimmer, MFCCs, duration, intensity
- NLP: type-token ratio, hesitations, vague terms, sentence coherence

## 💡 Usage:
Deploy on Render/Heroku and test via Postman or `curl`:
```bash
curl -X POST -F 'file=@audio.wav' https://your-app-url.onrender.com/predict
```
