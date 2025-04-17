# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn, tempfile, shutil
import numpy as np
import parselmouth
from parselmouth.praat import call
import librosa
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer, util
import joblib

nltk.download('punkt')
model = SentenceTransformer('all-MiniLM-L6-v2')
clf = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI(title="Cognitive Impairment Detection API")

def extract_acoustic_features(file_path):
    snd = parselmouth.Sound(file_path)
    pitch = call(snd, "To Pitch", 0.0, 75, 600)
    mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)
    jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    duration = snd.get_total_duration()
    intensity = call(snd, "To Intensity", 75, 0.0)
    mean_intensity = call(intensity, "Get mean", 0, 0, "energy")
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfccs.mean(axis=1)
    return mean_pitch, jitter, shimmer, duration, mean_intensity, mfcc_mean

def extract_text_features(transcript):
    sentences = sent_tokenize(transcript)
    words = word_tokenize(transcript)
    ttr = len(set(words)) / len(words) if words else 0
    hesitations = sum(1 for w in words if w.lower() in ["uh", "um", "erm"])
    avg_words_sent = len(words) / len(sentences) if sentences else 0
    recall_issues = sum(1 for w in words if w.lower() in ["thing", "stuff", "what's it called"])
    return ttr, hesitations, avg_words_sent, recall_issues

def semantic_coherence(text):
    sents = sent_tokenize(text)
    if len(sents) < 2:
        return 1.0
    embeddings = model.encode(sents)
    sim_scores = [util.cos_sim(embeddings[i], embeddings[i+1]).item() for i in range(len(sents)-1)]
    return np.mean(sim_scores)

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        transcript = "This is a test case. The patient seems hesitant."
        pitch, jitter, shimmer, duration, intensity, mfcc_mean = extract_acoustic_features(tmp_path)
        ttr, hesitations, avg_words_sent, recall_issues = extract_text_features(transcript)
        semantic = semantic_coherence(transcript)

        row = [pitch, jitter, shimmer, duration, intensity, ttr, hesitations, avg_words_sent, recall_issues, semantic] + list(mfcc_mean)
        row = np.array(row).reshape(1, -1)
        row_scaled = scaler.transform(row)
        pred = clf.predict(row_scaled)[0]
        return {"prediction": int(pred)}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
