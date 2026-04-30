from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

import os
import re
import string
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# =========================
# BASE DIR
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# NLTK SETUP
# =========================
NLTK_DATA_DIR = os.path.join(BASE_DIR, "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

ps = PorterStemmer()
FALLBACK_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "or", "that", "the", "to",
    "was", "were", "will", "with", "you", "your", "we", "our", "this", "i",
    "me", "my", "they", "them", "their", "have", "had", "do", "does", "did",
}

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    stop_words = FALLBACK_STOP_WORDS


# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Email Spam Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# LOAD MODEL
# =========================
MODEL_PATH = os.path.join(BASE_DIR, "model", "spam_model.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "model", "tfidf.pkl")

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)


# =========================
# TEXT CLEANING
# =========================
def transform_text(text: str) -> str:
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)

    tokens = [
        ps.stem(t)
        for t in tokens
        if t not in stop_words and t not in string.punctuation
    ]

    return " ".join(tokens)


# =========================
# SCHEMA
# =========================
class EmailInput(BaseModel):
    text: str


# =========================
# FRONTEND
# =========================
@app.get("/", response_class=HTMLResponse)
def home():
    with open(os.path.join(BASE_DIR, "front.html"), encoding="utf-8") as f:
        return f.read()


# =========================
# PREDICT API
# =========================
@app.post("/predict")
def predict(data: EmailInput):
    if not data.text.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "Empty email"},
        )

    processed = transform_text(data.text)
    vector = tfidf.transform([processed])

    if hasattr(model, "predict_proba"):
        spam_prob = float(model.predict_proba(vector)[0][1])
    else:
        spam_prob = float(model.predict(vector)[0])

    result = "SPAM" if spam_prob >= 0.5 else "NOT SPAM"

    return {
        "prediction": result,
        "spam_probability": round(spam_prob, 2),
    }
