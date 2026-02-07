from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import joblib

import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import os
import nltk

# 🔥 Tell NLTK where to store data (Render-compatible)
NLTK_DATA_DIR = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)


# ✅ CREATE APP ONLY ONCE
app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ NLTK DOWNLOAD ON STARTUP (PRODUCTION SAFE)
@app.on_event("startup")
def download_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", download_dir=NLTK_DATA_DIR)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", download_dir=NLTK_DATA_DIR)


# -----------------------------
# Load Model and Vectorizer
# -----------------------------
model = joblib.load("model/spam_model.pkl")
tfidf = joblib.load("model/tfidf.pkl")

ps = PorterStemmer()

# -----------------------------
# Text Preprocessing
# -----------------------------
def transform_text(text: str) -> str:
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    tokens = [i for i in tokens if i.isalnum()]
    tokens = [
        ps.stem(i)
        for i in tokens
        if i not in stopwords.words("english")
        and i not in string.punctuation
    ]

    return " ".join(tokens)

# -----------------------------
# Request Model
# -----------------------------
class EmailInput(BaseModel):
    text: str

# -----------------------------
# Frontend Route
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    with open("fronted/front.html", "r", encoding="utf-8") as f:
        return f.read()

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(data: EmailInput):
    transformed = transform_text(data.text)
    vector = tfidf.transform([transformed]).toarray()
    proba = model.predict_proba(vector)[0][1]

    spam_keywords = [
        "guaranteed",
        "cash reward",
        "reply yes",
        "urgent",
        "winner",
        "claim now",
        "limited offer",
        "call now",
    ]

    text_lower = data.text.lower()

    if any(k in text_lower for k in spam_keywords) or proba >= 0.6:
        result = "SPAM"
        prediction_id = 1
    else:
        result = "NOT SPAM"
        prediction_id = 0

    return {
        "prediction": result,
        "spam_probability": round(proba, 2),
        "prediction_id": prediction_id,
    }
