from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

import os
import string
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# ============================
# BASE DIRECTORY
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================
# NLTK DATA SETUP
# ============================
NLTK_DATA_DIR = os.path.join(BASE_DIR, "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)


# ============================
# FASTAPI APP
# ============================
app = FastAPI(title="Email Spam Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================
# DOWNLOAD NLTK DATA
# ============================
@app.on_event("startup")
def download_nltk():
    nltk.download("punkt", download_dir=NLTK_DATA_DIR)
    nltk.download("stopwords", download_dir=NLTK_DATA_DIR)


# ============================
# LOAD MODEL & TFIDF
# ============================
MODEL_PATH = os.path.join(BASE_DIR, "model", "spam_model.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "model", "tfidf.pkl")

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))


# ============================
# TEXT PREPROCESSING
# ============================
def transform_text(text: str) -> str:
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    tokens = [t for t in tokens if t.isalnum()]
    tokens = [
        ps.stem(t)
        for t in tokens
        if t not in stop_words and t not in string.punctuation
    ]

    return " ".join(tokens)


# ============================
# REQUEST SCHEMA
# ============================
class EmailInput(BaseModel):
    text: str


# ============================
# FRONTEND ROUTE
# ============================
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    with open(os.path.join(BASE_DIR, "front.html"), encoding="utf-8") as f:
        return f.read()


# ============================
# PREDICTION ROUTE
# ============================
@app.post("/predict")
def predict(data: EmailInput):
    if not data.text.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "Empty email text"},
        )

    processed = transform_text(data.text)
    vector = tfidf.transform([processed])

    if hasattr(model, "predict_proba"):
        spam_prob = float(model.predict_proba(vector)[0][1])
    else:
        spam_prob = float(model.predict(vector)[0])

    prediction = "SPAM" if spam_prob >= 0.5 else "NOT SPAM"

    return {
        "prediction": prediction,
        "spam_probability": round(spam_prob, 2),
    }
