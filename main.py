from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import os
import string
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# ==================================================
# NLTK DATA PATH (RENDER SAFE)
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NLTK_DATA_DIR = os.path.join(BASE_DIR, "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)


# =================================================
# FASTAPI APP (ONLY ONCE)
# ==================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# NLTK DOWNLOAD ON STARTUP (ONE TIME)
# ==================================================
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


# ==================================================
# LOAD MODEL & VECTORIZER
# ==================================================
model = joblib.load(os.path.join(BASE_DIR, "model", "spam_model.pkl"))
tfidf = joblib.load(os.path.join(BASE_DIR, "model", "tfidf.pkl"))

ps = PorterStemmer()
STOPWORDS = set(stopwords.words("english"))


# ==================================================
# TEXT PREPROCESSING
# ==================================================
def transform_text(text: str) -> str:
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    cleaned = []
    for word in tokens:
        if word.isalnum() and word not in STOPWORDS:
            cleaned.append(ps.stem(word))

    return " ".join(cleaned)


# ==================================================
# REQUEST SCHEMA
# ==================================================
class EmailInput(BaseModel):
    text: str


# ==================================================
# FRONTEND ROUTE
# ==================================================
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    with open(os.path.join(BASE_DIR, "fronted", "front.html"), "r", encoding="utf-8") as f:
        return f.read()


# ==================================================
# PREDICTION ROUTE (CRASH-PROOF)
# ==================================================
@app.post("/predict")
def predict(data: EmailInput):
    transformed = transform_text(data.text)
    vector = tfidf.transform([transformed]).toarray()

    # 🔒 SAFE probability handling
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vector)[0][1]
    else:
        pred = model.predict(vector)[0]
        proba = float(pred)

    spam_keywords = [
        "urgent",
        "winner",
        "cash",
        "reward",
        "guaranteed",
        "claim",
        "reply yes",
        "call now",
        "limited offer",
    ]

    text_lower = data.text.lower()

    if any(k in text_lower for k in spam_keywords) or proba >= 0.6:
        return {
            "prediction": "SPAM",
            "spam_probability": round(proba, 2),
            "prediction_id": 1,
        }
    else:
        return {
            "prediction": "NOT SPAM",
            "spam_probability": round(proba, 2),
            "prediction_id": 0,
        }
