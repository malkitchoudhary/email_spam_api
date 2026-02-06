from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # frontend se request allow
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Download nltk data only once

ps = PorterStemmer()

# -----------------------------
# Load Model and Vectorizer
# -----------------------------
model = joblib.load("model/spam_model.pkl")
tfidf = joblib.load("model/tfidf.pkl")

# -----------------------------
# Preprocessing Function
# -----------------------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# -----------------------------
# Request Model
# -----------------------------
class EmailInput(BaseModel):
    text: str


# -----------------------------
# API Home Route
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

    # 🔥 STRONG SPAM KEYWORDS (override)
    spam_keywords = [
        "guaranteed", "cash reward", "reply yes", "urgent",
        "winner", "claim now", "limited offer"
    ]

    text_lower = data.text.lower()

    if any(keyword in text_lower for keyword in spam_keywords):
        result = "SPAM"
        prediction_id = 1
    elif proba >= 0.6:
        result = "SPAM"
        prediction_id = 1
    else:
        result = "NOT SPAM"
        prediction_id = 0

    return {
        "input": data.text,
        "prediction": result,
        "spam_probability": round(proba, 2),
        "prediction_id": prediction_id
    }
