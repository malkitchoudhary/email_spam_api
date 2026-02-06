from fastapi.middleware.cors import CORSMiddleware

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
nltk.download('punkt')
nltk.download('stopwords')

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
@app.get("/")
def home():
    return {"message": "Email Spam Detection API Working Successfully!"}


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(data: EmailInput):
    # Preprocess
    transformed = transform_text(data.text)
    vector = tfidf.transform([transformed]).toarray()

    # Predict
    prediction = model.predict(vector)[0]

    result = "SPAM" if prediction == 1 else "NOT SPAM"

    return {
        "input": data.text,
        "prediction": result,
        "prediction_id": int(prediction)
    }
