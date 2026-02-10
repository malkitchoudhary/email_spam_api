import pandas as pd
import nltk
import string
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import os

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# -----------------------------
# TEXT PREPROCESSING FUNCTION
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
# LOAD DATA
# -----------------------------
df = pd.read_csv("spam.csv", encoding="latin1")

df = df.rename(columns={"v1": "target", "v2": "text"})
df = df[["text", "target"]]

df["target"] = df["target"].map({"ham": 0, "spam": 1})

df["transformed_text"] = df["text"].apply(transform_text)

# -----------------------------
# TF-IDF VECTORIZER
# -----------------------------
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df["transformed_text"]).toarray()
y = df["target"].values

# -----------------------------
# TRAIN MODEL
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# -----------------------------
# SAVE MODEL + TFIDF
# -----------------------------
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/spam_model.pkl")
joblib.dump(tfidf, "model/tfidf.pkl")

print("\nModel training complete!")
print("Files saved in /model/")
