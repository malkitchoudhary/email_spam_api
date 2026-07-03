# 📧 Spam Email Detector


# 📌 Overview

Spam Email Detector is a Machine Learning web application that classifies email messages as **Spam** or **Not Spam (Ham)** using Natural Language Processing (NLP).

The application combines a **Logistic Regression** machine learning model with a **FastAPI backend** and a responsive **HTML/CSS/JavaScript frontend** to provide real-time spam detection.

Users simply enter an email message into the web interface, and the application instantly predicts whether the email is spam or legitimate.

---

# ✨ Features

## 🤖 Machine Learning

- Spam Email Classification
- Text Preprocessing using NLP
- TF-IDF Vectorization
- Logistic Regression Model
- High-Speed Prediction
- Pickle Model Serialization

---

## 🌐 Web Application

- Responsive User Interface
- FastAPI REST API
- HTML, CSS & JavaScript Frontend
- Real-Time Prediction
- Clean Modern Design
- Lightweight and Fast

---

# 🎥 Project Demo

<p align="center">
<img src="spam_detector.gif" width="100%">
</p>

---

# 📸 Screenshots

| Home Page | Prediction |
|-----------|------------|
| ![](images/home.png) | ![](images/prediction.png) |

---

# 🏗️ System Architecture

```text
                    spam.csv Dataset
                           │
                           ▼
                  Data Preprocessing
                           │
                           ▼
                  Text Cleaning (NLP)
                           │
                           ▼
                  TF-IDF Vectorizer
                           │
                           ▼
            Logistic Regression Model
                           │
                           ▼
              Trained Model (.pkl Files)
                           │
                           ▼
                 FastAPI Backend (main.py)
                           │
          ┌────────────────┴────────────────┐
          ▼                                 ▼
     Load ML Model                 Load TF-IDF Vectorizer
          │                                 │
          └────────────────┬────────────────┘
                           ▼
                Spam Prediction API
                           │
                           ▼
              HTML/CSS/JavaScript Frontend
                           │
                           ▼
               Spam / Not Spam Prediction
```

---

# 🛠️ Tech Stack

| Category | Technology |
|-----------|------------|
| Programming Language | Python |
| Backend Framework | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Machine Learning | Scikit-learn |
| NLP | NLTK |
| Feature Extraction | TF-IDF Vectorizer |
| Classification Model | Logistic Regression |
| Data Processing | Pandas, NumPy |
| Model Serialization | Pickle |
| Deployment | Vercel |

---

# 📂 Dataset

The project is trained using the **SMS Spam Collection Dataset**.

Dataset contains:

- Email/SMS Message
- Spam Label
- Ham (Not Spam) Label

---

# 📁 Project Structure

```text
email_spam_api/
│
├── model/
│   ├── logistic_regression.pkl
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
│
├── nltk_data/
│
├── front.html
├── main.py
├── train_model.py
├── spam.csv
├── requirements.txt
├── vercel.json
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/malkitchoudhary/email_spam_api.git
```

### Navigate to the Project Folder

```bash
cd email_spam_api
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the FastAPI Server

```bash
uvicorn main:app --reload
```

Open your browser:

```
http://127.0.0.1:8000
```

API Documentation:

```
http://127.0.0.1:8000/docs
```

---

# 🚀 Application Workflow

### Step 1

Load the trained Logistic Regression model.

↓

### Step 2

Load the saved TF-IDF Vectorizer.

↓

### Step 3

User enters an email message.

↓

### Step 4

FastAPI receives the request.

↓

### Step 5

The email is preprocessed using NLP techniques.

↓

### Step 6

Convert the text into TF-IDF feature vectors.

↓

### Step 7

The Logistic Regression model predicts whether the email is Spam or Not Spam.

↓

### Step 8

The prediction is returned to the frontend.

↓

### Step 9

Display the prediction instantly.

---

# 💡 Example

### Input

```text
Congratulations!

You have won a FREE iPhone.

Click the link below to claim your prize.
```

### Output

```text
Prediction : Spam 🚨
```

---

### Input

```text
Hi John,

Can we meet tomorrow at 5 PM?
```

### Output

```text
Prediction : Not Spam ✅
```

---

# 🔮 Future Improvements

- Deep Learning (LSTM/Bi-LSTM)
- BERT-Based Spam Detection
- Email Attachment Analysis
- Explainable AI (SHAP)
- Multilingual Spam Detection
- User Authentication
- Spam Score Percentage
- Cloud Deployment
- Docker Support
- Email API Integration

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository.

2. Create a feature branch.

```bash
git checkout -b feature-name
```

3. Commit your changes.

```bash
git commit -m "Added new feature"
```

4. Push to your branch.

```bash
git push origin feature-name
```

5. Open a Pull Request.

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Malkit Choudhary**

- 💼 Aspiring Data Scientist
- 🤖 Machine Learning Enthusiast
- 🐍 Python Developer

---

# ⭐ Support

If you found this project useful, please consider giving it a **⭐ Star** on GitHub.

Your support motivates me to build more open-source Machine Learning projects.
