# 📰 Fake News Detector

A Machine Learning + Flask based web application that detects whether a news article is **Real** or **Fake** using Natural Language Processing (NLP).

Because apparently the internet needed one more system to verify whether random humans are lying. Historic species behavior.

---

# 📌 Features

- 🔍 Detects Fake or Real News
- ⚡ Fast Prediction System
- 🧠 NLP-based Text Processing
- 🎨 Modern Responsive UI
- ⌨️ Press Enter to Analyze News
- 🔄 Auto Redirect on Page Refresh

---

# 🛠️ Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- HTML5
- CSS3
- NLP (TF-IDF Vectorization)

---

# 📂 Project Structure

```bash
Fake_News_Detector/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│   ├── style.css
│   └── result.css
│
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/sagar2508/Fake_News_Detector.git
```

---

## 2️⃣ Move into Project Folder

```bash
cd Fake_News_Detector
```

---

## 3️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

# 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

Humanity truly looked at dependency conflicts and decided, “this is acceptable.”

---

# ▶️ Run the Application

```bash
python app.py
```

Now open this in your browser:

```bash
http://127.0.0.1:8000
```

---

# 🧠 How It Works

1. User enters a news headline/article.
2. Text is cleaned using NLP preprocessing.
3. TF-IDF Vectorizer converts text into numerical format.
4. Trained ML model predicts:
   - ✅ Real News
   - ❌ Fake News

---

# 📸 Screenshots

## Home Page

<img width="2873" height="1631" alt="Screenshot 2026-05-06 140655" src="https://github.com/user-attachments/assets/ca22ba65-4bb8-40a1-88bf-f3ecd9e58cf1" />


---

## Result Page

<img width="2879" height="1532" alt="Screenshot 2026-05-06 140705" src="https://github.com/user-attachments/assets/000a7ed5-2b3b-41fd-af2c-e8cb59003885" />


---

# 📊 Machine Learning Model

This project uses:

- TF-IDF Vectorizer
- Passive Aggressive Classifier
- NLP preprocessing techniques

The model is trained on Fake and Real news datasets for prediction accuracy.

---
# 👨‍💻 Author

Developed by **Sagar**

GitHub: https://github.com/sagar2508

---
# ⭐ Support

If you liked this project:

- ⭐ Star the repository
- 🍴 Fork the project
- 🧠 Improve the model
- 📰 Help fight misinformation
