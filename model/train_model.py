import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score  # 

# ==============================
# PATH SETUP
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../dataset/news.csv')

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(DATA_PATH, low_memory=False)

# Keep only needed columns
df = df[['title', 'text', 'target']]

# Drop missing values
df = df.dropna()

# ==============================
# CLEAN TARGET COLUMN
# ==============================
df['target'] = df['target'].astype(str).str.strip().str.lower()

# Keep only valid labels
df = df[df['target'].isin(['0', '1', 'real', 'fake'])]

# Convert labels
df['target'] = df['target'].map({
    '0': 0,
    '1': 1,
    'real': 0,
    'fake': 1
})

df['target'] = df['target'].astype(int)

# ==============================
# CREATE TEXT FEATURE
# ==============================
df['content'] = df['title'] + " " + df['text']

X = df['content']
y = df['target']

# ==============================
# SPLIT DATA
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# TEXT TO NUMBERS (TF-IDF)
# ==============================
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==============================
# TRAIN MODEL
# ==============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ==============================
# EVALUATE MODEL
# ==============================
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# ==============================
# SAVE MODEL
# ==============================
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

with open(VECTORIZER_PATH, 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully.")