from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

@app.route("/")
def home():
    return "Fake News Detector is running"

# ==============================
# LOAD MODEL + VECTORIZER
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, 'model/model.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'model/vectorizer.pkl')

model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))

# ==============================
# HOME PAGE
# ==============================
@app.route('/')
def home():
    return render_template('index.html')

# ==============================
# PREDICTION
# ==============================
@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']

    # Convert text → vector
    data = vectorizer.transform([news])

    # Prediction
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data).max()

    # Convert output
    if prediction == 0:
        result = "Real News"
    else:
        result = "Fake News"

    confidence = round(probability * 100, 2)

    return render_template(
        'result.html',
        prediction=result,
        confidence=confidence
    )

# ==============================
# RUN APP
# ==============================
if __name__ == '__main__':
    app.run(debug=True, port = 8000)