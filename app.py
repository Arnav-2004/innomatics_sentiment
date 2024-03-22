from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

app = Flask(__name__)

model = joblib.load("model.h5")

try:
    vocabulary = joblib.load("vocabulary.pkl")
except FileNotFoundError:
    print("Error: Vocabulary file not found.")
    exit()

cv = CountVectorizer(max_features=100)

cv.vocabulary_ = vocabulary

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        review_cv = cv.transform([review])
        sentiment = model.predict(review_cv)[0]
        if sentiment == 1:
            result = 'Positive'
        else:
            result = 'Negative'
        return render_template('index.html', review=review, sentiment=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
