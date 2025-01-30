import nltk
from flask import Flask, request, jsonify, render_template_string
import joblib
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Initialize stopwords and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Cleaning and stemming function
def cleaning_and_stemming(text):
    text = BeautifulSoup(text, 'html.parser').get_text()  # Remove HTML tags
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    # Remove stopwords
    text = " ".join(word for word in text.split() if word not in stop_words)
    # Perform stemming
    text = ' '.join(stemmer.stem(wrd) for wrd in text.split())
    return text

# HTML template for user input
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Sentiment Analysis</title>
</head>
<body>
    <h1>Sentiment Analysis</h1>
    <form method="post" action="/predict">
        <label for="review_text">Enter your review:</label><br>
        <textarea id="review_text" name="review_text" rows="4" cols="50"></textarea><br><br>
        <input type="submit" value="Analyze Sentiment">
    </form>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(html_template)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data
    review_text = request.form.get("review_text", "")  # For form input
    if not review_text:
        return jsonify({"error": "No review text provided!"}), 400

    # Preprocess the review text
    cleaned_text = cleaning_and_stemming(review_text)
    review_vector = vectorizer.transform([cleaned_text])

    # Predict sentiment
    prediction = model.predict(review_vector)
    sentiment = "positive" if prediction[0] == 1 else "negative"

    return jsonify({"review_text": review_text, "sentiment_prediction": sentiment})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
