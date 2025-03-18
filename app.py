from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
data = pd.read_csv("news_dataset.csv")
X = data['text']
# Reinitialize the vectorizer with the same parameters used before
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
vectorizer.fit(X)
# Initialize Flask app
app = Flask(__name__)

# Load trained ML model
model = joblib.load("model.pkl")

# Define API route for predictions
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
    
        if 'news_text' not in data:
            return jsonify({'error': 'Missing key: news_text'}), 400
    
        news_text = data['news_text']
        news_vector = vectorizer.transform([news_text])  # Convert text to numeric vector
        prediction = model.predict(news_vector)
        return jsonify({"prediction": prediction.tolist()})  # Send response
    except Exception as e:
        return jsonify({"error": str(e)})  # Handle errors

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
