from flask import Flask, request, jsonify
import joblib
import numpy as np

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
    
        # Corrected part: Make sure input is 2D
        news_vector = vectorizer.transform([news_text])  # Extract features from request
            prediction = model.predict([np.array(data)])  # Make prediction
            return jsonify({"prediction": prediction.tolist()})  # Send response
    except Exception as e:
        return jsonify({"error": str(e)})  # Handle errors

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
