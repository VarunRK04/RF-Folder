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
        data = request.json["features"]  # Extract features from request
        prediction = model.predict([np.array(data)])  # Make prediction
        return jsonify({"prediction": prediction.tolist()})  # Send response
    except Exception as e:
        return jsonify({"error": str(e)})  # Handle errors

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
