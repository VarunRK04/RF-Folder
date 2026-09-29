import logging
import os

from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_TEXT_LENGTH = 20_000  # characters; guards against pathological/oversized payloads

# Initialize Flask app
app = Flask(__name__)
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*")
CORS(app, origins=allowed_origins.split(",") if allowed_origins != "*" else "*")

# Load trained pipeline (TF-IDF vectorizer + classifier bundled together)
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")
model = joblib.load(MODEL_PATH)
logger.info("Loaded model from %s", MODEL_PATH)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    if "news_text" not in payload:
        return jsonify({"error": "Missing key: news_text"}), 400

    news_text = payload["news_text"]
    if not isinstance(news_text, str) or not news_text.strip():
        return jsonify({"error": "news_text must be a non-empty string"}), 400

    if len(news_text) > MAX_TEXT_LENGTH:
        return jsonify({"error": f"news_text exceeds max length of {MAX_TEXT_LENGTH} characters"}), 400

    try:
        prediction = model.predict([news_text])[0]
        response = {"prediction": prediction}

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([news_text])[0]
            response["confidence"] = {
                cls: round(float(p), 4) for cls, p in zip(model.classes_, proba)
            }

        return jsonify(response)
    except Exception:
        logger.exception("Prediction failed")
        return jsonify({"error": "Internal error while generating prediction"}), 500


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=debug, host=host, port=port)
