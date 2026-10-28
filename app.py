
from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__)

# Paths to the trained model and vectorizer
MODEL_PATH = "model/sentiment_model.pkl"
VECTORIZER_PATH = "model/tfidf_vectorizer.pkl"

# Load the trained model
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(
        "Model files not found. Run 'python train_model.py' first."
    )

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def predict_sentiment(review):
    # Convert review text into TF-IDF features
    review_vector = vectorizer.transform([review])

    # Predict sentiment
    prediction = model.predict(review_vector)[0]

    # Get prediction confidence, if supported
    probabilities = model.predict_proba(review_vector)[0]
    confidence = round(float(max(probabilities)) * 100, 2)

    return str(prediction).capitalize(), confidence


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    review = data.get("review", "")

    if not isinstance(review, str) or not review.strip():
        return jsonify({"error": "Please provide a valid review."}), 400

    if len(review) > 10000:
        return jsonify({"error": "Review must be 10,000 characters or fewer."}), 400

    sentiment, confidence = predict_sentiment(review.strip())

    return jsonify({
        "review": review.strip(),
        "sentiment": sentiment,
        "confidence": confidence
    })


@app.route("/predict", methods=["POST"])
def predict():
    review = request.form.get("review", "").strip()

    if not review:
        return render_template(
            "index.html",
            error="Please enter a review."
        )

    if len(review) > 10000:
        return render_template(
            "index.html",
            error="Review must be 10,000 characters or fewer."
        )

    sentiment, confidence = predict_sentiment(review)

    return render_template(
        "index.html",
        review=review,
        sentiment=sentiment,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)
