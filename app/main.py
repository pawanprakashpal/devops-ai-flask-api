from flask import Flask, request, jsonify
import pickle
import os
load_dotenv()
app = Flask(__name__)

# Load model (dummy model for now)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data.get("features", [])
    prediction = model.predict([features])[0]
    return jsonify({"prediction": prediction})


@app.route("/health", methods=["GET"])
def health():
    return "Healthy", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
