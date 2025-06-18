from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load("crop_recommendation_model.pkl")


app = Flask(__name__)

@app.route("/")
def home():
    return "🌾 Crop Recommendation API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    try:
        # Extract features from input JSON
        features = [
            data["N"],
            data["P"],
            data["K"],
            data["temperature"],
            data["humidity"],
            data["ph"],
            data["rainfall"]
        ]

        # Convert to NumPy array
        input_array = np.array([features])

        # Predict using model
        prediction = model.predict(input_array)[0]

        return jsonify({"recommended_crop": prediction})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
