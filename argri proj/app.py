from flask import Flask
import joblib
import numpy as np

# Load the trained model
model = joblib.load("crop_recommendation_model.pkl")


app = Flask(__name__)

@app.route("/")
def home():
    return "🌾 Crop Recommendation API is running!"



if __name__ == "__main__":
    app.run(debug=True)
