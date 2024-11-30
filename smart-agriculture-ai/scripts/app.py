from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging
import os
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)

# Model and Scaler Loading with Robust Error Handling
def load_models_and_scalers():
    models = {}
    scalers = {}

    try:
        # Soil Analysis Model
        soil_model_path = "models/soil_analysis/soil_tabular_model.h5"
        if os.path.exists(soil_model_path):
            models['soil'] = load_model(soil_model_path)
            logger.info(f"Soil analysis model loaded successfully from {soil_model_path}")
        else:
            logger.error(f"Soil model not found at {soil_model_path}")

        # Soil Analysis Scaler
        soil_scaler_path = "models/soil_analysis/scaler.pkl"
        if os.path.exists(soil_scaler_path):
            scalers['soil'] = joblib.load(soil_scaler_path)
            logger.info(f"Soil scaler loaded successfully from {soil_scaler_path}")
        else:
            logger.error(f"Soil scaler not found at {soil_scaler_path}")

        # Weather Forecast Model
        weather_model_path = "models/weather_forecast/weather_model.h5"
        if os.path.exists(weather_model_path):
            models['weather'] = load_model(
                weather_model_path, 
                custom_objects={
                    "mse": MeanSquaredError(),
                    "mae": MeanAbsoluteError()
                }
            )
            logger.info(f"Weather model loaded successfully from {weather_model_path}")
        else:
            logger.error(f"Weather model not found at {weather_model_path}")

        # Weather Scaler
        weather_scaler_path = "models/weather_forecast/scaler.pkl"
        if os.path.exists(weather_scaler_path):
            scalers['weather'] = joblib.load(weather_scaler_path)
            logger.info(f"Weather scaler loaded successfully from {weather_scaler_path}")
        else:
            logger.error(f"Weather scaler not found at {weather_scaler_path}")

    except Exception as e:
        logger.error(f"Error loading models/scalers: {e}")
        logger.error(traceback.format_exc())
        models = {}
        scalers = {}

    return models, scalers

# Load models and scalers at startup
MODELS, SCALERS = load_models_and_scalers()

@app.route("/", methods=["GET"])
def index():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": {
            "soil_analysis": "soil" in MODELS,
            "weather_prediction": "weather" in MODELS
        }
    }), 200

@app.route("/soil-analysis", methods=["POST"])
def soil_analysis():
    """Soil fertility analysis endpoint"""
    try:
        # Validate model is loaded
        if 'soil' not in MODELS or 'soil' not in SCALERS:
            return jsonify({"error": "Soil analysis model not loaded"}), 500

        # Extract features from request
        # Assuming JSON input with soil parameters
        data = request.json
        if not data:
            return jsonify({"error": "No soil data provided"}), 400

        # Required feature names
        feature_names = ["N", "P", "K", "pH", "EC", "OC", "S", "Zn", "Fe", "Cu", "Mn", "B"]
        
        # Validate input features
        if not all(feature in data for feature in feature_names):
            return jsonify({"error": "Missing required soil features"}), 400

        # Prepare features
        features = [data[feature] for feature in feature_names]
        features_array = np.array([features])

        # Scale features
        scaled_features = SCALERS['soil'].transform(features_array)

        # Make prediction
        prediction = MODELS['soil'].predict(scaled_features)
        
        # Convert prediction to interpretable result
        fertility_status = "High" if prediction[0][0] > 0.5 else "Low"
        confidence = float(prediction[0][0] * 100)

        return jsonify({
            "fertility_status": fertility_status,
            "confidence": confidence
        }), 200

    except Exception as e:
        logger.error(f"Soil analysis error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error during soil analysis"}), 500

@app.route("/weather-prediction", methods=["POST"])
def weather_prediction():
    """Weather prediction endpoint (1,1,4)"""
    try:
        # Validate model is loaded
        if 'weather' not in MODELS or 'weather' not in SCALERS:
            return jsonify({"error": "Weather prediction model not loaded"}), 500

        # Extract features from request
        data = request.json
        if isinstance(data, dict):  # If `data` is a dictionary, extract "data"
            data = data.get("data", None)
        elif isinstance(data, list):  # If it's a list, treat it as raw input
            pass
        else:
            return jsonify({"error": "Invalid input format. Expected a JSON array or object."}), 400

        # Validate the input size
        if not data or not isinstance(data, list) or len(data) != 4:
            return jsonify({
                "error": f"Invalid input size. Expected a JSON array with exactly 4 values, but got {len(data)} values."
            }), 400

        # Convert to numpy array and reshape to (1, 1, 4)
        input_array = np.array(data).reshape(1, 1, 4)

        # Normalize input using the scaler
        scaled_features = SCALERS['weather'].transform(input_array.reshape(-1, 4))
        scaled_features = scaled_features.reshape(1, 1, 4)

        # Make prediction
        prediction = MODELS['weather'].predict(scaled_features)

        # Return the predicted temperature
        return jsonify({
            "predicted_temperature": float(prediction[0][0])
        }), 200

    except Exception as e:
        logger.error(f"Weather prediction error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error during weather prediction"}), 500


if __name__ == "__main__":
    app.run(debug=True)
