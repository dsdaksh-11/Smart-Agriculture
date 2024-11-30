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

@app.route("/weather-prediction", methods=["GET"])
def weather_prediction():
    """Weather prediction endpoint"""
    try:
        # Validate model is loaded
        if 'weather' not in MODELS or 'weather' not in SCALERS:
            return jsonify({"error": "Weather prediction model not loaded"}), 500

        # Get location from query parameters
        location = request.args.get('location')
        if not location:
            return jsonify({"error": "Location is required"}), 400

        # In a real-world scenario, you'd fetch location-specific weather data
        # For now, using a placeholder prediction approach
        # You'd replace this with actual feature extraction for your specific model
        sample_features = np.random.rand(1, 10)  # Replace with actual feature extraction
        scaled_features = SCALERS['weather'].transform(sample_features)

        # Make prediction
        prediction = MODELS['weather'].predict(scaled_features)

        return jsonify({
            "temperature": float(prediction[0][0]),
            "precipitation": float(prediction[0][1]),
            "humidity": float(prediction[0][2])
        }), 200

    except Exception as e:
        logger.error(f"Weather prediction error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error during weather prediction"}), 500

if __name__ == "__main__":
    app.run(debug=True)
