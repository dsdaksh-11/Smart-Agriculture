from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError
from sklearn.preprocessing import MinMaxScaler
import joblib

try:
    scaler_path = "models/weather_forecast/scaler.pkl"  # Update the path if necessary
    scaler = joblib.load(scaler_path)
    print("Scaler loaded successfully.")
except FileNotFoundError:
    print("Error: Scaler file not found. Please check the path or save the scaler during training.")
    scaler = None
except Exception as e:
    print(f"Error loading scaler: {e}")
    scaler = None

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained soil model
soil_model_path = "models/soil_analysis/soil_tabular_model.h5"  # Update path if needed
soil_model = load_model(soil_model_path)

# Load the pre-trained weather model
weather_model_path = "models/weather_forecast/weather_model.h5"
weather_model = load_model(weather_model_path, custom_objects={
    "mse": MeanSquaredError(),
    "mae": MeanAbsoluteError()
})

# Root route to check if the server is running
@app.route("/", methods=["GET"])
def index():
    return "Welcome to the Soil Fertility Classification and Weather Prediction API!"

# Classification route for soil fertility
@app.route("/classify", methods=["POST"])
def classify():
    feature_names = ["N", "P", "K", "pH", "EC", "OC", "S", "Zn", "Fe", "Cu", "Mn", "B"]

    try:
        # Parse input data
        data = request.get_json()
        if not isinstance(data, list) or len(data) != len(feature_names):
            return jsonify({
                "error": f"Input must be an array of {len(feature_names)} values: {feature_names}"
            }), 400

        # Convert data to NumPy array
        features = np.array([float(val) for val in data]).reshape(1, -1)

        # Predict using the soil model
        prediction = soil_model.predict(features)
        classification = "High Fertility" if prediction[0][0] > 0.5 else "Low Fertility"

        return jsonify({
            "classification": classification
        })

    except ValueError:
        return jsonify({"error": "All values in the input array must be numbers."}), 400
    except Exception as e:
        return jsonify({"error": f"Error during classification: {str(e)}"}), 500

# Weather prediction route
@app.route("/predict_weather", methods=["POST"])
def predict_weather():
    """
    Predict future weather conditions based on the provided 10-timestep data.
    """
    try:
        # Get JSON input
        data = request.get_json()

        # Validate input format
        if not isinstance(data, list):
            return jsonify({"error": "Input must be a list of 10 rows of weather data."}), 400

        if len(data) != 10:
            return jsonify({"error": "Input must contain exactly 10 rows of weather data."}), 400

        for row in data:
            if not isinstance(row, list) or len(row) != 3:
                return jsonify({"error": "Each row must be a list with 3 numeric values: [Temperature, Humidity, WindSpeed]."}), 400
            if not all(isinstance(val, (int, float)) for val in row):
                return jsonify({"error": "All values in the input array must be numbers."}), 400

        # Convert data to numpy array
        input_data = np.array(data)
        print(f"Input data shape before reshaping: {input_data.shape}")  # Debugging log

        # Reshape to (1, 10, 3) for LSTM input
        input_data = input_data.reshape(1, 10, 3)  # Ensure correct shape for model

        # Predict using the loaded model
        normalized_prediction = weather_model.predict(input_data)[0][0]  # Extract single value

        # Reverse normalization to get actual temperature
        predicted_temperature = scaler.inverse_transform([[normalized_prediction, 0, 0]])[0][0]

        # Return response
        return jsonify({"predicted_temperature": float(predicted_temperature)})

    except Exception as e:
        # Catch and return any unexpected errors
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    except ValueError:
        return jsonify({"error": "All values in the input array must be numbers."}), 400

if __name__ == "__main__":
    app.run(debug=True)
