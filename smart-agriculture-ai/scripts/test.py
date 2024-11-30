import numpy as np
from tensorflow.keras.models import load_model
import joblib
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError

# Load the model with custom objects
model = load_model(
    'models/weather_forecast/weather_model.h5',
    custom_objects={
        "mse": MeanSquaredError(),  # Alias for 'mse'
        "mae": MeanAbsoluteError()  # Alias for 'mae' if necessary
    }
)


# Load the model and scaler
model = load_model('models/weather_forecast/weather_model.h5')
scaler = joblib.load('models/weather_forecast/scaler.pkl')

# Create a test input (shape: 1, 10, 4)
test_input = np.array([
    [25.0, 60.0, 10.0, 5.0],
    [24.5, 58.0, 9.0, 5.2],
    [26.0, 62.0, 12.0, 4.8],
    [25.5, 61.0, 10.5, 5.0],
    [27.0, 65.0, 11.0, 5.1],
    [26.5, 63.0, 9.8, 4.9],
    [28.0, 68.0, 13.0, 5.3],
    [27.5, 66.0, 12.5, 4.7],
    [29.0, 70.0, 14.0, 5.0],
    [28.5, 68.5, 13.2, 4.9]
]).reshape(1, 10, 4)

# Predict using the model
prediction = model.predict(test_input)
print(f"Normalized Prediction: {prediction}")

# Reverse scaling if needed
if scaler:
    unscaled = scaler.inverse_transform([[prediction[0][0], 0, 0, 0]])
    print(f"Unscaled Prediction: {unscaled[0][0]}")
else:
    print("Scaler not available for inverse transformation.")