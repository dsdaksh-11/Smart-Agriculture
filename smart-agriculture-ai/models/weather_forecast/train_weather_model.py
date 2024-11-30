import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import sys

sys.path.append("D:/Code/Smart-Agriculture/smart-agriculture-ai") # SYSTEM FILE PATH NECESSARY
from scripts.preprocess_weather import load_data, create_sequences

# Build the LSTM model
def build_model(input_shape):
    """
    Build an LSTM model for weather forecasting.
    :param input_shape: Shape of the input data (timesteps, features).
    :return: Compiled LSTM model.
    """
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(1)  # Predict one value (e.g., next day's temperature)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Train the model
def train_model(X_train, y_train, X_val, y_val):
    """
    Train the LSTM model on the weather dataset.
    :param X_train: Training input sequences.
    :param y_train: Training targets.
    :param X_val: Validation input sequences.
    :param y_val: Validation targets.
    :return: Trained model.
    """
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)
    # Save the model without the compilation state
    model.save('models/weather_forecast/weather_model.h5', save_format='h5')

    print("Model saved at 'models/weather_forecast/weather_model.h5'")
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on test data.
    :param model: Trained LSTM model.
    :param X_test: Test input sequences.
    :param y_test: Test targets.
    """
    loss, mae = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, Test MAE: {mae}")

if __name__ == "__main__":
    # Load and preprocess data
    file_path = 'data/weather/weather_data.csv'
    data, scaler = load_data(file_path)
    X, y = create_sequences(data, sequence_length=10)
    
    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Train and evaluate the model
    model = train_model(X_train, y_train, X_val, y_val)
    evaluate_model(model, X_test, y_test)
