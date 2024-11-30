import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def save_model_metadata(scaler, sequence_length, X_train_shape, epochs, save_path):
    """
    Save model metadata for consistent future use.
    :param scaler: Scaler object used for normalization.
    :param sequence_length: Length of input sequences.
    :param X_train_shape: Shape of the training data input.
    :param epochs: Number of epochs used during training.
    :param save_path: File path to save the metadata JSON.
    """
    metadata = {
        "input_shape": list(X_train_shape[1:]),
        "scaler_min": scaler.data_min_.tolist(),
        "scaler_max": scaler.data_max_.tolist(),
        "sequence_length": sequence_length,
        "epochs": epochs
    }

    with open(save_path, 'w') as f:
        json.dump(metadata, f)
    print(f"Metadata saved to {save_path}")

if __name__ == "__main__":
    # Example usage: Adjust these parameters to match your model
    scaler = MinMaxScaler()
    scaler.data_min_ = np.array([0, 0, 0, 0])  # Replace with actual scaler min values
    scaler.data_max_ = np.array([50, 100, 30, 20])  # Replace with actual scaler max values
    sequence_length = 10
    X_train_shape = (None, 10, 4)  # Example input shape
    epochs = 20

    save_model_metadata(scaler, sequence_length, X_train_shape, epochs, 'models/weather_forecast/model_metadata.json')
