import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load and preprocess data
def load_data(file_path):
    """
    Load and normalize weather data.
    :param file_path: Path to the weather data CSV file.
    :return: Normalized data array and the scaler object.
    """
    # Load CSV data
    data = pd.read_csv(file_path)
    
    # Convert 'Date' column to datetime and sort
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)  # Use dayfirst=True for DD/MM/YYYY format
    data.sort_values('Date', inplace=True)
    
    # Select columns to scale
    features = ['Temperature', 'Humidity', 'WindSpeed', 'Visibility']
    data = data[features]  # Exclude non-numeric or non-relevant columns

    # Initialize and fit MinMaxScaler
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)  # Scale the selected columns

    return normalized_data, scaler

# Create sequences for supervised learning
def create_sequences(data, sequence_length=10):
    """
    Convert the dataset into sequences for time-series prediction.
    :param data: Normalized data array.
    :param sequence_length: Number of past timesteps to use for predicting the future.
    :return: Input sequences (X) and target values (y).
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length, :-1])  # Features (exclude target column)
        y.append(data[i + sequence_length, 0])     # Target (e.g., next day's temperature)
    return np.array(X), np.array(y)

# Main execution block
if __name__ == "__main__":
    # Example usage
    file_path = 'data/weather/weather_data.csv'  # Adjust path as needed
    
    try:
        # Load and preprocess data
        data, scaler = load_data(file_path)
        
        # Save the scaler for later use
        joblib.dump(scaler, "models/weather_forecast/scaler.pkl")
        print("Scaler saved successfully.")

        # Create sequences
        X, y = create_sequences(data, sequence_length=10)
        print("Data loaded and preprocessed.")
        print(f"Input shape: {X.shape}, Target shape: {y.shape}")
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
