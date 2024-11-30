import requests
import json

# Flask base URL
BASE_URL = "http://127.0.0.1:5000"

# Test Soil Fertility Classification
def test_soil_fertility_classification():
    soil_data = {
        "N": 210,
        "P": 50,
        "K": 300,
        "pH": 6.8,
        "EC": 1.5,
        "OC": 0.9,
        "S": 15,
        "Zn": 1.2,
        "Fe": 5.0,
        "Cu": 1.8,
        "Mn": 1.0,
        "B": 0.5
    }
    try:
        soil_response = requests.post(f"{BASE_URL}/soil-analysis", json=soil_data)
        soil_response.raise_for_status()  # Raise HTTPError for bad responses
        print("Soil Fertility Classification Response:")
        print(soil_response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error during soil classification request: {e}")
    except json.JSONDecodeError:
        print("Error decoding JSON response for soil classification.")

# Test Weather Prediction
def test_weather_prediction():
    # Updated weather data to match the expected format
    weather_data = [25.0, 60.0, 10.0, 5.0]
    try:
        weather_response = requests.post(f"{BASE_URL}/weather-prediction", json=weather_data)
        weather_response.raise_for_status()  # Raise HTTPError for bad responses
        print("\nWeather Prediction Response:")
        print(weather_response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error during weather prediction request: {e}")
    except json.JSONDecodeError:
        print("Error decoding JSON response for weather prediction.")

# Main execution
if __name__ == "__main__":
    print("Testing Soil Fertility Classification...")
    test_soil_fertility_classification()

    print("\nTesting Weather Prediction...")
    test_weather_prediction()
