import requests
import json

# Flask base URL
BASE_URL = "http://127.0.0.1:5000"

# Test Soil Fertility Classification
def test_soil_fertility_classification():
    soil_data = [210, 50, 300, 6.8, 1.5, 0.9, 15, 1.2, 5.0, 1.8, 1.0, 0.5]
    try:
        soil_response = requests.post(f"{BASE_URL}/classify", json=soil_data)
        soil_response.raise_for_status()  # Raise HTTPError for bad responses
        print("Soil Fertility Classification Response:")
        print(soil_response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error during soil classification request: {e}")
    except json.JSONDecodeError:
        print("Error decoding JSON response for soil classification.")

# Test Weather Prediction
def test_weather_prediction():
    weather_data = [
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
    ]
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
