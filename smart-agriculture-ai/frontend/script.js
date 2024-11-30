const API_BASE_URL = 'http://localhost:5000';

// Navigation and Page Management
function setupNavigation() {
    const homeTitle = document.getElementById('home-title');
    const soilAnalysisNav = document.getElementById('soil-analysis-nav');
    const weatherPredictionNav = document.getElementById('weather-prediction-nav');
    const dashboardItems = document.querySelectorAll('.dashboard-item');

    function showPage(pageId) {
        const pages = document.querySelectorAll('.page');
        const dashboard = document.getElementById('dashboard');
        
        pages.forEach(page => page.classList.add('hidden'));
        dashboard.classList.add('hidden');
        
        const targetPage = document.getElementById(pageId);
        if (targetPage) {
            targetPage.classList.remove('hidden');
        } else {
            dashboard.classList.remove('hidden');
        }
    }

    homeTitle.addEventListener('click', () => showPage('dashboard'));
    soilAnalysisNav.addEventListener('click', () => showPage('soil-analysis'));
    weatherPredictionNav.addEventListener('click', () => showPage('weather-prediction'));

    dashboardItems.forEach(item => {
        item.addEventListener('click', () => {
            const page = item.getAttribute('data-page');
            showPage(page);
        });
    });
}

// Soil Analysis
function setupSoilAnalysis() {
    const submitButton = document.getElementById('soil-submit-btn');
    const soilInputs = document.querySelectorAll('.soil-input');
    const soilAnalysisLoader = document.getElementById('soil-analysis-loader');
    const soilAnalysisResult = document.getElementById('soil-analysis-result');
    const soilAnalysisError = document.getElementById('soil-analysis-error');
    const fertilityStatus = document.getElementById('fertility-status');

    submitButton.addEventListener('click', async () => {
        // Reset previous state
        soilAnalysisError.textContent = '';
        soilAnalysisError.classList.add('hidden');
        soilAnalysisResult.classList.add('hidden');

        // Collect input values
        const inputValues = Array.from(soilInputs).map(input => {
            const value = parseFloat(input.value);
            if (isNaN(value)) {
                throw new Error(`Invalid input for ${input.name}`);
            }
            return value;
        });

        // Validate inputs
        if (inputValues.length !== 12) {
            soilAnalysisError.textContent = 'Please fill in all soil parameters.';
            soilAnalysisError.classList.remove('hidden');
            return;
        }

        // Show loader
        soilAnalysisLoader.classList.remove('hidden');

        try {
            const response = await fetch(`${API_BASE_URL}/classify`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(inputValues)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Soil analysis failed. Please try again.');
            }

            const data = await response.json();
            
            // Display results
            fertilityStatus.textContent = `Soil Fertility: ${data.classification}`;
            
            soilAnalysisResult.classList.remove('hidden');
        } catch (error) {
            soilAnalysisError.textContent = error.message || 'An unexpected error occurred';
            soilAnalysisError.classList.remove('hidden');
            console.error('Soil analysis error:', error);
        } finally {
            soilAnalysisLoader.classList.add('hidden');
        }
    });
}

// Weather Prediction
function setupWeatherPrediction() {
    const weatherInputs = document.querySelectorAll('.weather-input-field');
    const weatherPredictionBtn = document.getElementById('get-weather-prediction');
    const weatherLoader = document.getElementById('weather-loader');
    const weatherResult = document.getElementById('weather-result');
    const weatherError = document.getElementById('weather-error');
    const temperatureDisplay = document.getElementById('weather-temperature');

    weatherPredictionBtn.addEventListener('click', async () => {
        // Reset previous state
        weatherError.textContent = '';
        weatherError.classList.add('hidden');
        weatherResult.classList.add('hidden');

        // Collect input values
        const inputValues = [];
        for (let i = 0; i < 10; i++) {
            const rowValues = [];
            for (let j = 0; j < 3; j++) {
                const inputIndex = i * 3 + j;
                
                // Check if the input element exists
                const inputElement = weatherInputs[inputIndex];
                if (!inputElement) {
                    weatherError.textContent = `Input field missing at Row ${i + 1}, Column ${j + 1}`;
                    weatherError.classList.remove('hidden');
                    return;
                }

                const value = parseFloat(inputElement.value);
                
                if (isNaN(value)) {
                    weatherError.textContent = `Please fill in all weather input fields (Row ${i+1}, Column ${j+1})`;
                    weatherError.classList.remove('hidden');
                    return;
                }
                
                rowValues.push(value);
            }
            inputValues.push(rowValues);
        }

        // Show loader
        weatherLoader.classList.remove('hidden');

        try {
            const response = await fetch(`${API_BASE_URL}/weather-prediction`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(inputValues)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Weather prediction failed. Please try again.');
            }

            const data = await response.json();
            
            // Display results
            temperatureDisplay.textContent = `Predicted Temperature: ${data.predicted_temperature.toFixed(2)}°C`;
            
            weatherResult.classList.remove('hidden');
        } catch (error) {
            weatherError.textContent = error.message || 'An unexpected error occurred';
            weatherError.classList.remove('hidden');
            console.error('Weather prediction error:', error);
        } finally {
            weatherLoader.classList.add('hidden');
        }
    });
}

// Initialize the application
function init() {
    setupNavigation();
    setupSoilAnalysis();
    setupWeatherPrediction();
}

// Run initialization when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', init);