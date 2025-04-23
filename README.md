# ML Healthcare System

A Streamlit-based ML healthcare system for disease prediction and hospital resource management.

## Features

- **Disease Prediction**: Uses machine learning models to predict diseases based on patient data
- **Risk Assessment**: Evaluates the risk level and provides confidence scores
- **Medical Recommendations**: Offers personalized medical advice based on the predicted condition
- **Specialist Recommendations**: Suggests appropriate medical specialists for consultation
- **Hospital Management**: Visualizes and manages hospital bed capacity and resource allocation
- **Doctor Finder**: Helps locate specialists based on medical needs

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/ml-healthcare-system.git
   cd ml-healthcare-system
   ```

2. Install the required packages:
   ```
   pip install -r github_requirements.txt
   ```

3. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

## System Architecture

The application is organized into several modules:

- `app.py`: Main Streamlit application with UI components
- `data/`: Contains data generation and processing modules
  - `medical_data.py`: Handles patient data and disease predictions
  - `hospital_data.py`: Manages hospital information and bed availability
- `utils/`: Contains utility functions
  - `data_preprocessing.py`: Preprocesses data for machine learning models
  - `model_training.py`: Trains and manages ML models
  - `prediction.py`: Handles disease prediction logic
  - `recommendation.py`: Generates recommendations based on predictions
  - `hospital_management.py`: Manages hospital resources

## Machine Learning Models

The system uses two primary machine learning models:

1. **Random Forest**: An ensemble learning method that constructs multiple decision trees
2. **XGBoost**: A gradient boosting framework known for its performance and efficiency

## Usage

- Navigate through different sections using the sidebar
- In the Disease Prediction section, input patient information and symptoms
- View detailed predictions with risk levels and recommendations
- Explore hospital bed availability and resource allocation
- Find specialists based on medical needs

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This system is for educational and demonstration purposes only. It should not be used for real medical diagnosis or patient care decisions. Always consult qualified healthcare professionals for medical advice.