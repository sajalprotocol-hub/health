import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_patient_data(patient_data, feature_names):
    """
    Preprocess the patient data for model prediction.
    
    Args:
        patient_data (dict): Dictionary containing patient information and test results
        feature_names (list): List of feature names expected by the model
        
    Returns:
        numpy.ndarray: Preprocessed data ready for model prediction
    """
    # Convert patient data dictionary to DataFrame
    patient_df = pd.DataFrame([patient_data])
    
    # Ensure all expected features are present
    for feature in feature_names:
        if feature not in patient_df.columns:
            patient_df[feature] = 0  # Default value for missing features
    
    # Keep only the features expected by the model
    patient_df = patient_df[feature_names]
    
    # Handle categorical features
    categorical_features = ['gender']
    categorical_indices = [feature_names.index(feature) for feature in categorical_features if feature in feature_names]
    
    # Handle numerical features
    numerical_features = [f for f in feature_names if f not in categorical_features]
    numerical_indices = [feature_names.index(feature) for feature in numerical_features if feature in feature_names]
    
    # Create preprocessing pipelines
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_indices),
            ('cat', categorical_transformer, categorical_indices)
        ],
        remainder='passthrough'
    )
    
    # Apply preprocessing
    preprocessed_data = preprocessor.fit_transform(patient_df)
    
    return preprocessed_data

def normalize_vital_signs(patient_data):
    """
    Normalize vital signs based on age and gender.
    
    Args:
        patient_data (dict): Dictionary containing patient information and vital signs
        
    Returns:
        dict: Dictionary with normalized vital signs
    """
    normalized_data = patient_data.copy()
    
    # Age-based adjustment for heart rate
    age = patient_data['age']
    if age < 1:  # Infant
        normal_heart_rate = 120
    elif age < 10:  # Child
        normal_heart_rate = 100
    elif age < 18:  # Adolescent
        normal_heart_rate = 85
    else:  # Adult
        normal_heart_rate = 75
    
    # Adjust heart rate based on normal range for age
    normalized_data['heart_rate_normalized'] = (patient_data['heart_rate'] - normal_heart_rate) / 20
    
    # Age and gender-based adjustment for blood pressure
    gender = patient_data['gender']  # 0: Male, 1: Female
    
    if age < 10:
        normal_systolic = 100
        normal_diastolic = 65
    elif age < 18:
        normal_systolic = 110 if gender == 1 else 115
        normal_diastolic = 70 if gender == 1 else 75
    else:  # Adult
        normal_systolic = 120 if gender == 1 else 125
        normal_diastolic = 80 if gender == 1 else 85
    
    # Adjust blood pressure
    normalized_data['systolic_bp_normalized'] = (patient_data['systolic_bp'] - normal_systolic) / 20
    normalized_data['diastolic_bp_normalized'] = (patient_data['diastolic_bp'] - normal_diastolic) / 10
    
    # Normalize temperature (37°C is normal)
    normalized_data['body_temp_normalized'] = patient_data['body_temp'] - 37.0
    
    # Normalize respiratory rate (adult normal: 12-20)
    normal_resp_rate = 30 if age < 1 else 25 if age < 5 else 20 if age < 12 else 16
    normalized_data['respiratory_rate_normalized'] = (patient_data['respiratory_rate'] - normal_resp_rate) / 5
    
    return normalized_data

def extract_symptom_features(patient_data):
    """
    Extract advanced features from symptom combinations.
    
    Args:
        patient_data (dict): Dictionary containing patient symptoms
        
    Returns:
        dict: Dictionary with additional symptom-based features
    """
    enhanced_data = patient_data.copy()
    
    # Respiratory symptom cluster
    if patient_data.get('cough', 0) and patient_data.get('difficulty_breathing', 0):
        enhanced_data['respiratory_symptoms'] = 1
    else:
        enhanced_data['respiratory_symptoms'] = 0
    
    # Cardiovascular symptom cluster
    if patient_data.get('chest_pain', 0) and patient_data.get('fatigue', 0):
        enhanced_data['cardiovascular_symptoms'] = 1
    else:
        enhanced_data['cardiovascular_symptoms'] = 0
    
    # Gastrointestinal symptom cluster
    if patient_data.get('nausea', 0) and patient_data.get('abdominal_pain', 0):
        enhanced_data['gastrointestinal_symptoms'] = 1
    else:
        enhanced_data['gastrointestinal_symptoms'] = 0
    
    # Symptom severity score (simple count of symptoms)
    symptoms = ['fever', 'cough', 'fatigue', 'difficulty_breathing', 'headache',
                'weight_loss', 'chest_pain', 'nausea', 'abdominal_pain', 
                'jaundice', 'frequent_urination']
    
    symptom_count = sum(patient_data.get(s, 0) for s in symptoms)
    enhanced_data['symptom_severity'] = symptom_count / len(symptoms)  # Normalized count
    
    # Flag for critical combinations
    if (patient_data.get('fever', 0) and 
        patient_data.get('difficulty_breathing', 0) and 
        patient_data.get('chest_pain', 0)):
        enhanced_data['critical_combination'] = 1
    else:
        enhanced_data['critical_combination'] = 0
    
    return enhanced_data

def calculate_risk_factors(patient_data):
    """
    Calculate risk factors based on patient data.
    
    Args:
        patient_data (dict): Dictionary containing patient information
        
    Returns:
        dict: Dictionary with calculated risk factors
    """
    risk_data = patient_data.copy()
    
    # Calculate BMI if height and weight are available
    if 'height' in patient_data and 'weight' in patient_data:
        height_m = patient_data['height'] / 100  # Convert cm to m
        weight_kg = patient_data['weight']
        bmi = weight_kg / (height_m * height_m)
        risk_data['bmi'] = bmi
        
        # BMI risk categories
        if bmi < 18.5:
            risk_data['bmi_risk'] = 'underweight'
        elif bmi < 25:
            risk_data['bmi_risk'] = 'normal'
        elif bmi < 30:
            risk_data['bmi_risk'] = 'overweight'
        else:
            risk_data['bmi_risk'] = 'obese'
    
    # Cardiovascular risk based on cholesterol
    if all(k in patient_data for k in ['cholesterol', 'hdl_cholesterol', 'ldl_cholesterol']):
        total_chol = patient_data['cholesterol']
        hdl = patient_data['hdl_cholesterol']
        ldl = patient_data['ldl_cholesterol']
        
        risk_data['cholesterol_ratio'] = total_chol / hdl if hdl > 0 else float('inf')
        
        # Cholesterol risk assessment
        if risk_data['cholesterol_ratio'] < 3.5:
            risk_data['cholesterol_risk'] = 'low'
        elif risk_data['cholesterol_ratio'] < 5:
            risk_data['cholesterol_risk'] = 'moderate'
        else:
            risk_data['cholesterol_risk'] = 'high'
    
    # Diabetes risk based on glucose
    if 'glucose_level' in patient_data:
        glucose = patient_data['glucose_level']
        
        if glucose < 100:
            risk_data['glucose_risk'] = 'normal'
        elif glucose < 126:
            risk_data['glucose_risk'] = 'prediabetic'
        else:
            risk_data['glucose_risk'] = 'diabetic'
    
    # Age risk category
    age = patient_data['age']
    if age < 18:
        risk_data['age_risk'] = 'youth'
    elif age < 40:
        risk_data['age_risk'] = 'young_adult'
    elif age < 65:
        risk_data['age_risk'] = 'middle_age'
    else:
        risk_data['age_risk'] = 'senior'
    
    return risk_data
