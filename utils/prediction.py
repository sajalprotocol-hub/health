import numpy as np
import pandas as pd

def predict_disease(processed_data, models, model_choice="Random Forest"):
    """
    Predict disease based on processed patient data using the selected model.
    Enhanced to use both ML predictions and clinical rule-based adjustments.
    
    Args:
        processed_data (numpy.ndarray): Preprocessed patient data
        models (dict): Dictionary of trained ML models
        model_choice (str): Name of the model to use for prediction
        
    Returns:
        tuple: (prediction, probability, all_probabilities)
            - prediction: Predicted disease name
            - probability: Confidence score of the prediction as a percentage
            - all_probabilities: Dictionary with probabilities for all diseases
    """
    # Select the appropriate model
    if model_choice in models:
        model = models[model_choice]
    else:
        # Default to the first model if the chosen one isn't available
        model = next(iter(models.values()))
    
    # Use disease classes from session state if available (set during model training)
    import streamlit as st
    
    if 'disease_classes' in st.session_state:
        # Get mapping from session state
        disease_classes = st.session_state.disease_classes
        disease_mapping = {i: disease_name for i, disease_name in enumerate(disease_classes)}
    else:
        # Fallback mapping if session state is not available
        disease_mapping = {
            0: "Normal",
            1: "Pneumonia",
            2: "Diabetes",
            3: "Heart Disease",
            4: "Liver Disease",
            5: "Kidney Disease",
            6: "Influenza",
            7: "Hypertension",
            8: "Asthma",
            9: "COVID-19"
        }
    
    # Get the original patient data for rule-based adjustments
    original_patient_data = {}
    if 'last_patient_data' in st.session_state:
        original_patient_data = st.session_state.last_patient_data
    
    # Make prediction
    try:
        # Get probability predictions if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(processed_data)
            predicted_class = np.argmax(probabilities, axis=1)[0]
            prediction_probability = probabilities[0, predicted_class] * 100
            
            # Create dictionary of all disease probabilities for visualization
            all_probabilities = {}
            for i, prob in enumerate(probabilities[0]):
                disease_name = disease_mapping.get(i, f"Unknown-{i}")
                all_probabilities[disease_name] = round(prob * 100, 1)
        else:
            # Fallback for models without probability prediction
            predicted_class = model.predict(processed_data)[0]
            prediction_probability = 85.0  # Default confidence
            all_probabilities = {disease_mapping.get(predicted_class, "Unknown"): 85.0}
        
        # Map numeric prediction to disease name
        disease_prediction = disease_mapping.get(predicted_class, "Unknown")
        
        # Add small random variation for real-time feedback
        import random
        random.seed(sum(original_patient_data.values()) if original_patient_data else 42)
        prediction_probability = min(100, max(0, prediction_probability + random.uniform(-2, 2)))
        
        #######################
        # CLINICAL RULE-BASED ADJUSTMENTS
        #######################
        # These override ML predictions when specific symptom patterns are detected
        # This ensures medically appropriate predictions for known symptom combinations
        if original_patient_data:
            # Save input data in session state for symptom pattern matching
            st.session_state.last_patient_data = original_patient_data
            
            # Create a dictionary to track matched rules and their strength (0-100)
            # This allows us to determine which condition is most strongly indicated
            condition_matches = {}
            
            # LIVER DISEASE DETECTION
            # Jaundice is a strong indicator of liver disease
            if original_patient_data.get('jaundice', 0) == 1:
                liver_probability = max(80, all_probabilities.get("Liver Disease", 0))
                if original_patient_data.get('abdominal_pain', 0) == 1:
                    liver_probability += 15  # Abdominal pain with jaundice strongly suggests liver issues
                if original_patient_data.get('nausea', 0) == 1:
                    liver_probability += 10  # Nausea with jaundice strengthens liver disease probability
                
                # If liver enzymes are elevated, this is a very strong indicator
                if (original_patient_data.get('alt', 0) > 40 or 
                    original_patient_data.get('ast', 0) > 40):
                    liver_probability = min(95, liver_probability + 20)
                
                condition_matches["Liver Disease"] = min(100, liver_probability)
            
            # DIABETES DETECTION
            # High glucose levels are indicative of diabetes
            if original_patient_data.get('glucose_level', 0) > 150:
                diabetes_probability = max(70, all_probabilities.get("Diabetes", 0))
                if original_patient_data.get('frequent_urination', 0) == 1:
                    diabetes_probability += 20  # Frequent urination with high glucose strongly suggests diabetes
                
                condition_matches["Diabetes"] = min(100, diabetes_probability)
            
            # RESPIRATORY CONDITIONS DETECTION
            # Check for respiratory symptom patterns
            if original_patient_data.get('cough', 0) == 1:
                # Cough alone indicates potential respiratory issues
                respiratory_probability = 40
                
                # Cough with difficulty breathing strongly suggests respiratory condition
                if original_patient_data.get('difficulty_breathing', 0) == 1:
                    respiratory_probability += 30
                
                # Determine which respiratory condition is most likely
                if original_patient_data.get('fever', 0) == 1:
                    if original_patient_data.get('body_temp', 37) >= 38.5:
                        # High fever with respiratory symptoms suggests pneumonia
                        condition_matches["Pneumonia"] = min(100, respiratory_probability + 25)
                    else:
                        # Mild fever with respiratory symptoms suggests influenza
                        condition_matches["Influenza"] = min(100, respiratory_probability + 30)
                else:
                    # Respiratory symptoms without fever - could be asthma
                    # Especially if there's just difficulty breathing
                    condition_matches["Asthma"] = min(100, respiratory_probability + 15)
            
            # HEART DISEASE DETECTION
            # Chest pain is a key symptom of heart disease
            if original_patient_data.get('chest_pain', 0) == 1:
                heart_probability = max(60, all_probabilities.get("Heart Disease", 0))
                
                # With high BP, heart disease becomes more likely
                if (original_patient_data.get('systolic_bp', 120) > 140 or 
                    original_patient_data.get('diastolic_bp', 80) > 90):
                    heart_probability += 20
                
                # Higher cholesterol increases the probability
                if original_patient_data.get('cholesterol', 0) > 220:
                    heart_probability += 15
                
                condition_matches["Heart Disease"] = min(100, heart_probability)
            
            # KIDNEY DISEASE DETECTION
            # Elevated kidney markers indicate kidney issues
            if original_patient_data.get('creatinine', 0) > 1.2:
                kidney_probability = max(60, all_probabilities.get("Kidney Disease", 0))
                
                # BUN elevation strengthens the case
                if original_patient_data.get('bun', 0) > 20:
                    kidney_probability += 25
                
                condition_matches["Kidney Disease"] = min(100, kidney_probability)
            
            # HYPERTENSION DETECTION 
            # Consistently elevated BP indicates hypertension
            if (original_patient_data.get('systolic_bp', 120) > 140 or 
                original_patient_data.get('diastolic_bp', 80) > 90):
                
                hypertension_probability = 70
                
                # If both systolic and diastolic are high, even stronger indication
                if (original_patient_data.get('systolic_bp', 120) > 140 and 
                    original_patient_data.get('diastolic_bp', 80) > 90):
                    hypertension_probability += 15
                
                condition_matches["Hypertension"] = min(100, hypertension_probability)
            
            # COVID/FLU DETECTION
            # Fever with multiple other symptoms suggests viral infection
            if original_patient_data.get('fever', 0) == 1:
                flu_probability = 30
                
                # Count additional symptoms to differentiate between flu and COVID
                symptom_count = 0
                for symptom in ['cough', 'fatigue', 'headache', 'difficulty_breathing']:
                    if original_patient_data.get(symptom, 0) == 1:
                        symptom_count += 1
                        flu_probability += 15
                
                # With 3+ symptoms, strongly suggests influenza
                if symptom_count >= 2:
                    condition_matches["Influenza"] = min(100, flu_probability)
            
            # If any condition matches were found, select the highest probability one
            if condition_matches:
                # Find the condition with the highest probability
                best_match = max(condition_matches.items(), key=lambda x: x[1])
                disease_name, match_probability = best_match
                
                # Only override if the match is strong enough (prevents weak matches from trumping ML)
                if match_probability >= 60:
                    disease_prediction = disease_name
                    prediction_probability = match_probability
                    
                    # Update all probabilities based on our clinical rules
                    for condition, prob in condition_matches.items():
                        all_probabilities[condition] = prob
                    
                    # Reduce Normal probability accordingly for all matches
                    all_probabilities["Normal"] = max(0, all_probabilities.get("Normal", 0) - 40)
            
            # Normalize all_probabilities to sum to 100 (approximately)
            total_prob = sum(all_probabilities.values())
            if total_prob > 0:
                factor = 100 / total_prob
                all_probabilities = {k: min(100, v * factor) for k, v in all_probabilities.items()}
        
        return disease_prediction, prediction_probability, all_probabilities
    
    except Exception as e:
        # Return a safe default in case of errors
        print(f"Prediction error: {e}")
        empty_probs = {disease: 0.0 for disease in disease_mapping.values()}
        empty_probs["Uncertain"] = 50.0
        return "Uncertain", 50.0, empty_probs

def get_risk_level(disease, probability):
    """
    Determine the risk level based on the predicted disease and probability.
    
    Args:
        disease (str): Predicted disease name
        probability (float): Confidence score of the prediction as a percentage
        
    Returns:
        str: Risk level ("Low", "Moderate", or "High")
    """
    import streamlit as st
    
    # Define high risk diseases that are immediately concerning
    high_risk_diseases = [
        "Pneumonia", 
        "Heart Disease", 
        "COVID-19",
        "Liver Disease"  # Adding liver disease as high risk since symptoms like jaundice are serious
    ]
    
    # Define moderate risk diseases that need medical attention but aren't immediately life-threatening
    moderate_risk_diseases = [
        "Diabetes", 
        "Kidney Disease", 
        "Hypertension", 
        "Asthma",
        "Influenza"
    ]
    
    # Get original patient data for context-specific risk assessment if available
    original_data = {}
    if 'last_patient_data' in st.session_state:
        original_data = st.session_state.last_patient_data
    
    # Determine base risk level based on disease
    if disease in high_risk_diseases:
        base_risk = "High"
    elif disease in moderate_risk_diseases:
        base_risk = "Moderate"
    elif disease == "Normal":
        base_risk = "Low"
    else:
        base_risk = "Moderate"  # Default for unknown diseases
    
    # Apply symptom-specific risk adjustments
    if original_data:
        # Liver disease with jaundice is high risk regardless of confidence
        if disease == "Liver Disease" and original_data.get('jaundice', 0) == 1:
            return "High"
        
        # Heart disease with chest pain is high risk
        if disease == "Heart Disease" and original_data.get('chest_pain', 0) == 1:
            return "High"
        
        # Pneumonia with high fever and difficulty breathing is high risk
        if disease == "Pneumonia" and original_data.get('difficulty_breathing', 0) == 1 and original_data.get('body_temp', 37) > 38.5:
            return "High"
        
        # Diabetes with very high glucose is high risk
        if disease == "Diabetes" and original_data.get('glucose_level', 0) > 250:
            return "High"
    
    # Standard probability-based adjustments for other cases
    if probability >= 80:
        # High confidence - keep the base risk
        risk_level = base_risk
    elif probability >= 65:
        # Moderate confidence - potentially lower the risk for high risk diseases
        if base_risk == "High":
            risk_level = "Moderate"
        else:
            risk_level = base_risk
    else:
        # Low confidence - lower the risk unless it's already low
        if base_risk == "Low":
            risk_level = "Low"
        elif base_risk == "Moderate":
            risk_level = "Low"
        else:  # High
            risk_level = "Moderate"
    
    return risk_level

def predict_comorbidities(patient_data, predicted_disease):
    """
    Predict potential comorbidities based on patient data and primary predicted disease.
    
    Args:
        patient_data (dict): Dictionary containing patient information
        predicted_disease (str): Primary predicted disease
        
    Returns:
        list: List of potential comorbidities
    """
    comorbidities = []
    
    # Check for hypertension
    if patient_data.get('systolic_bp', 0) > 140 or patient_data.get('diastolic_bp', 0) > 90:
        if predicted_disease != "Hypertension":
            comorbidities.append("Hypertension")
    
    # Check for diabetes
    if patient_data.get('glucose_level', 0) > 126:
        if predicted_disease != "Diabetes":
            comorbidities.append("Diabetes")
    
    # Check for potential heart disease
    if (patient_data.get('chest_pain', 0) == 1 and 
        patient_data.get('fatigue', 0) == 1 and 
        (patient_data.get('cholesterol', 0) > 240 or patient_data.get('systolic_bp', 0) > 140)):
        if predicted_disease != "Heart Disease":
            comorbidities.append("Heart Disease")
    
    # Check for potential kidney disease
    if patient_data.get('creatinine', 0) > 1.2 and patient_data.get('bun', 0) > 20:
        if predicted_disease != "Kidney Disease":
            comorbidities.append("Kidney Disease")
    
    # Check for potential liver disease
    if (patient_data.get('alt', 0) > 50 or patient_data.get('ast', 0) > 40) and patient_data.get('jaundice', 0) == 1:
        if predicted_disease != "Liver Disease":
            comorbidities.append("Liver Disease")
    
    # Check for potential respiratory conditions
    if patient_data.get('difficulty_breathing', 0) == 1 and patient_data.get('cough', 0) == 1:
        if predicted_disease not in ["Pneumonia", "Asthma", "COVID-19"]:
            comorbidities.append("Respiratory Condition")
    
    return comorbidities

def calculate_severity_score(patient_data, disease):
    """
    Calculate a severity score for the predicted disease based on patient data.
    
    Args:
        patient_data (dict): Dictionary containing patient information
        disease (str): Predicted disease
        
    Returns:
        tuple: (severity_score, severity_factors)
            - severity_score: Numeric score representing disease severity (0-100)
            - severity_factors: List of factors contributing to severity
    """
    severity_score = 0
    severity_factors = []
    
    # Common factors affecting severity
    age = patient_data.get('age', 30)
    if age > 65:
        severity_score += 15
        severity_factors.append("Advanced age (65+)")
    elif age > 50:
        severity_score += 10
        severity_factors.append("Age over 50")
    
    # Vital signs scoring
    if patient_data.get('body_temp', 37) > 39:
        severity_score += 10
        severity_factors.append("High fever")
    elif patient_data.get('body_temp', 37) > 38:
        severity_score += 5
        severity_factors.append("Moderate fever")
    
    if patient_data.get('respiratory_rate', 16) > 24:
        severity_score += 15
        severity_factors.append("Elevated respiratory rate")
    
    if patient_data.get('heart_rate', 75) > 100:
        severity_score += 10
        severity_factors.append("Tachycardia")
    
    if patient_data.get('systolic_bp', 120) < 90 or patient_data.get('systolic_bp', 120) > 160:
        severity_score += 15
        severity_factors.append("Abnormal blood pressure")
    
    # Disease-specific factors
    if disease == "Pneumonia" or disease == "COVID-19":
        if patient_data.get('difficulty_breathing', 0) == 1:
            severity_score += 20
            severity_factors.append("Difficulty breathing")
        if patient_data.get('oxygen_saturation', 98) < 92:
            severity_score += 25
            severity_factors.append("Low oxygen saturation")
    
    elif disease == "Heart Disease":
        if patient_data.get('chest_pain', 0) == 1:
            severity_score += 20
            severity_factors.append("Chest pain")
        if patient_data.get('cholesterol', 180) > 240:
            severity_score += 15
            severity_factors.append("High cholesterol")
    
    elif disease == "Diabetes":
        if patient_data.get('glucose_level', 100) > 200:
            severity_score += 20
            severity_factors.append("Severely elevated glucose")
        elif patient_data.get('glucose_level', 100) > 150:
            severity_score += 10
            severity_factors.append("Moderately elevated glucose")
    
    elif disease == "Liver Disease" or disease == "Kidney Disease":
        if patient_data.get('fatigue', 0) == 1 and patient_data.get('nausea', 0) == 1:
            severity_score += 15
            severity_factors.append("Multiple symptoms affecting quality of life")
    
    # Cap the score at 100
    severity_score = min(severity_score, 100)
    
    return severity_score, severity_factors
