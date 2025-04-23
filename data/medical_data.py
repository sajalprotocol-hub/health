import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import streamlit as st

def load_medical_data():
    """
    Load and preprocess medical data for model training.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
            - X_train, X_test: Training and testing features
            - y_train, y_test: Training and testing labels
    """
    # Create synthetic medical dataset (in a real application, this would load from files)
    # We're generating synthetic data here since we don't have an actual dataset
    data = generate_synthetic_medical_data()
    
    # Split features and target
    X = data.drop('disease', axis=1)
    y = data['disease']
    
    # Encode the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Store the label encoder classes for later use
    st.session_state.disease_classes = list(label_encoder.classes_)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    return X_train, X_test, y_train, y_test

def generate_synthetic_medical_data(n_samples=1000):
    """
    Generate synthetic medical data for model training.
    
    Args:
        n_samples (int): Number of samples to generate
        
    Returns:
        pandas.DataFrame: DataFrame containing synthetic medical data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define disease classes
    diseases = [
        "Normal", "Pneumonia", "Diabetes", "Heart Disease", "Liver Disease",
        "Kidney Disease", "Influenza", "Hypertension", "Asthma", "COVID-19"
    ]
    
    # Initialize data dictionary
    data = {
        'age': np.random.randint(1, 100, n_samples),
        'gender': np.random.randint(0, 2, n_samples),  # 0: Male, 1: Female
        'fever': np.random.randint(0, 2, n_samples),
        'cough': np.random.randint(0, 2, n_samples),
        'fatigue': np.random.randint(0, 2, n_samples),
        'difficulty_breathing': np.random.randint(0, 2, n_samples),
        'headache': np.random.randint(0, 2, n_samples),
        'weight_loss': np.random.randint(0, 2, n_samples),
        'chest_pain': np.random.randint(0, 2, n_samples),
        'nausea': np.random.randint(0, 2, n_samples),
        'abdominal_pain': np.random.randint(0, 2, n_samples),
        'jaundice': np.random.randint(0, 2, n_samples),
        'frequent_urination': np.random.randint(0, 2, n_samples),
    }
    
    # Generate vital signs with realistic ranges
    data['body_temp'] = np.random.uniform(36.5, 39.5, n_samples)
    data['heart_rate'] = np.random.randint(60, 130, n_samples)
    data['respiratory_rate'] = np.random.randint(12, 30, n_samples)
    data['systolic_bp'] = np.random.randint(90, 180, n_samples)
    data['diastolic_bp'] = np.random.randint(60, 110, n_samples)
    
    # Generate blood test results
    data['glucose_level'] = np.random.randint(70, 300, n_samples)
    data['wbc_count'] = np.random.randint(4000, 15000, n_samples)
    data['rbc_count'] = np.random.uniform(3.5, 6.0, n_samples)
    data['cholesterol'] = np.random.randint(120, 300, n_samples)
    data['hdl_cholesterol'] = np.random.randint(30, 80, n_samples)
    data['ldl_cholesterol'] = np.random.randint(70, 200, n_samples)
    data['triglycerides'] = np.random.randint(50, 300, n_samples)
    data['creatinine'] = np.random.uniform(0.6, 2.0, n_samples)
    data['bun'] = np.random.randint(8, 40, n_samples)
    data['alt'] = np.random.randint(10, 100, n_samples)
    data['ast'] = np.random.randint(10, 100, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Assign diseases with specific symptom patterns to make the data more realistic
    
    # Normal condition
    normal_indices = np.random.choice(n_samples, n_samples // 10, replace=False)
    df.loc[normal_indices, 'fever'] = 0
    df.loc[normal_indices, 'cough'] = 0
    df.loc[normal_indices, 'difficulty_breathing'] = 0
    df.loc[normal_indices, 'body_temp'] = np.random.uniform(36.5, 37.2, len(normal_indices))
    df.loc[normal_indices, 'glucose_level'] = np.random.randint(70, 110, len(normal_indices))
    
    # Pneumonia
    pneumonia_indices = np.random.choice(
        [i for i in range(n_samples) if i not in normal_indices], 
        n_samples // 10, 
        replace=False
    )
    df.loc[pneumonia_indices, 'fever'] = 1
    df.loc[pneumonia_indices, 'cough'] = 1
    df.loc[pneumonia_indices, 'difficulty_breathing'] = np.random.randint(0, 2, len(pneumonia_indices))
    df.loc[pneumonia_indices, 'chest_pain'] = np.random.randint(0, 2, len(pneumonia_indices))
    df.loc[pneumonia_indices, 'body_temp'] = np.random.uniform(38.0, 39.5, len(pneumonia_indices))
    df.loc[pneumonia_indices, 'respiratory_rate'] = np.random.randint(20, 30, len(pneumonia_indices))
    
    # Diabetes
    diabetes_indices = np.random.choice(
        [i for i in range(n_samples) if i not in normal_indices and i not in pneumonia_indices],
        n_samples // 10,
        replace=False
    )
    df.loc[diabetes_indices, 'frequent_urination'] = 1
    df.loc[diabetes_indices, 'fatigue'] = np.random.randint(0, 2, len(diabetes_indices))
    df.loc[diabetes_indices, 'glucose_level'] = np.random.randint(126, 300, len(diabetes_indices))
    
    # Heart Disease
    heart_indices = np.random.choice(
        [i for i in range(n_samples) if i not in normal_indices and i not in pneumonia_indices and i not in diabetes_indices],
        n_samples // 10,
        replace=False
    )
    df.loc[heart_indices, 'chest_pain'] = 1
    df.loc[heart_indices, 'fatigue'] = np.random.randint(0, 2, len(heart_indices))
    df.loc[heart_indices, 'difficulty_breathing'] = np.random.randint(0, 2, len(heart_indices))
    df.loc[heart_indices, 'cholesterol'] = np.random.randint(200, 300, len(heart_indices))
    df.loc[heart_indices, 'systolic_bp'] = np.random.randint(130, 180, len(heart_indices))
    
    # Assign remaining diseases based on specific patterns
    remaining_indices = [i for i in range(n_samples) if i not in normal_indices and 
                         i not in pneumonia_indices and i not in diabetes_indices and 
                         i not in heart_indices]
    
    remaining_diseases = [disease for disease in diseases 
                          if disease not in ["Normal", "Pneumonia", "Diabetes", "Heart Disease"]]
    
    # Assign remaining disease labels and patterns
    for i, disease in enumerate(remaining_diseases):
        # Select a portion of the remaining indices for this disease
        if i < len(remaining_diseases) - 1:
            count = len(remaining_indices) // (len(remaining_diseases) - i)
            disease_indices = np.random.choice(remaining_indices, count, replace=False)
            remaining_indices = [idx for idx in remaining_indices if idx not in disease_indices]
        else:
            # Assign all remaining indices to the last disease
            disease_indices = remaining_indices
        
        # Set specific patterns for each disease
        if disease == "Liver Disease":
            df.loc[disease_indices, 'jaundice'] = 1
            df.loc[disease_indices, 'nausea'] = np.random.randint(0, 2, len(disease_indices))
            df.loc[disease_indices, 'abdominal_pain'] = np.random.randint(0, 2, len(disease_indices))
            df.loc[disease_indices, 'alt'] = np.random.randint(50, 100, len(disease_indices))
            df.loc[disease_indices, 'ast'] = np.random.randint(50, 100, len(disease_indices))
            
        elif disease == "Kidney Disease":
            df.loc[disease_indices, 'fatigue'] = 1
            df.loc[disease_indices, 'nausea'] = np.random.randint(0, 2, len(disease_indices))
            df.loc[disease_indices, 'creatinine'] = np.random.uniform(1.3, 2.0, len(disease_indices))
            df.loc[disease_indices, 'bun'] = np.random.randint(20, 40, len(disease_indices))
            
        elif disease == "Influenza":
            df.loc[disease_indices, 'fever'] = 1
            df.loc[disease_indices, 'cough'] = np.random.randint(0, 2, len(disease_indices))
            df.loc[disease_indices, 'fatigue'] = 1
            df.loc[disease_indices, 'headache'] = np.random.randint(0, 2, len(disease_indices))
            df.loc[disease_indices, 'body_temp'] = np.random.uniform(38.0, 39.0, len(disease_indices))
            
        elif disease == "Hypertension":
            df.loc[disease_indices, 'headache'] = np.random.randint(0, 2, len(disease_indices))
            df.loc[disease_indices, 'systolic_bp'] = np.random.randint(140, 180, len(disease_indices))
            df.loc[disease_indices, 'diastolic_bp'] = np.random.randint(90, 110, len(disease_indices))
            
        elif disease == "Asthma":
            df.loc[disease_indices, 'difficulty_breathing'] = 1
            df.loc[disease_indices, 'cough'] = np.random.randint(0, 2, len(disease_indices))
            df.loc[disease_indices, 'chest_pain'] = np.random.randint(0, 2, len(disease_indices))
            df.loc[disease_indices, 'respiratory_rate'] = np.random.randint(20, 30, len(disease_indices))
            
        elif disease == "COVID-19":
            df.loc[disease_indices, 'fever'] = 1
            df.loc[disease_indices, 'cough'] = 1
            df.loc[disease_indices, 'fatigue'] = np.random.randint(0, 2, len(disease_indices))
            df.loc[disease_indices, 'difficulty_breathing'] = np.random.randint(0, 2, len(disease_indices))
            df.loc[disease_indices, 'body_temp'] = np.random.uniform(37.8, 39.5, len(disease_indices))
            df.loc[disease_indices, 'respiratory_rate'] = np.random.randint(18, 30, len(disease_indices))
    
    # Assign disease labels - use strings for all values
    disease_labels = np.full(n_samples, "Unknown", dtype=object)  # Initialize with strings
    disease_labels[normal_indices] = "Normal"
    disease_labels[pneumonia_indices] = "Pneumonia"
    disease_labels[diabetes_indices] = "Diabetes"
    disease_labels[heart_indices] = "Heart Disease"
    
    # Assign remaining disease labels
    remaining_indices = [i for i in range(n_samples) if disease_labels[i] == "Unknown"]
    
    # Distribute remaining indices evenly among remaining diseases
    if remaining_indices and remaining_diseases:
        indices_per_disease = len(remaining_indices) // len(remaining_diseases)
        
        for i, disease in enumerate(remaining_diseases):
            # Calculate start and end indices for this disease
            start = i * indices_per_disease
            end = (i + 1) * indices_per_disease if i < len(remaining_diseases) - 1 else len(remaining_indices)
            
            # Assign this disease to the corresponding indices
            for idx in remaining_indices[start:end]:
                disease_labels[idx] = disease
    
    df['disease'] = disease_labels
    
    return df

def get_patient_history(patient_id=None):
    """
    Get patient medical history for a given patient ID.
    For demo purposes, this generates synthetic history.
    
    Args:
        patient_id (str, optional): Patient ID to get history for
        
    Returns:
        dict: Dictionary containing patient medical history
    """
    # In a real application, this would query a database
    # For this demo, we'll generate synthetic history
    
    if patient_id is None:
        # Generate a random patient ID
        patient_id = f"P{np.random.randint(10000, 99999)}"
    
    # Generate basic patient info
    patient_info = {
        'patient_id': patient_id,
        'name': f"Patient {patient_id}",
        'age': np.random.randint(18, 80),
        'gender': np.random.choice(['Male', 'Female']),
        'blood_type': np.random.choice(['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'])
    }
    
    # Generate medical history entries
    num_entries = np.random.randint(3, 10)
    
    # List of possible conditions and treatments
    conditions = [
        "Common Cold", "Influenza", "Bronchitis", "Sinusitis", "Hypertension", 
        "Type 2 Diabetes", "Asthma", "Allergic Rhinitis", "Migraine", "Gastritis",
        "Urinary Tract Infection", "Dermatitis", "Anxiety", "Depression", "Insomnia"
    ]
    
    treatments = [
        "Prescribed medication", "Rest and fluids", "Physical therapy", 
        "Dietary changes", "Surgery", "Counseling", "Antibiotics",
        "Pain management", "Lifestyle modifications", "Monitoring"
    ]
    
    # Generate history entries
    history = []
    
    # Start date (between 1-5 years ago)
    start_date = datetime.datetime.now() - datetime.timedelta(days=np.random.randint(365, 365*5))
    
    for i in range(num_entries):
        # Generate a random date after the start date
        days_since_start = (datetime.datetime.now() - start_date).days
        visit_date = start_date + datetime.timedelta(days=np.random.randint(0, days_since_start))
        
        # Generate a random condition and treatment
        condition = np.random.choice(conditions)
        treatment = np.random.choice(treatments)
        
        # Generate random vital signs
        vitals = {
            'temperature': round(np.random.uniform(36.5, 38.5), 1),
            'heart_rate': np.random.randint(60, 100),
            'blood_pressure': f"{np.random.randint(110, 140)}/{np.random.randint(70, 90)}",
            'respiratory_rate': np.random.randint(12, 20)
        }
        
        # Create history entry
        entry = {
            'date': visit_date.strftime('%Y-%m-%d'),
            'condition': condition,
            'symptoms': generate_symptoms_for_condition(condition),
            'treatment': treatment,
            'vitals': vitals,
            'doctor': f"Dr. {np.random.choice(['Smith', 'Johnson', 'Williams', 'Jones', 'Brown', 'Davis', 'Miller'])}"
        }
        
        history.append(entry)
    
    # Sort history by date (newest first)
    history.sort(key=lambda x: datetime.datetime.strptime(x['date'], '%Y-%m-%d'), reverse=True)
    
    # Create complete patient history
    patient_history = {
        'patient_info': patient_info,
        'medical_history': history
    }
    
    return patient_history

def generate_symptoms_for_condition(condition):
    """
    Generate a list of symptoms for a given medical condition.
    
    Args:
        condition (str): Medical condition
        
    Returns:
        list: List of symptoms
    """
    # Map of conditions to common symptoms
    symptom_map = {
        "Common Cold": ["Runny nose", "Sneezing", "Cough", "Sore throat", "Congestion"],
        "Influenza": ["Fever", "Body aches", "Fatigue", "Headache", "Cough", "Sore throat"],
        "Bronchitis": ["Persistent cough", "Mucus production", "Fatigue", "Shortness of breath", "Chest discomfort"],
        "Sinusitis": ["Facial pain", "Nasal congestion", "Headache", "Post-nasal drip", "Reduced sense of smell"],
        "Hypertension": ["Headache", "Shortness of breath", "Nosebleeds", "Visual changes", "Dizziness"],
        "Type 2 Diabetes": ["Increased thirst", "Frequent urination", "Hunger", "Fatigue", "Blurred vision"],
        "Asthma": ["Wheezing", "Shortness of breath", "Chest tightness", "Coughing", "Trouble sleeping due to breathing"],
        "Allergic Rhinitis": ["Sneezing", "Itchy eyes", "Runny nose", "Nasal congestion", "Watery eyes"],
        "Migraine": ["Severe headache", "Nausea", "Sensitivity to light", "Sensitivity to sound", "Visual disturbances"],
        "Gastritis": ["Abdominal pain", "Nausea", "Vomiting", "Bloating", "Indigestion"],
        "Urinary Tract Infection": ["Painful urination", "Frequent urination", "Urgency", "Cloudy urine", "Lower abdominal pain"],
        "Dermatitis": ["Skin rash", "Itching", "Redness", "Swelling", "Dry skin"],
        "Anxiety": ["Excessive worry", "Restlessness", "Fatigue", "Difficulty concentrating", "Irritability"],
        "Depression": ["Persistent sadness", "Loss of interest", "Sleep changes", "Low energy", "Feelings of worthlessness"],
        "Insomnia": ["Difficulty falling asleep", "Difficulty staying asleep", "Waking up too early", "Daytime fatigue", "Irritability"]
    }
    
    # Get symptoms for the given condition, or return generic symptoms
    symptoms = symptom_map.get(condition, ["Fatigue", "Discomfort", "General malaise"])
    
    # Select a random subset of symptoms (at least 2)
    num_symptoms = np.random.randint(2, len(symptoms) + 1)
    selected_symptoms = np.random.choice(symptoms, num_symptoms, replace=False)
    
    return list(selected_symptoms)
