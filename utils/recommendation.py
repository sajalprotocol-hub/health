def get_recommendations(disease, risk_level):
    """
    Get medical recommendations based on the predicted disease and risk level.
    Enhanced with more specific, actionable advice and symptom-specific guidance.
    
    Args:
        disease (str): Predicted disease name
        risk_level (str): Risk level ("Low", "Moderate", or "High")
        
    Returns:
        list: List of recommendation strings
    """
    # Common recommendations for all conditions
    common_recommendations = [
        "Stay hydrated by drinking 8-10 glasses of water daily",
        "Get 7-8 hours of quality sleep each night",
        "Maintain a balanced diet rich in fruits, vegetables, and whole grains",
        "Avoid alcohol consumption and smoking entirely while experiencing symptoms"
    ]
    
    # Disease specific recommendations
    disease_recommendations = {
        "Normal": [
            "Continue annual health check-ups with your primary care provider",
            "Maintain a healthy lifestyle with 150+ minutes of exercise weekly",
            "Monitor your vital signs periodically (BP, pulse, weight)"
        ],
        
        "Pneumonia": [
            "Complete the full course of prescribed antibiotics (typically 5-10 days)",
            "Use a humidifier to loosen mucus and ease breathing",
            "Take acetaminophen or NSAIDs for fever/discomfort as directed",
            "Practice deep breathing exercises (10 breaths every hour while awake)",
            "Avoid lying flat; elevate your head with 2-3 pillows while sleeping"
        ],
        
        "Diabetes": [
            "Monitor blood glucose levels 2-4 times daily, keeping a detailed log",
            "Follow a diabetes-friendly diet with controlled carbohydrate portions",
            "Engage in 30 minutes of physical activity at least 5 days per week",
            "Take insulin or oral medications exactly as prescribed",
            "Inspect feet daily for cuts, blisters, redness, or swelling",
            "Schedule quarterly A1C tests to monitor long-term glucose control"
        ],
        
        "Heart Disease": [
            "Limit sodium to less than 2,300 mg daily (about 1 teaspoon of salt)",
            "Engage in cardiac rehabilitation or physician-approved exercise",
            "Maintain a heart-healthy diet rich in omega-3s and low in saturated fats",
            "Take cardiac medications at the same time each day for consistent effects",
            "Monitor and record your blood pressure daily",
            "Learn the warning signs of a heart attack and create an emergency plan"
        ],
        
        "Liver Disease": [
            "Completely avoid alcohol consumption and liver-toxic substances",
            "Follow a liver-friendly diet low in processed foods and added sugars",
            "Take hepatic medications as prescribed and at consistent times",
            "Schedule liver function tests every 1-3 months",
            "Report increased jaundice, abdominal swelling, or confusion immediately",
            "Avoid medications that can harm the liver (ask your doctor about all OTC meds)"
        ],
        
        "Kidney Disease": [
            "Follow a renal diet, limiting protein, phosphorus, potassium and sodium",
            "Take all prescribed kidney medications consistently and on schedule",
            "Monitor and potentially restrict fluid intake (ask your doctor for daily limits)",
            "Track weight daily to detect fluid retention",
            "Get regular kidney function and electrolyte tests every 1-3 months",
            "Avoid NSAIDs and other medications that can further damage kidneys"
        ],
        
        "Influenza": [
            "Rest at home for 5-7 days and until fever-free for 24 hours",
            "Take antiviral medications within 48 hours of symptom onset if prescribed",
            "Use acetaminophen or ibuprofen for fever and body aches",
            "Isolate yourself to prevent spreading the virus to others",
            "Monitor for signs of complications like persistent high fever or difficulty breathing"
        ],
        
        "Hypertension": [
            "Follow the DASH diet (rich in fruits, vegetables, whole grains, low-fat dairy)",
            "Limit sodium to 1,500-2,300 mg per day (check food labels carefully)",
            "Monitor your blood pressure at the same time daily, recording all readings",
            "Take blood pressure medications consistently, even when feeling well",
            "Engage in 30 minutes of moderate aerobic exercise most days",
            "Limit caffeine and practice regular stress-reduction techniques"
        ],
        
        "Asthma": [
            "Identify and document all personal asthma triggers to avoid",
            "Use controller medications daily as prescribed, even when feeling well",
            "Keep rescue inhalers with you at all times and check expiration dates",
            "Follow your written asthma action plan from your healthcare provider",
            "Use a peak flow meter daily to monitor lung function",
            "Keep living spaces clean and free of common allergens (dust, pet dander)"
        ],
        
        "COVID-19": [
            "Isolate for at least 5 days and until symptoms improve significantly",
            "Monitor oxygen levels with a pulse oximeter 2-3 times daily (seek care if below 94%)",
            "Rest in a prone position (lying on stomach) if experiencing breathing difficulties",
            "Take prescribed antiviral medications exactly as directed if applicable",
            "Stay in contact with your healthcare provider and report worsening symptoms",
            "Seek emergency care for severe symptoms (difficulty breathing, chest pain, confusion)"
        ]
    }
    
    # Risk-level specific recommendations
    risk_recommendations = {
        "Low": [
            "Follow up with your healthcare provider at your next routine appointment",
            "Consider preventive health measures appropriate for your age and gender",
            "Continue monitoring your symptoms and note any changes"
        ],
        
        "Moderate": [
            "Schedule a follow-up appointment within the next 1-2 weeks",
            "Begin recommended lifestyle and medication changes immediately",
            "Keep a detailed log of your symptoms, including timing and severity",
            "Contact your healthcare provider promptly if your condition worsens"
        ],
        
        "High": [
            "⚠️ URGENT: Seek medical attention within the next 24-48 hours",
            "For severe symptoms, consider emergency care or urgent care facilities",
            "Have someone stay with you to monitor your condition if possible",
            "Bring a detailed list of your current symptoms, medications, and allergies"
        ]
    }
    
    # Add symptom-specific recommendations
    import streamlit as st
    symptom_recs = []
    
    if 'last_patient_data' in st.session_state:
        patient_data = st.session_state.last_patient_data
        
        # Jaundice - specific advice
        if patient_data.get('jaundice', 0) == 1:
            symptom_recs.append("For jaundice: Seek immediate medical evaluation as this is a serious sign of liver dysfunction")
            symptom_recs.append("Avoid all alcohol, acetaminophen, and potentially hepatotoxic medications")
        
        # Abdominal pain - specific advice
        if patient_data.get('abdominal_pain', 0) == 1:
            symptom_recs.append("For abdominal pain: Document location, severity (1-10), timing, and any food/activity triggers")
            symptom_recs.append("Avoid fatty, spicy foods and alcohol while experiencing abdominal discomfort")
        
        # Fever - specific advice
        if patient_data.get('fever', 0) == 1:
            symptom_recs.append("For fever: Monitor temperature every 4 hours and take fever-reducing medication as directed")
            symptom_recs.append("Stay well-hydrated with cool liquids and use light clothing/bedding")
            
        # Difficulty breathing - specific advice
        if patient_data.get('difficulty_breathing', 0) == 1:
            symptom_recs.append("For breathing difficulty: Sit upright, practice pursed-lip breathing, and use a fan for airflow")
            symptom_recs.append("Seek urgent care if breathing worsens, especially when accompanied by chest pain or confusion")
        
        # High glucose - specific advice
        if patient_data.get('glucose_level', 0) > 150:
            symptom_recs.append("For elevated glucose: Monitor blood sugar levels frequently and follow a strict low-carb diet")
            symptom_recs.append("Stay well-hydrated with water (not sugary drinks) and be alert for signs of diabetic complications")
            
        # High blood pressure - specific advice
        if patient_data.get('systolic_bp', 0) > 140 or patient_data.get('diastolic_bp', 0) > 90:
            symptom_recs.append("For high blood pressure: Limit sodium intake, practice daily relaxation techniques, and monitor BP daily")
            symptom_recs.append("Take medications exactly as prescribed and position yourself properly during BP readings")
            
        # Abnormal liver enzymes
        if patient_data.get('alt', 0) > 40 or patient_data.get('ast', 0) > 40:
            symptom_recs.append("For elevated liver enzymes: Avoid alcohol completely and limit processed foods high in fructose")
            symptom_recs.append("Discuss all medications with your doctor, including supplements and over-the-counter products")
    
    # Compile recommendations based on disease and risk level
    recommendations = []
    
    if risk_level == "High":
        # For high risk, prioritize risk and symptom-specific recommendations
        recommendations.extend(risk_recommendations.get(risk_level, []))
        recommendations.extend(symptom_recs)
        if disease in disease_recommendations:
            recommendations.extend(disease_recommendations[disease][:3])  # Add top 3 disease recommendations
    elif risk_level == "Moderate":
        # For moderate risk, mix risk, symptom and disease recommendations
        recommendations.extend(risk_recommendations.get(risk_level, []))
        recommendations.extend(symptom_recs)
        if disease in disease_recommendations:
            recommendations.extend(disease_recommendations[disease])
    else:  # Low risk
        # For low risk, emphasize general health recommendations
        recommendations.extend(risk_recommendations.get(risk_level, []))
        recommendations.extend(common_recommendations)
        recommendations.extend(symptom_recs)
        if disease in disease_recommendations:
            recommendations.extend(disease_recommendations[disease][:2])  # Add top 2 disease recommendations
    
    # If we don't have disease-specific recommendations, add fallback
    if disease not in disease_recommendations:
        recommendations.extend([
            "Consult with a healthcare professional for specific advice about your condition",
            "Monitor and document your symptoms carefully, noting any changes",
            "Avoid self-medication without professional guidance"
        ])
    
    # If we have too many recommendations, trim to a reasonable number
    if len(recommendations) > 8:
        recommendations = recommendations[:8]
    
    return recommendations

def get_specialist_recommendation(disease):
    """
    Recommend medical specialists based on the predicted disease.
    
    Args:
        disease (str): Predicted disease name
        
    Returns:
        list: List of recommended specialists
    """
    specialist_mapping = {
        "Normal": ["General Practitioner/Family Medicine"],
        
        "Pneumonia": [
            "Pulmonologist (Lung Specialist)",
            "Infectious Disease Specialist",
            "Internal Medicine Specialist"
        ],
        
        "Diabetes": [
            "Endocrinologist (Diabetes Specialist)",
            "Dietitian/Nutritionist",
            "Podiatrist (Foot Care Specialist)",
            "Ophthalmologist (Eye Specialist)"
        ],
        
        "Heart Disease": [
            "Cardiologist (Heart Specialist)",
            "Cardiac Surgeon",
            "Interventional Cardiologist",
            "Cardiovascular Rehabilitation Specialist"
        ],
        
        "Liver Disease": [
            "Hepatologist (Liver Specialist)",
            "Gastroenterologist",
            "Transplant Surgeon (for severe cases)"
        ],
        
        "Kidney Disease": [
            "Nephrologist (Kidney Specialist)",
            "Dialysis Specialist (for advanced cases)",
            "Renal Dietitian"
        ],
        
        "Influenza": [
            "Infectious Disease Specialist",
            "General Practitioner/Family Medicine"
        ],
        
        "Hypertension": [
            "Cardiologist",
            "Nephrologist",
            "Hypertension Specialist"
        ],
        
        "Asthma": [
            "Pulmonologist",
            "Allergist/Immunologist",
            "Respiratory Therapist"
        ],
        
        "COVID-19": [
            "Infectious Disease Specialist",
            "Pulmonologist",
            "Critical Care Specialist (for severe cases)"
        ]
    }
    
    # Return specialists for the given disease, or general recommendation if not found
    return specialist_mapping.get(disease, ["General Practitioner/Family Medicine"])

def get_medication_recommendations(disease, risk_level, patient_data=None):
    """
    Provide general medication guidance based on disease and risk level.
    Note: This is for informational purposes only and not a substitute for medical advice.
    
    Args:
        disease (str): Predicted disease name
        risk_level (str): Risk level ("Low", "Moderate", or "High")
        patient_data (dict, optional): Patient data for customized recommendations
        
    Returns:
        list: List of medication guidance notes
    """
    # Disclaimer for medication recommendations
    disclaimer = [
        "NOTE: The following medication information is for educational purposes only. " 
        "Always consult with a healthcare professional before starting any medication."
    ]
    
    medication_info = {
        "Normal": [
            "No specific medications recommended at this time",
            "Focus on preventive health measures and lifestyle adjustments"
        ],
        
        "Pneumonia": [
            "Antibiotics are typically prescribed for bacterial pneumonia",
            "Antiviral medications may be prescribed for viral pneumonia",
            "Over-the-counter pain relievers and fever reducers may help manage symptoms",
            "Cough medicine may be recommended to control persistent coughing"
        ],
        
        "Diabetes": [
            "Oral diabetes medications (e.g., metformin) are common first-line treatments",
            "Insulin therapy may be necessary depending on type and severity",
            "Medications to protect kidney function may be recommended",
            "Medications to manage cholesterol and blood pressure are often prescribed alongside diabetes treatment"
        ],
        
        "Heart Disease": [
            "Blood pressure medications (e.g., ACE inhibitors, beta-blockers)",
            "Cholesterol-lowering medications (e.g., statins)",
            "Blood thinners to prevent clots (e.g., aspirin, warfarin)",
            "Medications to control heart rhythm or reduce chest pain may be prescribed"
        ],
        
        "Liver Disease": [
            "Treatment depends on the specific cause of liver disease",
            "Antiviral medications for viral hepatitis",
            "Medications to control itching and other symptoms",
            "Avoid acetaminophen (Tylenol) and other potentially liver-toxic medications"
        ],
        
        "Kidney Disease": [
            "Blood pressure medications to slow kidney damage",
            "Medications to control phosphorus levels",
            "Erythropoietin for anemia related to kidney disease",
            "Vitamin D supplements as prescribed"
        ],
        
        "Influenza": [
            "Antiviral medications (e.g., oseltamivir/Tamiflu) if started early",
            "Over-the-counter fever reducers and pain relievers",
            "Cough suppressants as recommended by a healthcare provider"
        ],
        
        "Hypertension": [
            "Diuretics (water pills)",
            "ACE inhibitors or ARBs",
            "Calcium channel blockers",
            "Beta-blockers",
            "Medication combinations are often more effective than single drugs"
        ],
        
        "Asthma": [
            "Quick-relief/rescue inhalers (e.g., albuterol)",
            "Long-term control medications (e.g., inhaled corticosteroids)",
            "Leukotriene modifiers for some patients",
            "Always carry rescue medication and follow your asthma action plan"
        ],
        
        "COVID-19": [
            "Treatment varies based on severity and is rapidly evolving",
            "Antiviral medications may be prescribed in certain cases",
            "Over-the-counter medications for symptom relief",
            "Follow the most current CDC and healthcare provider guidance"
        ]
    }
    
    # Compile medication information
    med_recommendations = disclaimer.copy()
    
    if disease in medication_info:
        med_recommendations.extend(medication_info[disease])
    else:
        med_recommendations.append("Consult with a healthcare provider for appropriate medication recommendations")
    
    # Add risk-level specific notes
    if risk_level == "High":
        med_recommendations.append("Due to your high-risk assessment, prescription medications will likely be necessary")
    elif risk_level == "Moderate":
        med_recommendations.append("A combination of lifestyle changes and medications may be recommended")
    else:  # Low
        med_recommendations.append("Medications may be minimal, with a focus on lifestyle modifications")
    
    return med_recommendations

def get_lifestyle_recommendations(disease, patient_data=None):
    """
    Provide lifestyle recommendations based on the predicted disease.
    
    Args:
        disease (str): Predicted disease name
        patient_data (dict, optional): Patient data for customized recommendations
        
    Returns:
        list: List of lifestyle recommendation strings
    """
    # General lifestyle recommendations for all patients
    general_recommendations = [
        "Maintain a balanced diet rich in fruits, vegetables, whole grains, and lean proteins",
        "Stay physically active with at least 150 minutes of moderate exercise per week",
        "Get 7-9 hours of quality sleep each night",
        "Manage stress through relaxation techniques, mindfulness, or meditation",
        "Limit alcohol consumption and avoid smoking"
    ]
    
    # Disease-specific lifestyle recommendations
    disease_lifestyle = {
        "Normal": [
            "Continue with regular preventive health check-ups",
            "Consider incorporating strength training into your exercise routine",
            "Stay up to date with recommended vaccinations"
        ],
        
        "Pneumonia": [
            "Practice good respiratory hygiene, including covering coughs and sneezes",
            "Avoid exposure to airborne irritants like smoke and pollutants",
            "Use a humidifier to maintain optimal air moisture",
            "Gradually increase physical activity as recovery progresses"
        ],
        
        "Diabetes": [
            "Monitor carbohydrate intake and follow a consistent meal schedule",
            "Choose foods with a low glycemic index",
            "Incorporate regular physical activity to help manage blood sugar",
            "Check your feet daily for any signs of problems",
            "Maintain a healthy weight through diet and exercise"
        ],
        
        "Heart Disease": [
            "Follow a heart-healthy diet low in sodium and saturated fats",
            "Engage in regular cardiovascular exercise as approved by your doctor",
            "Maintain a healthy weight to reduce strain on your heart",
            "Monitor and manage stress levels through relaxation techniques",
            "Know the warning signs of heart attack and when to seek emergency care"
        ],
        
        "Liver Disease": [
            "Avoid alcohol and recreational drugs",
            "Limit processed foods, added sugars, and saturated fats",
            "Stay hydrated with plenty of water",
            "Avoid unnecessary medications, especially acetaminophen (Tylenol)",
            "Get vaccinated against hepatitis A and B if appropriate"
        ],
        
        "Kidney Disease": [
            "Follow a kidney-friendly diet as recommended by your healthcare provider",
            "Monitor and limit fluid intake if advised",
            "Reduce sodium, potassium, and phosphorus in your diet as directed",
            "Control blood pressure through diet, exercise, and medication",
            "Avoid NSAIDs and other medications that can stress the kidneys"
        ],
        
        "Influenza": [
            "Rest and limit activities until fever subsides",
            "Stay home to prevent spreading illness to others",
            "Consider annual flu vaccination to prevent future infections",
            "Practice frequent hand washing during flu season"
        ],
        
        "Hypertension": [
            "Reduce sodium intake to less than 2,300 mg per day",
            "Follow the DASH diet (Dietary Approaches to Stop Hypertension)",
            "Limit alcohol consumption to moderate levels",
            "Engage in regular aerobic exercise",
            "Monitor your blood pressure regularly at home"
        ],
        
        "Asthma": [
            "Identify and avoid your specific asthma triggers",
            "Use air purifiers in your home, especially in the bedroom",
            "Maintain a clean home environment to reduce dust and allergens",
            "Consider using hypoallergenic bedding",
            "Develop and follow an asthma action plan with your healthcare provider"
        ],
        
        "COVID-19": [
            "Isolate yourself to prevent spreading the virus to others",
            "Rest and stay hydrated to support recovery",
            "Monitor your oxygen levels with a pulse oximeter if available",
            "Practice deep breathing exercises to maintain lung function",
            "Follow current CDC guidelines for isolation and recovery"
        ]
    }
    
    # Compile lifestyle recommendations
    lifestyle_recs = []
    
    # Add disease-specific recommendations
    if disease in disease_lifestyle:
        lifestyle_recs.extend(disease_lifestyle[disease])
    
    # Add general recommendations
    lifestyle_recs.extend(general_recommendations)
    
    # Customize based on patient data if available
    if patient_data:
        age = patient_data.get('age', 30)
        
        # Age-specific additions
        if age > 65:
            lifestyle_recs.append("Consider balance exercises to prevent falls")
            lifestyle_recs.append("Stay mentally active with puzzles, reading, or social activities")
        
        # Add based on BMI if available
        if 'weight' in patient_data and 'height' in patient_data:
            height_m = patient_data['height'] / 100  # Convert cm to m
            weight_kg = patient_data['weight']
            bmi = weight_kg / (height_m * height_m)
            
            if bmi > 30:
                lifestyle_recs.append("Focus on gradual weight loss through sustainable diet and exercise changes")
                lifestyle_recs.append("Consider consulting with a dietitian for a personalized meal plan")
            elif bmi < 18.5:
                lifestyle_recs.append("Focus on nutrient-dense foods to achieve a healthy weight")
                lifestyle_recs.append("Consider working with a dietitian to develop a weight gain plan")
    
    return lifestyle_recs
