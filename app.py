import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import datetime
from utils.data_preprocessing import preprocess_patient_data
from utils.model_training import train_models
from utils.prediction import predict_disease, get_risk_level
from utils.recommendation import get_recommendations, get_specialist_recommendation
from utils.hospital_management import get_hospital_bed_status, recommend_hospitals, book_hospital_bed, get_doctors_by_specialty
from data.medical_data import load_medical_data
from data.hospital_data import get_hospital_data

# Page configuration
st.set_page_config(
    page_title="ML Healthcare System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application title
st.title("🏥 ML-Based Healthcare System")
st.markdown("### Disease Prediction & Hospital Resource Management")

# Initialize session state variables if they don't exist
if 'trained_models' not in st.session_state:
    with st.spinner("Training ML models... This might take a moment."):
        # Load and preprocess data
        X_train, X_test, y_train, y_test = load_medical_data()

        # Train models
        models, feature_names = train_models(X_train, y_train, X_test, y_test)
        st.session_state.trained_models = models
        st.session_state.feature_names = feature_names
        st.success("Models trained successfully!")

# Initialize booking state
if 'booking_status' not in st.session_state:
    st.session_state.booking_status = None

# Initialize prediction results
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Disease Prediction", "Hospital Management", "Bed Booking", "Doctor Finder", "About"])

# Initialize database tables if they don't exist
try:
    from utils.database import init_db
    init_db()
    if 'database_initialized' not in st.session_state:
        st.session_state.database_initialized = True
        st.sidebar.success("Database connected successfully")
except Exception as e:
    if 'database_initialized' not in st.session_state:
        st.sidebar.error(f"Database connection error: {e}")
        st.session_state.database_initialized = False

if page == "Disease Prediction":
    st.header("Disease Prediction System")

    # Create tabs for basic and advanced input
    basic_tab, advanced_tab = st.tabs(["Basic Information", "Advanced Tests"])

    with basic_tab:
        # Create columns for patient information
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Patient Information")
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])

            st.subheader("Common Symptoms")
            fever = st.checkbox("Fever")
            cough = st.checkbox("Cough")
            fatigue = st.checkbox("Fatigue")
            difficulty_breathing = st.checkbox("Difficulty Breathing")
            headache = st.checkbox("Headache")
            weight_loss = st.checkbox("Weight Loss")

        with col2:
            st.subheader("Additional Symptoms")
            chest_pain = st.checkbox("Chest Pain")
            nausea = st.checkbox("Nausea")
            abdominal_pain = st.checkbox("Abdominal Pain")
            jaundice = st.checkbox("Jaundice/Yellowing of Skin")
            frequent_urination = st.checkbox("Frequent Urination")

            st.subheader("Vital Signs")
            body_temp = st.slider("Body Temperature (°C)", 35.0, 42.0, 37.0, 0.1)
            heart_rate = st.slider("Heart Rate (bpm)", 40, 200, 75)
            respiratory_rate = st.slider("Respiratory Rate (breaths/min)", 8, 40, 16)
            systolic_bp = st.slider("Systolic Blood Pressure (mmHg)", 70, 200, 120)
            diastolic_bp = st.slider("Diastolic Blood Pressure (mmHg)", 40, 120, 80)

    with advanced_tab:
        # Create columns for advanced tests
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Blood Sugar Tests")
            glucose_level = st.slider("Glucose Level (mg/dL)", 50, 500, 100)

            st.subheader("Blood Cell Counts")
            wbc_count = st.slider("White Blood Cell Count (cells/mcL)", 3000, 20000, 7500)
            rbc_count = st.slider("Red Blood Cell Count (million cells/mcL)", 3.0, 7.0, 4.8, 0.1)

            st.subheader("Cholesterol Panel")
            cholesterol = st.slider("Total Cholesterol (mg/dL)", 100, 350, 180)
            hdl_cholesterol = st.slider("HDL Cholesterol (mg/dL)", 20, 100, 50)
            ldl_cholesterol = st.slider("LDL Cholesterol (mg/dL)", 40, 250, 100)
            triglycerides = st.slider("Triglycerides (mg/dL)", 50, 500, 150)

        with col2:
            st.subheader("Kidney Function Tests")
            creatinine = st.slider("Creatinine (mg/dL)", 0.5, 3.0, 1.0, 0.1)
            bun = st.slider("Blood Urea Nitrogen (mg/dL)", 5, 50, 15)

            st.subheader("Liver Function Tests")
            alt = st.slider("ALT (U/L)", 5, 200, 30)
            ast = st.slider("AST (U/L)", 5, 200, 25)

            st.subheader("Model Selection")
            model_choice = st.selectbox("Select AI Model", ["Random Forest", "XGBoost"])

    # Create a dictionary of the input data
    patient_data = {
        'age': age,
        'gender': 0 if gender == "Male" else 1 if gender == "Female" else 2,
        'fever': 1 if fever else 0,
        'cough': 1 if cough else 0,
        'fatigue': 1 if fatigue else 0,
        'difficulty_breathing': 1 if difficulty_breathing else 0,
        'headache': 1 if headache else 0,
        'weight_loss': 1 if weight_loss else 0,
        'chest_pain': 1 if chest_pain else 0,
        'nausea': 1 if nausea else 0,
        'abdominal_pain': 1 if abdominal_pain else 0,
        'jaundice': 1 if jaundice else 0,
        'frequent_urination': 1 if frequent_urination else 0,
        'body_temp': body_temp,
        'heart_rate': heart_rate,
        'respiratory_rate': respiratory_rate,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'glucose_level': glucose_level,
        'wbc_count': wbc_count,
        'rbc_count': rbc_count,
        'cholesterol': cholesterol,
        'hdl_cholesterol': hdl_cholesterol,
        'ldl_cholesterol': ldl_cholesterol,
        'triglycerides': triglycerides,
        'creatinine': creatinine,
        'bun': bun,
        'alt': alt,
        'ast': ast
    }

    # Create a toggle for real-time prediction mode
    realtime_mode = st.checkbox("Enable Real-time Prediction Mode", value=True, 
                              help="When enabled, predictions will update as you change symptoms or values")

    # Function to make prediction
    def make_prediction():
        with st.spinner("Analyzing patient data..."):
            # Process the patient data
            processed_data = preprocess_patient_data(patient_data, st.session_state.feature_names)

            # Predict disease
            prediction, probability, all_probabilities = predict_disease(processed_data, st.session_state.trained_models, model_choice)
            risk_level = get_risk_level(prediction, probability)

            # Get recommendations
            recommendations = get_recommendations(prediction, risk_level)

            # Get specialist recommendations
            specialists = get_specialist_recommendation(prediction)

            # Save results to session state
            st.session_state.prediction_results = {
                "prediction": prediction,
                "risk_level": risk_level,
                "probability": probability,
                "all_probabilities": all_probabilities,  # Store all disease probabilities
                "recommendations": recommendations,
                "specialists": specialists,
                "patient_data": patient_data.copy(),  # Make a copy to prevent reference issues
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
            }

    # Make real-time prediction if enabled and symptoms or vitals have changed
    if realtime_mode:
        # Create a hash of the current patient data to check for changes
        current_data_hash = hash(frozenset(patient_data.items()))

        # Check if we need to update the prediction
        needs_update = False

        # If this is the first prediction or data has changed since last prediction
        if 'last_prediction_hash' not in st.session_state:
            needs_update = True
            st.session_state.last_prediction_hash = current_data_hash
        elif st.session_state.last_prediction_hash != current_data_hash:
            needs_update = True
            st.session_state.last_prediction_hash = current_data_hash

        # Store the original patient data for clinical rule-based prediction adjustments
        st.session_state.last_patient_data = patient_data

        if needs_update:
            make_prediction()

    # Button to predict disease (always available as fallback)
    predict_col1, predict_col2, predict_col3 = st.columns([1, 1, 1])
    with predict_col2:
        if st.button("Update Prediction", use_container_width=True):
            make_prediction()

    # Display results if prediction was made
    if st.session_state.prediction_results:
        prediction = st.session_state.prediction_results["prediction"]
        risk_level = st.session_state.prediction_results["risk_level"]
        probability = st.session_state.prediction_results["probability"]
        recommendations = st.session_state.prediction_results["recommendations"]
        specialists = st.session_state.prediction_results["specialists"]

        # Divider
        st.markdown("---")

        # Create main result container with background color based on risk level
        risk_colors = {
            "Low": "#d4edda",      # Light green
            "Moderate": "#fff3cd", # Light yellow
            "High": "#f8d7da"      # Light red
        }

        result_container = st.container()
        with result_container:
            # Header with risk-based styling
            st.markdown(f"""
            <div style="
                padding: 15px;
                border-radius: 10px;
                background-color: {risk_colors.get(risk_level, '#ffffff')};
                margin-bottom: 20px;
                ">
                <h2 style="text-align: center; margin: 0;">Disease Prediction Results</h2>
            </div>
            """, unsafe_allow_html=True)

            # Create columns for primary results and recommendations
            col1, col2 = st.columns([1, 2])

            with col1:
                # Display prediction with appropriate color
                if prediction == "Normal":
                    st.success(f"📊 Prediction: {prediction}")
                elif risk_level == "Moderate":
                    st.warning(f"📊 Prediction: {prediction}")
                else:  # High
                    st.error(f"📊 Prediction: {prediction}")

                # Display risk level with appropriate color
                if risk_level == "Low":
                    st.success(f"⚠️ Risk Level: {risk_level}")
                elif risk_level == "Moderate":
                    st.warning(f"⚠️ Risk Level: {risk_level}")
                else:  # High
                    st.error(f"⚠️ Risk Level: {risk_level}")

                # Display probability with progress bar
                st.info(f"🔍 Confidence: {probability:.2f}%")
                st.progress(probability/100)

                # Create a visualization of all disease probabilities
                if "all_probabilities" in st.session_state.prediction_results:
                    st.subheader("Disease Likelihood")

                    # Sort probabilities by value in descending order
                    all_probs = st.session_state.prediction_results["all_probabilities"]
                    sorted_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))

                    # Only show top 5 diseases for cleaner display
                    top_diseases = list(sorted_probs.keys())[:5]
                    top_probs = [sorted_probs[disease] for disease in top_diseases]

                    # Create colors for the bars (highlight the predicted disease)
                    bar_colors = ['#FF9999' if disease == prediction else '#9ECAE1' for disease in top_diseases]

                    # Create horizontal bar chart using plotly
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        y=top_diseases,
                        x=top_probs,
                        orientation='h',
                        marker_color=bar_colors,
                        text=[f"{p:.1f}%" for p in top_probs],
                        textposition='outside'
                    ))

                    fig.update_layout(
                        height=200,
                        margin=dict(l=0, r=0, t=0, b=0),
                        xaxis_title=None,
                        yaxis_title=None,
                        xaxis=dict(range=[0, 100])
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # Recommended specialists
                st.subheader("👨‍⚕️ Recommended Specialists")
                for specialist in specialists:
                    st.markdown(f"- {specialist}")

                # Create a button to find nearby hospitals/book bed if risk is high
                if risk_level == "High":
                    if st.button("Find Available Hospital Beds"):
                        # Navigate to the bed booking page
                        st.session_state.booking_requested = True
                        st.rerun()  # Use modern rerun instead of experimental_rerun

            with col2:
                # Display last update time if available
                if "timestamp" in st.session_state.prediction_results:
                    st.caption(f"Last updated: {st.session_state.prediction_results['timestamp']}")

                # Display factors influencing prediction
                st.subheader("🔬 Key Factors in Prediction")

                # Show significant symptoms and values that influenced the prediction
                factors_tab, advice_tab = st.tabs(["Contributing Factors", "Medical Advice"])

                with factors_tab:
                    # Convert binary symptoms to readable format
                    symptom_list = []

                    if st.session_state.prediction_results["patient_data"]["fever"] == 1:
                        symptom_list.append("Fever")
                    if st.session_state.prediction_results["patient_data"]["cough"] == 1:
                        symptom_list.append("Cough")
                    if st.session_state.prediction_results["patient_data"]["fatigue"] == 1:
                        symptom_list.append("Fatigue")
                    if st.session_state.prediction_results["patient_data"]["difficulty_breathing"] == 1:
                        symptom_list.append("Difficulty Breathing")
                    if st.session_state.prediction_results["patient_data"]["headache"] == 1:
                        symptom_list.append("Headache")
                    if st.session_state.prediction_results["patient_data"]["chest_pain"] == 1:
                        symptom_list.append("Chest Pain")
                    if st.session_state.prediction_results["patient_data"]["nausea"] == 1:
                        symptom_list.append("Nausea")
                    if st.session_state.prediction_results["patient_data"]["jaundice"] == 1:
                        symptom_list.append("Jaundice")
                    if st.session_state.prediction_results["patient_data"]["frequent_urination"] == 1:
                        symptom_list.append("Frequent Urination")

                    # Display list of active symptoms
                    if symptom_list:
                        st.write("**Symptoms detected:**")
                        for symptom in symptom_list:
                            st.markdown(f"- {symptom}")
                    else:
                        st.write("No specific symptoms detected.")

                    # Show abnormal vital signs and test results
                    abnormal_values = []

                    # Check vitals
                    if st.session_state.prediction_results["patient_data"]["body_temp"] > 37.5:
                        abnormal_values.append(f"Elevated body temperature: {st.session_state.prediction_results['patient_data']['body_temp']}°C")
                    if st.session_state.prediction_results["patient_data"]["heart_rate"] > 100:
                        abnormal_values.append(f"Elevated heart rate: {st.session_state.prediction_results['patient_data']['heart_rate']} bpm")
                    if st.session_state.prediction_results["patient_data"]["respiratory_rate"] > 20:
                        abnormal_values.append(f"Elevated respiratory rate: {st.session_state.prediction_results['patient_data']['respiratory_rate']} breaths/min")
                    if st.session_state.prediction_results["patient_data"]["systolic_bp"] > 140 or st.session_state.prediction_results["patient_data"]["systolic_bp"] < 90:
                        abnormal_values.append(f"Abnormal blood pressure: {st.session_state.prediction_results['patient_data']['systolic_bp']}/{st.session_state.prediction_results['patient_data']['diastolic_bp']} mmHg")

                    # Check specific disease indicators
                    if st.session_state.prediction_results["patient_data"]["glucose_level"] > 126:
                        abnormal_values.append(f"Elevated glucose: {st.session_state.prediction_results['patient_data']['glucose_level']} mg/dL")
                    if st.session_state.prediction_results["patient_data"]["cholesterol"] > 200:
                        abnormal_values.append(f"Elevated cholesterol: {st.session_state.prediction_results['patient_data']['cholesterol']} mg/dL")
                    if st.session_state.prediction_results["patient_data"]["creatinine"] > 1.2:
                        abnormal_values.append(f"Elevated creatinine: {st.session_state.prediction_results['patient_data']['creatinine']} mg/dL")
                    if st.session_state.prediction_results["patient_data"]["alt"] > 40 or st.session_state.prediction_results["patient_data"]["ast"] > 40:
                        abnormal_values.append(f"Elevated liver enzymes: ALT={st.session_state.prediction_results['patient_data']['alt']} U/L, AST={st.session_state.prediction_results['patient_data']['ast']} U/L")

                    # Display abnormal values
                    if abnormal_values:
                        st.write("**Abnormal test results and vital signs:**")
                        for value in abnormal_values:
                            st.markdown(f"- {value}")
                    else:
                        st.write("No significant abnormal values detected.")

                with advice_tab:
                    # Display recommendations
                    for rec in recommendations:
                        st.markdown(f"✅ {rec}")

                # If high risk, recommend hospitals
                if risk_level == "High":
                    hospitals = get_hospital_data()
                    nearby_hospitals = recommend_hospitals(hospitals, prediction)

                    with st.expander("Nearby Hospitals with Available Beds", expanded=True):
                        if nearby_hospitals:
                            st.markdown("### Recommended Hospitals:")

                            # Create a nicer hospital display
                            for i, hospital in enumerate(nearby_hospitals):
                                st.markdown(f"""
                                <div style="
                                    background-color: #f8f9fa;
                                    border-radius: 10px;
                                    padding: 15px;
                                    margin-bottom: 10px;
                                    border-left: 5px solid {'#4CAF50' if hospital['available_beds'] > 30 else '#FFC107' if hospital['available_beds'] > 10 else '#F44336'};
                                ">
                                    <h4>{hospital['name']}</h4>
                                    <p>
                                        <span style="font-weight: bold;">Distance:</span> {hospital['distance']:.2f} km | 
                                        <span style="font-weight: bold;">Available Beds:</span> {hospital['available_beds']}/{hospital['total_beds']} | 
                                        <span style="font-weight: bold;">Rating:</span> {hospital['rating']}/5
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)

elif page == "Hospital Management":
    st.header("Hospital Resource Management")

    # Create tabs for different hospital resources
    tabs = st.tabs(["Bed Status", "ICU Status", "Hospital Occupancy", "Staff Status"])

    with tabs[0]:  # Bed Status
        st.subheader("Hospital Bed Status")

        # Get hospital data
        hospitals = get_hospital_data()
        bed_status = get_hospital_bed_status(hospitals)

        # Create columns for different displays
        col1, col2 = st.columns([1, 1])

        with col1:
            # Create a line chart for hospital occupancy
            fig = px.bar(
                bed_status, 
                x='hospital_name', 
                y=['available_beds', 'occupied_beds'], 
                title='Bed Availability by Hospital',
                labels={'value': 'Number of Beds', 'hospital_name': 'Hospital', 'variable': 'Status'},
                color_discrete_map={'available_beds': '#4CAF50', 'occupied_beds': '#F44336'},
                barmode='stack'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Create a table view of bed status
            st.subheader("Detailed Bed Status")
            status_df = pd.DataFrame({
                'Hospital': bed_status['hospital_name'],
                'Available Beds': bed_status['available_beds'],
                'Occupied Beds': bed_status['occupied_beds'],
                'Total Beds': bed_status['total_beds'],
                'Occupancy Rate (%)': (bed_status['occupied_beds'] / bed_status['total_beds'] * 100).round(1)
            })

            st.dataframe(status_df, use_container_width=True)

            # Show overall statistics
            total_beds = status_df['Total Beds'].sum()
            total_available = status_df['Available Beds'].sum()
            total_occupied = status_df['Occupied Beds'].sum()
            overall_occupancy = (total_occupied / total_beds * 100).round(1)

            st.metric(
                label="Overall Bed Occupancy Rate", 
                value=f"{overall_occupancy}%",
                delta=f"{total_available} beds available"
            )

    with tabs[1]:  # ICU Status
        st.subheader("ICU Bed Status")

        icu_status = pd.DataFrame({
            'hospital_name': [h['name'] for h in hospitals],
            'available_icu': [h.get('available_icu', 0) for h in hospitals],
            'total_icu': [h.get('total_icu', 0) for h in hospitals],
            'occupied_icu': [h.get('total_icu', 0) - h.get('available_icu', 0) for h in hospitals]
        })

        # Create columns for different displays
        col1, col2 = st.columns([1, 1])

        with col1:
            # Create a line chart for ICU occupancy
            fig = px.bar(
                icu_status, 
                x='hospital_name', 
                y=['available_icu', 'occupied_icu'], 
                title='ICU Availability by Hospital',
                labels={'value': 'Number of ICU Beds', 'hospital_name': 'Hospital', 'variable': 'Status'},
                color_discrete_map={'available_icu': '#4CAF50', 'occupied_icu': '#F44336'},
                barmode='stack'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Create a table view of ICU status
            st.subheader("Detailed ICU Status")
            icu_df = pd.DataFrame({
                'Hospital': icu_status['hospital_name'],
                'Available ICU': icu_status['available_icu'],
                'Occupied ICU': icu_status['occupied_icu'],
                'Total ICU Beds': icu_status['total_icu'],
                'ICU Occupancy Rate (%)': (icu_status['occupied_icu'] / icu_status['total_icu'] * 100).round(1)
            })

            st.dataframe(icu_df, use_container_width=True)

            # Show overall statistics
            total_icu = icu_df['Total ICU Beds'].sum()
            total_available_icu = icu_df['Available ICU'].sum()
            total_occupied_icu = icu_df['Occupied ICU'].sum()
            overall_icu_occupancy = (total_occupied_icu / total_icu * 100).round(1)

            st.metric(
                label="Overall ICU Occupancy Rate", 
                value=f"{overall_icu_occupancy}%",
                delta=f"{total_available_icu} ICU beds available"
            )

    with tabs[2]:  # Hospital Occupancy
        st.subheader("Hospital Occupancy Trends")

        # Create a hospital occupancy trend chart (simulated data)
        date_range = pd.date_range(start=datetime.datetime.now() - datetime.timedelta(days=30), 
                                   end=datetime.datetime.now(), 
                                   freq='D')

        # Create random trend lines for different hospitals
        hospital_trends = {}
        for hospital in [h['name'] for h in hospitals]:
            base_occupancy = np.random.randint(40, 80)
            trend = np.random.normal(loc=0, scale=5, size=len(date_range))
            occupancy = np.clip(base_occupancy + np.cumsum(trend)/10, 10, 100)
            hospital_trends[hospital] = occupancy

        # Create a dataframe
        trend_df = pd.DataFrame(hospital_trends, index=date_range)
        trend_df_melted = trend_df.reset_index().melt(id_vars='index', var_name='Hospital', value_name='Occupancy Rate')

        # Plot the data
        fig = px.line(
            trend_df_melted, 
            x='index', 
            y='Occupancy Rate', 
            color='Hospital',
            title='Hospital Occupancy Trends (Last 30 Days)',
            labels={'index': 'Date', 'Occupancy Rate': 'Occupancy Rate (%)'}
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Add some insights
        st.subheader("Occupancy Insights")

        # Calculate some statistics from the trend data
        avg_occupancy = trend_df.mean().sort_values(ascending=False)
        most_occupied = avg_occupancy.index[0]
        least_occupied = avg_occupancy.index[-1]

        # Create columns for displaying insights
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"🏆 Highest average occupancy: **{most_occupied}** ({avg_occupancy[most_occupied]:.1f}%)")

            # Find hospital with most fluctuation
            most_variable = trend_df.std().sort_values(ascending=False).index[0]
            st.warning(f"📊 Most variable occupancy: **{most_variable}** (Standard Deviation: {trend_df.std()[most_variable]:.1f}%)")

        with col2:
            st.success(f"🌟 Lowest average occupancy: **{least_occupied}** ({avg_occupancy[least_occupied]:.1f}%)")

            # Calculate 7-day trend
            current = trend_df.iloc[-1].mean()
            week_ago = trend_df.iloc[-8].mean()
            change = current - week_ago

            if change > 0:
                st.error(f"📈 Overall 7-day trend: **+{change:.1f}%** increase in occupancy")
            else:
                st.success(f"📉 Overall 7-day trend: **{change:.1f}%** decrease in occupancy")

    with tabs[3]:  # Staff Status
        st.subheader("Hospital Staff Status")

        # Create a staff status dataframe
        staff_data = []
        for hospital in hospitals:
            doctors = hospital.get('doctors', np.random.randint(20, 100))
            nurses = hospital.get('nurses', np.random.randint(40, 200))
            ratio = round(nurses / doctors, 1)

            staff_data.append({
                'Hospital': hospital['name'],
                'Doctors': doctors,
                'Nurses': nurses,
                'Nurse-to-Doctor Ratio': ratio,
                'Staff Availability (%)': hospital.get('staff_availability', np.random.randint(70, 100))
            })

        staff_df = pd.DataFrame(staff_data)

        # Create columns for display
        col1, col2 = st.columns([1, 1])

        with col1:
            # Create a bar chart for staff counts
            staff_melted = staff_df.melt(id_vars='Hospital', value_vars=['Doctors', 'Nurses'], 
                                         var_name='Staff Type', value_name='Count')

            fig = px.bar(
                staff_melted,
                x='Hospital',
                y='Count',
                color='Staff Type',
                title='Medical Staff Distribution by Hospital',
                barmode='group'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Show staff availability as a gauge chart
            fig = go.Figure()

            for i, row in staff_df.iterrows():
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=row['Staff Availability (%)'],
                    title={'text': row['Hospital']},
                    domain={'x': [0, 1], 'y': [i/len(staff_df), (i+0.9)/len(staff_df)]},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': 'green' if row['Staff Availability (%)'] >= 85 else 
                                         'orange' if row['Staff Availability (%)'] >= 70 else 'red'},
                        'steps': [
                            {'range': [0, 70], 'color': 'lightgray'},
                            {'range': [70, 85], 'color': 'gray'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 85
                        }
                    }
                ))

            fig.update_layout(
                height=500,
                title_text="Staff Availability by Hospital (%)"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Display the staff data as a table
        st.subheader("Detailed Staff Information")
        st.dataframe(staff_df, use_container_width=True)

        # Add some insights
        total_doctors = staff_df['Doctors'].sum()
        total_nurses = staff_df['Nurses'].sum()
        avg_ratio = staff_df['Nurse-to-Doctor Ratio'].mean()

        st.markdown(f"""
        ### Staffing Insights:
        - Total Medical Staff: **{total_doctors + total_nurses}** ({total_doctors} doctors, {total_nurses} nurses)
        - Average Nurse-to-Doctor Ratio: **{avg_ratio:.1f}**
        - Recommended Ratio (WHO): **3.0**
        """)

elif page == "Bed Booking":
    st.header("Hospital Bed Booking System")

    # Check if booking was requested from disease prediction
    if 'booking_requested' in st.session_state and st.session_state.booking_requested:
        st.info("Booking requested based on high-risk disease prediction")
        if 'prediction_results' in st.session_state and st.session_state.prediction_results:
            st.markdown(f"**Predicted Condition:** {st.session_state.prediction_results['prediction']}")
            st.markdown(f"**Risk Level:** {st.session_state.prediction_results['risk_level']}")

        # Reset the booking requested flag
        st.session_state.booking_requested = False

    # Create columns for input form
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Patient Information")
        patient_name = st.text_input("Patient Name")
        patient_age = st.number_input("Patient Age", min_value=0, max_value=120, value=30)
        patient_gender = st.selectbox("Patient Gender", ["Male", "Female", "Other"])
        contact_number = st.text_input("Contact Number")

        st.subheader("Medical Information")
        medical_condition = st.text_input("Medical Condition", 
                                         value=st.session_state.prediction_results['prediction'] if 
                                         'prediction_results' in st.session_state and st.session_state.prediction_results else "")

        urgency_level = st.selectbox("Urgency Level", 
                                    ["Critical", "High", "Medium", "Low"],
                                    index=0 if 'prediction_results' in st.session_state and 
                                    st.session_state.prediction_results and 
                                    st.session_state.prediction_results['risk_level'] == "High" else 2)

        st.subheader("Special Requirements")
        needs_ventilator = st.checkbox("Requires Ventilator")
        needs_icu = st.checkbox("Requires ICU")
        special_notes = st.text_area("Special Notes or Requirements")

    with col2:
        st.subheader("Hospital Selection")

        # Get hospital data
        hospitals = get_hospital_data()

        # Create filtered list of hospitals based on bed availability
        available_hospitals = [h for h in hospitals if h['available_beds'] > 0]

        if needs_icu:
            available_hospitals = [h for h in available_hospitals if h.get('available_icu', 0) > 0]

        if not available_hospitals:
            st.error("No hospitals with available beds match your requirements.")
            st.stop()

        # Create a hospital selector
        hospital_options = [h['name'] for h in available_hospitals]
        selected_hospital = st.selectbox("Select Hospital", hospital_options)

        # Get the selected hospital data
        selected_hospital_data = next(h for h in available_hospitals if h['name'] == selected_hospital)

        # Display hospital information
        st.markdown(f"""
        #### Hospital Information:
        - **Address**: {selected_hospital_data.get('address', 'Not available')}
        - **Available Beds**: {selected_hospital_data['available_beds']}
        - **ICU Beds Available**: {selected_hospital_data.get('available_icu', 0)}
        - **Rating**: {selected_hospital_data['rating']}/5
        """)

        # Calculate distance (for display purposes)
        distance = selected_hospital_data.get('distance', 0)
        st.info(f"📍 This hospital is approximately {distance:.1f} km away.")

        # Bed type selection
        bed_type = st.radio("Bed Type", ["General Ward", "Semi-Private", "Private", "ICU"], 
                           index=3 if needs_icu else 0,
                           disabled=needs_icu)  # Force ICU if needed

        # Date selection
        booking_date = st.date_input("Admission Date", value=datetime.datetime.now().date())

        # Display estimated cost
        bed_costs = {
            "General Ward": 500,
            "Semi-Private": 1200,
            "Private": 2000,
            "ICU": 5000
        }

        daily_cost = bed_costs[bed_type]
        st.markdown(f"""
        #### Estimated Costs:
        - **Daily Rate**: ${daily_cost}/day
        - **Expected Stay**: Varies based on condition
        """)

        # Booking button
        booking_button = st.button("Book Hospital Bed", use_container_width=True, type="primary")

    # Process booking
    if booking_button:
        if not patient_name or not contact_number:
            st.error("Please fill in all required patient information.")
        else:
            with st.spinner("Processing your booking..."):
                # Create a booking dictionary
                booking_data = {
                    'patient_name': patient_name,
                    'patient_age': patient_age,
                    'patient_gender': patient_gender,
                    'contact_number': contact_number,
                    'medical_condition': medical_condition,
                    'urgency_level': urgency_level,
                    'needs_ventilator': needs_ventilator,
                    'needs_icu': needs_icu,
                    'special_notes': special_notes,
                    'hospital_name': selected_hospital,
                    'bed_type': bed_type,
                    'booking_date': booking_date
                }

                # Call the booking function
                booking_result = book_hospital_bed(booking_data, selected_hospital_data)

                if booking_result['success']:
                    st.session_state.booking_status = {
                        'success': True,
                        'booking_id': booking_result['booking_id'],
                        'hospital': selected_hospital,
                        'bed_type': bed_type,
                        'admission_date': booking_date
                    }
                else:
                    st.session_state.booking_status = {
                        'success': False,
                        'error': booking_result['error']
                    }

    # Display booking status if available
    if st.session_state.booking_status:
        st.markdown("---")

        if st.session_state.booking_status['success']:
            st.success("Bed booked successfully!")

            st.markdown(f"""
            ### Booking Confirmation

            **Booking ID**: {st.session_state.booking_status['booking_id']}  
            **Hospital**: {st.session_state.booking_status['hospital']}  
            **Bed Type**: {st.session_state.booking_status['bed_type']}  
            **Admission Date**: {st.session_state.booking_status['admission_date'].strftime('%B %d, %Y')}

            Please arrive at the hospital 30 minutes before your scheduled time with all necessary medical documents 
            and identification. Contact the hospital for any changes to your booking.
            """)

            # Add a map of the hospital location (simulated)
            st.subheader("Hospital Location")
            hospital_data = next(h for h in hospitals if h['name'] == st.session_state.booking_status['hospital'])

            # Create a simple map using plotly
            fig = px.scatter_mapbox(
                pd.DataFrame([{
                    'lat': hospital_data.get('latitude', 40.7128), 
                    'lon': hospital_data.get('longitude', -74.0060),
                    'name': hospital_data['name']
                }]),
                lat="lat", lon="lon", hover_name="name", zoom=13, height=300
            )
            fig.update_layout(mapbox_style="open-street-map")
            fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)

            # Reset button
            if st.button("Make Another Booking"):
                st.session_state.booking_status = None
                st.rerun()

        else:
            st.error(f"Booking failed: {st.session_state.booking_status['error']}")

            # Suggestion for alternatives
            st.info("Please try another hospital or bed type, or contact hospital directly.")

    # Display general information about bed availability
    if not st.session_state.booking_status:
        st.markdown("---")
        st.subheader("Current Bed Availability Overview")

        # Create a bar chart showing bed availability across hospitals
        bed_data = pd.DataFrame({
            'Hospital': [h['name'] for h in hospitals],
            'Available Beds': [h['available_beds'] for h in hospitals],
            'Available ICU': [h.get('available_icu', 0) for h in hospitals]
        })

        fig = px.bar(
            bed_data,
            x='Hospital',
            y=['Available Beds', 'Available ICU'],
            title='Current Bed Availability',
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "Doctor Finder":
    st.header("Doctor Finder")

    # Set up columns
    col1, col2 = st.columns([1, 2])

    with col1:
        # Specialty filter
        specialty_options = [
            "General Medicine", "Cardiology", "Neurology", "Oncology", 
            "Pediatrics", "Orthopedics", "Dermatology", "ENT", 
            "Ophthalmology", "Psychiatry", "Pulmonology", "Endocrinology",
            "Gastroenterology", "Nephrology", "Urology", "Gynecology"
        ]

        # Get specialty from prediction if available
        default_specialty = None
        if 'prediction_results' in st.session_state and st.session_state.prediction_results:
            specialists = st.session_state.prediction_results['specialists']
            if specialists:
                # Extract the specialty part from the specialist title
                for specialist in specialists:
                    for option in specialty_options:
                        if option.lower() in specialist.lower():
                            default_specialty = option
                            break
                    if default_specialty:
                        break

        # If a specialty is found in prediction, set it as default
        specialty_index = specialty_options.index(default_specialty) if default_specialty else 0
        specialty = st.selectbox("Select Specialty", specialty_options, index=specialty_index)

        # Hospital filter
        hospitals = get_hospital_data()
        hospital_options = ["All Hospitals"] + [h['name'] for h in hospitals]
        selected_hospital = st.selectbox("Select Hospital", hospital_options)

        # Experience filter
        min_experience = st.slider("Minimum Years of Experience", 0, 40, 0)

        # Rating filter
        min_rating = st.slider("Minimum Rating", 1.0, 5.0, 3.0, 0.5)

        # Gender preference
        gender_preference = st.radio("Doctor Gender Preference", ["Any", "Male", "Female"])

        # Available today filter
        available_today = st.checkbox("Available Today")

        # Language preference
        language_options = ["English", "Spanish", "French", "Mandarin", "Hindi", "Arabic", "Russian"]
        selected_languages = st.multiselect("Preferred Languages", language_options, default=["English"])

        # Apply filters button
        apply_filters = st.button("Apply Filters", use_container_width=True)

    with col2:
        # Get doctors by specialty
        all_doctors = get_doctors_by_specialty(specialty)

        # Apply filters
        filtered_doctors = all_doctors.copy()

        if selected_hospital != "All Hospitals":
            filtered_doctors = [d for d in filtered_doctors if d['hospital'] == selected_hospital]

        filtered_doctors = [d for d in filtered_doctors if d['experience'] >= min_experience]
        filtered_doctors = [d for d in filtered_doctors if d['rating'] >= min_rating]

        if gender_preference != "Any":
            filtered_doctors = [d for d in filtered_doctors if d['gender'] == gender_preference]

        if available_today:
            filtered_doctors = [d for d in filtered_doctors if d['available_today']]

        if selected_languages:
            filtered_doctors = [d for d in filtered_doctors if any(lang in d['languages'] for lang in selected_languages)]

        # Display filtered doctors
        st.subheader(f"Found {len(filtered_doctors)} doctors matching your criteria")

        if not filtered_doctors:
            st.warning("No doctors match your selected criteria. Try adjusting your filters.")

        # Display doctors
        for i, doctor in enumerate(filtered_doctors):
            with st.container():
                color = '#4CAF50' if doctor['rating'] >= 4.5 else '#FFC107' if doctor['rating'] >= 3.5 else '#F44336'
                available_badge = ' • <span style="color: green;">Available Today</span>' if doctor.get('available_today') else ''

                st.markdown(f"""
                <div style="
                    background-color: #f8f9fa;
                    border-radius: 10px;
                    padding: 15px;
                    margin-bottom: 10px;
                    border-left: 5px solid {color};
                ">
                    <h4 style="margin-top: 0;">{doctor['name']}</h4>
                    <p style="margin-bottom: 8px;">
                        <span style="font-weight: bold;">Hospital:</span> {doctor['hospital']} | 
                        <span style="font-weight: bold;">Experience:</span> {doctor.get('experience', 'N/A')} years | 
                        <span style="font-weight: bold;">Rating:</span> {doctor.get('rating', 4.0)}/5
                    </p>
                    <p style="margin-bottom: 8px;">
                        <span style="font-weight: bold;">Languages:</span> {', '.join(doctor.get('languages', ['English']))}
                    </p>
                    <p style="margin-bottom: 0;">
                        <span style="font-weight: bold;">Speciality:</span> {doctor.get('specialty', specialty)}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Create columns for buttons
                btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])

                with btn_col1:
                    st.button(f"View Profile #{i}", key=f"profile_{i}")

                with btn_col2:
                    st.button(f"Book Appointment #{i}", key=f"book_{i}")

                with btn_col3:
                    st.button(f"Contact #{i}", key=f"contact_{i}")

        # Pagination (basic implementation)
        if len(filtered_doctors) > 5:
            st.markdown("---")
            pag_col1, pag_col2, pag_col3, pag_col4, pag_col5 = st.columns([1, 1, 1, 1, 5])
            with pag_col1:
                st.button("« Previous")
            with pag_col2:
                st.button("1", type="primary")
            with pag_col3:
                st.button("2")
            with pag_col4:
                st.button("Next »")

elif page == "About":
    st.header("About ML Healthcare System")

    st.markdown("""
    ## About This System

    This ML-based Healthcare System combines advanced machine learning algorithms with healthcare data to provide accurate disease predictions and efficient hospital resource management.

    ### Key Features:

    - **Disease Prediction**: Analyzes patient symptoms and medical test results to predict potential diseases.
    - **Risk Assessment**: Evaluates the severity and urgency of medical conditions.
    - **Medical Recommendations**: Provides tailored advice based on the predicted condition.
    - **Hospital Management**: Tracks bed availability and resource utilization across hospitals.
    - **Bed Booking System**: Facilitates seamless hospital bed reservations.
    - **Doctor Finder**: Helps patients find specialists based on various criteria.

    ### How It Works:

    1. **Data Collection**: Patient symptoms, vital signs, and test results are gathered.
    2. **ML Processing**: The system processes this data using trained models.
    3. **Prediction**: The system predicts potential medical conditions and their risk levels.
    4. **Recommendation**: Based on predictions, the system provides medical advice and specialist recommendations.
    5. **Resource Management**: If needed, the system facilitates hospital resource allocation.

    ### Technologies Used:

    - Python
    - Streamlit
    - Scikit-learn and XGBoost for ML
    - Pandas and NumPy for data processing
    - Plotly and Matplotlib for data visualization

    ### Data Privacy Notice:

    Patient data entered into this system is processed locally and is not stored permanently. The system complies with healthcare data protection regulations.
    """)

    # Create columns for technical info and contact
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Model Information")
        st.markdown("""
        This system uses two primary machine learning models:

        1. **Random Forest Classifier**
           - Ensemble learning method
           - Uses multiple decision trees for robust predictions
           - Good for handling various types of medical data

        2. **XGBoost**
           - Advanced implementation of gradient boosting
           - High performance on structured medical data
           - Better handling of imbalanced classes for rare conditions

        Both models are trained on comprehensive medical datasets including patient symptoms, vital signs, and laboratory tests. Regular updates ensure the models maintain high accuracy with the latest medical knowledge.
        """)

    with col2:
        st.subheader("System Information")

        if 'trained_models' in st.session_state:
            models = st.session_state.trained_models
            features = len(st.session_state.feature_names) if 'feature_names' in st.session_state else 0

            st.markdown(f"""
            **Models Loaded**: ✅

            **Available Models**: 
            - Random Forest
            - XGBoost

            **Features Used**: {features}

            **Last Updated**: {datetime.datetime.now().strftime('%Y-%m-%d')}
            """)
        else:
            st.warning("Models not yet loaded. Please return to the prediction page.")

        st.info("For technical support or questions, please contact the system administrator.")
