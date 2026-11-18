
import streamlit as st
import pandas as pd
import joblib

# Load trained model and preprocessing objects
model = joblib.load('KNN_heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("Heart Disease Prediction App")
st.markdown("Provide the following details:")

# User inputs
age = st.slider("Age", 18, 100, 40)

sex = st.selectbox("Sex", ["Male", "Female"])

chest_pain_type = st.selectbox(
    "Chest Pain Type",
    ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
)

resting_blood_pressure = st.number_input(
    "Resting Blood Pressure (mm Hg)", 80, 200, 120
)

cholesterol = st.number_input(
    "Cholesterol (mg/dl)", 100, 600, 200
)

fasting_blood_sugar = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
)

max_heart_rate = st.slider("Max Heart Rate Achieved", 60, 220, 150)

exercise_induced_angina = st.selectbox(
    "Exercise Induced Angina", ["Yes", "No"]
)

oldpeak = st.number_input(
    "Oldpeak (ST depression induced by exercise)", 0.0, 10.0, 1.0
)

st_depression_slope = st.selectbox(
    "Slope of the peak exercise ST segment",
    ["Upsloping", "Flat", "Downsloping"]
)

if st.button("Predict"):

    # Create input using the original training dataset column names
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [
            {
                "Typical Angina": "TA",
                "Atypical Angina": "ATA",
                "Non-Anginal Pain": "NAP",
                "Asymptomatic": "ASY"
            }[chest_pain_type]
        ],
        'RestingBP': [resting_blood_pressure],
        'Cholesterol': [cholesterol],
        'FastingBS': [
            1 if fasting_blood_sugar == "Yes" else 0
        ],
        'RestingECG': [
            {
                "Normal": "Normal",
                "ST-T Wave Abnormality": "ST",
                "Left Ventricular Hypertrophy": "LVH"
            }[resting_ecg]
        ],
        'MaxHR': [max_heart_rate],
        'ExerciseAngina': [
            "Y" if exercise_induced_angina == "Yes" else "N"
        ],
        'Oldpeak': [oldpeak],
        'ST_Slope': [
            {
                "Upsloping": "Up",
                "Flat": "Flat",
                "Downsloping": "Down"
            }[st_depression_slope]
        ]
    })

    # Apply the same one-hot encoding used during training
    input_data = pd.get_dummies(input_data)

    # Match the exact training columns and their order
    input_data = input_data.reindex(
        columns=model_columns,
        fill_value=0
    )

    # Scale the input
    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data_scaled)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("The patient is likely to have heart disease.")
    else:
        st.success("The patient is unlikely to have heart disease.")
