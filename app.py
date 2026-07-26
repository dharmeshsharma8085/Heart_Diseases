import streamlit as st
import pandas as pd
import joblib

model = joblib.load("le_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("🫀 Heart Disease Prediction System")
st.caption("Machine Learning Project by Dharmesh Sharma")
st.write("Please provide the following details:")

age = st.slider("Age", 1, 150, 40)

sex = st.selectbox("Sex", ["M", "F"])

Chest_pain = st.selectbox(
    "Chest Pain Type", ["ATA", "NAP", "TA", "ASY"]
)

Resting_BP = st.number_input(
    "Resting Blood Pressure (mm Hg)", 80, 200, 120
)

Cholesterol = st.number_input(
    "Cholesterol (mg/dL)", 100, 600, 200
)

FastingBS = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dL", [0, 1]
)

Resting_ecg = st.selectbox(
    "Resting ECG", ["Normal", "ST", "LVH"]
)

Max_hr = st.slider("Maximum Heart Rate", 60, 220, 150)

Exercise_angina = st.selectbox(
    "Exercise-Induced Angina", ["Yes", "No"]
)

Oldpeak = st.slider(
    "Oldpeak (ST Depression)", 0.0, 6.0, 1.0, 0.1
)

St_slope = st.selectbox(
    "ST Slope", ["Up", "Flat", "Down"]
)

if st.button("Predict"):

    exercise_code = "Y" if Exercise_angina == "Yes" else "N"

    raw_input = {
        "Age": age,
        "Sex_" + sex: 1,
        "ChestPainType_" + Chest_pain: 1,
        "RestingBP": Resting_BP,
        "Cholesterol": Cholesterol,
        "FastingBS": FastingBS,
        "RestingECG_" + Resting_ecg: 1,
        "MaxHR": Max_hr,
        "ExerciseAngina_" + exercise_code: 1,
        "Oldpeak": Oldpeak,
        "ST_Slope_" + St_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("⚠️ Higher predicted risk of heart disease.")
    else:
        st.success("✅ Lower predicted risk of heart disease.")

st.caption(
    "⚠️ Educational ML project only. This prediction is not a medical diagnosis."
)