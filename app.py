import streamlit as st
import pandas as pd
import requests

st.title("Obesity Level Prediction")

def get_user_input():
    return {
        "Gender": st.selectbox("Gender", ["Male", "Female"]),
        "Age": st.number_input("Age", 1, 100),
        "Height": st.number_input("Height (m)", 1.0, 2.5),
        "Weight": st.number_input("Weight (kg)", 20.0, 200.0),
        "family_history_with_overweight": st.selectbox("Family History Overweight", ["yes", "no"]),
        "FAVC": st.selectbox("Frequent High Calorie Food", ["yes", "no"]),
        "FCVC": st.slider("Vegetable Intake Frequency", 1.0, 3.0),
        "NCP": st.slider("Number of Meals", 1.0, 4.0),
        "CAEC": st.selectbox("Snacking Frequency", ["no", "Sometimes", "Frequently", "Always"]),
        "SMOKE": st.selectbox("Smoke?", ["yes", "no"]),
        "CH2O": st.slider("Daily Water Intake", 1.0, 3.0),
        "SCC": st.selectbox("Monitor Calories?", ["yes", "no"]),
        "FAF": st.slider("Physical Activity", 0.0, 3.0),
        "TUE": st.slider("Technology Use", 0.0, 3.0),
        "CALC": st.selectbox("Alcohol Intake", ["no", "Sometimes", "Frequently", "Always"]),
        "MTRANS": st.selectbox("Main Transport", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])
    }

input_data = get_user_input()

if st.button("Predict Obesity Level"):
    response = requests.post("http://localhost:8000/predict", json=input_data)

    if response.status_code == 200:
        result = response.json()
        st.success(f"Predicted Class: {result['label']} (code {result['prediction']})")
    else:
        st.error("Prediction failed. Please check your FastAPI backend.")