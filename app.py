import streamlit as st
import pandas as pd
import pickle

st.title("Obesity Level Prediction")

@st.cache_resource
def load_model():
    with open('best_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['encoders']

model, encoders = load_model()

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

def preprocess_input(input_dict, encoders):
    df = pd.DataFrame([input_dict])
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])
    return df

input_data = get_user_input()

if st.button("Predict Obesity Level"):
    processed = preprocess_input(input_data, encoders)
    prediction = model.predict(processed)[0]

    if 'NObeyesdad' in encoders:
        label = encoders['NObeyesdad'].inverse_transform([prediction])[0]
    else:
        label = str(prediction)

    st.success(f"Predicted Class: {label} (code {prediction})")