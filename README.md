# Obesity Level Prediction - Model Deployment

## Objective
Predict obesity levels based on lifestyle and demographic features, then deploy the model with FastAPI + Streamlit.

## Dataset
- Source:
- Features: Age, Weight, Family History, Diet Habits, Physical Activity, etc.
- Target: NObeyesedad (7 levels of obesity)

## Methods
- Preprocessing (encoding, scaling)
- Trained Logistic Regression & Random Forest
- Selected best-performing model (saved as `best_model.pkl`)

## Deployment
- Backend: FastAPI API (`inference.py`, `app.py`)
- Frontend: Streamlit form for user input & predictions
- Requirements: see `requirements.txt`

## Files
- `2702224811_Leony Sani Winata.ipynb`: training pipeline
- `best_model.pkl`: trained model
- `app.py`: Streamlit frontend
- `inference.py`: prediction script
- `requirements.txt`: dependencies

## Results
- Accuracy: 95%
