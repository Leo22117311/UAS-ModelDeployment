from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

with open("best_model.pkl", "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    encoders = data["encoders"]

app = FastAPI()

class ObesityInput(BaseModel):
    Gender: str
    Age: float
    Height: float
    Weight: float
    family_history_with_overweight: str
    FAVC: str
    FCVC: float
    NCP: float
    CAEC: str
    SMOKE: str
    CH2O: float
    SCC: str
    FAF: float
    TUE: float
    CALC: str
    MTRANS: str

@app.post("/predict")
def predict(data_in: ObesityInput):
    input_dict = data_in.dict()
    df = pd.DataFrame([input_dict])

    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    prediction = model.predict(df)[0]

    if "NObeyesdad" in encoders:
        label_encoder = encoders["NObeyesdad"]
        label = label_encoder.inverse_transform([prediction])[0]
    else:
        label = str(prediction)

    return {
        "prediction": int(prediction),
        "label": label
    }