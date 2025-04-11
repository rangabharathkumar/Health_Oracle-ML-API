from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load Models
obesity_model = pickle.load(open("Obesity_Prediction_Model.pkl", "rb"))
obesity_encoder = pickle.load(open("Obesity_Label_Encoder.pkl", "rb"))

insomnia_model = pickle.load(open("Insomnia_model.pkl", "rb"))
insomnia_encoder = pickle.load(open("Insomnia_label_encoders.pkl", "rb"))

diabetes_model = pickle.load(open("diabetes_prediction_model.pkl", "rb"))

app = FastAPI()

# Input Schemas
class ObesityInput(BaseModel):
    Gender: str
    Age: int
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

class InsomniaInput(BaseModel):
    Gender: str
    Age: int
    Occupation: str
    Sleep_Duration: float
    Quality_of_Sleep: int
    Physical_Activity_Level: int
    Stress_Level: int
    BMI_Category: str
    Blood_Pressure: str
    Heart_Rate: int
    Daily_Steps: int

class DiabetesInput(BaseModel):
    gender: str
    age: int
    occupation: str
    sleep_duration: float
    quality_of_sleep: int
    physical_activity: int
    stress_level: int
    bmi_category: str
    blood_pressure: str
    heart_rate: int
    daily_steps: int

@app.get("/")
def root():
    return {"message": "HealthOracle FastAPI is up!"}

@app.post("/predict/obesity")
def predict_obesity(input: ObesityInput):
    input_dict = input.dict()
    input_df = obesity_encoder.transform([input_dict])
    prediction = obesity_model.predict(input_df)
    return {"prediction": prediction[0]}

@app.post("/predict/insomnia")
def predict_insomnia(input: InsomniaInput):
    input_dict = input.dict()
    input_df = insomnia_encoder.transform([input_dict])
    prediction = insomnia_model.predict(input_df)
    return {"prediction": prediction[0]}

@app.post("/predict/diabetes")
def predict_diabetes(input: DiabetesInput):
    input_list = list(input.dict().values())
    prediction = diabetes_model.predict([input_list])
    return {"prediction": prediction[0]}
