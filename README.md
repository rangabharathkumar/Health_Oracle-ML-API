# Health Oracle — ML API

Lightweight FastAPI service that serves three pre-trained health prediction models:
- Obesity prediction
- Insomnia prediction
- Diabetes prediction

---

## Repository contents
- main.py — FastAPI application and route definitions
- requirements.txt — Python dependencies
- Obesity_Prediction_Model.pkl — obesity model (default)
- Obesity_Label_Encoder.pkl — encoder(s) for obesity model (default)
- Insomnia_model.pkl — insomnia model (default)
- Insomnia_label_encoders.pkl — encoder(s) for insomnia model (default)
- diabetes_prediction_model.pkl — diabetes model (default)

---

## Requirements
- Python 3.8+
- pip
- (Optional) virtualenv/venv

Install dependencies:
pip install -r requirements.txt

---

## Environment variables
main.py loads model file paths from environment variables with fallback defaults:

- OBESITY_MODEL — path to obesity model (default: Obesity_Prediction_Model.pkl)  
- OBESITY_ENCODER — path to obesity encoder(s) (default: Obesity_Label_Encoder.pkl)  
- INSOMNIA_MODEL — path to insomnia model (default: Insomnia_model.pkl)  
- INSOMNIA_ENCODER — path to insomnia encoder(s) (default: Insomnia_label_encoders.pkl)  
- DIABETES_MODEL — path to diabetes model (default: diabetes_prediction_model.pkl)

You can place these in a .env file for local development or configure them in your deployment environment.

---

## Run locally
Start the app (FastAPI app object name is `app` in main.py):
uvicorn main:app --reload --host 0.0.0.0 --port 8000

Health check:
GET http://localhost:8000/
Response:
{
  "message": "HealthOracle FastAPI is up!"
}

---

## API Endpoints

All endpoints accept JSON and return JSON with a "prediction" key.

1) POST /predict/obesity
- Input model: ObesityInput (field names and casing must match exactly)
{
  "Gender": "Female",
  "Age": 29,
  "Height": 160.0,
  "Weight": 60.0,
  "family_history_with_overweight": "no",
  "FAVC": "no",
  "FCVC": 2.0,
  "NCP": 2.0,
  "CAEC": "Sometimes",
  "SMOKE": "no",
  "CH2O": 2.0,
  "SCC": "no",
  "FAF": 1.0,
  "TUE": 1.0,
  "CALC": "Sometimes",
  "MTRANS": "Walking"
}

Example curl:
curl -X POST http://localhost:8000/predict/obesity \
  -H "Content-Type: application/json" \
  -d '{"Gender":"Female","Age":29,"Height":160.0,"Weight":60.0,"family_history_with_overweight":"no","FAVC":"no","FCVC":2.0,"NCP":2.0,"CAEC":"Sometimes","SMOKE":"no","CH2O":2.0,"SCC":"no","FAF":1.0,"TUE":1.0,"CALC":"Sometimes","MTRANS":"Walking"}'

Example Python:
import requests
resp = requests.post("http://localhost:8000/predict/obesity", json={...})
print(resp.json())

Response:
{"prediction": "<label_or_value>"}

Notes:
- main.py uses obesity_encoder.transform([input_dict]) before predicting — the encoder must accept a dict-like mapping for transform. If you get errors, convert the input into a pandas DataFrame with the correct column order as used during training.

2) POST /predict/insomnia
- Input model: InsomniaInput
{
  "Gender": "Male",
  "Age": 40,
  "Occupation": "Office",
  "Sleep_Duration": 6.0,
  "Quality_of_Sleep": 2,
  "Physical_Activity_Level": 2,
  "Stress_Level": 3,
  "BMI_Category": "Overweight",
  "Blood_Pressure": "Normal",
  "Heart_Rate": 72,
  "Daily_Steps": 4000
}

Example curl:
curl -X POST http://localhost:8000/predict/insomnia \
  -H "Content-Type: application/json" \
  -d '{"Gender":"Male","Age":40,"Occupation":"Office","Sleep_Duration":6.0,"Quality_of_Sleep":2,"Physical_Activity_Level":2,"Stress_Level":3,"BMI_Category":"Overweight","Blood_Pressure":"Normal","Heart_Rate":72,"Daily_Steps":4000}'

Response:
{"prediction": "<label_or_value>"}

Notes:
- main.py uses insomnia_encoder.transform([input_dict]) — ensure the saved encoder expects a mapping or adjust preprocessing to match training pipeline.

3) POST /predict/diabetes
- Input model: DiabetesInput (note: fields are lowercase)
{
  "gender": "Female",
  "age": 50,
  "occupation": "Teacher",
  "sleep_duration": 6.5,
  "quality_of_sleep": 2,
  "physical_activity": 1,
  "stress_level": 2,
  "bmi_category": "Obese",
  "blood_pressure": "High",
  "heart_rate": 80,
  "daily_steps": 2000
}

Example curl:
curl -X POST http://localhost:8000/predict/diabetes \
  -H "Content-Type: application/json" \
  -d '{"gender":"Female","age":50,"occupation":"Teacher","sleep_duration":6.5,"quality_of_sleep":2,"physical_activity":1,"stress_level":2,"bmi_category":"Obese","blood_pressure":"High","heart_rate":80,"daily_steps":2000}'

Response:
{"prediction": "<label_or_value>"}

Notes:
- main.py creates a list from input.dict().values() and passes it to diabetes_model.predict([input_list]). The order of values in input.dict().values() is insertion order based on the Pydantic model definition — ensure the model expects the same feature order. If results look wrong, explicitly reorder fields into the expected model input order before predicting.

---

## Troubleshooting & tips
- If you get pickling/unpickling errors, confirm the Python and library versions used during serialization match your runtime.
- Large model files (~60 MB) are included. For deployments, consider downloading models from object storage at container startup or using Git LFS.
- If encoder.transform(...) raises an error, inspect how encoders were saved (scikit-learn encoders typically expect arrays or DataFrame input) and adapt preprocessing accordingly.
- Validate incoming data strictly (add Pydantic field constraints or additional validation if needed).

---

## Testing
- Use curl or Postman for quick testing.
- Add unit tests for:
  - Pydantic validation
  - Prediction pipeline (preprocessing → model)
  - Model loading at startup
  - Error handling for bad input

---

## Deployment recommendations
- Serve with a production ASGI server: uvicorn + gunicorn (for FastAPI) behind a reverse proxy (nginx).
- Set model file environment variables in the deployment environment or download model artifacts at container startup.
- Lock dependency versions in requirements.txt.

---

## Contributing
- Fork, create a branch, add tests, and open a pull request.
- Include a description of changes and how to test them locally.
