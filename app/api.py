from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Hospital Readmission Risk API")

MODEL_PATH = Path("models/random_forest.joblib")
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

class Encounter(BaseModel):
    age: str | None = None
    gender: str | None = None
    race: str | None = None
    admission_type_id: int | None = None
    discharge_disposition_id: int | None = None
    admission_source_id: int | None = None
    time_in_hospital: int | None = None
    num_lab_procedures: int | None = None
    num_procedures: int | None = None
    num_medications: int | None = None
    number_outpatient: int | None = None
    number_emergency: int | None = None
    number_inpatient: int | None = None
    number_diagnoses: int | None = None
    diagnosis_group: str | None = None
    prior_admissions: int | None = None
    medication_count: int | None = None

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(encounter: Encounter):
    if model is None:
        return {"error": "Model not found. Run src/train_models.py first."}
    X = pd.DataFrame([encounter.model_dump()])
    probability = float(model.predict_proba(X)[:, 1][0])
    return {
        "readmission_30d_probability": probability,
        "prediction": int(probability >= 0.5),
        "disclaimer": "Academic decision-support prototype; not clinically validated."
    }
