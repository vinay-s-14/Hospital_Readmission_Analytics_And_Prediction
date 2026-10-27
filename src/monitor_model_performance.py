from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

DATA_PATH = Path("data/curated/encounters.csv")
MODEL_PATH = Path("models/xgboost.joblib")
OUTPUT_DIR = Path("reports/monitoring")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)

features = [
    "age",
    "gender",
    "race",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "diagnosis_group",
    "prior_admissions",
    "medication_count"
]

target = "readmitted_30d"

X = df[features]
y = df[target]

print("Creating test split...")

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Loading XGBoost model...")
model = joblib.load(MODEL_PATH)

print("Generating predictions...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

metrics = {
    "model": "xgboost",
    "test_records": int(len(X_test)),
    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, y_prob))
}

output_file = OUTPUT_DIR / "model_performance_report.json"

with open(output_file, "w") as file:
    json.dump(metrics, file, indent=4)

print("\nModel Performance")
print("-----------------")
print(f"Model      : {metrics['model']}")
print(f"Test rows  : {metrics['test_records']}")
print(f"Precision  : {metrics['precision']:.4f}")
print(f"Recall     : {metrics['recall']:.4f}")
print(f"F1 Score   : {metrics['f1']:.4f}")
print(f"ROC-AUC    : {metrics['roc_auc']:.4f}")

print(f"\nSaved: {output_file}")
