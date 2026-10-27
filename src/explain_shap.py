from pathlib import Path
import joblib
import pandas as pd
import shap

MODEL_PATH = Path("models/xgboost.joblib")
DATA_PATH = Path("data/curated/encounters.csv")
OUTPUT_DIR = Path("reports/shap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading XGBoost model...")
pipeline = joblib.load(MODEL_PATH)

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

X = df[features]

print("Transforming features...")
preprocessor = pipeline.named_steps["preprocess"]
model = pipeline.named_steps["model"]

X_transformed = preprocessor.transform(X)

feature_names = preprocessor.get_feature_names_out()

print(f"Original features: {len(features)}")
print(f"Transformed features: {X_transformed.shape[1]}")

print("Creating SHAP explainer...")
explainer = shap.TreeExplainer(model)

print("Calculating SHAP values...")
shap_values = explainer.shap_values(X_transformed)

print("Creating global feature importance...")

importance = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": abs(shap_values).mean(axis=0)
})

importance = importance.sort_values(
    "mean_abs_shap",
    ascending=False
)

importance.to_csv(
    OUTPUT_DIR / "global_feature_importance.csv",
    index=False
)

print("\nTop 20 features:")
print(importance.head(20).to_string(index=False))

print("\nSHAP analysis completed successfully.")
print(f"Saved: {OUTPUT_DIR / 'global_feature_importance.csv'}")
