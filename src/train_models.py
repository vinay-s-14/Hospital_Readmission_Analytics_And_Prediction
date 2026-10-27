from pathlib import Path
import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier


# ============================================================
# 1. FILE LOCATIONS
# ============================================================

DATA = "data/curated/encounters.csv"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA, low_memory=False)

print(f"Dataset loaded successfully: {len(df)} rows")


# ============================================================
# 3. TARGET
# ============================================================

target = "readmitted_30d"

if target not in df.columns:
    raise ValueError(
        f"Target column '{target}' was not found in the dataset."
    )

y = df[target].astype(int)


# ============================================================
# 4. FEATURES
# ============================================================

possible_features = [
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

features = [
    column
    for column in possible_features
    if column in df.columns
]

print("\nFeatures used:")
for feature in features:
    print(" -", feature)

X = df[features].copy()


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

print("\nCreating train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training records: {len(X_train)}")
print(f"Testing records:  {len(X_test)}")


# ============================================================
# 6. IDENTIFY CATEGORICAL AND NUMERICAL FEATURES
# ============================================================

# This avoids the Pandas warning in newer versions.
cat = X.select_dtypes(
    include=["object", "str", "string"]
).columns.tolist()

num = [
    column
    for column in X.columns
    if column not in cat
]

print("\nCategorical features:")
print(cat)

print("\nNumerical features:")
print(num)


# ============================================================
# 7. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]),
            num
        ),

        (
            "cat",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]),
            cat
        )
    ]
)


# ============================================================
# 8. MACHINE LEARNING MODELS
# ============================================================

models = {

    "logistic_regression":
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

    "random_forest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1
        ),

    "xgboost":
        XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )
}


# ============================================================
# 9. MLFLOW SETUP
# ============================================================

# IMPORTANT:
# Use SQLite instead of the old file:./mlruns backend.

mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)

mlflow.set_experiment(
    "hospital-readmission"
)


# ============================================================
# 10. TRAIN MODELS
# ============================================================

results = {}


for name, model in models.items():

    print("\n" + "=" * 60)
    print(f"TRAINING MODEL: {name}")
    print("=" * 60)

    pipeline = Pipeline([
        (
            "preprocess",
            preprocessor
        ),
        (
            "model",
            model
        )
    ])

    with mlflow.start_run(
        run_name=name
    ):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        print("Training...")

        pipeline.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        predictions = pipeline.predict(
            X_test
        )

        probabilities = pipeline.predict_proba(
            X_test
        )[:, 1]

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

        metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc)
        }

        # ----------------------------------------------------
        # PRINT RESULTS
        # ----------------------------------------------------

        print("\nModel Results:")

        print(
            f"Precision : {precision:.4f}"
        )

        print(
            f"Recall    : {recall:.4f}"
        )

        print(
            f"F1 Score  : {f1:.4f}"
        )

        print(
            f"ROC-AUC   : {roc_auc:.4f}"
        )

        # ----------------------------------------------------
        # MLFLOW PARAMETERS
        # ----------------------------------------------------

        mlflow.log_params(
            model.get_params()
        )

        # ----------------------------------------------------
        # MLFLOW METRICS
        # ----------------------------------------------------

        mlflow.log_metrics(
            metrics
        )

        # ----------------------------------------------------
        # SAVE MODEL USING JOBLIB
        # ----------------------------------------------------

        model_path = (
            MODEL_DIR /
            f"{name}.joblib"
        )

        joblib.dump(
            pipeline,
            model_path
        )

        print(
            f"\nModel saved: {model_path}"
        )

        # ----------------------------------------------------
        # MLFLOW MODEL LOGGING
        # ----------------------------------------------------

        # Use cloudpickle serialization.
        # This avoids the skops trusted-type problem
        # that occurred with XGBoost.

        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            serialization_format="cloudpickle"
        )

        results[name] = metrics



# ============================================================
# 11. CALIBRATED XGBOOST
# ============================================================

print("\n" + "=" * 60)
print("CALIBRATING XGBOOST")
print("=" * 60)

calibrated_xgb = CalibratedClassifierCV(
    estimator=XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    ),
    method="sigmoid",
    cv=3
)

calibrated_pipeline = Pipeline([
    (
        "preprocess",
        preprocessor
    ),
    (
        "model",
        calibrated_xgb
    )
])

with mlflow.start_run(run_name="xgboost_calibrated"):

    print("Training calibrated XGBoost...")

    calibrated_pipeline.fit(
        X_train,
        y_train
    )

    calibrated_probabilities = calibrated_pipeline.predict_proba(
        X_test
    )[:, 1]

    calibrated_predictions = (
        calibrated_probabilities >= 0.5
    ).astype(int)

    calibrated_precision = precision_score(
        y_test,
        calibrated_predictions,
        zero_division=0
    )

    calibrated_recall = recall_score(
        y_test,
        calibrated_predictions,
        zero_division=0
    )

    calibrated_f1 = f1_score(
        y_test,
        calibrated_predictions,
        zero_division=0
    )

    calibrated_roc_auc = roc_auc_score(
        y_test,
        calibrated_probabilities
    )

    from sklearn.metrics import brier_score_loss

    calibrated_brier = brier_score_loss(
        y_test,
        calibrated_probabilities
    )

    calibration_metrics = {
        "precision": float(calibrated_precision),
        "recall": float(calibrated_recall),
        "f1": float(calibrated_f1),
        "roc_auc": float(calibrated_roc_auc),
        "brier_score": float(calibrated_brier)
    }

    print("\nCalibrated XGBoost Results:")

    print(
        f"Precision : {calibrated_precision:.4f}"
    )

    print(
        f"Recall    : {calibrated_recall:.4f}"
    )

    print(
        f"F1 Score  : {calibrated_f1:.4f}"
    )

    print(
        f"ROC-AUC   : {calibrated_roc_auc:.4f}"
    )

    print(
        f"Brier     : {calibrated_brier:.4f}"
    )

    mlflow.log_params({
        "calibration_method": "sigmoid",
        "cv": 3,
        "base_model": "xgboost"
    })

    mlflow.log_metrics(
        calibration_metrics
    )

    calibrated_model_path = (
        MODEL_DIR /
        "xgboost_calibrated.joblib"
    )

    joblib.dump(
        calibrated_pipeline,
        calibrated_model_path
    )

    print(
        f"\nCalibrated model saved: {calibrated_model_path}"
    )

    mlflow.sklearn.log_model(
        calibrated_pipeline,
        name="calibrated_model",
        serialization_format="cloudpickle"
    )

results["xgboost_calibrated"] = calibration_metrics


# ============================================================
# 12. SAVE ALL METRICS
# ============================================================

metrics_file = (
    MODEL_DIR /
    "metrics.json"
)

metrics_file.write_text(
    json.dumps(
        results,
        indent=4
    ),
    encoding="utf-8"
)


# ============================================================
# 13. DISPLAY FINAL RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("ALL MODELS TRAINED SUCCESSFULLY")
print("=" * 60)

for model_name, metrics in results.items():

    print(f"\n{model_name}")

    print(
        f"  Precision : {metrics['precision']:.4f}"
    )

    print(
        f"  Recall    : {metrics['recall']:.4f}"
    )

    print(
        f"  F1        : {metrics['f1']:.4f}"
    )

    print(
        f"  ROC-AUC   : {metrics['roc_auc']:.4f}"
    )


print("\nMetrics saved to:")
print(metrics_file)

print("\nProject training completed successfully!")