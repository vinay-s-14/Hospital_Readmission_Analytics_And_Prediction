# Hospital Readmission Analytics and Prediction

**Course:** Data Engineering and MLOps  
**Project:** Hospital Readmission Analytics and Prediction Using Public Data

This project implements the assignment brief using the **UCI Diabetes 130-US Hospitals for Years 1999–2008** public dataset.

## What is included

- Reproducible public-data ingestion with Python/Pandas.
- Raw → staging → curated ETL layers.
- PostgreSQL analytical schema.
- Apache Airflow DAG for orchestration.
- Data-quality checks and rejected-record logging.
- Streamlit dashboard with five+ analytical views.
- Part 2 ML extension: Logistic Regression, Random Forest and XGBoost comparison.
- MLflow experiment tracking and model registration hooks.
- SHAP/feature-importance explainability.
- FastAPI inference service.
- Dockerfile for the inference service.
- Project report template with architecture, data dictionary, validation rules and limitations.

## Dataset

Download the official dataset from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008

Place `diabetic_data.csv` and `IDS_mapping.csv` in `data/raw/`, or run:

```bash
python src/download_data.py
```

The dataset is public and de-identified, but it contains demographic fields such as age, gender and race. Do not add any private patient information.

## Part 1 quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
python src/download_data.py
python src/etl.py
python src/load_postgres.py
streamlit run app/dashboard.py
```

For a smoke test without the real dataset:

```bash
python src/generate_sample.py
python src/etl.py --input data/raw/diabetic_data_sample.csv
```

## PostgreSQL

Set credentials through environment variables:

```text
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=hospital_readmission
POSTGRES_USER=postgres
POSTGRES_PASSWORD=change_me
```

Never hard-code credentials.

## Airflow

The DAG is in:

`dags/hospital_readmission_pipeline.py`

It runs:

1. ingestion/availability check
2. ETL
3. data-quality validation
4. PostgreSQL load

## Part 2 ML

```bash
python src/train_models.py
streamlit run app/dashboard.py
uvicorn app.api:app --reload
```

The training script compares:

- Logistic Regression
- Random Forest
- XGBoost

Metrics include recall, precision, F1 and ROC-AUC. MLflow tracking is enabled when an MLflow server is available.

## Important modeling note

The UCI target is the dataset-defined `readmitted` field. The project converts `<30` to the positive class and `>30` / `NO` to the negative class for a binary early-readmission prototype.

This is an academic decision-support prototype, **not a clinical tool**. It has not been clinically validated.

## Submission checklist

- [x] Source code and ETL scripts
- [x] Airflow DAG
- [x] Database schema
- [x] Dataset source instructions
- [x] Architecture diagram specification
- [x] Data dictionary
- [x] Validation rules
- [x] Streamlit dashboard
- [x] Part 2 ML pipeline
- [x] FastAPI + Docker
- [x] README
- [ ] Add screenshots after running locally
- [ ] Add actual model metrics after training
