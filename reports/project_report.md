# Hospital Readmission Analytics and Prediction Using Public Data

## 1. Abstract

This project implements a privacy-conscious healthcare data engineering and MLOps workflow for analyzing hospital admission and readmission patterns and building an academic readmission-risk prediction prototype. The implementation follows the assignment requirement to use anonymized public data, reproducible ingestion, ETL, PostgreSQL storage, dashboarding, and a later MLOps extension.

The selected source is the UCI Diabetes 130-US Hospitals for Years 1999–2008 dataset. The pipeline creates raw, staging, curated and analytical layers, applies data-quality rules, produces an error log, and exposes analytical views through Streamlit. Part 2 compares Logistic Regression, Random Forest and XGBoost, tracks experiments with MLflow, and provides a FastAPI inference service.

## 2. Problem Statement

Hospital readmission is an important operational and analytical problem. The objective is to prepare de-identified hospital encounter records, identify patterns associated with early readmission, and provide a reproducible foundation for an academic prediction model.

The project is not intended for clinical diagnosis, treatment, triage, or patient-level medical decision-making.

## 3. Objectives

1. Prepare anonymized hospital records for analysis.
2. Create a reproducible healthcare ETL workflow.
3. Store curated data in PostgreSQL.
4. Create dashboards for admission and readmission patterns.
5. Build a readmission-risk modeling pipeline.
6. Track and operationalize the model as an academic prototype.

## 4. Dataset

**Dataset:** Diabetes 130-US Hospitals for Years 1999–2008  
**Source:** UCI Machine Learning Repository  
**Dataset ID:** 296  
**Coverage:** 130 US hospitals/integrated delivery networks, 1999–2008  
**Rows:** 101,766  
**Features:** 47 listed in the UCI repository metadata, with encounter records containing additional source fields in the distributed CSV.

The dataset includes demographic information, admission/discharge attributes, diagnoses, procedures, medications, laboratory-related counts and readmission outcomes.

## 5. Architecture

```text
UCI Public Dataset
       |
       v
Python/Pandas Ingestion
       |
       v
Raw Landing Layer
       |
       v
Airflow DAG
       |
       +--> Validation --> Rejected Record Log
       |
       v
Staging / Cleaning
       |
       v
Curated Encounter Table
       |
       v
PostgreSQL Healthcare Mart
       |
       +--------------------+
       |                    |
       v                    v
Streamlit Dashboard      ML Pipeline
                            |
                            v
                    MLflow Experiments
                            |
                            v
                     Model Registry
                            |
                            v
                       FastAPI API
                            |
                            v
                         Docker
```

## 6. ETL Design

### Extraction
The ingestion layer downloads the UCI dataset through the `ucimlrepo` package or accepts a manually downloaded official CSV.

### Transformation
- Replace special `?` values with missing values.
- Remove exact duplicate records.
- Reject records missing required identifiers or target values.
- Create deterministic patient-safe surrogate keys.
- Convert age brackets to numeric midpoints.
- Create broad diagnosis groups.
- Calculate prior admission count.
- Create medication-count and readmission indicators.
- Preserve the raw source copy.

### Validation
Required fields include encounter identifier, patient identifier, length of stay and readmission label. Rejected records are written to `data/logs/rejected_records.csv`.

## 7. Analytical Tables

The main curated table is `healthcare.encounters`.

Important analytical fields include:
- `patient_key`
- `encounter_key`
- `age`
- `gender`
- `race`
- `admission_type_id`
- `discharge_disposition_id`
- `admission_source_id`
- `time_in_hospital`
- `diagnosis_group`
- `prior_admissions`
- `medication_count`
- `readmitted_30d`

## 8. Dashboard

The Streamlit application contains at least five meaningful views:

1. Readmission rate by age group.
2. Readmission rate by diagnosis group.
3. Length-of-stay distribution.
4. Admission-type pattern.
5. High-risk cohort table.

Screenshots should be added after local execution.

## 9. MLOps Extension

The binary target is derived from the dataset-defined `readmitted` field:
- `<30` → 1
- `>30` → 0
- `NO` → 0

The project compares:
- Logistic Regression
- Random Forest
- XGBoost

Metrics:
- Precision
- Recall
- F1
- ROC-AUC

MLflow records parameters, metrics and model artifacts. The best model can be registered and served through FastAPI.

## 10. Explainability and Monitoring

### 10.1 Explainability

SHAP (SHapley Additive Explanations) was implemented to explain the XGBoost model. The trained model pipeline transforms the 17 original input features into 41 encoded features. SHAP TreeExplainer was used to calculate the contribution of these features to the model predictions.

The global SHAP analysis identified the following as the most influential model features:

- discharge_disposition_id
- number_inpatient
- prior_admissions
- time_in_hospital
- num_medications
- num_lab_procedures
- number_diagnoses
- number_emergency
- diagnosis_group

The global feature importance results are saved in:

`reports/shap/global_feature_importance.csv`

The SHAP visualization is saved as:

`reports/shap/global_feature_importance.png`

These results describe which features contribute most to the model's predictions and should not be interpreted as causal relationships.

### 10.2 Data Quality Monitoring

A data-quality monitoring script was implemented to check the curated dataset.

The current monitoring result is:

- Rows: 101,766
- Columns: 57
- Missing cells: 374,017
- Duplicate rows: 0
- Missing required columns: 0
- Invalid time-in-hospital values: 0
- Invalid readmission values: 0
- Quality status: PASS

The monitoring report is saved as:

`reports/monitoring/data_quality_report.json`

Missing values are present in the source data, but the dataset passes the defined quality checks because missing values are handled during ETL preprocessing.

### 10.3 Model Performance Monitoring

Model performance monitoring was implemented for the XGBoost model using the test dataset.

Current results:

- Precision: 0.5000
- Recall: 0.0185
- F1 Score: 0.0357
- ROC-AUC: 0.6777

The report is saved as:

`reports/monitoring/model_performance_report.json`

The low recall and F1 score at the default classification threshold indicate that threshold tuning and further model improvement would be required before operational clinical use.

### 10.4 API Monitoring

The FastAPI `/predict` endpoint was monitored by sending 10 test prediction requests.

Results:

- Total requests: 10
- Successful requests: 10
- Failed requests: 0
- Success rate: 100.00%
- Average latency: 84.71 ms
- Maximum latency: 128.32 ms

The API monitoring report is saved as:

`reports/monitoring/api_monitoring_report.json`

### 10.5 Retraining Criteria

A retraining decision script was implemented using predefined monitoring thresholds:

- Minimum ROC-AUC: 0.60
- Minimum F1: 0.05
- Maximum API latency: 500 ms
- Minimum API success rate: 95%

The current monitoring result triggered retraining because the F1 score of 0.0357 is below the minimum threshold of 0.05.

The decision is saved as:

`reports/monitoring/retraining_decision.json`

This provides a documented, metric-based retraining criterion instead of relying only on an arbitrary schedule.

## 11. Data Dictionary

| Field | Meaning | Type |
|---|---|---|
| patient_key | Patient-safe surrogate key | string |
| encounter_key | Encounter identifier | string |
| race | Recorded race category | categorical |
| gender | Recorded gender | categorical |
| age | Original age bracket | categorical |
| age_midpoint | Numeric midpoint of age bracket | numeric |
| admission_type_id | Admission type code | integer |
| discharge_disposition_id | Discharge disposition code | integer |
| admission_source_id | Admission source code | integer |
| time_in_hospital | Length of stay in days | integer |
| num_lab_procedures | Laboratory procedure count | integer |
| num_procedures | Non-lab procedure count | integer |
| num_medications | Medication count | integer |
| number_outpatient | Prior-year outpatient visits | integer |
| number_emergency | Prior-year emergency visits | integer |
| number_inpatient | Prior-year inpatient visits | integer |
| number_diagnoses | Number of diagnoses | integer |
| diagnosis_group | Broad group derived from diagnosis code | categorical |
| prior_admissions | Prior encounter count within dataset | integer |
| medication_count | Count of medication fields populated | integer |
| readmitted_30d | Binary early-readmission target | integer |

## 12. Privacy and Ethics

Only public, de-identified data are used. No private patient records, names, addresses, contact details, credentials or confidential institutional data should be added. Database credentials are supplied through environment variables.

The UCI metadata notes that the public dataset still contains demographic information such as age, gender and race. These variables should therefore be handled carefully and used only for legitimate academic analysis.

## 13. Limitations

- The source covers 1999–2008 and may not represent current hospital practice.
- The data are selected inpatient encounters and are not a general population sample.
- The model is an academic prototype and has not been clinically validated.
- A random split is included for convenience; a patient-level or chronological split is preferable for a stronger evaluation.
- Readmission labels are dataset-defined and should not be interpreted as a complete clinical outcome definition.
- Fairness, calibration and drift need explicit evaluation before any real-world deployment.

## 14. Conclusion

The project demonstrates an end-to-end data engineering and MLOps architecture for hospital readmission analytics. It separates raw data, transformation, validation, analytical storage, visualization, model development and deployment. The design is reproducible and privacy-conscious and provides a foundation for the required Part 1 submission and Part 2 MLOps extension.

## 15. Run Evidence

The following execution evidence was generated during project validation:

1. Successful Airflow DAG execution showing `check_source`, `transform_and_validate`, and `load_postgres` tasks.
2. PostgreSQL verification showing the `healthcare.encounters` and `healthcare.readmission_summary` tables.
3. Streamlit dashboard showing 101,766 encounters, 11,357 30-day readmissions, an 11.16% readmission rate, and an average stay of 4.40 days.
4. MLflow experiment runs showing model parameters, metrics and artifacts.
5. MLflow Model Registry verification for the `hospital-readmission-xgboost` model.
6. FastAPI `/health` endpoint returning a successful model-loaded response.
7. FastAPI `/predict` endpoint returning a prediction and probability.
8. SHAP global feature-importance analysis and visualization.
9. Data-quality monitoring report with quality status PASS.
10. Model-performance monitoring report containing Precision, Recall, F1 Score and ROC-AUC.
11. API monitoring report showing 100% request success and measured latency.
12. Retraining decision report based on predefined performance thresholds.

## 16. Source

UCI Machine Learning Repository, Diabetes 130-US Hospitals for Years 1999–2008, Dataset ID 296, DOI 10.24432/C5230J.
