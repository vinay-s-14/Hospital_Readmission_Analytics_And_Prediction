"""
Hospital Readmission Analytics and Clinical Risk Prediction Dashboard.

Academic prototype using public, de-identified hospital encounter data
(UCI Diabetes 130-US Hospitals dataset, 1999–2008).
"""

import os

# Prevent OpenBLAS thread allocation failure on Windows
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib
from sqlalchemy import create_engine
from dotenv import load_dotenv


# ============================================================
# 1. PAGE CONFIGURATION & DARK THEME SETUP
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="Hospital Readmission Risk Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Dark Healthcare Clinical Analytics CSS
CUSTOM_CSS = """
<style>
    /* Global Background & Typography */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Main container padding */
    .block-container {
        padding-top: 1.75rem;
        padding-bottom: 3rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        max-width: 100% !important;
    }

    /* Header styling */
    .dashboard-header {
        border-bottom: 1px solid #1e293b;
        padding-bottom: 1.25rem;
        margin-bottom: 1.5rem;
    }
    .dashboard-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.025em;
        margin: 0 0 0.35rem 0;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .dashboard-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin: 0 0 0.35rem 0;
        font-weight: 400;
    }
    .academic-pill {
        display: inline-block;
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        color: #cbd5e1;
        font-size: 0.78rem;
        padding: 0.2rem 0.65rem;
        border-radius: 9999px;
    }

    /* KPI Cards */
    .kpi-container {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-container:hover {
        border-color: #374151;
        transform: translateY(-2px);
    }
    .kpi-label {
        color: #9ca3af;
        font-size: 0.82rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        color: #f9fafb;
        font-size: 1.85rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .kpi-sub {
        color: #6b7280;
        font-size: 0.76rem;
        margin-top: 0.25rem;
    }

    /* Clinical Cards */
    .clinical-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 1.4rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
        margin-bottom: 1.25rem;
    }
    .card-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #f3f4f6;
        margin-bottom: 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Risk Badges */
    .risk-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.45rem 1rem;
        border-radius: 9999px;
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .risk-low {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .risk-moderate {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .risk-high {
        background: rgba(239, 68, 68, 0.18);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.45);
    }

    /* Risk Detail Box */
    .risk-detail-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.75rem;
        margin-top: 1.1rem;
        padding-top: 1.1rem;
        border-top: 1px solid #1f2937;
    }
    .risk-detail-item {
        background: #182234;
        padding: 0.65rem 0.85rem;
        border-radius: 8px;
        border: 1px solid #243247;
    }
    .risk-detail-label {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-bottom: 0.2rem;
    }
    .risk-detail-value {
        font-size: 0.92rem;
        font-weight: 600;
        color: #e2e8f0;
    }

    /* Academic Guidance Callout */
    .academic-guidance-box {
        background: rgba(30, 41, 59, 0.5);
        border-left: 4px solid #38bdf8;
        padding: 0.85rem 1rem;
        border-radius: 4px 8px 8px 4px;
        font-size: 0.85rem;
        color: #cbd5e1;
        line-height: 1.45;
        margin-top: 1.1rem;
    }

    /* Encounter Profile Metrics */
    .patient-metric-box {
        background: #182234;
        border: 1px solid #243247;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        text-align: left;
    }
    .patient-metric-label {
        font-size: 0.74rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    .patient-metric-value {
        font-size: 1.15rem;
        color: #f8fafc;
        font-weight: 700;
    }

    /* Sidebar adjustments */
    section[data-testid="stSidebar"] {
        background-color: #0d131f;
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
    }
    
    /* Tabs custom styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0b0f19;
        border-bottom: 1px solid #1f2937;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px 8px 0px 0px;
        color: #94a3b8;
        font-size: 0.9rem;
        font-weight: 500;
        padding: 0.55rem 1.15rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        border-color: #38bdf8 !important;
        color: #38bdf8 !important;
        font-weight: 600;
    }

    /* Primary calculate button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        border: 1px solid #3b82f6;
        color: #ffffff;
        font-weight: 700;
        border-radius: 8px;
        padding: 0.65rem 1.25rem;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4);
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
        border-color: #60a5fa;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6);
        transform: translateY(-1px);
    }
    
    /* Footer Notice */
    .academic-disclaimer-footer {
        background: rgba(17, 24, 39, 0.8);
        border: 1px solid #1f2937;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        font-size: 0.82rem;
        color: #cbd5e1;
        line-height: 1.5;
        margin-top: 2.5rem;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# 2. CACHED DATA & MODEL LOADERS
# ============================================================

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load curated encounters dataset from CSV or PostgreSQL."""
    csv_path = Path("data/curated/encounters.csv")
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path, low_memory=False)
        except Exception as e:
            st.error(f"Error reading curated CSV: {e}")

    # Fallback to PostgreSQL
    try:
        user = os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("POSTGRES_PASSWORD", "change_me")
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        database = os.getenv("POSTGRES_DB", "hospital_readmission")

        connection_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        engine = create_engine(connection_url)
        return pd.read_sql("SELECT * FROM healthcare.encounters", engine)
    except Exception as e:
        st.error(f"Failed to connect to database or read CSV: {e}")
        return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_ml_model() -> Tuple[Optional[Any], str]:
    """
    Load trained ML model pipeline.
    Prefers calibrated XGBoost, falling back to standard XGBoost or Random Forest.
    """
    calibrated_path = Path("models/xgboost_calibrated.joblib")
    xgb_path = Path("models/xgboost.joblib")
    rf_path = Path("models/random_forest.joblib")

    if calibrated_path.exists():
        try:
            model = joblib.load(calibrated_path)
            return model, "Calibrated XGBoost (Sigmoid, 3-Fold CV)"
        except Exception as e:
            st.warning(f"Failed loading calibrated model: {e}. Trying fallback.")

    if xgb_path.exists():
        try:
            model = joblib.load(xgb_path)
            return model, "XGBoost Classifier"
        except Exception:
            pass

    if rf_path.exists():
        try:
            model = joblib.load(rf_path)
            return model, "Random Forest Classifier"
        except Exception:
            pass

    return None, "No Model Loaded"


@st.cache_data(show_spinner=False)
def load_metrics_json() -> Dict[str, Any]:
    """Load model performance metrics."""
    metrics_path = Path("models/metrics.json")
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


@st.cache_data(show_spinner=False)
def load_shap_feature_importance() -> Optional[pd.DataFrame]:
    """Load global SHAP feature importance CSV."""
    shap_path = Path("reports/shap/global_feature_importance.csv")
    if shap_path.exists():
        try:
            return pd.read_csv(shap_path)
        except Exception:
            return None
    return None


# Helper for feature name translation to clinical terms
def format_clinical_feature_name(raw_name: str) -> str:
    cleaned = raw_name.replace("num__", "").replace("cat__", "")
    mapping = {
        "discharge_disposition_id": "Discharge Disposition ID",
        "number_inpatient": "Prior Inpatient Admissions",
        "prior_admissions": "Cumulative Prior Admissions",
        "time_in_hospital": "Length of Stay (Days)",
        "num_medications": "Number of Medications",
        "num_lab_procedures": "Number of Lab Procedures",
        "number_diagnoses": "Number of Diagnoses",
        "number_emergency": "Emergency Room Visits",
        "number_outpatient": "Outpatient Encounters",
        "num_procedures": "Number of Procedures",
        "admission_source_id": "Admission Source ID",
        "admission_type_id": "Admission Type ID",
        "medication_count": "Distinct Medication Count",
        "age": "Age Group",
        "gender": "Patient Gender",
        "race": "Patient Race",
        "diagnosis_group": "Primary Diagnosis Group"
    }
    if cleaned in mapping:
        return mapping[cleaned]
    if cleaned.startswith("diagnosis_group_"):
        return f"Diagnosis: {cleaned.replace('diagnosis_group_', '')}"
    if cleaned.startswith("age_"):
        return f"Age: {cleaned.replace('age_', '')}"
    if cleaned.startswith("gender_"):
        return f"Gender: {cleaned.replace('gender_', '')}"
    if cleaned.startswith("race_"):
        return f"Race: {cleaned.replace('race_', '')}"
    return cleaned.replace("_", " ").title()


# Plotly Dark Theme Styling Helper
def apply_dark_theme(fig: go.Figure, title: Optional[str] = None, height: int = 380) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        font=dict(
            family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
            color="#cbd5e1",
            size=12
        ),
        margin=dict(l=40, r=30, t=50 if title else 25, b=40),
        height=height,
        xaxis=dict(
            gridcolor="#1f2937",
            zerolinecolor="#1f2937",
            tickfont=dict(color="#94a3b8")
        ),
        yaxis=dict(
            gridcolor="#1f2937",
            zerolinecolor="#1f2937",
            tickfont=dict(color="#94a3b8")
        )
    )
    if title:
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=15, color="#f8fafc", weight="bold"),
                x=0.01,
                y=0.96
            )
        )
    return fig


# ============================================================
# 3. LOAD DATA & INITIAL VALIDATIONS
# ============================================================

df = load_data()
model, model_name = load_ml_model()
metrics_dict = load_metrics_json()
shap_df = load_shap_feature_importance()

if df.empty:
    st.error("No hospital encounter data was found. Please ensure `data/curated/encounters.csv` or PostgreSQL is available.")
    st.stop()

required_core_columns = ["readmitted_30d", "time_in_hospital"]
missing = [c for c in required_core_columns if c not in df.columns]
if missing:
    st.error(f"Curated dataset is missing essential clinical columns: {missing}")
    st.stop()


# ============================================================
# 4. SIDEBAR ANALYTICAL FILTERS
# ============================================================

st.sidebar.markdown("## 🔎 Analytical Filters")
st.sidebar.caption("Filter the population cohort and select encounters for clinical risk prediction.")

filtered_df = df.copy()

# 1. Demographics Expander
with st.sidebar.expander("👤 Patient Demographics", expanded=True):
    # Age filter
    if "age" in df.columns:
        all_ages = sorted(df["age"].dropna().astype(str).unique().tolist())
        selected_age = st.multiselect("Age Group", all_ages, default=all_ages)
        if selected_age:
            filtered_df = filtered_df[filtered_df["age"].astype(str).isin(selected_age)]

    # Gender filter
    if "gender" in df.columns:
        all_genders = sorted(df["gender"].dropna().astype(str).unique().tolist())
        selected_gender = st.multiselect("Gender", all_genders, default=all_genders)
        if selected_gender:
            filtered_df = filtered_df[filtered_df["gender"].astype(str).isin(selected_gender)]

    # Race filter
    if "race" in df.columns:
        all_races = sorted(df["race"].dropna().astype(str).unique().tolist())
        selected_race = st.multiselect("Race / Ethnicity", all_races, default=all_races)
        if selected_race:
            filtered_df = filtered_df[filtered_df["race"].astype(str).isin(selected_race)]

# 2. Clinical & Diagnosis Expander
with st.sidebar.expander("🩺 Clinical & Encounter Attributes", expanded=True):
    # Diagnosis group filter
    if "diagnosis_group" in df.columns:
        all_diag = sorted(df["diagnosis_group"].dropna().astype(str).unique().tolist())
        selected_diag = st.multiselect("Diagnosis Group", all_diag, default=all_diag)
        if selected_diag:
            filtered_df = filtered_df[filtered_df["diagnosis_group"].astype(str).isin(selected_diag)]

    # Admission type filter
    if "admission_type_id" in df.columns:
        all_adm_types = sorted(df["admission_type_id"].dropna().unique().tolist())
        selected_adm = st.multiselect("Admission Type ID", all_adm_types, default=all_adm_types)
        if selected_adm:
            filtered_df = filtered_df[filtered_df["admission_type_id"].isin(selected_adm)]

# 3. Utilization & Length of Stay Sliders
with st.sidebar.expander("📊 Utilization & Stay Bounds", expanded=False):
    # Length of stay
    min_stay, max_stay = int(df["time_in_hospital"].min()), int(df["time_in_hospital"].max())
    stay_range = st.slider("Length of Stay (Days)", min_stay, max_stay, (min_stay, max_stay))
    filtered_df = filtered_df[
        (filtered_df["time_in_hospital"] >= stay_range[0]) & 
        (filtered_df["time_in_hospital"] <= stay_range[1])
    ]

    # Prior admissions
    if "prior_admissions" in df.columns:
        min_p, max_p = int(df["prior_admissions"].min()), int(df["prior_admissions"].max())
        if max_p > min_p:
            prior_range = st.slider("Cumulative Prior Admissions", min_p, max_p, (min_p, max_p))
            filtered_df = filtered_df[
                (filtered_df["prior_admissions"] >= prior_range[0]) & 
                (filtered_df["prior_admissions"] <= prior_range[1])
            ]

    # Number of medications
    if "num_medications" in df.columns:
        min_med, max_med = int(df["num_medications"].min()), int(df["num_medications"].max())
        med_range = st.slider("Number of Medications", min_med, max_med, (min_med, max_med))
        filtered_df = filtered_df[
            (filtered_df["num_medications"] >= med_range[0]) & 
            (filtered_df["num_medications"] <= med_range[1])
        ]

st.sidebar.markdown(f"**Cohort Size:** `{len(filtered_df):,}` of `{len(df):,}` encounters")


# ============================================================
# 5. SIDEBAR ENCOUNTER SELECTION & RISK PREDICTION TRIGGER
# ============================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Risk Score Calculation")
st.sidebar.caption("Select an encounter from the filtered cohort to compute patient-specific readmission risk.")

if filtered_df.empty:
    st.sidebar.warning("Current filters return zero encounters. Broaden filter criteria.")
    selected_row = None
else:
    # Sort options to provide immediate high-value clinical examples
    sort_mode = st.sidebar.selectbox(
        "Candidate Encounter Sorting:",
        ["High Inpatient Admissions", "Longest Length of Stay", "Recent Sequence", "Random Sample"],
        index=0
    )

    if sort_mode == "High Inpatient Admissions" and "number_inpatient" in filtered_df.columns:
        candidates = filtered_df.sort_values(
            ["number_inpatient", "prior_admissions", "time_in_hospital"], 
            ascending=False
        ).head(80)
    elif sort_mode == "Longest Length of Stay":
        candidates = filtered_df.sort_values("time_in_hospital", ascending=False).head(80)
    elif sort_mode == "Random Sample":
        candidates = filtered_df.sample(min(80, len(filtered_df)), random_state=42)
    else:
        candidates = filtered_df.head(80)

    # Build clear readable options
    encounter_options = candidates["encounter_id"].tolist()
    
    def format_encounter_label(eid):
        row_match = candidates[candidates["encounter_id"] == eid].iloc[0]
        age_str = row_match.get("age", "N/A")
        diag_str = str(row_match.get("diagnosis_group", "N/A"))
        stay_str = f"{row_match.get('time_in_hospital', 0)}d"
        inpat_str = f"{row_match.get('number_inpatient', 0)} inpat"
        return f"ID {eid} | {age_str} | {diag_str} | {stay_str} | {inpat_str}"

    selected_encounter_id = st.sidebar.selectbox(
        "Select Encounter Record:",
        options=encounter_options,
        format_func=format_encounter_label
    )

    # Optional manual lookup
    direct_lookup = st.sidebar.text_input("Or enter exact Encounter ID:", placeholder="e.g. 154315056")
    if direct_lookup.strip():
        try:
            lookup_id = int(direct_lookup.strip())
            if lookup_id in df["encounter_id"].values:
                selected_encounter_id = lookup_id
                st.sidebar.success(f"Encounter #{lookup_id} loaded!")
            else:
                st.sidebar.warning(f"Encounter ID {lookup_id} not found.")
        except ValueError:
            st.sidebar.warning("Enter a valid numerical ID.")

    selected_row = df[df["encounter_id"] == selected_encounter_id].iloc[0]

    # Quick sidebar summary box
    st.sidebar.markdown(f"""
    <div style="background: #182234; border: 1px solid #243247; border-radius: 8px; padding: 0.75rem; margin: 0.6rem 0 1rem 0; font-size: 0.82rem;">
        <div style="color: #94a3b8; font-weight: 600; margin-bottom: 0.3rem;">Selected Record Summary:</div>
        <div>• <b>Encounter ID:</b> <span style="color:#38bdf8;">{selected_row['encounter_id']}</span></div>
        <div>• <b>Age:</b> {selected_row.get('age', 'N/A')} | <b>Gender:</b> {selected_row.get('gender', 'N/A')}</div>
        <div>• <b>Diagnosis:</b> {selected_row.get('diagnosis_group', 'N/A')}</div>
        <div>• <b>Stay:</b> {selected_row.get('time_in_hospital', 'N/A')} days | <b>Medications:</b> {selected_row.get('num_medications', 'N/A')}</div>
        <div>• <b>Prior Inpatient:</b> {selected_row.get('number_inpatient', 0)}</div>
    </div>
    """, unsafe_allow_html=True)

    calc_clicked = st.sidebar.button("🧮 CALCULATE RISK SCORE", type="primary", use_container_width=True)


# ============================================================
# 6. RISK PREDICTION LOGIC (REAL MODEL INFERENCE)
# ============================================================

def compute_risk_prediction(model_obj, record: pd.Series) -> Dict[str, Any]:
    """
    Execute real model inference using predict_proba.
    Applies prototype stratification:
      0.0% - 20.0%: Low Risk
      20.0% - 40.0%: Moderate Risk
      40.0% - 100.0%: High Risk
    """
    if model_obj is None:
        return {
            "error": "Trained model pipeline not available.",
            "probability": None,
            "risk_pct": 0.0,
            "level": "UNAVAILABLE",
            "icon": "⚠️",
            "badge_class": "risk-moderate",
            "color": "#94a3b8",
            "explanation": "No trained model found. Verify models/xgboost_calibrated.joblib exists.",
            "recommendation": "Prediction pipeline inactive."
        }

    try:
        # Pass 1-row DataFrame through saved pipeline containing ColumnTransformer & classifier
        input_df = pd.DataFrame([record])
        prob = float(model_obj.predict_proba(input_df)[0][1])
        risk_pct = round(prob * 100, 1)

        if risk_pct < 20.0:
            level = "LOW RISK"
            icon = "🟢"
            badge_class = "risk-low"
            color = "#10b981"
            explanation = (
                f"The model estimates a {risk_pct:.1f}% probability of 30-day readmission, "
                "placing this encounter in the LOW RISK tier (< 20%) under Prototype Risk Stratification Thresholds."
            )
            recommendation = (
                "Routine discharge planning and standard outpatient primary care follow-up "
                "within 14–30 days are typically appropriate."
            )
        elif risk_pct < 40.0:
            level = "MODERATE RISK"
            icon = "🟡"
            badge_class = "risk-moderate"
            color = "#f59e0b"
            explanation = (
                f"The model estimates a {risk_pct:.1f}% probability of 30-day readmission, "
                "placing this encounter in the MODERATE RISK tier (20–40%) under Prototype Risk Stratification Thresholds."
            )
            recommendation = (
                "Enhanced follow-up review, medication reconciliation, and scheduled clinical "
                "check-in within 7–14 days may be considered based on appropriate clinical judgment."
            )
        else:
            level = "HIGH RISK"
            icon = "🔴"
            badge_class = "risk-high"
            color = "#ef4444"
            explanation = (
                f"The model estimates a {risk_pct:.1f}% probability of 30-day readmission, "
                "placing this encounter in the HIGH RISK tier (40–100%) under Prototype Risk Stratification Thresholds."
            )
            recommendation = (
                "High-touch care coordination, multidisciplinary case review, prompt post-discharge "
                "outreach (48–72 hours), and targeted disease management are advised under clinical judgment."
            )

        return {
            "error": None,
            "probability": prob,
            "risk_pct": risk_pct,
            "level": level,
            "icon": icon,
            "badge_class": badge_class,
            "color": color,
            "explanation": explanation,
            "recommendation": recommendation
        }
    except Exception as e:
        return {
            "error": f"Inference execution error: {str(e)}",
            "probability": None,
            "risk_pct": 0.0,
            "level": "ERROR",
            "icon": "⚠️",
            "badge_class": "risk-high",
            "color": "#ef4444",
            "explanation": f"Prediction failed: {e}",
            "recommendation": "Verify feature schema compatibility with the trained model."
        }


# Maintain prediction result in session state so it persists across tab switches
if "last_prediction" not in st.session_state or calc_clicked or (selected_row is not None and st.session_state.get("last_encounter_id") != selected_row["encounter_id"]):
    if selected_row is not None:
        st.session_state["last_prediction"] = compute_risk_prediction(model, selected_row)
        st.session_state["last_encounter_id"] = selected_row["encounter_id"]

current_pred = st.session_state.get("last_prediction", None)


# ============================================================
# 7. DASHBOARD HEADER
# ============================================================

st.markdown(f"""
<div class="dashboard-header">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
        <div>
            <h1 class="dashboard-title">🏥 Hospital Readmission Risk Analytics</h1>
            <p class="dashboard-subtitle">30-Day Readmission Prediction & Clinical Risk Stratification</p>
            <span class="academic-pill">Academic predictive analytics prototype using public, de-identified hospital encounter data.</span>
        </div>
        <div style="text-align: right;">
            <div style="background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 0.5rem 0.85rem; display: inline-block;">
                <span style="color: #10b981; font-size: 0.82rem;">● Active Model:</span>
                <span style="color: #f8fafc; font-weight: 600; font-size: 0.85rem;"> {model_name}</span>
                <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 0.15rem;">
                    ROC-AUC: <b>{metrics_dict.get('xgboost_calibrated', {}).get('roc_auc', 0.678):.3f}</b> | 
                    Brier: <b>{metrics_dict.get('xgboost_calibrated', {}).get('brier_score', 0.095):.3f}</b>
                </div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 8. ANALYTICAL TABS (ORGANIZED DASHBOARD VIEWS)
# ============================================================

tab_risk, tab_age, tab_diag, tab_stay, tab_adm, tab_cohort = st.tabs([
    "🎯 Risk Overview",
    "📊 Age Risk",
    "🩺 Diagnosis Analysis",
    "⏱️ Length of Stay",
    "🏥 Admission Pattern",
    "🚨 High-Risk Cohort"
])


# ============================================================
# TAB 1: 🎯 RISK OVERVIEW (MAIN CLINICAL RISK PREDICTION)
# ============================================================

with tab_risk:
    # 1. TOP KEY COHORT METRICS
    total_enc = len(filtered_df)
    tot_readm = int(filtered_df["readmitted_30d"].fillna(0).sum()) if total_enc > 0 else 0
    readm_rate = (tot_readm / total_enc * 100) if total_enc > 0 else 0.0
    avg_los = float(filtered_df["time_in_hospital"].mean()) if total_enc > 0 else 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">Filtered Encounters</div>
            <div class="kpi-value">{total_enc:,}</div>
            <div class="kpi-sub">Total active patient encounters</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">30-Day Readmissions</div>
            <div class="kpi-value" style="color: #f87171;">{tot_readm:,}</div>
            <div class="kpi-sub">Unplanned returns within 30 days</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">Cohort Readmission Rate</div>
            <div class="kpi-value" style="color: #fbbf24;">{readm_rate:.2f}%</div>
            <div class="kpi-sub">Baseline population risk level</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">Avg. Length of Stay</div>
            <div class="kpi-value" style="color: #38bdf8;">{avg_los:.2f} <span style="font-size: 1.05rem; font-weight: normal; color: #94a3b8;">days</span></div>
            <div class="kpi-sub">Inpatient hospitalization duration</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # 2. MAIN RISK PREDICTION: GAUGE & RISK CARD
    if current_pred is None or selected_row is None:
        st.info("Please select an encounter in the sidebar to view predictive risk assessment.")
    elif current_pred.get("error"):
        st.error(current_pred["error"])
    else:
        risk_score_pct = current_pred["risk_pct"]
        risk_level = current_pred["level"]
        risk_color = current_pred["color"]
        badge_class = current_pred["badge_class"]
        risk_icon = current_pred["icon"]

        col_gauge, col_card = st.columns([5, 7])

        with col_gauge:
            st.markdown('<div class="card-title">🎯 Predicted 30-Day Readmission Risk</div>', unsafe_allow_html=True)

            # High-end Plotly Dark Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score_pct,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "30-Day Readmission Probability", 'font': {'size': 14, 'color': '#94a3b8'}},
                number={
                    'suffix': "%",
                    'font': {'size': 44, 'color': risk_color, 'family': 'Arial, sans-serif'}
                },
                gauge={
                    'axis': {
                        'range': [0, 100],
                        'tickwidth': 1,
                        'tickcolor': "#475569",
                        'tickmode': 'array',
                        'tickvals': [0, 20, 40, 60, 80, 100],
                        'ticktext': ['0%', '20%', '40%', '60%', '80%', '100%'],
                        'tickfont': {'color': '#94a3b8', 'size': 11}
                    },
                    'bar': {'color': risk_color, 'thickness': 0.28},
                    'bgcolor': "rgba(15, 23, 42, 0.7)",
                    'borderwidth': 1,
                    'bordercolor': "#334155",
                    'steps': [
                        {'range': [0, 20], 'color': 'rgba(16, 185, 129, 0.22)'},
                        {'range': [20, 40], 'color': 'rgba(245, 158, 11, 0.22)'},
                        {'range': [40, 100], 'color': 'rgba(239, 68, 68, 0.22)'}
                    ],
                    'threshold': {
                        'line': {'color': risk_color, 'width': 3},
                        'thickness': 0.8,
                        'value': risk_score_pct
                    }
                }
            ))

            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': '#f8fafc'},
                margin=dict(l=25, r=25, t=35, b=25),
                height=295
            )

            st.plotly_chart(fig_gauge, use_container_width=True)

            # Prototype Threshold Legend
            gauge_legend_html = (
                '<div style="background: #182234; border: 1px solid #243247; border-radius: 8px; padding: 0.65rem 0.85rem; text-align: center; margin-top: -0.5rem;">'
                '<div style="font-size: 0.72rem; color: #94a3b8; margin-bottom: 0.35rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">'
                'Prototype Risk Stratification Thresholds'
                '</div>'
                '<div style="display: flex; justify-content: space-around; font-size: 0.78rem;">'
                '<div><span style="color:#34d399;">●</span> 0–20%: <b>Low Risk</b></div>'
                '<div><span style="color:#fbbf24;">●</span> 20–40%: <b>Moderate Risk</b></div>'
                '<div><span style="color:#f87171;">●</span> 40–100%: <b>High Risk</b></div>'
                '</div>'
                '</div>'
            )
            st.markdown(gauge_legend_html, unsafe_allow_html=True)

        with col_card:
            historical_outcome_html = ""
            if selected_row is not None and "readmitted_30d" in selected_row and pd.notna(selected_row["readmitted_30d"]):
                outcome_val = int(selected_row["readmitted_30d"])
                outcome_text = "Readmitted <30 Days (1)" if outcome_val == 1 else "Not Readmitted <30 Days (0)"
                historical_outcome_html = (
                    '<div class="risk-detail-item">'
                    '<div class="risk-detail-label">Historical Outcome (for selected encounter)</div>'
                    f'<div class="risk-detail-value">{outcome_text}</div>'
                    '</div>'
                )

            card_html = (
                f'<div class="clinical-card" style="height: 100%;">'
                f'<div class="card-title">📋 Clinical Risk Assessment & Stratification</div>'
                f'<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.9rem;">'
                f'<span class="risk-badge {badge_class}">{risk_icon} {risk_level}</span>'
                f'<span style="font-size: 1.55rem; font-weight: 700; color: {risk_color};">{risk_score_pct:.1f}% Risk Score</span>'
                f'</div>'
                f'<p style="font-size: 0.95rem; color: #e2e8f0; line-height: 1.5; margin-bottom: 0.8rem;">{current_pred["explanation"]}</p>'
                f'<div class="risk-detail-grid">'
                f'<div class="risk-detail-item">'
                f'<div class="risk-detail-label">ACTIVE MODEL</div>'
                f'<div class="risk-detail-value">{model_name}</div>'
                f'</div>'
                f'<div class="risk-detail-item">'
                f'<div class="risk-detail-label">PREDICTION TARGET</div>'
                f'<div class="risk-detail-value">30-Day Readmission Risk</div>'
                f'</div>'
                f'<div class="risk-detail-item">'
                f'<div class="risk-detail-label">PROTOTYPE RISK STRATIFICATION</div>'
                f'<div class="risk-detail-value" style="color: {risk_color}; font-weight: 600;">{risk_level} (0–20% / 20–40% / 40–100%)</div>'
                f'</div>'
                f'{historical_outcome_html}'
                f'</div>'
                f'<div class="academic-guidance-box">'
                f'<b>💡 Academic Decision-Support Guidance:</b><br>'
                f'{current_pred["recommendation"]}'
                f'</div>'
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

        # 3. SELECTED ENCOUNTER PROFILE METRICS
        st.markdown(f'<div class="card-title" style="margin-top: 1rem;">🩺 Selected Encounter Clinical Profile (ID: {selected_row["encounter_id"]})</div>', unsafe_allow_html=True)

        p1, p2, p3, p4, p5, p6, p7, p8 = st.columns(8)
        with p1:
            st.markdown(f'<div class="patient-metric-box"><div class="patient-metric-label">Age Group</div><div class="patient-metric-value">{selected_row.get("age", "N/A")}</div></div>', unsafe_allow_html=True)
        with p2:
            st.markdown(f'<div class="patient-metric-box"><div class="patient-metric-label">Gender / Race</div><div class="patient-metric-value" style="font-size: 0.95rem;">{selected_row.get("gender", "N/A")}<br><span style="color:#94a3b8; font-size:0.75rem;">{selected_row.get("race", "N/A")}</span></div></div>', unsafe_allow_html=True)
        with p3:
            st.markdown(f'<div class="patient-metric-box"><div class="patient-metric-label">Length of Stay</div><div class="patient-metric-value">{selected_row.get("time_in_hospital", "N/A")} <span style="font-size: 0.8rem; font-weight: normal; color: #94a3b8;">days</span></div></div>', unsafe_allow_html=True)
        with p4:
            inpatient_val = selected_row.get("number_inpatient", 0)
            inpatient_color = "#f87171" if int(inpatient_val) > 1 else "#f8fafc"
            st.markdown(f'<div class="patient-metric-box"><div class="patient-metric-label">Prior Inpatient</div><div class="patient-metric-value" style="color: {inpatient_color};">{inpatient_val}</div></div>', unsafe_allow_html=True)
        with p5:
            st.markdown(f'<div class="patient-metric-box"><div class="patient-metric-label">Prior Admissions</div><div class="patient-metric-value">{selected_row.get("prior_admissions", 0)}</div></div>', unsafe_allow_html=True)
        with p6:
            st.markdown(f'<div class="patient-metric-box"><div class="patient-metric-label">Medications</div><div class="patient-metric-value">{selected_row.get("num_medications", "N/A")}</div></div>', unsafe_allow_html=True)
        with p7:
            st.markdown(f'<div class="patient-metric-box"><div class="patient-metric-label">Lab Procedures</div><div class="patient-metric-value">{selected_row.get("num_lab_procedures", "N/A")}</div></div>', unsafe_allow_html=True)
        with p8:
            st.markdown(f'<div class="patient-metric-box"><div class="patient-metric-label">Primary Diagnosis</div><div class="patient-metric-value" style="font-size: 0.92rem; color: #38bdf8;">{selected_row.get("diagnosis_group", "N/A")}</div></div>', unsafe_allow_html=True)

        # 4. TOP CLINICAL FEATURE DRIVERS (GLOBAL SHAP IMPORTANCE)
        st.markdown("""
        <div class="clinical-card">
            <div class="card-title">🔎 Top Clinical Feature Drivers (Global Model Feature Importance)</div>
            <p style="color: #94a3b8; font-size: 0.86rem; margin-top: -0.4rem; margin-bottom: 0.8rem;">
                Population-level mean absolute SHAP values reflecting the primary clinical drivers utilized by the trained classifier.
            </p>
        """, unsafe_allow_html=True)

        if shap_df is not None and not shap_df.empty:
            top_shap = shap_df.head(10).copy()
            top_shap["clinical_feature"] = top_shap["feature"].apply(format_clinical_feature_name)
            top_shap = top_shap.sort_values("mean_abs_shap", ascending=True)

            fig_shap = px.bar(
                top_shap,
                x="mean_abs_shap",
                y="clinical_feature",
                orientation="h",
                labels={
                    "mean_abs_shap": "Mean |SHAP Value| (Impact on Log-Odds)",
                    "clinical_feature": "Clinical Feature"
                },
                color="mean_abs_shap",
                color_continuous_scale=[[0, "#0284c7"], [0.5, "#38bdf8"], [1, "#818cf8"]]
            )
            fig_shap = apply_dark_theme(fig_shap, height=330)
            fig_shap.update_layout(coloraxis_showscale=False)
            fig_shap.update_traces(
                hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>"
            )
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.info("SHAP feature importance report (`reports/shap/global_feature_importance.csv`) is not available.")

        st.markdown("""
            <div style="font-size: 0.77rem; color: #64748b; line-height: 1.4; border-top: 1px solid #1f2937; padding-top: 0.6rem;">
                <b>Note on Explainability:</b> The horizontal chart illustrates <i>global population-level</i> feature importance computed across the test cohort. 
                Discharge disposition, prior inpatient visits, and cumulative prior admissions serve as dominant predictive signals. Global SHAP values should not be interpreted as causal guarantees for individual patients.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# TAB 2: 📊 READMISSION RATE BY AGE GROUP
# ============================================================

with tab_age:
    st.markdown("""
    <div class="clinical-card">
        <div class="card-title">📊 30-Day Readmission Rate by Age Group</div>
    """, unsafe_allow_html=True)

    if "age" in filtered_df.columns:
        age_summary = (
            filtered_df.groupby("age", observed=True)
            .agg(
                encounters=("readmitted_30d", "count"),
                readmissions=("readmitted_30d", "sum"),
                mean_rate=("readmitted_30d", "mean"),
            )
            .reset_index()
        )
        age_summary["readmission_rate"] = age_summary["mean_rate"] * 100

        fig_age = px.bar(
            age_summary,
            x="age",
            y="readmission_rate",
            text=age_summary["readmission_rate"].apply(lambda v: f"{v:.1f}%"),
            labels={
                "age": "Age Bracket",
                "readmission_rate": "Readmission Rate (%)"
            },
            color="readmission_rate",
            color_continuous_scale=[[0, "#10b981"], [0.5, "#f59e0b"], [1, "#ef4444"]]
        )
        fig_age = apply_dark_theme(fig_age, height=380)
        fig_age.update_layout(coloraxis_showscale=False)
        fig_age.update_traces(
            textposition="outside",
            hovertemplate="<b>Age: %{x}</b><br>Rate: %{y:.2f}%<extra></extra>"
        )
        st.plotly_chart(fig_age, use_container_width=True)

        col_tbl, col_insights = st.columns([6, 6])
        with col_tbl:
            st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #cbd5e1; margin-bottom: 0.4rem;'>Detailed Breakdown</div>", unsafe_allow_html=True)
            display_age = age_summary.rename(columns={
                "age": "Age Group",
                "encounters": "Total Encounters",
                "readmissions": "30d Readmissions",
                "readmission_rate": "Readmission Rate (%)"
            })[["Age Group", "Total Encounters", "30d Readmissions", "Readmission Rate (%)"]]
            st.dataframe(
                display_age.style.format({
                    "Total Encounters": "{:,}",
                    "30d Readmissions": "{:,}",
                    "Readmission Rate (%)": "{:.2f}%"
                }),
                use_container_width=True,
                hide_index=True
            )

        with col_insights:
            max_age_row = age_summary.sort_values("readmission_rate", ascending=False).iloc[0]
            st.markdown(f"""
            <div style="background: #182234; border: 1px solid #243247; border-radius: 8px; padding: 1rem; height: 100%;">
                <div style="color: #38bdf8; font-weight: 600; margin-bottom: 0.5rem;">Age Cohort Clinical Observations:</div>
                <div style="font-size: 0.86rem; color: #cbd5e1; line-height: 1.5;">
                    • Highest readmission frequency is observed in the <b>{max_age_row['age']}</b> bracket at <b>{max_age_row['readmission_rate']:.2f}%</b>.<br>
                    • Senior diabetic cohorts (60–90 years) account for the vast majority of total readmission events.<br>
                    • Transitional care planning and geriatric pharmacy monitoring should prioritize these age categories.
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Age information is not present in the dataset.")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# TAB 3: 🩺 READMISSION BY DIAGNOSIS GROUP
# ============================================================

with tab_diag:
    st.markdown("""
    <div class="clinical-card">
        <div class="card-title">🩺 30-Day Readmission Rate by Primary Diagnosis Group</div>
    """, unsafe_allow_html=True)

    if "diagnosis_group" in filtered_df.columns:
        diag_summary = (
            filtered_df.groupby("diagnosis_group", dropna=False)
            .agg(
                encounters=("readmitted_30d", "count"),
                readmissions=("readmitted_30d", "sum"),
                mean_rate=("readmitted_30d", "mean"),
            )
            .reset_index()
        )
        diag_summary["readmission_rate"] = diag_summary["mean_rate"] * 100
        diag_summary = diag_summary.sort_values("readmission_rate", ascending=False)

        fig_diag = px.bar(
            diag_summary,
            x="diagnosis_group",
            y="readmission_rate",
            text=diag_summary["readmission_rate"].apply(lambda v: f"{v:.1f}%"),
            labels={
                "diagnosis_group": "Primary Diagnosis Category",
                "readmission_rate": "Readmission Rate (%)"
            },
            color="readmission_rate",
            color_continuous_scale=[[0, "#38bdf8"], [0.5, "#f59e0b"], [1, "#f43f5e"]]
        )
        fig_diag = apply_dark_theme(fig_diag, height=390)
        fig_diag.update_layout(coloraxis_showscale=False)
        fig_diag.update_traces(
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Readmission Rate: %{y:.2f}%<extra></extra>"
        )
        st.plotly_chart(fig_diag, use_container_width=True)

        col_dtbl, col_dcard = st.columns([7, 5])
        with col_dtbl:
            display_diag = diag_summary.rename(columns={
                "diagnosis_group": "Diagnosis Category",
                "encounters": "Encounters",
                "readmissions": "30d Readmissions",
                "readmission_rate": "Readmission Rate (%)"
            })[["Diagnosis Category", "Encounters", "30d Readmissions", "Readmission Rate (%)"]]
            st.dataframe(
                display_diag.style.format({
                    "Encounters": "{:,}",
                    "30d Readmissions": "{:,}",
                    "Readmission Rate (%)": "{:.2f}%"
                }),
                use_container_width=True,
                hide_index=True
            )

        with col_dcard:
            top_diag = diag_summary.iloc[0]
            st.markdown(f"""
            <div style="background: #182234; border: 1px solid #243247; border-radius: 8px; padding: 1rem; height: 100%;">
                <div style="color: #f87171; font-weight: 600; margin-bottom: 0.5rem;">Diagnostic Risk Hotspots:</div>
                <div style="font-size: 0.86rem; color: #cbd5e1; line-height: 1.5;">
                    • <b>{top_diag['diagnosis_group']}</b> admissions exhibit the peak 30-day readmission rate at <b>{top_diag['readmission_rate']:.2f}%</b> ({top_diag['readmissions']:,} readmissions).<br>
                    • Circulatory, Respiratory, and Diabetes categories constitute the largest volume of high-risk patients.<br>
                    • Protocolized post-discharge disease management clinics are recommended for these cohorts.
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Diagnosis group information is not available.")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# TAB 4: ⏱️ LENGTH OF STAY ANALYSIS
# ============================================================

with tab_stay:
    st.markdown("""
    <div class="clinical-card">
        <div class="card-title">⏱️ Hospital Length of Stay Distribution & Readmission Impact</div>
    """, unsafe_allow_html=True)

    stay_df = filtered_df.copy()
    stay_df["readmission_status"] = stay_df["readmitted_30d"].map({
        0: "No 30-Day Readmission",
        1: "30-Day Readmission"
    }).fillna("Unknown")

    fig_stay = px.histogram(
        stay_df,
        x="time_in_hospital",
        color="readmission_status",
        barmode="overlay",
        nbins=14,
        labels={
            "time_in_hospital": "Length of Stay (Days)",
            "readmission_status": "Outcome Group"
        },
        color_discrete_map={
            "No 30-Day Readmission": "rgba(56, 189, 248, 0.65)",
            "30-Day Readmission": "rgba(244, 63, 94, 0.85)"
        }
    )
    fig_stay = apply_dark_theme(fig_stay, height=390)
    fig_stay.update_layout(
        xaxis_title="Length of Stay (Days)",
        yaxis_title="Encounter Count",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)"
        )
    )
    st.plotly_chart(fig_stay, use_container_width=True)

    # Average Length of Stay Contrast
    stay_0 = stay_df[stay_df["readmitted_30d"] == 0]["time_in_hospital"].mean()
    stay_1 = stay_df[stay_df["readmitted_30d"] == 1]["time_in_hospital"].mean()

    c_s1, c_s2, c_s3 = st.columns(3)
    with c_s1:
        st.markdown(f"""
        <div class="patient-metric-box">
            <div class="patient-metric-label">Mean Stay (Non-Readmitted)</div>
            <div class="patient-metric-value" style="color:#38bdf8;">{stay_0:.2f} <span style="font-size:0.8rem; font-weight:normal; color:#94a3b8;">days</span></div>
        </div>""", unsafe_allow_html=True)
    with c_s2:
        st.markdown(f"""
        <div class="patient-metric-box">
            <div class="patient-metric-label">Mean Stay (30-Day Readmitted)</div>
            <div class="patient-metric-value" style="color:#f43f5e;">{stay_1:.2f} <span style="font-size:0.8rem; font-weight:normal; color:#94a3b8;">days</span></div>
        </div>""", unsafe_allow_html=True)
    with c_s3:
        diff_days = stay_1 - stay_0
        st.markdown(f"""
        <div class="patient-metric-box">
            <div class="patient-metric-label">Stay Duration Differential</div>
            <div class="patient-metric-value" style="color:#fbbf24;">+{diff_days:.2f} <span style="font-size:0.8rem; font-weight:normal; color:#94a3b8;">days higher</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# TAB 5: 🏥 ADMISSION PATTERN ANALYSIS
# ============================================================

with tab_adm:
    st.markdown("""
    <div class="clinical-card">
        <div class="card-title">🏥 Hospital Encounters by Admission Type Pattern</div>
    """, unsafe_allow_html=True)

    if "admission_type_id" in filtered_df.columns:
        adm_summary = (
            filtered_df.groupby("admission_type_id")
            .agg(
                encounters=("readmitted_30d", "count"),
                readmissions=("readmitted_30d", "sum"),
                mean_rate=("readmitted_30d", "mean"),
            )
            .reset_index()
        )
        adm_summary["readmission_rate"] = adm_summary["mean_rate"] * 100
        adm_summary["admission_label"] = adm_summary["admission_type_id"].apply(lambda x: f"Admission Type {x}")

        fig_adm = px.bar(
            adm_summary,
            x="admission_label",
            y="encounters",
            text="encounters",
            labels={
                "admission_label": "Admission Type Classification",
                "encounters": "Total Inpatient Encounters"
            },
            color="readmission_rate",
            color_continuous_scale=[[0, "#38bdf8"], [0.5, "#f59e0b"], [1, "#f43f5e"]]
        )
        fig_adm = apply_dark_theme(fig_adm, height=390)
        fig_adm.update_traces(
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Encounters: %{y:,}<br>Readmission Rate: %{marker.color:.2f}%<extra></extra>"
        )
        fig_adm.update_layout(coloraxis_colorbar=dict(title="Readm. %"))
        st.plotly_chart(fig_adm, use_container_width=True)

        display_adm = adm_summary.rename(columns={
            "admission_type_id": "Type ID",
            "admission_label": "Classification Label",
            "encounters": "Total Encounters",
            "readmissions": "30-Day Readmissions",
            "readmission_rate": "Readmission Rate (%)"
        })[["Type ID", "Classification Label", "Total Encounters", "30-Day Readmissions", "Readmission Rate (%)"]]
        st.dataframe(
            display_adm.style.format({
                "Total Encounters": "{:,}",
                "30-Day Readmissions": "{:,}",
                "Readmission Rate (%)": "{:.2f}%"
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("Admission type information is not available.")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# TAB 6: 🚨 HIGH-RISK COHORT EXPLORER
# ============================================================

with tab_cohort:
    st.markdown("""
    <div class="clinical-card">
        <div class="card-title">🚨 Highest-Risk Encounter Cohort (Clinical Prioritization)</div>
        <p style="color: #94a3b8; font-size: 0.86rem; margin-top: -0.4rem; margin-bottom: 0.8rem;">
            Identifies complex encounters prioritized by prior inpatient hospitalizations, cumulative admissions, and prolonged length of stay.
        </p>
    """, unsafe_allow_html=True)

    risk_cols = [
        "encounter_id",
        "patient_key",
        "age",
        "gender",
        "diagnosis_group",
        "time_in_hospital",
        "prior_admissions",
        "number_inpatient",
        "num_medications",
        "readmitted_30d"
    ]
    avail_risk_cols = [c for c in risk_cols if c in filtered_df.columns]

    sort_criteria = [c for c in ["number_inpatient", "prior_admissions", "time_in_hospital"] if c in filtered_df.columns]

    if avail_risk_cols and sort_criteria:
        high_risk_cohort = (
            filtered_df.sort_values(sort_criteria, ascending=False)[avail_risk_cols]
            .head(50)
            .copy()
        )

        st.dataframe(
            high_risk_cohort.rename(columns={
                "encounter_id": "Encounter ID",
                "patient_key": "Patient Surrogate Key",
                "age": "Age",
                "gender": "Gender",
                "diagnosis_group": "Primary Diagnosis",
                "time_in_hospital": "Stay (Days)",
                "prior_admissions": "Prior Admissions",
                "number_inpatient": "Prior Inpatient",
                "num_medications": "Medications",
                "readmitted_30d": "Actual 30d Readmission"
            }),
            use_container_width=True,
            hide_index=True
        )

        csv_data = high_risk_cohort.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download High-Risk Cohort CSV",
            data=csv_data,
            file_name="high_risk_encounter_cohort.csv",
            mime="text/csv"
        )
    else:
        st.warning("Insufficient clinical features to assemble the prioritized risk cohort.")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# 9. FOOTER & ACADEMIC NOTICE
# ============================================================

st.markdown("""
<div class="academic-disclaimer-footer">
    <b>ACADEMIC PROTOTYPE NOTICE:</b><br>
    This automated prediction is generated for research and educational purposes. It is not a clinical diagnosis or treatment recommendation. 
    Clinical decisions must be made by qualified healthcare professionals.
</div>
""", unsafe_allow_html=True)