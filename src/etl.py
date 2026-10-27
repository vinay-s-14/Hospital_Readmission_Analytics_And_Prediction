from pathlib import Path
import argparse
import json
import re
import pandas as pd
import numpy as np

def age_to_midpoint(value):
    if pd.isna(value):
        return np.nan
    m = re.search(r"(\d+)-(\d+)", str(value))
    return (int(m.group(1)) + int(m.group(2))) / 2 if m else np.nan

def diagnosis_group(value):
    if pd.isna(value):
        return "Unknown"
    s = str(value).strip()
    if s in {"?", "nan", ""}:
        return "Unknown"
    try:
        x = float(s)
        if 390 <= x < 460 or x in [785]:
            return "Circulatory"
        if 250 <= x < 251:
            return "Diabetes"
        if 460 <= x < 520 or x in [786]:
            return "Respiratory"
        if 520 <= x < 580:
            return "Digestive"
        if 800 <= x < 1000:
            return "Injury"
        if 710 <= x < 740:
            return "Musculoskeletal"
        if 580 <= x < 630:
            return "Genitourinary"
        if 140 <= x < 240:
            return "Neoplasms"
    except ValueError:
        pass
    return "Other"

def clean(input_path):
    raw = pd.read_csv(input_path, low_memory=False)
    raw_rows = len(raw)

    raw = raw.replace("?", pd.NA)
    raw = raw.drop_duplicates()

    required = ["encounter_id", "patient_nbr", "time_in_hospital", "readmitted"]
    missing_required = raw[required].isna().any(axis=1)
    rejected = raw.loc[missing_required].copy()
    raw = raw.loc[~missing_required].copy()

    # Patient-safe surrogate key: deterministic hash of the source patient identifier.
    raw["patient_key"] = pd.util.hash_pandas_object(
        raw["patient_nbr"].astype(str), index=False
    ).astype("uint64").astype(str)

    raw["encounter_key"] = raw["encounter_id"].astype(str)
    raw["age_midpoint"] = raw["age"].map(age_to_midpoint)
    raw["diagnosis_group"] = raw["diag_1"].map(diagnosis_group)

    count_cols = [
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient", "number_diagnoses"
    ]
    for c in count_cols:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

    raw["readmitted_30d"] = (raw["readmitted"].astype(str) == "<30").astype(int)

    # Patient-level summary fields.
    raw = raw.sort_values(["patient_key", "encounter_key"])
    raw["prior_admissions"] = raw.groupby("patient_key").cumcount()

    med_cols = [c for c in raw.columns if c.startswith("drug_") or c in {
        "metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "acetohexamide", "glipizide", "glyburide",
        "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
        "miglitol", "troglitazone", "tolazamide", "examide",
        "citoglipton", "insulin", "glyburide-metformin",
        "glipizide-metformin", "glimepiride-pioglitazone",
        "metformin-rosiglitazone", "metformin-pioglitazone"
    }]
    if med_cols:
        raw["medication_count"] = raw[med_cols].notna().sum(axis=1)
    else:
        raw["medication_count"] = raw.get("num_medications", 0)

    out = raw.copy()
    Path("data/staging").mkdir(parents=True, exist_ok=True)
    Path("data/curated").mkdir(parents=True, exist_ok=True)
    Path("data/logs").mkdir(parents=True, exist_ok=True)

    out.to_csv("data/curated/encounters.csv", index=False)
    rejected.to_csv("data/logs/rejected_records.csv", index=False)

    manifest = {
        "source_file": str(input_path),
        "extraction_rows": int(raw_rows),
        "curated_rows": int(len(out)),
        "rejected_rows": int(len(rejected)),
        "extraction_status": "success"
    }
    Path("data/logs/etl_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/diabetic_data.csv")
    args = parser.parse_args()
    clean(args.input)
