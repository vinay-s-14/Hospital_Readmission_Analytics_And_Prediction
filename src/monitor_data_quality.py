from pathlib import Path
import json
import pandas as pd

DATA_PATH = Path("data/curated/encounters.csv")
OUTPUT_DIR = Path("reports/monitoring")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

checks = {}

# 1. Missing values
missing_values = int(df.isna().sum().sum())
checks["missing_values"] = missing_values

# 2. Duplicate records
duplicate_rows = int(df.duplicated().sum())
checks["duplicate_rows"] = duplicate_rows

# 3. Required columns
required_columns = [
    "encounter_id",
    "patient_nbr",
    "readmitted",
    "time_in_hospital"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

checks["missing_required_columns"] = missing_columns

# 4. Length of stay validation
invalid_stay = int(
    ((df["time_in_hospital"] < 1) |
     (df["time_in_hospital"] > 14)).sum()
)

checks["invalid_time_in_hospital"] = invalid_stay

# 5. Readmission value validation
allowed_readmitted = {"NO", "<30", ">30"}

invalid_readmitted = int(
    (~df["readmitted"].isin(allowed_readmitted)).sum()
)

checks["invalid_readmitted_values"] = invalid_readmitted

# Overall status
quality_pass = (
    missing_columns == []
    and duplicate_rows == 0
    and invalid_stay == 0
    and invalid_readmitted == 0
)

checks["quality_status"] = "PASS" if quality_pass else "FAIL"

# Save report
output_file = OUTPUT_DIR / "data_quality_report.json"

with open(output_file, "w") as file:
    json.dump(checks, file, indent=4)

print("\nData Quality Results")
print("--------------------")
for key, value in checks.items():
    print(f"{key}: {value}")

print(f"\nData quality status: {checks['quality_status']}")
print(f"Saved: {output_file}")
