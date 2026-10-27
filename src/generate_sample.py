from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1000

ages = rng.choice(
    ["[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)"],
    size=n
)

df = pd.DataFrame({
    "encounter_id": np.arange(100000, 100000+n),
    "patient_nbr": rng.integers(1000, 1300, n),
    "race": rng.choice(["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other", "?"], n),
    "gender": rng.choice(["Male", "Female"], n),
    "age": ages,
    "admission_type_id": rng.integers(1, 9, n),
    "discharge_disposition_id": rng.integers(1, 30, n),
    "admission_source_id": rng.integers(1, 22, n),
    "time_in_hospital": rng.integers(1, 15, n),
    "num_lab_procedures": rng.integers(1, 130, n),
    "num_procedures": rng.integers(0, 7, n),
    "num_medications": rng.integers(1, 60, n),
    "number_outpatient": rng.integers(0, 10, n),
    "number_emergency": rng.integers(0, 8, n),
    "number_inpatient": rng.integers(0, 8, n),
    "number_diagnoses": rng.integers(1, 10, n),
    "diag_1": rng.choice(["250.00", "428.0", "414.01", "486", "786.05"], n),
    "diag_2": rng.choice(["250.00", "401.9", "428.0", "427.31", "584.9"], n),
    "diag_3": rng.choice(["250.00", "272.4", "428.0", "585", "780.2"], n),
    "readmitted": rng.choice(["NO", ">30", "<30"], n, p=[0.65, 0.24, 0.11]),
})
df.to_csv("data/raw/diabetic_data_sample.csv", index=False)
print("Created data/raw/diabetic_data_sample.csv")
