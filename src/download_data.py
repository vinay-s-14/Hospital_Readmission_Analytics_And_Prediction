from pathlib import Path
from ucimlrepo import fetch_ucirepo

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

dataset = fetch_ucirepo(id=296)
features = dataset.data.features.copy()
target = dataset.data.targets.copy()

# Recombine into the source-style encounter table.
df = features.copy()
if "readmitted" in target.columns:
    df["readmitted"] = target["readmitted"]

df.to_csv(RAW / "diabetic_data.csv", index=False)

# The ETL can operate from the downloaded source table. IDS_mapping.csv is optional
# for this project because the dashboard can use the numeric IDs directly.
print(f"Saved {len(df):,} rows to {RAW / 'diabetic_data.csv'}")
print("Extraction source: UCI Machine Learning Repository dataset 296")
