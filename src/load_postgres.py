import os
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

url = (
    f"postgresql+psycopg2://{os.getenv('POSTGRES_USER','postgres')}:"
    f"{os.getenv('POSTGRES_PASSWORD','change_me')}@"
    f"{os.getenv('POSTGRES_HOST','localhost')}:"
    f"{os.getenv('POSTGRES_PORT','5432')}/"
    f"{os.getenv('POSTGRES_DB','hospital_readmission')}"
)

engine = create_engine(url)
df = pd.read_csv("data/curated/encounters.csv", low_memory=False)
uint_cols = df.select_dtypes(include=["uint64"]).columns
df[uint_cols] = df[uint_cols].astype("int64")

with engine.begin() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS healthcare;"))

df.to_sql("encounters", engine, schema="healthcare", if_exists="replace", index=False, chunksize=5000)
print(f"Loaded {len(df):,} curated encounters into healthcare.encounters")

# Create analytical data mart
summary_sql = """
DROP TABLE IF EXISTS healthcare.readmission_summary;

CREATE TABLE healthcare.readmission_summary AS
SELECT
    diagnosis_group,
    age AS age_group,
    admission_type_id,
    COUNT(*) AS encounters,
    SUM(readmitted_30d) AS readmissions_30d,
    ROUND(AVG(readmitted_30d)::numeric * 100, 4) AS readmission_rate,
    ROUND(AVG(time_in_hospital)::numeric, 2) AS avg_length_of_stay
FROM healthcare.encounters
GROUP BY
    diagnosis_group,
    age,
    admission_type_id;
"""

with engine.begin() as conn:
    conn.execute(text(summary_sql))

print("Created healthcare.readmission_summary analytical data mart")
