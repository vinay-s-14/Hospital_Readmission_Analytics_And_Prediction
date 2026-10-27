import os
import shutil
from datetime import datetime
from pathlib import Path

from airflow import DAG
try:
    from airflow.providers.standard.operators.bash import BashOperator
except ImportError:
    from airflow.operators.bash import BashOperator

# Detect project root directory (WSL mount path or container fallback)
WSL_PROJECT_DIR = "/mnt/c/Users/DELL/OneDrive/Desktop/Hospital_Readmission_Analytics_And_Prediction/hospital_readmission_project"
PROJECT_DIR = os.getenv(
    "PROJECT_DIR",
    WSL_PROJECT_DIR if Path(WSL_PROJECT_DIR).exists() else "/opt/project"
)

# Detect Python interpreter (use Windows Python if available for direct DB and library access)
python_resolved = shutil.which("python.exe") or shutil.which("python3") or "python"
PYTHON_BIN = os.getenv("PYTHON_BIN", python_resolved)

with DAG(
    dag_id="hospital_readmission_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["healthcare", "etl", "mlops"],
) as dag:

    check_source = BashOperator(
        task_id="check_source",
        bash_command=f'test -f "{PROJECT_DIR}/data/raw/diabetic_data.csv"',
        cwd=PROJECT_DIR,
    )

    transform = BashOperator(
        task_id="transform_and_validate",
        bash_command=f'cd "{PROJECT_DIR}" && "{PYTHON_BIN}" src/etl.py --input data/raw/diabetic_data.csv',
        cwd=PROJECT_DIR,
    )

    load = BashOperator(
        task_id="load_postgres",
        bash_command=f'cd "{PROJECT_DIR}" && "{PYTHON_BIN}" src/load_postgres.py',
        cwd=PROJECT_DIR,
    )

    check_source >> transform >> load
