CREATE SCHEMA IF NOT EXISTS healthcare;

CREATE TABLE IF NOT EXISTS healthcare.readmission_summary (
    diagnosis_group TEXT,
    age_group TEXT,
    admission_type_id INTEGER,
    encounters BIGINT,
    readmissions_30d BIGINT,
    readmission_rate NUMERIC(8,4),
    avg_length_of_stay NUMERIC(8,2)
);

CREATE INDEX IF NOT EXISTS idx_encounters_patient
ON healthcare.encounters(patient_key);

CREATE INDEX IF NOT EXISTS idx_encounters_readmission
ON healthcare.encounters(readmitted_30d);
