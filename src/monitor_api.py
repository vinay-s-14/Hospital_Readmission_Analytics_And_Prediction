from pathlib import Path
import json
import time
import requests

OUTPUT_DIR = Path("reports/monitoring")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

URL = "http://127.0.0.1:8000/predict"

sample_data = {
    "age": "[70-80)",
    "gender": "Female",
    "race": "Caucasian",
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 5,
    "num_lab_procedures": 40,
    "num_procedures": 1,
    "num_medications": 10,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "number_diagnoses": 5,
    "diagnosis_group": "Diabetes",
    "prior_admissions": 0,
    "medication_count": 10
}

total_requests = 10
successful_requests = 0
failed_requests = 0
latencies = []

print("Starting API monitoring...")
print(f"Endpoint: {URL}")
print(f"Requests: {total_requests}")

for i in range(total_requests):
    try:
        start = time.perf_counter()

        response = requests.post(
            URL,
            json=sample_data,
            timeout=10
        )

        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

        if response.status_code == 200:
            successful_requests += 1
        else:
            failed_requests += 1

    except Exception:
        failed_requests += 1

average_latency = (
    sum(latencies) / len(latencies)
    if latencies
    else 0
)

max_latency = (
    max(latencies)
    if latencies
    else 0
)

success_rate = (
    successful_requests / total_requests
    if total_requests
    else 0
)

report = {
    "endpoint": URL,
    "total_requests": total_requests,
    "successful_requests": successful_requests,
    "failed_requests": failed_requests,
    "success_rate": success_rate,
    "average_latency_ms": average_latency,
    "max_latency_ms": max_latency
}

output_file = OUTPUT_DIR / "api_monitoring_report.json"

with open(output_file, "w") as file:
    json.dump(report, file, indent=4)

print("\nAPI Monitoring Results")
print("----------------------")
print(f"Total requests      : {total_requests}")
print(f"Successful requests : {successful_requests}")
print(f"Failed requests     : {failed_requests}")
print(f"Success rate        : {success_rate:.2%}")
print(f"Average latency     : {average_latency:.2f} ms")
print(f"Maximum latency     : {max_latency:.2f} ms")

print(f"\nSaved: {output_file}")
