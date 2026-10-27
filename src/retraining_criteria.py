from pathlib import Path
import json

MONITORING_DIR = Path("reports/monitoring")

# Retraining thresholds
MIN_ROC_AUC = 0.60
MIN_F1 = 0.05
MAX_API_LATENCY_MS = 500.0
MIN_API_SUCCESS_RATE = 0.95


def load_json(filename):
    path = MONITORING_DIR / filename
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


model_report = load_json("model_performance_report.json")
api_report = load_json("api_monitoring_report.json")

reasons = []

if model_report:
    if model_report.get("roc_auc", 0) < MIN_ROC_AUC:
        reasons.append("ROC-AUC below threshold")

    if model_report.get("f1_score", 0) < MIN_F1:
        reasons.append("F1 score below threshold")

if api_report:
    if api_report.get("average_latency_ms", 0) > MAX_API_LATENCY_MS:
        reasons.append("API latency above threshold")

    if api_report.get("success_rate", 0) < MIN_API_SUCCESS_RATE:
        reasons.append("API success rate below threshold")


retrain_required = len(reasons) > 0

result = {
    "retraining_required": retrain_required,
    "reasons": reasons,
    "criteria": {
        "min_roc_auc": MIN_ROC_AUC,
        "min_f1": MIN_F1,
        "max_api_latency_ms": MAX_API_LATENCY_MS,
        "min_api_success_rate": MIN_API_SUCCESS_RATE
    }
}

output_file = MONITORING_DIR / "retraining_decision.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=4)

print("Retraining Decision")
print("-------------------")
print("Retraining required:", retrain_required)

if reasons:
    print("Reasons:")
    for reason in reasons:
        print("-", reason)
else:
    print("No retraining trigger detected.")

print(f"Saved: {output_file}")

































