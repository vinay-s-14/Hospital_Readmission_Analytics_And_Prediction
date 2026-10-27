from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = Path("reports/shap/global_feature_importance.csv")
OUTPUT_FILE = Path("reports/shap/global_feature_importance.png")

df = pd.read_csv(INPUT_FILE)

top = df.head(15).sort_values("mean_abs_shap")

plt.figure(figsize=(10, 7))
plt.barh(top["feature"], top["mean_abs_shap"])
plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Feature")
plt.title("XGBoost Global Feature Importance using SHAP")
plt.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=300)
plt.close()

print("SHAP plot created successfully.")
print(f"Saved: {OUTPUT_FILE}")
