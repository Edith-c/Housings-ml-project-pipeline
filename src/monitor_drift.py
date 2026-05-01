import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


# ---------------------------
# CONFIG
# ---------------------------
DATA_PATH = "/Users/caroletene/California-Housing/datasets/housing.csv"
DRIFT_THRESHOLD = 0.2  
REPORT_PATH = "reports/drift_report.html"


# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv(DATA_PATH)

# simulate production data (split)
reference_data, production_data = train_test_split(
    df,
    test_size=0.3,
    random_state=42
)

# OPTIONAL: simulate stronger drift
production_data["median_income"] *= np.random.normal(1.2, 0.1, len(production_data))


# ---------------------------
# RUN DRIFT REPORT
# ---------------------------
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_data, current_data=production_data)

# Save HTML report
import os
os.makedirs("reports", exist_ok=True)
report.save_html(REPORT_PATH)

print(f"\n Drift report saved to: {REPORT_PATH}")


# ---------------------------
# EXTRACT DRIFT METRICS
# ---------------------------
result = report.as_dict()

drift_result = result["metrics"][0]["result"]

# dataset-level drift
drift_share = drift_result.get("dataset_drift")

print("\n=== DRIFT SUMMARY ===")
print(f"Dataset drift detected: {drift_share}")

# ✅ correct way: use "drift_by_columns" IF available, else fallback
drift_by_columns = drift_result.get("drift_by_columns", None)

drifted_features = []

if drift_by_columns:
    for col, stats in drift_by_columns.items():
        if stats.get("drift_detected"):
            drifted_features.append(col)
else:
    print("\nColumn-level drift not available in this Evidently version.")
# ---------------------------
# EXIT CONDITION
# ---------------------------
if drift_share > DRIFT_THRESHOLD:
    print("\n DRIFT THRESHOLD EXCEEDED → EXITING WITH CODE 1")
    sys.exit(1)

print("\n Drift within acceptable range → OK")
sys.exit(0)