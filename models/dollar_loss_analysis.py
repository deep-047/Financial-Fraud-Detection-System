
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix

# Project paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(ROOT, "data", "raw", "bank_fraud.csv")
MODEL_PATH = os.path.join(ROOT, "models", "lightgbm.pkl")
FEATURE_PATH = os.path.join(ROOT, "models", "lgbm_features.pkl")
THRESHOLD_PATH = os.path.join(ROOT, "models", "lgbm_threshold.pkl")

OUTPUT_PATH = os.path.join(
    ROOT, "models", "dollar_loss_analysis.csv"
)

TARGET = "is_fraud"

print("Loading dataset and model...")

df = pd.read_csv(DATA_PATH)

df["transaction_date"] = pd.to_datetime(
    df["transaction_date"], errors="coerce"
)

df["transaction_amount"] = pd.to_numeric(
    df["transaction_amount"], errors="coerce"
)

df = df.dropna(
    subset=["transaction_date", "transaction_amount", TARGET]
)

df = df.sort_values(
    ["customer_id", "transaction_date", "transaction_time"]
).reset_index(drop=True)

# Feature engineering
df["Hist_Mean_Spend"] = (
    df.groupby("customer_id")["transaction_amount"]
    .transform(lambda x: x.shift(1).expanding().mean())
    .fillna(0)
)

df["Spend_Ratio"] = (
    df["transaction_amount"] /
    (df["Hist_Mean_Spend"] + 1)
)

df["transaction_month"] = df["transaction_date"].dt.month
df["transaction_day"] = df["transaction_date"].dt.day

df["high_failed_attempts"] = (
    df["failed_attempts"] >= 2
).astype(int)

df["high_distance"] = (
    df["distance_from_home_km"] > 100
).astype(int)

# Temporal split: 70% train, 15% validation, 15% test
dates = np.sort(df["transaction_date"].unique())

if len(dates) < 3:
    raise ValueError("Not enough unique dates for temporal splitting.")

val_date = dates[int(len(dates) * 0.85)]

test = df[
    df["transaction_date"] >= val_date
].copy()

if test.empty:
    raise ValueError("Temporal test dataset is empty.")

print("Test rows:", len(test))
print("Test fraud records:", int(test[TARGET].sum()))

# Prepare test features
drop_cols = [
    "transaction_id",
    "customer_id",
    "transaction_date",
    "transaction_time",
    "fraud_type",
    TARGET
]

X_test = test.drop(
    columns=drop_cols,
    errors="ignore"
)

X_test = pd.get_dummies(
    X_test,
    dummy_na=True
)

features = joblib.load(FEATURE_PATH)

X_test = X_test.reindex(
    columns=features,
    fill_value=0
)

X_test = X_test.replace(
    [np.inf, -np.inf], np.nan
).fillna(0)

y_test = test[TARGET].astype(int)

# Load model
model = joblib.load(MODEL_PATH)

threshold_artifact = joblib.load(THRESHOLD_PATH)

if isinstance(threshold_artifact, dict):
    threshold = float(
        threshold_artifact.get(
            "threshold",
            threshold_artifact.get("best_threshold", 0.5)
        )
    )
else:
    threshold = float(
        np.asarray(threshold_artifact).reshape(-1)[0]
    )

threshold = float(np.clip(threshold, 0, 1))

print("Using threshold:", threshold)

# Predict test transactions
probabilities = model.predict_proba(X_test)[:, 1]

predictions = (
    probabilities >= threshold
).astype(int)

# Financial impact
amounts = test["transaction_amount"].astype(float)

actual_fraud = y_test == 1

detected_fraud = actual_fraud & (predictions == 1)
missed_fraud = actual_fraud & (predictions == 0)

total_fraud_amount = float(
    amounts[actual_fraud].sum()
)

detected_fraud_amount = float(
    amounts[detected_fraud].sum()
)

missed_fraud_amount = float(
    amounts[missed_fraud].sum()
)

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
).ravel()

potential_coverage = (
    detected_fraud_amount / total_fraud_amount * 100
    if total_fraud_amount > 0 else 0
)

# Results
print("\n" + "=" * 55)
print("TEMPORAL TEST - FINANCIAL IMPACT ANALYSIS")
print("=" * 55)

print("Test transactions:", len(test))
print("Threshold:", threshold)

print(f"Total fraud amount: {total_fraud_amount:,.2f}")
print(f"Potentially preventable amount: {detected_fraud_amount:,.2f}")
print(f"Missed fraud amount: {missed_fraud_amount:,.2f}")

print(f"Potential amount coverage: {potential_coverage:.2f}%")

print("\nConfusion Matrix:")
print("TP:", tp)
print("FP:", fp)
print("FN:", fn)
print("TN:", tn)

print(
    "\nNote: Potentially preventable amount is an estimate, "
    "not confirmed realized savings. Currency and operational "
    "cost assumptions are not available in the dataset."
)

# Save results
results = {
    "Evaluation Set": "Temporal Test",
    "Test Transactions": len(test),
    "Threshold": threshold,
    "Total Fraud Amount": total_fraud_amount,
    "Potentially Preventable Amount": detected_fraud_amount,
    "Missed Fraud Amount": missed_fraud_amount,
    "Potential Amount Coverage (%)": potential_coverage,
    "True Positives": int(tp),
    "False Positives": int(fp),
    "False Negatives": int(fn),
    "True Negatives": int(tn)
}

pd.DataFrame([results]).to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nResults saved:", OUTPUT_PATH)
print("TEMPORAL TEST FINANCIAL ANALYSIS COMPLETED!")
