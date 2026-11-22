
import os
import time
import joblib
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(ROOT, "data", "raw", "bank_fraud.csv")
MODEL_PATH = os.path.join(ROOT, "models", "lightgbm.pkl")
FEATURE_PATH = os.path.join(ROOT, "models", "lgbm_features.pkl")

print("Loading model and dataset...")

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURE_PATH)

df = pd.read_csv(DATA_PATH, nrows=100)

df["transaction_date"] = pd.to_datetime(
    df["transaction_date"], errors="coerce"
)

df = df.dropna(subset=["transaction_date"])

df = df.sort_values(
    ["customer_id", "transaction_date", "transaction_time"]
).reset_index(drop=True)

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

drop_cols = [
    "transaction_id",
    "customer_id",
    "transaction_date",
    "transaction_time",
    "fraud_type",
    "is_fraud"
]

X = df.drop(columns=drop_cols, errors="ignore")

X = pd.get_dummies(X, dummy_na=True)

X = X.reindex(columns=features, fill_value=0)

X = X.replace(
    [np.inf, -np.inf], np.nan
).fillna(0)

if len(X) == 0:
    raise ValueError("No valid transaction rows available.")

# Warm-up
print("Warming up model...")

model.predict_proba(X.iloc[[0]])

# Single transaction inference benchmark
latencies = []

print("Testing single transaction predictions...")

for i in range(len(X)):

    transaction = X.iloc[[i]]

    start = time.perf_counter()

    model.predict_proba(transaction)

    end = time.perf_counter()

    latency_ms = (end - start) * 1000

    latencies.append(latency_ms)

# Results
latencies = np.array(latencies)

avg_latency = np.mean(latencies)
median_latency = np.median(latencies)
p95_latency = np.percentile(latencies, 95)
max_latency = np.max(latencies)

print("\n" + "=" * 45)
print("API INFERENCE LATENCY RESULTS")
print("=" * 45)

print(f"Transactions tested: {len(latencies)}")
print(f"Average latency: {avg_latency:.4f} ms")
print(f"Median latency: {median_latency:.4f} ms")
print(f"P95 latency: {p95_latency:.4f} ms")
print(f"Maximum latency: {max_latency:.4f} ms")

if p95_latency < 25:
    print("\nP95 LATENCY TARGET: PASSED")
else:
    print("\nP95 LATENCY TARGET: NOT MET")

print("\nNote: This measures local model inference only.")
print("It does not include API/network/preprocessing overhead.")

print("\nLATENCY TEST COMPLETED!")
