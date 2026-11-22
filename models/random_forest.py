
import pandas as pd
import numpy as np
import joblib

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    confusion_matrix
)

print("Loading dataset...")

df = pd.read_csv("data/raw/bank_fraud.csv")

df["transaction_date"] = pd.to_datetime(df["transaction_date"])

# Sort chronologically
df = df.sort_values(
    ["customer_id", "transaction_date", "transaction_time"]
).copy()

# Feature engineering
df["Hist_Mean_Spend"] = (
    df.groupby("customer_id")["transaction_amount"]
    .transform(lambda x: x.shift(1).expanding().mean())
    .fillna(0)
)

df["Spend_Ratio"] = (
    df["transaction_amount"] / (df["Hist_Mean_Spend"] + 1)
)

df["transaction_month"] = df["transaction_date"].dt.month
df["transaction_day"] = df["transaction_date"].dt.day

df["high_failed_attempts"] = (
    df["failed_attempts"] >= 2
).astype(int)

df["high_distance"] = (
    df["distance_from_home_km"] > 100
).astype(int)

# Temporal split
dates = np.sort(df["transaction_date"].dropna().unique())

train_date = dates[int(len(dates) * 0.70)]
validation_date = dates[int(len(dates) * 0.85)]

train = df[df["transaction_date"] < train_date]
validation = df[
    (df["transaction_date"] >= train_date) &
    (df["transaction_date"] < validation_date)
]

# Prepare features
drop_columns = [
    "transaction_id",
    "customer_id",
    "transaction_date",
    "transaction_time",
    "fraud_type",
    "is_fraud"
]

X_train = train.drop(columns=drop_columns)
y_train = train["is_fraud"]

X_val = validation.drop(columns=drop_columns)
y_val = validation["is_fraud"]

# Encode categorical features using training data
X_train = pd.get_dummies(X_train, dummy_na=True)
X_val = pd.get_dummies(X_val, dummy_na=True)

X_train, X_val = X_train.align(
    X_val,
    join="left",
    axis=1,
    fill_value=0
)

# Train Balanced Random Forest
print("\nTraining Balanced Random Forest...")

model = BalancedRandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    sampling_strategy="all",
    replacement=True,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Validation predictions
probabilities = model.predict_proba(X_val)[:, 1]

pr_auc = average_precision_score(y_val, probabilities)

# Find threshold for 95% precision
thresholds = np.linspace(0.01, 0.99, 99)

valid_thresholds = []

for threshold in thresholds:
    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_val, predictions, zero_division=0
    )

    if precision >= 0.95 and predictions.sum() > 0:
        valid_thresholds.append(threshold)

threshold = min(valid_thresholds) if valid_thresholds else 0.50

predictions = (probabilities >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_val, predictions, labels=[0, 1]
).ravel()

fpr = fp / (fp + tn) if (fp + tn) else 0

print("\nRANDOM FOREST RESULTS")
print("PR-AUC:", round(pr_auc, 4))
print("Threshold:", round(threshold, 4))
print("Precision:", round(
    precision_score(y_val, predictions, zero_division=0), 4
))
print("Recall:", round(
    recall_score(y_val, predictions, zero_division=0), 4
))
print("FPR:", round(fpr, 4))

# Save model and feature columns
joblib.dump(model, "models/balanced_random_forest.pkl")
joblib.dump(list(X_train.columns), "models/rf_features.pkl")
joblib.dump(threshold, "models/rf_threshold.pkl")

print("\nBALANCED RANDOM FOREST SAVED!")
