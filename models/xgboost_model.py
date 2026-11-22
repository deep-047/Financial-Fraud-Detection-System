
import pandas as pd
import numpy as np
import joblib

from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    confusion_matrix
)

print("Loading dataset...")

df = pd.read_csv("data/raw/bank_fraud.csv")
df["transaction_date"] = pd.to_datetime(df["transaction_date"])

df = df.sort_values(
    ["customer_id", "transaction_date", "transaction_time"]
).copy()

# Historical spending feature
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

# Chronological split
dates = np.sort(df["transaction_date"].dropna().unique())

train_date = dates[int(len(dates) * 0.70)]
val_date = dates[int(len(dates) * 0.85)]

train = df[df["transaction_date"] < train_date]
val = df[
    (df["transaction_date"] >= train_date) &
    (df["transaction_date"] < val_date)
]

drop_cols = [
    "transaction_id", "customer_id",
    "transaction_date", "transaction_time",
    "fraud_type", "is_fraud"
]

X_train = train.drop(columns=drop_cols)
y_train = train["is_fraud"]

X_val = val.drop(columns=drop_cols)
y_val = val["is_fraud"]

# One-hot encoding
X_train = pd.get_dummies(X_train, dummy_na=True)
X_val = pd.get_dummies(X_val, dummy_na=True)

X_train, X_val = X_train.align(
    X_val, join="left", axis=1, fill_value=0
)

# Class imbalance ratio
scale_pos_weight = (
    (y_train == 0).sum() / (y_train == 1).sum()
)

print("Scale Pos Weight:", scale_pos_weight)

# Train XGBoost
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="aucpr",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

print("\nTraining XGBoost...")
model.fit(X_train, y_train)

# Validation
prob = model.predict_proba(X_val)[:, 1]

pr_auc = average_precision_score(y_val, prob)

# Threshold selection
thresholds = np.linspace(0.01, 0.99, 99)
valid_thresholds = []

for threshold in thresholds:
    pred = (prob >= threshold).astype(int)

    precision = precision_score(
        y_val, pred, zero_division=0
    )

    if precision >= 0.95 and pred.sum() > 0:
        valid_thresholds.append(threshold)

threshold = min(valid_thresholds) if valid_thresholds else 0.50

pred = (prob >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_val, pred, labels=[0, 1]
).ravel()

fpr = fp / (fp + tn) if (fp + tn) else 0

print("\nXGBOOST RESULTS")
print("PR-AUC:", round(pr_auc, 4))
print("Threshold:", round(threshold, 4))
print("Precision:", round(
    precision_score(y_val, pred, zero_division=0), 4
))
print("Recall:", round(
    recall_score(y_val, pred, zero_division=0), 4
))
print("FPR:", round(fpr, 4))

joblib.dump(model, "models/xgboost.pkl")
joblib.dump(list(X_train.columns), "models/xgb_features.pkl")
joblib.dump(threshold, "models/xgb_threshold.pkl")

print("\nXGBOOST MODEL SAVED!")
