
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(ROOT, "data", "raw", "bank_fraud.csv")
MODEL_PATH = os.path.join(ROOT, "models", "lightgbm.pkl")
FEATURE_PATH = os.path.join(ROOT, "models", "lgbm_features.pkl")
OUTPUT_PATH = os.path.join(ROOT, "models", "precision_recall_95.csv")

TARGET = "is_fraud"
TARGET_PRECISION = 0.95

print("Loading dataset and model...")

df = pd.read_csv(DATA_PATH)

df["transaction_date"] = pd.to_datetime(
    df["transaction_date"], errors="coerce"
)

df = df.dropna(subset=["transaction_date"])

df = df.sort_values(
    ["customer_id", "transaction_date", "transaction_time"]
).reset_index(drop=True)

# Feature Engineering
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

# Temporal Split
dates = np.sort(df["transaction_date"].unique())

train_date = dates[int(len(dates) * 0.70)]
val_date = dates[int(len(dates) * 0.85)]

val = df[
    (df["transaction_date"] >= train_date) &
    (df["transaction_date"] < val_date)
].copy()

test = df[
    df["transaction_date"] >= val_date
].copy()

drop_cols = [
    "transaction_id",
    "customer_id",
    "transaction_date",
    "transaction_time",
    "fraud_type",
    TARGET
]

features = joblib.load(FEATURE_PATH)
model = joblib.load(MODEL_PATH)

def prepare_data(data):

    X = data.drop(
        columns=drop_cols,
        errors="ignore"
    )

    y = data[TARGET].astype(int)

    X = pd.get_dummies(X, dummy_na=True)

    X = X.reindex(
        columns=features,
        fill_value=0
    )

    X = X.replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

    return X, y


X_val, y_val = prepare_data(val)
X_test, y_test = prepare_data(test)

print("Generating Validation Predictions...")

val_prob = model.predict_proba(X_val)[:, 1]

pr_auc_val = average_precision_score(y_val, val_prob)

# Select threshold using Validation data only
precisions, recalls, thresholds = precision_recall_curve(
    y_val, val_prob
)

valid_indices = np.where(
    precisions[:-1] >= TARGET_PRECISION
)[0]

if len(valid_indices) > 0:

    best_index = valid_indices[
        np.argmax(recalls[valid_indices])
    ]

    best_threshold = float(thresholds[best_index])
    target_achieved = True

else:

    best_index = np.argmax(precisions[:-1])
    best_threshold = float(thresholds[best_index])
    target_achieved = False

print("Selected Threshold:", best_threshold)

def evaluate(name, y_true, probabilities):

    predictions = (
        probabilities >= best_threshold
    ).astype(int)

    precision = precision_score(
        y_true, predictions, zero_division=0
    )

    recall = recall_score(
        y_true, predictions, zero_division=0
    )

    f1 = f1_score(
        y_true, predictions, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0

    pr_auc = average_precision_score(
        y_true, probabilities
    )

    return {
        "Dataset": name,
        "Target Precision": TARGET_PRECISION,
        "Target Achieved on Validation": target_achieved,
        "Actual Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "PR-AUC": pr_auc,
        "Threshold": best_threshold,
        "FPR": fpr,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn
    }


print("Generating Test Predictions...")

test_prob = model.predict_proba(X_test)[:, 1]

val_results = evaluate(
    "Validation", y_val, val_prob
)

test_results = evaluate(
    "Test", y_test, test_prob
)

results = pd.DataFrame([
    val_results,
    test_results
])

results.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 50)
print("RECALL AT 95% PRECISION - FINAL RESULTS")
print("=" * 50)

print(results.to_string(index=False))

print("\nResults saved:", OUTPUT_PATH)
print("\nEVALUATION COMPLETED!")
