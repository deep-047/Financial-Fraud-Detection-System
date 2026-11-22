
import os
import joblib
import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    confusion_matrix
)

# 1. Project paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(ROOT, "data", "raw", "bank_fraud.csv")
MODEL_DIR = os.path.join(ROOT, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

TARGET = "is_fraud"

# 2. Load dataset
print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

df["transaction_date"] = pd.to_datetime(
    df["transaction_date"], errors="coerce"
)

df = df.dropna(subset=["transaction_date"])

df = df.sort_values(
    ["customer_id", "transaction_date", "transaction_time"]
).reset_index(drop=True)

# 3. Feature engineering
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

# 4. Temporal train-validation split
dates = np.sort(df["transaction_date"].unique())

if len(dates) < 3:
    raise ValueError("Not enough unique dates for temporal splitting.")

train_date = dates[int(len(dates) * 0.70)]
val_date = dates[int(len(dates) * 0.85)]

train = df[df["transaction_date"] < train_date].copy()

val = df[
    (df["transaction_date"] >= train_date) &
    (df["transaction_date"] < val_date)
].copy()

print("\nDataset Distribution:")
print("Train:", train.shape)
print("Validation:", val.shape)

if train.empty or val.empty:
    raise ValueError("Train or validation dataset is empty.")

if train[TARGET].nunique() < 2 or val[TARGET].nunique() < 2:
    raise ValueError("Both classes must exist in train and validation data.")

# 5. Prepare features
drop_cols = [
    "transaction_id",
    "customer_id",
    "transaction_date",
    "transaction_time",
    "fraud_type",
    TARGET
]

X_train = train.drop(columns=drop_cols, errors="ignore")
X_val = val.drop(columns=drop_cols, errors="ignore")

y_train = train[TARGET].astype(int)
y_val = val[TARGET].astype(int)

# Consistent categorical encoding
X_train = pd.get_dummies(X_train, dummy_na=True)
X_val = pd.get_dummies(X_val, dummy_na=True)

X_train, X_val = X_train.align(
    X_val, join="left", axis=1, fill_value=0
)

X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)

# 6. Handle class imbalance
fraud_count = int((y_train == 1).sum())
normal_count = int((y_train == 0).sum())

if fraud_count == 0:
    raise ValueError("No fraud transactions in training data.")

scale_pos_weight = normal_count / fraud_count

print("\nClass Distribution:")
print("Normal:", normal_count)
print("Fraud:", fraud_count)
print("Scale Pos Weight:", round(scale_pos_weight, 4))

# 7. Train LightGBM
model = LGBMClassifier(
    objective="binary",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

print("\nTraining LightGBM...")
model.fit(X_train, y_train)

# 8. Validation probabilities
probabilities = model.predict_proba(X_val)[:, 1]

pr_auc = average_precision_score(y_val, probabilities)

# 9. Select threshold
precision_values, recall_values, thresholds = precision_recall_curve(
    y_val, probabilities
)

target_precision = 0.95

valid_indices = np.where(
    precision_values[:-1] >= target_precision
)[0]

if len(valid_indices) > 0:
    best_index = valid_indices[
        np.argmax(recall_values[valid_indices])
    ]
    threshold = float(thresholds[best_index])
    target_achieved = True

else:
    f1_scores = (
        2 * precision_values[:-1] * recall_values[:-1]
    ) / (
        precision_values[:-1] + recall_values[:-1] + 1e-12
    )

    best_index = int(np.argmax(f1_scores))
    threshold = float(thresholds[best_index])
    target_achieved = False

# 10. Evaluate model
predictions = (probabilities >= threshold).astype(int)

precision = precision_score(
    y_val, predictions, zero_division=0
)

recall = recall_score(
    y_val, predictions, zero_division=0
)

tn, fp, fn, tp = confusion_matrix(
    y_val, predictions, labels=[0, 1]
).ravel()

fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

print("\n" + "=" * 50)
print("LIGHTGBM VALIDATION RESULTS")
print("=" * 50)

print("PR-AUC:", round(pr_auc, 4))
print("Threshold:", round(threshold, 6))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("FPR:", round(fpr, 4))

print("True Positives:", tp)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Negatives:", tn)

if target_achieved:
    print("\n95% PRECISION TARGET ACHIEVED!")
else:
    print("\n95% PRECISION TARGET NOT ACHIEVED.")
    print("Using the best F1 threshold.")

# 11. Save model and metadata
joblib.dump(model, os.path.join(MODEL_DIR, "lightgbm.pkl"))

joblib.dump(
    list(X_train.columns),
    os.path.join(MODEL_DIR, "lgbm_features.pkl")
)

joblib.dump(
    threshold,
    os.path.join(MODEL_DIR, "lgbm_threshold.pkl")
)

joblib.dump(
    {
        "target_precision": target_precision,
        "target_achieved": target_achieved,
        "validation_pr_auc": float(pr_auc),
        "validation_precision": float(precision),
        "validation_recall": float(recall),
        "validation_fpr": float(fpr)
    },
    os.path.join(MODEL_DIR, "lgbm_metrics.pkl")
)

print("\nLIGHTGBM MODEL SAVED SUCCESSFULLY!")
