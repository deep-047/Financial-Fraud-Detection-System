
import os
import joblib
import pandas as pd
import numpy as np

from imblearn.over_sampling import ADASYN
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve
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

# 4. Temporal split
dates = np.sort(df["transaction_date"].unique())

train_date = dates[int(len(dates) * 0.70)]
val_date = dates[int(len(dates) * 0.85)]

train = df[df["transaction_date"] < train_date].copy()

val = df[
    (df["transaction_date"] >= train_date) &
    (df["transaction_date"] < val_date)
].copy()

print("Training rows:", len(train))
print("Validation rows:", len(val))

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

# 6. Encode categorical features
X_train = pd.get_dummies(X_train, dummy_na=True)
X_val = pd.get_dummies(X_val, dummy_na=True)

X_train, X_val = X_train.align(
    X_val,
    join="left",
    axis=1,
    fill_value=0
)

X_train = X_train.replace(
    [np.inf, -np.inf], np.nan
).fillna(0)

X_val = X_val.replace(
    [np.inf, -np.inf], np.nan
).fillna(0)

# 7. Common LightGBM settings
normal_count = int((y_train == 0).sum())
fraud_count = int((y_train == 1).sum())

scale_pos_weight = normal_count / fraud_count

common_params = {
    "objective": "binary",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1
}

# 8. Train Original LightGBM
print("\nTraining Original LightGBM...")

original_model = LGBMClassifier(
    **common_params,
    scale_pos_weight=scale_pos_weight
)

original_model.fit(X_train, y_train)

# 9. Apply ADASYN only on training data
print("\nClass distribution BEFORE ADASYN:")
print(y_train.value_counts())

print("\nApplying ADASYN...")

adasyn = ADASYN(
    random_state=42,
    sampling_strategy="auto",
    n_neighbors=5
)

X_resampled, y_resampled = adasyn.fit_resample(
    X_train,
    y_train
)

print("\nClass distribution AFTER ADASYN:")
print(pd.Series(y_resampled).value_counts())

# 10. Train ADASYN LightGBM
print("\nTraining LightGBM with ADASYN...")

adasyn_model = LGBMClassifier(**common_params)

adasyn_model.fit(X_resampled, y_resampled)

# 11. Evaluation function
def evaluate_model(name, model, X_validation, y_validation):

    probabilities = model.predict_proba(X_validation)[:, 1]

    pr_auc = average_precision_score(
        y_validation, probabilities
    )

    precision_values, recall_values, thresholds = (
        precision_recall_curve(y_validation, probabilities)
    )

    f1_values = (
        2 * precision_values[:-1] * recall_values[:-1]
    ) / (
        precision_values[:-1] + recall_values[:-1] + 1e-12
    )

    best_index = int(np.argmax(f1_values))
    threshold = float(thresholds[best_index])

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_validation, predictions, zero_division=0
    )

    recall = recall_score(
        y_validation, predictions, zero_division=0
    )

    f1 = f1_score(
        y_validation, predictions, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_validation, predictions, labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0

    result = {
        "Model": name,
        "PR-AUC": pr_auc,
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "FPR": fpr,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn
    }

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    for key, value in result.items():
        if isinstance(value, (float, np.floating)):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    return result, threshold


# 12. Evaluate both models
print("\nEvaluating both models...")

original_result, original_threshold = evaluate_model(
    "Original LightGBM",
    original_model,
    X_val,
    y_val
)

adasyn_result, adasyn_threshold = evaluate_model(
    "ADASYN LightGBM",
    adasyn_model,
    X_val,
    y_val
)

# 13. Save comparison CSV
comparison = pd.DataFrame([
    original_result,
    adasyn_result
])

comparison_path = os.path.join(
    MODEL_DIR, "adasyn_comparison.csv"
)

comparison.to_csv(comparison_path, index=False)

print("\nFINAL ADASYN COMPARISON")
print(comparison.to_string(index=False))

# 14. Save experiment models separately
joblib.dump(
    original_model,
    os.path.join(MODEL_DIR, "lightgbm_adasyn_comparison.pkl")
)

joblib.dump(
    adasyn_model,
    os.path.join(MODEL_DIR, "adasyn_lightgbm.pkl")
)

joblib.dump(
    list(X_train.columns),
    os.path.join(MODEL_DIR, "adasyn_features.pkl")
)

joblib.dump(
    adasyn_threshold,
    os.path.join(MODEL_DIR, "adasyn_threshold.pkl")
)

joblib.dump(
    original_threshold,
    os.path.join(MODEL_DIR, "lightgbm_adasyn_threshold.pkl")
)

print("\nADASYN EXPERIMENT COMPLETED!")
print("Comparison saved:", comparison_path)
