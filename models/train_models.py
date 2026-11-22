
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    confusion_matrix
)


# 1. Load dataset
print("Loading dataset...")

df = pd.read_csv("data/raw/bank_fraud.csv")

df["transaction_date"] = pd.to_datetime(
    df["transaction_date"]
)

# 2. Feature engineering
df = df.sort_values(
    ["customer_id", "transaction_date", "transaction_time"]
).copy()

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

# 3. Chronological split using complete dates
dates = np.sort(df["transaction_date"].dropna().unique())

train_date = dates[int(len(dates) * 0.70)]
validation_date = dates[int(len(dates) * 0.85)]

train = df[df["transaction_date"] < train_date].copy()

validation = df[
    (df["transaction_date"] >= train_date) &
    (df["transaction_date"] < validation_date)
].copy()

test = df[df["transaction_date"] >= validation_date].copy()

print("Train:", train.shape)
print("Validation:", validation.shape)
print("Test:", test.shape)

# 4. Select features
target = "is_fraud"

drop_columns = [
    "transaction_id",
    "customer_id",
    "transaction_date",
    "transaction_time",
    "fraud_type",
    target
]

X_train = train.drop(columns=drop_columns)
y_train = train[target]

X_val = validation.drop(columns=drop_columns)
y_val = validation[target]

X_test = test.drop(columns=drop_columns)
y_test = test[target]

# 5. Identify numeric and categorical columns
numeric_features = X_train.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 6. Logistic Regression baseline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=500,
        solver="saga",
        random_state=42
    ))
])

print("\nTraining Logistic Regression...")
model.fit(X_train, y_train)

# 7. Validation evaluation
val_prob = model.predict_proba(X_val)[:, 1]

pr_auc = average_precision_score(y_val, val_prob)

# Select threshold with at least 95% precision
thresholds = np.linspace(0.01, 0.99, 99)

valid_thresholds = []

for threshold in thresholds:
    predictions = (val_prob >= threshold).astype(int)

    precision = precision_score(
        y_val, predictions, zero_division=0
    )

    if precision >= 0.95 and predictions.sum() > 0:
        valid_thresholds.append(threshold)

if valid_thresholds:
    threshold = min(valid_thresholds)
else:
    threshold = 0.50

val_pred = (val_prob >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_val, val_pred, labels=[0, 1]
).ravel()

fpr = fp / (fp + tn) if (fp + tn) else 0

print("\nVALIDATION RESULTS")
print("PR-AUC:", round(pr_auc, 4))
print("Threshold:", round(threshold, 4))
print("Precision:", round(
    precision_score(y_val, val_pred, zero_division=0), 4
))
print("Recall:", round(
    recall_score(y_val, val_pred, zero_division=0), 4
))
print("FPR:", round(fpr, 4))

# 8. Save model and threshold
joblib.dump(model, "models/logistic_regression.pkl")
joblib.dump(threshold, "models/threshold.pkl")

print("\nMODEL SAVED SUCCESSFULLY!")
