
import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "raw", "bank_fraud.csv")
MODEL_DIR = os.path.join(ROOT, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

DROP_COLS = [
    "transaction_id", "customer_id", "transaction_date",
    "transaction_time", "fraud_type", "is_fraud"
]

MODEL_CONFIG = {
    "Logistic Regression": {
        "model": "logistic_regression.pkl",
        "threshold": "threshold.pkl",
        "features": None,
        "encoded": False
    },
    "Balanced Random Forest": {
        "model": "balanced_random_forest.pkl",
        "threshold": "rf_threshold.pkl",
        "features": "rf_features.pkl",
        "encoded": True
    },
    "XGBoost": {
        "model": "xgboost.pkl",
        "threshold": "xgb_threshold.pkl",
        "features": "xgb_features.pkl",
        "encoded": True
    },
    "LightGBM": {
        "model": "lightgbm.pkl",
        "threshold": "lgbm_threshold.pkl",
        "features": "lgbm_features.pkl",
        "encoded": True
    }
}


def create_features(df):
    df = df.copy()

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

    return df


def prepare_features(model, config, X):
    if config["encoded"]:
        X = pd.get_dummies(X, dummy_na=True)

        feature_path = os.path.join(
            MODEL_DIR, config["features"]
        )

        saved_features = joblib.load(feature_path)

        X = X.reindex(
            columns=saved_features,
            fill_value=0
        )

    else:
        if hasattr(model, "feature_names_in_"):
            X = X.reindex(
                columns=list(model.feature_names_in_),
                fill_value=0
            )

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X


def select_best_f1_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    if len(thresholds) == 0:
        return 0.5

    f1_values = (
        2 * precision[:-1] * recall[:-1]
    ) / (
        precision[:-1] + recall[:-1] + 1e-12
    )

    best_index = int(np.argmax(f1_values))

    return float(thresholds[best_index])


def evaluate(name, config, X_val, y_val, X_test, y_test):
    model_path = os.path.join(MODEL_DIR, config["model"])
    threshold_path = os.path.join(MODEL_DIR, config["threshold"])

    if not os.path.exists(model_path):
        print(f"{name}: Model missing. Skipping.")
        return None

    model = joblib.load(model_path)

    Xv = prepare_features(model, config, X_val)
    Xt = prepare_features(model, config, X_test)

    print(f"\nEvaluating {name}...")

    # Select threshold ONLY on validation data
    val_prob = model.predict_proba(Xv)[:, 1]

    threshold = select_best_f1_threshold(
        y_val, val_prob
    )

    joblib.dump(threshold, threshold_path)

    # Evaluate on untouched test data
    start = time.perf_counter()

    test_prob = model.predict_proba(Xt)[:, 1]

    elapsed = time.perf_counter() - start

    predictions = (test_prob >= threshold).astype(int)

    pr_auc = average_precision_score(y_test, test_prob)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0

    latency = (elapsed / len(Xt)) * 1000

    result = {
        "Model": name,
        "PR-AUC": round(float(pr_auc), 4),
        "Threshold": round(threshold, 6),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "F1 Score": round(float(f1), 4),
        "FPR": round(float(fpr), 4),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "Batch Avg ms/transaction": round(float(latency), 4)
    }

    print("PR-AUC:", result["PR-AUC"])
    print("Threshold:", result["Threshold"])
    print("Precision:", result["Precision"])
    print("Recall:", result["Recall"])
    print("F1 Score:", result["F1 Score"])
    print("FPR:", result["FPR"])
    print("TP:", tp, "FP:", fp, "FN:", fn, "TN:", tn)

    return result


def main():
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)
    df = create_features(df)

    dates = np.sort(df["transaction_date"].unique())

    train_date = dates[int(len(dates) * 0.70)]
    val_date = dates[int(len(dates) * 0.85)]

    validation = df[
        (df["transaction_date"] >= train_date) &
        (df["transaction_date"] < val_date)
    ].copy()

    test = df[
        df["transaction_date"] >= val_date
    ].copy()

    if validation.empty or test.empty:
        raise ValueError("Validation or test dataset is empty.")

    X_val = validation.drop(columns=DROP_COLS, errors="ignore")
    y_val = validation["is_fraud"].astype(int)

    X_test = test.drop(columns=DROP_COLS, errors="ignore")
    y_test = test["is_fraud"].astype(int)

    print("\nDATA SPLIT")
    print("Validation:", len(X_val))
    print("Test:", len(X_test))
    print("Test Fraud:", int(y_test.sum()))
    print("Test Normal:", int((y_test == 0).sum()))

    results = []

    for name, config in MODEL_CONFIG.items():
        try:
            result = evaluate(
                name, config, X_val, y_val, X_test, y_test
            )

            if result is not None:
                results.append(result)

        except Exception as error:
            print(f"\nERROR in {name}: {error}")

    if not results:
        print("No models evaluated.")
        return

    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
        by="F1 Score", ascending=False
    )

    print("\n" + "=" * 90)
    print("FINAL TEST MODEL COMPARISON")
    print("=" * 90)
    print(comparison.to_string(index=False))

    output = os.path.join(MODEL_DIR, "model_comparison.csv")
    comparison.to_csv(output, index=False)

    print("\nComparison saved:", output)
    print("Threshold files updated successfully!")


if __name__ == "__main__":
    main()
