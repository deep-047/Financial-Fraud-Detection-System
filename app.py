import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Financial Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "bank_fraud.csv"
MODELS_DIR = ROOT / "models"
MODEL_RESULTS_PATH = MODELS_DIR / "model_comparison.csv"


# =========================================================
# MODEL ARTIFACT CONFIGURATION
# =========================================================

MODEL_CANDIDATES = [
    {
        "name": "LightGBM",
        "model": MODELS_DIR / "lightgbm.pkl",
        "features": MODELS_DIR / "lgbm_features.pkl",
        "threshold": MODELS_DIR / "lgbm_threshold.pkl",
    },
    {
        "name": "XGBoost",
        "model": MODELS_DIR / "xgboost.pkl",
        "features": MODELS_DIR / "xgb_features.pkl",
        "threshold": MODELS_DIR / "xgb_threshold.pkl",
    },
    {
        "name": "Balanced Random Forest",
        "model": MODELS_DIR / "balanced_random_forest.pkl",
        "features": MODELS_DIR / "rf_features.pkl",
        "threshold": MODELS_DIR / "rf_threshold.pkl",
    },
    {
        "name": "Logistic Regression",
        "model": MODELS_DIR / "logistic_regression.pkl",
        "features":  None,
        "threshold": MODELS_DIR / "threshold.pkl",
    },
]


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f4f7fb;
    }

    [data-testid="stSidebar"] {
        background-color: #101b35;
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    [data-testid="stMetric"] {
        background: white;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #e1e7f0;
    }

    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #14213d;
    }

    .section-caption {
        color: #64748b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DATA LOADING
# =========================================================

@st.cache_data
def load_data(path_string):

    data = pd.read_csv(path_string)

    if "transaction_date" in data.columns:
        data["transaction_date"] = pd.to_datetime(
            data["transaction_date"],
            errors="coerce"
        )

    if "is_fraud" in data.columns:
        data["is_fraud"] = pd.to_numeric(
            data["is_fraud"],
            errors="coerce"
        ).fillna(0).astype(int)

    if "transaction_amount" in data.columns:
        data["transaction_amount"] = pd.to_numeric(
            data["transaction_amount"],
            errors="coerce"
        )

    return data


@st.cache_data
def load_csv(path_string):

    if os.path.exists(path_string):
        return pd.read_csv(path_string)

    return pd.DataFrame()


def load_pickle(path):

    try:
        return joblib.load(path)

    except Exception as exc:
        st.warning(f"Could not load {path.name}: {exc}")
        return None


# =========================================================
# FEATURE NAME EXTRACTION
# =========================================================

def extract_feature_names(feature_artifact, model):

    if feature_artifact is not None:

        if isinstance(feature_artifact, dict):

            for key in [
                "features",
                "feature_names",
                "columns",
                "X_columns",
            ]:

                if key in feature_artifact:
                    return list(feature_artifact[key])

        if isinstance(
            feature_artifact,
            (list, tuple, np.ndarray, pd.Index)
        ):
            return list(feature_artifact)

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "feature_name_"):

        names = model.feature_name_()

        if names:
            return list(names)

    return None


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def create_dashboard_features(data):

    df = data.copy()

    df["_original_order"] = np.arange(len(df))

    # Convert numeric columns
    numeric_columns = [
        "transaction_amount",
        "failed_attempts",
        "distance_from_home_km",
    ]

    for col in numeric_columns:

        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    if "transaction_date" in df.columns:

        df["transaction_date"] = pd.to_datetime(
            df["transaction_date"],
            errors="coerce"
        )

    # Sort in the same chronological order used by training
    sort_columns = [
        col for col in [
            "customer_id",
            "transaction_date",
            "transaction_time",
        ]
        if col in df.columns
    ]

    if sort_columns:

        df = df.sort_values(
            sort_columns,
            kind="mergesort"
        )

    # Historical mean spend using previous transactions
    if {
        "customer_id",
        "transaction_amount",
    }.issubset(df.columns):

        df["Hist_Mean_Spend"] = (
            df.groupby("customer_id")["transaction_amount"]
            .transform(
                lambda x: (
                    x.shift(1)
                    .expanding()
                    .mean()
                )
            )
        )

        df["Hist_Mean_Spend"] = (
            df["Hist_Mean_Spend"].fillna(0)
        )

        df["Spend_Ratio"] = (
            df["transaction_amount"]
            / (df["Hist_Mean_Spend"] + 1)
        )

    else:

        df["Hist_Mean_Spend"] = 0
        df["Spend_Ratio"] = 0

    # Date-based features
    if "transaction_date" in df.columns:

        df["transaction_month"] = (
            df["transaction_date"].dt.month
        )

        df["transaction_day"] = (
            df["transaction_date"].dt.day
        )

    # Failed attempts feature
    if "failed_attempts" in df.columns:

        df["high_failed_attempts"] = (
            df["failed_attempts"] >= 2
        ).astype(int)

    else:

        df["high_failed_attempts"] = 0

    # Distance feature
    if "distance_from_home_km" in df.columns:

        df["high_distance"] = (
            df["distance_from_home_km"] > 100
        ).astype(int)

    else:

        df["high_distance"] = 0

    # Restore original row order
    df = df.sort_values("_original_order")

    df = df.drop(
        columns=["_original_order"],
        errors="ignore"
    )

    return df


# =========================================================
# MODEL INPUT PREPARATION
# =========================================================


def make_model_input(data, feature_names, model=None):

    if not feature_names:
        raise ValueError(
            "Saved model feature names are not available."
        )

    df = create_dashboard_features(data)

    drop_columns = [
        "transaction_id",
        "customer_id",
        "transaction_date",
        "transaction_time",
        "fraud_type",
        "is_fraud",
    ]

    X = df.drop(
        columns=drop_columns,
        errors="ignore"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Logistic Regression Pipeline:
    # Let its internal preprocessor handle encoding and imputation.
    is_pipeline = (
        hasattr(model, "named_steps")
        and "preprocessor" in model.named_steps
    )

    if is_pipeline:

        missing_features = [
            col for col in feature_names
            if col not in X.columns
        ]

        X = X.reindex(
            columns=feature_names
        )

    else:

        # Tree-based models use one-hot encoded features.
        X = pd.get_dummies(
            X,
            dummy_na=True
        )

        missing_features = [
            col for col in feature_names
            if col not in X.columns
        ]

        X = X.reindex(
            columns=feature_names,
            fill_value=0
        )

        X = X.fillna(0)

    return X, missing_features



# =========================================================
# FRAUD PROBABILITY
# =========================================================

def positive_class_probability(model, X):

    if hasattr(model, "predict_proba"):

        probabilities = np.asarray(
            model.predict_proba(X)
        )

        if probabilities.ndim == 2:

            classes = getattr(
                model,
                "classes_",
                None
            )

            if classes is not None:

                classes = list(classes)

                if 1 in classes:
                    fraud_index = classes.index(1)
                else:
                    fraud_index = 1

            else:
                fraud_index = 1

            return probabilities[:, fraud_index]

        return probabilities.reshape(-1)

    if hasattr(model, "decision_function"):

        scores = np.asarray(
            model.decision_function(X),
            dtype=float
        )

        scores = np.clip(
            scores,
            -30,
            30
        )

        return 1 / (1 + np.exp(-scores))

    return np.asarray(
        model.predict(X),
        dtype=float
    ).reshape(-1)


# =========================================================
# LOAD MODEL ARTIFACTS
# =========================================================

@st.cache_resource
def load_prediction_artifacts():

    loaded = []

    for item in MODEL_CANDIDATES:

        model_path = item["model"]
        feature_path = item["features"]
        threshold_path = item["threshold"]

        if not model_path.exists():
            continue

        model = load_pickle(model_path)

        if model is None:
            continue

        feature_artifact = None

        if feature_path is not None and feature_path.exists():
            feature_artifact = load_pickle(feature_path)

        feature_names = extract_feature_names(
            feature_artifact,
            model
        )

        # Do not run prediction without correct feature names
        if not feature_names:
            continue

        threshold = 0.5

        if threshold_path.exists():

            threshold_artifact = load_pickle(
                threshold_path
            )

            try:

                if isinstance(threshold_artifact, dict):

                    threshold = float(
                        threshold_artifact.get(
                            "threshold",
                            threshold_artifact.get(
                                "best_threshold",
                                0.5
                            )
                        )
                    )

                else:

                    threshold = float(
                        np.asarray(
                            threshold_artifact
                        ).reshape(-1)[0]
                    )

            except (
                TypeError,
                ValueError,
                IndexError,
            ):

                threshold = 0.5

        loaded.append({

            "name": item["name"],

            "model": model,

            "feature_names": feature_names,

            "threshold": float(
                np.clip(threshold, 0, 1)
            ),

        })

    return loaded


# =========================================================
# COMMON FUNCTIONS
# =========================================================

def format_money(value):

    return f"${value:,.2f}"


def safe_column(data, column):

    if column in data.columns:
        return data[column]

    return pd.Series(
        [0] * len(data),
        index=data.index
    )


# =========================================================
# LOAD REQUIRED DATA
# =========================================================

if not DATA_PATH.exists():

    st.error(
        "Dataset not found. Expected location: "
        "data/raw/bank_fraud.csv"
    )

    st.stop()


try:

    df = load_data(str(DATA_PATH))

except Exception as exc:

    st.error(f"Could not load dataset: {exc}")
    st.stop()


if "is_fraud" not in df.columns:

    st.error(
        "Dataset must contain an 'is_fraud' column."
    )

    st.stop()


model_results = load_csv(
    str(MODEL_RESULTS_PATH)
)

prediction_artifacts = load_prediction_artifacts()


# =========================================================
# COMMON METRICS
# =========================================================

total_transactions = len(df)

fraud_count = int(
    (df["is_fraud"] == 1).sum()
)

legitimate_count = int(
    (df["is_fraud"] == 0).sum()
)

fraud_rate = (
    fraud_count / total_transactions * 100
    if total_transactions else 0
)

fraud_rate_bps = fraud_rate * 100

fraud_amount = 0

if "transaction_amount" in df.columns:

    fraud_amount = pd.to_numeric(
        df.loc[
            df["is_fraud"] == 1,
            "transaction_amount"
        ],
        errors="coerce"
    ).fillna(0).sum()


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("🛡️ FRAUD DETECTION")

st.sidebar.caption(
    "Financial Risk Intelligence"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Executive Overview",
        "Risk Hotspots",
        "Exploratory Data Analysis",
        "Machine Learning",
        "Fraud Operations",
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Financial Fraud Detection System"
)

st.sidebar.caption(
    "Academic Project | 2026"
)


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '🛡️ Financial Fraud Detection System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="section-caption">'
    'AI-Powered Transaction Risk Monitoring'
    '</p>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# PAGE 1: EXECUTIVE OVERVIEW
# =========================================================

if page == "Executive Overview":

    st.header("Executive Overview")

    st.caption(
        "Overall financial transaction and fraud intelligence"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

    c2.metric(
        "Fraud Transactions",
        f"{fraud_count:,}"
    )

    c3.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%"
    )

    c4.metric(
        "Fraud Rate (bps)",
        f"{fraud_rate_bps:,.1f} bps"
    )

    left, right = st.columns(2)

    with left:

        chart_data = pd.DataFrame({

            "Transaction Type": [
                "Fraud",
                "Legitimate"
            ],

            "Count": [
                fraud_count,
                legitimate_count
            ],

        })

        fig = px.pie(
            chart_data,
            names="Transaction Type",
            values="Count",
            hole=0.55,
            title="Fraud vs Legitimate Transactions",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.subheader(
            "Fraud by Transaction Date"
        )

        if "transaction_date" in df.columns:

            daily = (
                df.dropna(
                    subset=["transaction_date"]
                )
                .set_index("transaction_date")
                ["is_fraud"]
                .resample("D")
                .sum()
                .reset_index()
            )

            daily.columns = [
                "Date",
                "Fraud Count"
            ]

            fig = px.line(
                daily,
                x="Date",
                y="Fraud Count",
                title="Daily Fraud Transactions",
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "Transaction date is unavailable."
            )

    st.divider()

    st.subheader("Financial Impact")

    c1, c2 = st.columns(2)

    c1.metric(
        "Recorded Fraud Transaction Amount",
        format_money(fraud_amount)
    )

    c2.metric(
        "Legitimate Transactions",
        f"{legitimate_count:,}"
    )

    st.info(
        "Dollars Saved is not reported as a confirmed value. "
        "A valid estimate requires held-out model predictions "
        "and explicit financial cost assumptions. "
        "Historical fraud amount is not the same as money saved."
    )

    st.subheader("Recent Transactions")

    recent_columns = [
        "transaction_id",
        "transaction_date",
        "transaction_amount",
        "merchant_category",
        "is_international",
        "is_fraud",
    ]

    available_columns = [
        col for col in recent_columns
        if col in df.columns
    ]

    st.dataframe(
        df[available_columns]
        .tail(10)
        .iloc[::-1],
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# PAGE 2: RISK HOTSPOTS
# =========================================================

elif page == "Risk Hotspots":

    st.header("Risk Hotspots")

    st.caption(
        "Merchant category and international transaction risk"
    )

    if "merchant_category" in df.columns:

        st.subheader(
            "Merchant Category Risk"
        )

        merchant_risk = (

            df.groupby(
                "merchant_category",
                dropna=False
            )

            .agg(

                Transactions=("is_fraud", "count"),

                Fraud_Count=("is_fraud", "sum"),

            )

            .reset_index()

        )

        merchant_risk["Fraud Rate (%)"] = (

            merchant_risk["Fraud_Count"]

            / merchant_risk["Transactions"]
            .replace(0, np.nan)

            * 100

        )

        merchant_risk = merchant_risk.sort_values(
            "Fraud Rate (%)",
            ascending=False
        )

        fig = px.bar(
            merchant_risk,
            x="merchant_category",
            y="Fraud Rate (%)",
            color="Fraud Rate (%)",
            text_auto=".2f",
            title="Fraud Rate by Merchant Category",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            merchant_risk,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Merchant category column is unavailable."
        )

    st.divider()

    if "is_international" in df.columns:

        st.subheader(
            "International Transaction Analysis"
        )

        international = (

            df.groupby(
                "is_international",
                dropna=False
            )

            .agg(

                Transactions=("is_fraud", "count"),

                Fraud_Count=("is_fraud", "sum"),

            )

            .reset_index()

        )

        international["Fraud Rate (%)"] = (

            international["Fraud_Count"]

            / international["Transactions"]
            .replace(0, np.nan)

            * 100

        )

        international["Transaction Type"] = (
            international["is_international"].astype(str)
        )

        fig = px.bar(
            international,
            x="Transaction Type",
            y="Transactions",
            color="Transaction Type",
            title="International Transaction Volume",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        fig = px.bar(
            international,
            x="Transaction Type",
            y="Fraud Rate (%)",
            color="Transaction Type",
            title="Fraud Rate: International vs Domestic",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            international,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "International transaction column is unavailable."
        )


# =========================================================
# PAGE 3: EXPLORATORY DATA ANALYSIS
# =========================================================

elif page == "Exploratory Data Analysis":

    st.header(
        "Exploratory Data Analysis"
    )

    st.caption(
        "Understanding transaction behaviour and fraud patterns"
    )

    if "transaction_amount" in df.columns:

        st.subheader(
            "Transaction Amount Density"
        )

        amount_data = df.copy()

        amount_data["Transaction Type"] = (
            amount_data["is_fraud"].map({
                0: "Legitimate",
                1: "Fraud"
            })
        )

        fig = px.histogram(
            amount_data,
            x="transaction_amount",
            color="Transaction Type",
            marginal="box",
            barmode="overlay",
            opacity=0.65,
            nbins=60,
            title="Transaction Amount: Fraud vs Legitimate",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        amount_stats = (

            df.groupby("is_fraud")["transaction_amount"]

            .agg([
                "count",
                "mean",
                "median",
                "min",
                "max"
            ])

            .reset_index()

        )

        amount_stats["Transaction Type"] = (
            amount_stats["is_fraud"].map({
                0: "Legitimate",
                1: "Fraud"
            })
        )

        st.subheader(
            "Transaction Amount Statistics"
        )

        st.dataframe(
            amount_stats.drop(
                columns=["is_fraud"]
            ),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Transaction amount column is unavailable."
        )

    st.divider()

    if "hour_of_day" in df.columns:

        st.subheader(
            "Fraud by Hour of Day"
        )

        hourly = (

            df.groupby("hour_of_day")["is_fraud"]

            .agg([
                "count",
                "sum"
            ])

            .reset_index()

        )

        hourly["Fraud Rate (%)"] = (

            hourly["sum"]

            / hourly["count"].replace(0, np.nan)

            * 100

        )

        fig = px.line(
            hourly,
            x="hour_of_day",
            y="Fraud Rate (%)",
            markers=True,
            title="Hourly Fraud Rate",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if "payment_method" in df.columns:

        st.subheader(
            "Fraud by Payment Method"
        )

        payment = (

            df.groupby("payment_method")["is_fraud"]

            .mean()

            .mul(100)

            .reset_index(
                name="Fraud Rate (%)"
            )

        )

        fig = px.bar(
            payment,
            x="payment_method",
            y="Fraud Rate (%)",
            color="Fraud Rate (%)",
            title="Fraud Rate by Payment Method",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# PAGE 4: MACHINE LEARNING
# =========================================================


elif page == "Machine Learning":

    st.header("Machine Learning Performance")

    st.caption(
        "Model evaluation, fraud detection performance "
        "and operational threshold analysis."
    )

    # -----------------------------------------
    # MODEL COMPARISON
    # -----------------------------------------

    if model_results.empty:

        st.warning(
            "Model comparison file not found. "
            "Run the model comparison script first."
        )

    else:

        st.subheader("Model Comparison")

        st.dataframe(
            model_results,
            use_container_width=True,
            hide_index=True
        )

        if {"Model", "PR-AUC"}.issubset(model_results.columns):

            fig = px.bar(
                model_results,
                x="Model",
                y="PR-AUC",
                color="Model",
                text_auto=".4f",
                title="Precision-Recall AUC Comparison",
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        metric_columns = [
            col for col in ["Precision", "Recall", "FPR"]
            if col in model_results.columns
        ]

        if metric_columns and "Model" in model_results.columns:

            metric_data = model_results.melt(
                id_vars="Model",
                value_vars=metric_columns,
                var_name="Metric",
                value_name="Score",
            )

            fig = px.bar(
                metric_data,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",
                title="Model Precision, Recall and FPR",
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        if "Inference Latency (ms)" in model_results.columns:

            fig = px.bar(
                model_results,
                x="Model",
                y="Inference Latency (ms)",
                color="Model",
                title="Batch Average Inference Latency",
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.caption(
                "Batch average latency is not single-transaction "
                "API latency. Assignment target: below 25 ms "
                "per API transaction call."
            )

    # -----------------------------------------
    # RECALL AT 95% PRECISION
    # -----------------------------------------

    st.divider()

    st.subheader("Recall at 95% Precision")

    precision_file = MODELS_DIR / "precision_recall_95.csv"

    if precision_file.exists():

        try:

            precision_results = pd.read_csv(precision_file)

            val_result = precision_results[
                precision_results["Dataset"] == "Validation"
            ]

            test_result = precision_results[
                precision_results["Dataset"] == "Test"
            ]

            if not val_result.empty and not test_result.empty:

                val_row = val_result.iloc[0]
                test_row = test_result.iloc[0]

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Validation Precision",
                        f"{val_row['Actual Precision'] * 100:.2f}%"
                    )

                with col2:
                    st.metric(
                        "Validation Recall",
                        f"{val_row['Recall'] * 100:.4f}%"
                    )

                with col3:
                    st.metric(
                        "Test Recall",
                        f"{test_row['Recall'] * 100:.4f}%"
                    )

                st.dataframe(
                    precision_results,
                    use_container_width=True,
                    hide_index=True
                )

                if bool(val_row["Target Achieved on Validation"]):

                    st.success(
                        "The 95% precision target was achieved "
                        "on validation data."
                    )

                else:

                    st.warning(
                        "The 95% precision target was not achieved "
                        "on validation data."
                    )

                if test_row["Actual Precision"] < 0.95:

                    st.warning(
                        "The selected validation threshold did not "
                        "achieve 95% precision on independent test data. "
                        "Do not claim that the test target was achieved."
                    )

                st.info(
                    f"Validation-selected threshold: "
                    f"{val_row['Threshold']:.6f}. "
                    "The same threshold was applied to test data."
                )

            else:

                st.warning(
                    "Validation or Test results are missing "
                    "from precision_recall_95.csv."
                )

        except Exception as exc:

            st.error(
                f"Could not load 95% precision results: {exc}"
            )

    else:

        st.warning(
            "Run models/precision_recall_95.py to generate "
            "the 95% precision evaluation report."
        )

    # -----------------------------------------
    # FALSE DECLINE RATE
    # -----------------------------------------
    

    # -----------------------------------------
    # FINAL TEST SET EVALUATION
    # -----------------------------------------

    st.divider()
    st.subheader("Final Temporal Test Evaluation")

    precision_file = MODELS_DIR / "precision_recall_95.csv"

    if precision_file.exists():

        test_results = pd.read_csv(precision_file)

        test_row = test_results[
            test_results["Dataset"] == "Test"
        ]

        if not test_row.empty:

            result = test_row.iloc[0]

            st.caption(
                "Independent temporal test-set evaluation "
                "using the threshold selected on validation data."
            )

            a, b, c, d = st.columns(4)

            a.metric(
                "Test Precision",
                f"{result['Actual Precision'] * 100:.2f}%"
            )

            b.metric(
                "Test Recall",
                f"{result['Recall'] * 100:.4f}%"
            )

            c.metric(
                "Test PR-AUC",
                f"{result['PR-AUC']:.4f}"
            )

            d.metric(
                "Test FPR",
                f"{result['FPR'] * 100:.2f}%"
            )

            st.write("**Test Confusion Matrix**")

            cm1, cm2, cm3, cm4 = st.columns(4)

            cm1.metric("TP", f"{int(result['TP']):,}")
            cm2.metric("FP", f"{int(result['FP']):,}")
            cm3.metric("FN", f"{int(result['FN']):,}")
            cm4.metric("TN", f"{int(result['TN']):,}")

            st.dataframe(
                test_results,
                use_container_width=True,
                hide_index=True
            )

            st.warning(
                "The 95% precision target was not achieved "
                "on the independent test set. Test Recall is "
                "0% at the validation-selected threshold."
            )

        else:

            st.warning("Test evaluation row not found.")

    else:

        st.warning(
            "Run models/precision_recall_95.py first."
        )

    # -----------------------------------------
    # FINANCIAL IMPACT ANALYSIS
    # -----------------------------------------

    st.divider()
    st.subheader("Test Set Financial Impact Analysis")

    dollar_file = MODELS_DIR / "dollar_loss_analysis.csv"

    if dollar_file.exists():

        dollar_results = pd.read_csv(dollar_file)

        st.dataframe(
            dollar_results,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "Potentially preventable fraud amount is an estimate, "
            "not confirmed realized savings. Dataset currency and "
            "operational cost assumptions are not established."
        )

    else:

        st.warning(
            "Dollar loss analysis file not found. "
            "Run models/dollar_loss_analysis.py first."
        )

    st.divider()

    st.subheader("False Decline Rate")

    st.caption(
        "False Decline Rate = Legitimate transactions incorrectly "
        "flagged as fraud / All legitimate transactions."
    )

    if not model_results.empty and "FPR" in model_results.columns:

        fdr_data = model_results[["Model", "FPR"]].copy()

        fdr_data["False Decline Rate (%)"] = (
            fdr_data["FPR"] * 100
        )

        st.dataframe(
            fdr_data[["Model", "False Decline Rate (%)"]],
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "False Decline Rate is calculated using FPR. "
            "It represents legitimate transactions incorrectly "
            "flagged by the model at its evaluated threshold."
        )

    # -----------------------------------------
    # PRECISION-RECALL CURVE
    # -----------------------------------------

    
    st.divider()

    st.subheader("Precision-Recall Curve")

    st.warning(
        "The curve and threshold marker below use the full dataset "
        "for diagnostic visualization only. They are not independent "
        "held-out test results. The threshold was selected using "
        "validation data."
    )

    if prediction_artifacts:

        selected_pr_model = st.selectbox(
            "Select model for PR Curve",
            [
                item["name"]
                for item in prediction_artifacts
            ],
            key="pr_model"
        )

        artifact = next(
            item for item in prediction_artifacts
            if item["name"] == selected_pr_model
        )

        try:

            X, missing_features = make_model_input(
                df,
                artifact["feature_names"],
                artifact["model"]
            )

            with st.spinner(
                "Generating Precision-Recall Curve..."
            ):

                probabilities = positive_class_probability(
                    artifact["model"],
                    X
                )

            y_true = df["is_fraud"].astype(int).to_numpy()

            precision, recall, thresholds = (
                precision_recall_curve(
                    y_true,
                    probabilities
                )
            )

            pr_auc = average_precision_score(
                y_true,
                probabilities
            )

            fig = go.Figure()

            # Main PR Curve
            fig.add_trace(
                go.Scatter(
                    x=recall,
                    y=precision,
                    mode="lines",
                    name=selected_pr_model,
                    fill="tozeroy",
                )
            )

            # Add 95% precision threshold marker for LightGBM
            if selected_pr_model == "LightGBM":

                threshold_file = (
                    MODELS_DIR / "precision_recall_95.csv"
                )

                if threshold_file.exists():

                    threshold_results = pd.read_csv(
                        threshold_file
                    )

                    validation_row = threshold_results[
                        threshold_results["Dataset"] == "Validation"
                    ]

                    if not validation_row.empty:

                        selected_threshold = float(
                            validation_row.iloc[0]["Threshold"]
                        )

                        threshold_predictions = (
                            probabilities >= selected_threshold
                        ).astype(int)

                        tp = int(
                            ((y_true == 1) &
                             (threshold_predictions == 1)).sum()
                        )

                        fp = int(
                            ((y_true == 0) &
                             (threshold_predictions == 1)).sum()
                        )

                        fn = int(
                            ((y_true == 1) &
                             (threshold_predictions == 0)).sum()
                        )

                        marker_precision = (
                            tp / (tp + fp)
                            if (tp + fp) > 0 else 0
                        )

                        marker_recall = (
                            tp / (tp + fn)
                            if (tp + fn) > 0 else 0
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=[marker_recall],
                                y=[marker_precision],
                                mode="markers+text",
                                name="95% Precision Threshold",
                                marker=dict(
                                    color="red",
                                    size=13,
                                    symbol="diamond",
                                    line=dict(
                                        color="white",
                                        width=2
                                    )
                                ),
                                text=[
                                    f"Threshold: {selected_threshold:.4f}"
                                ],
                                textposition="top right",
                            )
                        )

                        st.info(
                            f"Validation-selected threshold: "
                            f"{selected_threshold:.6f} | "
                            f"Full-data diagnostic Precision: "
                            f"{marker_precision:.4f} | "
                            f"Full-data diagnostic Recall: "
                            f"{marker_recall:.6f}"
                        )

                else:

                    st.warning(
                        "precision_recall_95.csv not found. "
                        "Run models/precision_recall_95.py first."
                    )

            fig.update_layout(
                title=(
                    f"{selected_pr_model} Precision-Recall Curve "
                    f"(AP = {pr_auc:.4f})"
                ),
                xaxis_title="Recall",
                yaxis_title="Precision",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        except Exception as exc:

            st.error(
                f"Could not generate PR Curve: {exc}"
            )

    else:

        st.info(
            "No compatible saved prediction model is available."
        )


# =========================================================
# PAGE 5: FRAUD OPERATIONS
# =========================================================

elif page == "Fraud Operations":

    st.header(
        "Fraud Operations"
    )

    st.caption(
        "Model-scored transaction triage and investigation queue"
    )

    st.info(
        "This is a local academic demonstration. "
        "It is not connected to a live bank or payment network."
    )

    if not prediction_artifacts:

        st.warning(
            "No compatible saved prediction model was found. "
            "Historical fraud records are shown as a fallback. "
            "These are NOT model predictions."
        )

        historical_queue = df[
            df["is_fraud"] == 1
        ].copy()

        st.metric(
            "Historical Fraud Records",
            f"{len(historical_queue):,}"
        )

        show_columns = [
            col for col in [
                "transaction_id",
                "transaction_date",
                "transaction_amount",
                "merchant_category",
                "payment_method",
                "is_international",
                "is_fraud",
            ]
            if col in historical_queue.columns
        ]

        st.dataframe(
            historical_queue[show_columns].head(500),
            use_container_width=True,
            hide_index=True
        )

    else:

        model_names = [
            item["name"]
            for item in prediction_artifacts
        ]

        selected_model = st.selectbox(
            "Select Prediction Model",
            model_names,
            key="operations_model"
        )

        artifact = next(
            item for item in prediction_artifacts
            if item["name"] == selected_model
        )

        try:

            X, missing_features = make_model_input(
                df,
                artifact["feature_names"],
                artifact["model"]
            )

            if missing_features:

                st.warning(
                    f"{len(missing_features)} expected model "
                    "features were not found in the generated input. "
                    "They have been filled with zero. "
                    "Verify that the training and dashboard "
                    "preprocessing pipelines match."
                )

            with st.spinner(
                "Scoring transactions..."
            ):

                start_time = time.perf_counter()

                probabilities = positive_class_probability(
                    artifact["model"],
                    X
                )

                elapsed_ms = (
                    time.perf_counter() - start_time
                ) * 1000

                per_transaction_ms = (
                    elapsed_ms / len(X)
                    if len(X) else 0
                )

            if len(probabilities) != len(df):

                st.error(
                    "Model output size does not match "
                    "the transaction count."
                )

                st.stop()

            queue = df.copy()

            queue["Fraud Probability (PD)"] = np.clip(
                probabilities,
                0,
                1
            )

            queue["Predicted Fraud"] = (
                queue["Fraud Probability (PD)"]
                >= artifact["threshold"]
            ).astype(int)

            # Assignment triage policy
            queue["Risk Level"] = np.select(
                [
                    queue["Fraud Probability (PD)"] < 0.15,

                    queue["Fraud Probability (PD)"] <= 0.65,
                ],
                [
                    "Low",
                    "Medium",
                ],
                default="High",
            )

            queue["Recommended Action"] = np.select(
                [
                    queue["Fraud Probability (PD)"] < 0.15,

                    queue["Fraud Probability (PD)"] <= 0.65,
                ],
                [
                    "Auto Approve",

                    "Step-Up Authentication",
                ],
                default="Hard Decline & Freeze Alert",
            )

            # Queue includes medium and high risk transactions
            high_queue = queue[
                queue["Fraud Probability (PD)"] >= 0.15
            ].copy()

            high_queue = high_queue.sort_values(
                "Fraud Probability (PD)",
                ascending=False
            )

            # Summary metrics
            low_count = int(
                (queue["Risk Level"] == "Low").sum()
            )

            medium_count = int(
                (queue["Risk Level"] == "Medium").sum()
            )

            high_count = int(
                (queue["Risk Level"] == "High").sum()
            )

            st.divider()

            st.subheader(
                "Transaction Risk Summary"
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Auto Approve",
                f"{low_count:,}"
            )

            c2.metric(
                "Step-Up Authentication",
                f"{medium_count:,}"
            )

            c3.metric(
                "Hard Decline",
                f"{high_count:,}"
            )

            c4.metric(
                "Batch Avg. Inference",
                f"{per_transaction_ms:.4f} ms"
            )

            st.caption(
                "Inference time is measured for the complete batch "
                "and divided by transaction count. It does not "
                "represent single-request API latency."
            )

            st.divider()

            st.subheader(
                "Risk Distribution"
            )

            risk_distribution = pd.DataFrame({

                "Risk Level": [
                    "Low",
                    "Medium",
                    "High"
                ],

                "Transactions": [
                    low_count,
                    medium_count,
                    high_count
                ]

            })

            fig = px.pie(
                risk_distribution,
                names="Risk Level",
                values="Transactions",
                hole=0.55,
                title="Transaction Risk Distribution",
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.divider()

            st.subheader(
                "Fraud Investigation Queue"
            )

            st.caption(
                "Transactions with PD >= 0.15 are displayed first."
            )

            # Risk filter
            risk_filter = st.multiselect(
                "Filter by Risk Level",
                [
                    "Medium",
                    "High"
                ],
                default=[
                    "Medium",
                    "High"
                ]
            )

            filtered_queue = high_queue[
                high_queue["Risk Level"].isin(risk_filter)
            ].copy()

            # Transaction amount filter
            if "transaction_amount" in filtered_queue.columns:

                amount_values = pd.to_numeric(
                    filtered_queue["transaction_amount"],
                    errors="coerce"
                ).fillna(0)

                max_amount = float(
                    amount_values.max()
                )

                if max_amount > 0:

                    minimum_amount = st.slider(
                        "Minimum Transaction Amount",
                        min_value=0.0,
                        max_value=max_amount,
                        value=0.0,
                    )

                    filtered_queue = filtered_queue[
                        pd.to_numeric(
                            filtered_queue["transaction_amount"],
                            errors="coerce"
                        ).fillna(0) >= minimum_amount
                    ]

            display_columns = [
                col for col in [
                    "transaction_id",
                    "transaction_date",
                    "transaction_time",
                    "transaction_amount",
                    "merchant_category",
                    "is_international",
                    "Fraud Probability (PD)",
                    "Risk Level",
                    "Recommended Action",
                    "Predicted Fraud",
                ]
                if col in filtered_queue.columns
            ]

            st.dataframe(
                filtered_queue[display_columns].head(1000),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                f"Showing {min(len(filtered_queue), 1000):,} "
                f"of {len(filtered_queue):,} matching transactions."
            )

            # CSV download
            csv_data = filtered_queue.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Fraud Investigation Queue",
                data=csv_data,
                file_name="fraud_operations_queue.csv",
                mime="text/csv",
            )

            st.divider()

            st.subheader(
                "Historical Label Comparison (Diagnostic)"
            )

            y_true = queue["is_fraud"].astype(int)

            y_pred = queue["Predicted Fraud"].astype(int)

            tp = int(
                ((y_true == 1) & (y_pred == 1)).sum()
            )

            fp = int(
                ((y_true == 0) & (y_pred == 1)).sum()
            )

            fn = int(
                ((y_true == 1) & (y_pred == 0)).sum()
            )

            tn = int(
                ((y_true == 0) & (y_pred == 0)).sum()
            )

            precision_value = (
                tp / (tp + fp) * 100
                if (tp + fp) else 0
            )

            recall_value = (
                tp / (tp + fn) * 100
                if (tp + fn) else 0
            )

            fpr_value = (
                fp / (fp + tn) * 100
                if (fp + tn) else 0
            )

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "True Positives",
                f"{tp:,}"
            )

            m2.metric(
                "False Positives",
                f"{fp:,}"
            )

            m3.metric(
                "False Negatives",
                f"{fn:,}"
            )

            m4.metric(
                "False Positive Rate",
                f"{fpr_value:.3f}%"
            )

            st.write(
                f"**Precision:** {precision_value:.2f}%"
            )

            st.write(
                f"**Recall:** {recall_value:.2f}%"
            )

            st.warning(
                "This diagnostic compares predictions with labels "
                "from the same dataset used to display the dashboard. "
                "It is not an unbiased held-out test result. "
                "Use a separate temporal test set for final evaluation."
            )


            st.divider()

            st.subheader("Operational Recommendations")

            st.caption(
                "Proposed fraud prevention policies. "
                "These are targets, not measured achievements."
            )

            st.markdown("### 1. Dynamic Step-Up Authentication")

            st.write(
                "Low Risk (PD < 0.15): Auto Approve"
            )

            st.write(
                "Medium Risk (0.15 to 0.65): "
                "Step-Up Authentication"
            )

            st.write(
                "Use 3D Secure 2.0 or biometric "
                "authentication for medium-risk transactions."
            )

            st.info(
                "Target: Authentication completion rate above 92%."
            )

            st.markdown(
                "### 2. High-Risk Merchant Monitoring"
            )

            st.write(
                "Monitor international Electronics and Crypto "
                "Exchange transaction frequency."
            )

            st.write(
                "Proposed limit: Maximum 2 attempts per hour "
                "for selected high-risk categories."
            )

            st.info(
                "Target: Chargeback rate below 0.9%."
            )

            st.markdown(
                "### 3. Fraud Investigation Strategy"
            )

            st.write(
                "Prioritize high-risk transactions for investigation "
                "and review medium-risk transactions through "
                "additional authentication."
            )

            st.markdown(
                "### 4. Continuous Model Monitoring"
            )

            st.write(
                "Monitor Precision, Recall, PR-AUC and FPR regularly. "
                "Retrain the model when fraud patterns change."
            )

            st.warning(
                "78% fraud prevention is an assignment target, "
                "not an achieved result in this project."
            )


        except Exception as exc:

            st.error(
                "Model prediction failed. Verify that the saved "
                "model, feature list, threshold and training "
                f"preprocessing match. Details: {exc}"
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Financial Fraud Detection System | Academic Project | 2026 | "
    "Local demonstration only — not connected to a live banking system."
)