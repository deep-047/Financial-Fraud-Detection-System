
import pandas as pd
import numpy as np


def create_features(df):
    df = df.copy()

    # Sort transactions chronologically
    df = df.sort_values(
        ["customer_id", "transaction_date", "transaction_time"]
    )

    # Customer historical mean spend (previous transactions only)
    df["Hist_Mean_Spend"] = (
        df.groupby("customer_id")["transaction_amount"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # Use a neutral fallback for customers with no prior transaction
    df["Hist_Mean_Spend"] = df["Hist_Mean_Spend"].fillna(0)

    # Required assignment feature
    df["Spend_Ratio"] = (
        df["transaction_amount"] /
        (df["Hist_Mean_Spend"] + 1)
    )

    # Date-based features
    df["transaction_date"] = pd.to_datetime(
        df["transaction_date"]
    )

    df["transaction_month"] = (
        df["transaction_date"].dt.month
    )

    df["transaction_day"] = (
        df["transaction_date"].dt.day
    )

    # High-risk indicators
    df["high_failed_attempts"] = (
        df["failed_attempts"] >= 2
    ).astype(int)

    df["high_distance"] = (
        df["distance_from_home_km"] > 100
    ).astype(int)

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/raw/bank_fraud.csv")

    df = create_features(df)

    print("FEATURE ENGINEERING COMPLETED!")
    print("\nNew Features:")
    print([
        "Hist_Mean_Spend",
        "Spend_Ratio",
        "transaction_month",
        "transaction_day",
        "high_failed_attempts",
        "high_distance"
    ])

    print("\nSample Feature Values:")
    print(
        df[
            ["transaction_amount", "Hist_Mean_Spend", "Spend_Ratio"]
        ].head(10)
    )
