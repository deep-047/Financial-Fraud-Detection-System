
import pandas as pd
import numpy as np


def load_data(file_path):
    """Load transaction dataset."""
    df = pd.read_csv(file_path)

    df["transaction_date"] = pd.to_datetime(
        df["transaction_date"],
        errors="coerce"
    )

    df["transaction_amount"] = pd.to_numeric(
        df["transaction_amount"],
        errors="coerce"
    )

    return df


def clean_data(df):
    """Clean missing and duplicate records."""
    df = df.drop_duplicates(subset=["transaction_id"]).copy()

    df = df.dropna(
        subset=[
            "transaction_date",
            "transaction_amount",
            "is_fraud"
        ]
    )

    df["is_fraud"] = df["is_fraud"].astype(int)

    return df


if __name__ == "__main__":
    df = load_data("data/raw/bank_fraud.csv")

    print("Original Shape:", df.shape)

    df = clean_data(df)

    print("Cleaned Shape:", df.shape)
    print("Missing Values:", df.isnull().sum().sum())
    print("Fraud Count:", df["is_fraud"].sum())
    print("Preprocessing Completed!")
