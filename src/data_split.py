
import pandas as pd
import numpy as np


def temporal_split(df):
    df = df.copy()

    df["transaction_date"] = pd.to_datetime(
        df["transaction_date"]
    )

    df = df.sort_values(
        ["transaction_date", "transaction_time"]
    ).reset_index(drop=True)

    # Split using unique dates to prevent same-day overlap
    unique_dates = np.sort(
        df["transaction_date"].dropna().unique()
    )

    total_dates = len(unique_dates)

    train_end = int(total_dates * 0.70)
    val_end = int(total_dates * 0.85)

    train_dates = unique_dates[:train_end]
    val_dates = unique_dates[train_end:val_end]
    test_dates = unique_dates[val_end:]

    train_df = df[
        df["transaction_date"].isin(train_dates)
    ].copy()

    val_df = df[
        df["transaction_date"].isin(val_dates)
    ].copy()

    test_df = df[
        df["transaction_date"].isin(test_dates)
    ].copy()

    return train_df, val_df, test_df


if __name__ == "__main__":

    df = pd.read_csv("data/raw/bank_fraud.csv")

    train, validation, test = temporal_split(df)

    print("TEMPORAL SPLITTING COMPLETED!")

    print("\nDataset Distribution:")
    print("Total:", len(df))
    print("Train:", len(train))
    print("Validation:", len(validation))
    print("Test:", len(test))

    print("\nDate Ranges:")

    for name, data in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test)
    ]:
        print(
            name,
            data["transaction_date"].min(),
            "to",
            data["transaction_date"].max()
        )

    print("\nFraud Distribution:")

    for name, data in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test)
    ]:
        print(
            name,
            data["is_fraud"].mean()
        )
