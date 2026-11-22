
import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/bank_fraud.csv")

print("\nDATASET LOADED SUCCESSFULLY!")
print("Dataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTOTAL ROWS AND COLUMNS:", df.shape)

print("\nALL COLUMNS:")
for col in df.columns:
    print(col)


print("\nFRAUD DISTRIBUTION:")
print(df["is_fraud"].value_counts())

print("\nFRAUD PERCENTAGE:")
print(df["is_fraud"].value_counts(normalize=True) * 100)

print("\nDATE RANGE:")
print("Start:", df["transaction_date"].min())
print("End:", df["transaction_date"].max())

print("\nMERCHANT CATEGORIES:")
print(df["merchant_category"].unique())
