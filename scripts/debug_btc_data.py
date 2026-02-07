import pandas as pd
import numpy as np

# Load BTC data
print("Loading BTC data...")
df = pd.read_csv('data/btc_global_insights.csv')

print(f"\nTotal rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

print("\n=== LAST 10 ROWS ===")
print(df.tail(10))

print("\n=== DATA TYPES ===")
print(df.dtypes)

print("\n=== NULL VALUES ===")
print(df.isnull().sum())

print("\n=== LAST 5 ROWS - RAW VALUES ===")
for idx, row in df.tail(5).iterrows():
    print(f"\nRow {idx}:")
    for col in df.columns:
        val = row[col]
        print(f"  {col}: {val} (type: {type(val).__name__}, isnan: {pd.isna(val)})")
