"""
Data Cleaning and Preprocessing for Heart Disease Dataset
"""
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('heart_disease.csv')

print("=" * 60)
print("INITIAL DATA OVERVIEW")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nBasic statistics:\n{df.describe()}")

# Check for missing values
print("\n" + "=" * 60)
print("HANDLING MISSING VALUES")
print("=" * 60)
missing_counts = df.isnull().sum()
print(f"Missing values per column:\n{missing_counts[missing_counts > 0]}")

# Handle missing values - drop rows with missing values (small dataset)
df_clean = df.dropna()
print(f"\nRows after dropping missing values: {df_clean.shape[0]} (removed {df.shape[0] - df_clean.shape[0]})")

# Convert target to binary (0 = no disease, 1-4 = disease)
print("\n" + "=" * 60)
print("TARGET VARIABLE TRANSFORMATION")
print("=" * 60)
print(f"Original target distribution:\n{df_clean['target'].value_counts().sort_index()}")
df_clean['target'] = (df_clean['target'] > 0).astype(int)
print(f"\nBinary target distribution:\n{df_clean['target'].value_counts()}")

# Check for outliers using IQR method
print("\n" + "=" * 60)
print("OUTLIER DETECTION")
print("=" * 60)
numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
for col in numeric_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df_clean[(df_clean[col] < lower) | (df_clean[col] > upper)]
    if len(outliers) > 0:
        print(f"{col}: {len(outliers)} outliers detected (range: {lower:.1f} - {upper:.1f})")

# Save cleaned data
df_clean.to_csv('heart_disease_clean.csv', index=False)
print("\n" + "=" * 60)
print(f"Cleaned data saved: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
print("=" * 60)
