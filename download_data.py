"""
Download UCI Heart Disease dataset
"""
import pandas as pd
import os

# Download from UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Column names based on UCI documentation
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# Download and save
df = pd.read_csv(url, names=column_names, na_values='?')
df.to_csv('heart_disease.csv', index=False)
print(f"Dataset downloaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nFirst few rows:")
print(df.head())
