import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv(r"C:\Users\debra\Downloads\Updated data - Sample 1.csv")  # Columns: Time, Voltage, Current

# Preview
print(df.head())

# Feature engineering
def extract_signal_features(df):
    features = {}
    for col in ['Voltage', 'Current']:
        features[f'{col}_mean'] = df[col].mean()
        features[f'{col}_std'] = df[col].std()
        features[f'{col}_min'] = df[col].min()
        features[f'{col}_max'] = df[col].max()
        features[f'{col}_range'] = df[col].max() - df[col].min()
        features[f'{col}_skew'] = df[col].skew()
        features[f'{col}_kurt'] = df[col].kurt()
    return pd.DataFrame([features])

signal_features = extract_signal_features(df)
print(signal_features)
