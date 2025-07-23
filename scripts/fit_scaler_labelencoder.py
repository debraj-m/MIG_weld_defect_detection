import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.abspath(os.path.join(BASE_DIR, '../defect_features_balanced.csv'))
data = pd.read_csv(csv_file)

# Feature columns (update as needed)
feature_columns = [
    'area', 'aspect_ratio', 'solidity', 'compactness',
    'circularity', 'fill_ratio', 'defects_count',
    'mean_intensity', 'std_intensity', 'lbp_uniformity'
]
target_column = 'class'

# Prepare features and target
df_features = data[feature_columns]
df_target = data[target_column]

# Fit scaler and label encoder
scaler = StandardScaler()
scaler.fit(df_features)

label_encoder = LabelEncoder()
label_encoder.fit(df_target)


models_dir = os.path.abspath(os.path.join(BASE_DIR, '../models'))
os.makedirs(models_dir, exist_ok=True)
joblib.dump(scaler, os.path.join(models_dir, 'defect_scaler.joblib'))
joblib.dump(label_encoder, os.path.join(models_dir, 'defect_label_encoder.joblib'))

print('Scaler and label encoder have been fitted and saved in the models directory.')
