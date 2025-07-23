import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the features CSV
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.abspath(os.path.join(BASE_DIR, '../defect_features_balanced.csv'))
df = pd.read_csv(csv_file)

# Show basic info
print(df.head())
print(df['class'].value_counts())

# List of features to analyze
features = [
    'area','aspect_ratio','solidity','compactness','circularity',
    'fill_ratio','defects_count','mean_intensity','std_intensity','lbp_uniformity'
]

# Plot distributions for each feature by class
for feat in features:
    plt.figure(figsize=(7,4))
    sns.histplot(data=df, x=feat, hue='class', kde=True, stat='density', common_norm=False, bins=30)
    plt.title(f'Distribution of {feat} by class')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Correlation heatmap
plt.figure(figsize=(10,7))
sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Boxplots for each feature
for feat in features:
    plt.figure(figsize=(7,4))
    sns.boxplot(data=df, x='class', y=feat)
    plt.title(f'Boxplot of {feat} by class')
    plt.tight_layout()
    plt.show()
