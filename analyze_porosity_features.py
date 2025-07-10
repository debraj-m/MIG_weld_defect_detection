import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
df = pd.read_csv('porosity_geometric_features.csv')

# Show basic statistics
desc = df.describe()
print(desc)

# Plot histograms for each geometric feature
features = ['area', 'aspect_ratio', 'solidity', 'compactness']
df[features].hist(bins=50, figsize=(12, 8))
plt.tight_layout()
plt.show()

# Pairplot to visualize relationships between features
sns.pairplot(df[features])
plt.show()

# Boxplot by class (if multiple classes)
if 'class' in df.columns:
    for feature in features:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x='class', y=feature, data=df)
        plt.title(f'{feature} by class')
        plt.show()
