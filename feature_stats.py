import pandas as pd
import numpy as np

class GeometricFeatureStats:
    """Loads and provides feature statistics for validation."""
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.stats = {}
        for feature in ['area', 'aspect_ratio', 'solidity', 'compactness']:
            self.stats[feature] = {
                'min': self.df[feature].quantile(0.05),
                'max': self.df[feature].quantile(0.95),
                'mean': self.df[feature].mean(),
                'std': self.df[feature].std()
            }
    def is_within_range(self, feature, value):
        return self.stats[feature]['min'] <= value <= self.stats[feature]['max']
    def get_stats(self, feature):
        return self.stats[feature]
