import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import pandas as pd

# Ensure results directory exists
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results'))
os.makedirs(RESULTS_DIR, exist_ok=True)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.abspath(os.path.join(BASE_DIR, '../models'))
features_csv = os.path.abspath(os.path.join(BASE_DIR, '../defect_features_balanced.csv'))

# Load data and models
print('Loading data and models...')
data = pd.read_csv(features_csv)
feature_columns = [
    'area', 'aspect_ratio', 'solidity', 'compactness',
    'circularity', 'fill_ratio', 'defects_count',
    'mean_intensity', 'std_intensity', 'lbp_uniformity'
]
X = data[feature_columns]
y = data['class']

scaler = joblib.load(os.path.join(models_dir, 'defect_scaler.joblib'))
label_encoder = joblib.load(os.path.join(models_dir, 'defect_label_encoder.joblib'))
clf = joblib.load(os.path.join(models_dir, 'defect_classifier.joblib'))

# Preprocess
X_scaled = scaler.transform(X)
y_encoded = label_encoder.transform(y)
class_names = label_encoder.classes_

# Predict
print('Predicting...')
y_pred = clf.predict(X_scaled)

# Confusion Matrix
cm = confusion_matrix(y_encoded, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'classifier_confusion_matrix.png'))
plt.close()

# Classification Report (as image)
report = classification_report(y_encoded, y_pred, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
plt.figure(figsize=(8, 4))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('Classification Report')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'classifier_classification_report.png'))
plt.close()

# (Optional) ROC Curves for each class (for multi-class)
if hasattr(clf, 'predict_proba'):
    y_score = clf.predict_proba(X_scaled)
    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_encoded == i, y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'classifier_roc_curves.png'))
    plt.close()

print('Saved:')
print(f'  {os.path.join(RESULTS_DIR, "classifier_confusion_matrix.png")}')
print(f'  {os.path.join(RESULTS_DIR, "classifier_classification_report.png")}')
if hasattr(clf, 'predict_proba'):
    print(f'  {os.path.join(RESULTS_DIR, "classifier_roc_curves.png")}')
