import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import cross_val_score
import pandas as pd

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

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

print(f'Classes found: {class_names}')
print(f'Dataset shape: {X.shape}')

# Predict
print('Predicting...')
y_pred = clf.predict(X_scaled)
y_pred_proba = clf.predict_proba(X_scaled) if hasattr(clf, 'predict_proba') else None

# 1. Enhanced Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_encoded, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted Class', fontsize=12)
plt.ylabel('True Class', fontsize=12)
plt.title('Confusion Matrix - Defect Classification', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix_enhanced.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Classification Report as Heatmap
report = classification_report(y_encoded, y_pred, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
plt.figure(figsize=(10, 6))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='RdYlGn', fmt='.3f',
            cbar_kws={'label': 'Score'})
plt.title('Classification Report - Performance Metrics', fontsize=14, fontweight='bold')
plt.xlabel('Metrics', fontsize=12)
plt.ylabel('Classes', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'classification_report_enhanced.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. ROC Curves (if probability predictions available)
if y_pred_proba is not None:
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_encoded == i, y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Binary Classification Performance', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curves_enhanced.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 4. Precision-Recall Curves
if y_pred_proba is not None:
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_encoded == i, y_pred_proba[:, i])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, linewidth=2, label=f'{class_name} (AUC = {pr_auc:.3f})')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'precision_recall_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 5. Feature Importance (if available)
if hasattr(clf, 'feature_importances_'):
    plt.figure(figsize=(12, 8))
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=True)
    
    sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
    plt.title('Feature Importance - Defect Classification', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 6. Class Distribution
plt.figure(figsize=(10, 6))
class_counts = data['class'].value_counts()
sns.barplot(x=class_counts.index, y=class_counts.values, palette='Set2')
plt.title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
plt.xlabel('Defect Classes', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
for i, v in enumerate(class_counts.values):
    plt.text(i, v + 50, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'class_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# 7. Model Performance Summary
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_encoded, y_pred)
precision = precision_score(y_encoded, y_pred, average='weighted')
recall = recall_score(y_encoded, y_pred, average='weighted')
f1 = f1_score(y_encoded, y_pred, average='weighted')

metrics_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Score': [accuracy, precision, recall, f1]
}
metrics_df = pd.DataFrame(metrics_data)

plt.figure(figsize=(10, 6))
bars = sns.barplot(data=metrics_df, x='Metric', y='Score', palette='coolwarm')
plt.title('Model Performance Summary', fontsize=14, fontweight='bold')
plt.ylabel('Score', fontsize=12)
plt.ylim(0, 1)
for i, bar in enumerate(bars.patches):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{metrics_df.iloc[i]["Score"]:.3f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'performance_summary.png'), dpi=300, bbox_inches='tight')
plt.close()

print('\n=== GENERATED METRICS VISUALIZATIONS ===')
print(f'✓ Enhanced Confusion Matrix: {os.path.join(RESULTS_DIR, "confusion_matrix_enhanced.png")}')
print(f'✓ Classification Report: {os.path.join(RESULTS_DIR, "classification_report_enhanced.png")}')
if y_pred_proba is not None:
    print(f'✓ ROC Curves: {os.path.join(RESULTS_DIR, "roc_curves_enhanced.png")}')
    print(f'✓ Precision-Recall Curves: {os.path.join(RESULTS_DIR, "precision_recall_curves.png")}')
if hasattr(clf, 'feature_importances_'):
    print(f'✓ Feature Importance: {os.path.join(RESULTS_DIR, "feature_importance.png")}')
print(f'✓ Class Distribution: {os.path.join(RESULTS_DIR, "class_distribution.png")}')
print(f'✓ Performance Summary: {os.path.join(RESULTS_DIR, "performance_summary.png")}')

print(f'\n=== MODEL PERFORMANCE ===')
print(f'Accuracy: {accuracy:.3f}')
print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')
print(f'F1-Score: {f1:.3f}')
