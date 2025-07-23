"""
Comprehensive Classifier Metrics Analysis
=========================================
This script generates detailed performance metrics for the weld defect classifier,
including accuracy scores, confusion matrices, prediction accuracy, correlation matrices,
and advanced visualization for porosity vs blowhole classification.

Author: Debraj Mukherjee
Project: MIG Weld Defect Detection
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, 
    precision_recall_curve, accuracy_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, cohen_kappa_score
)
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, '../results'))
models_dir = os.path.abspath(os.path.join(BASE_DIR, '../models'))
features_csv = os.path.abspath(os.path.join(BASE_DIR, '../defect_features_balanced.csv'))

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data_and_models():
    """Load dataset and trained models"""
    print("🔄 Loading data and trained models...")
    
    # Load dataset
    data = pd.read_csv(features_csv)
    feature_columns = [
        'area', 'aspect_ratio', 'solidity', 'compactness',
        'circularity', 'fill_ratio', 'defects_count',
        'mean_intensity', 'std_intensity', 'lbp_uniformity'
    ]
    
    X = data[feature_columns]
    y = data['class']
    
    # Load trained models
    scaler = joblib.load(os.path.join(models_dir, 'defect_scaler.joblib'))
    label_encoder = joblib.load(os.path.join(models_dir, 'defect_label_encoder.joblib'))
    classifier = joblib.load(os.path.join(models_dir, 'defect_classifier.joblib'))
    
    # Preprocess data
    X_scaled = scaler.transform(X)
    y_encoded = label_encoder.transform(y)
    
    print(f"✅ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"✅ Classes: {list(label_encoder.classes_)}")
    
    return X, X_scaled, y, y_encoded, classifier, scaler, label_encoder, feature_columns

def calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba, class_names):
    """Calculate comprehensive classification metrics"""
    print("📊 Calculating comprehensive metrics...")
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    
    # Advanced metrics
    metrics['matthews_corr'] = matthews_corrcoef(y_true, y_pred)
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    for i, class_name in enumerate(class_names):
        metrics[f'precision_{class_name}'] = precision_per_class[i]
        metrics[f'recall_{class_name}'] = recall_per_class[i]
        metrics[f'f1_{class_name}'] = f1_per_class[i]
    
    # AUC scores (for binary or multiclass)
    if y_pred_proba is not None:
        if len(class_names) == 2:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            metrics['auc_roc'] = auc(fpr, tpr)
        else:
            # Multi-class AUC (one-vs-rest)
            for i, class_name in enumerate(class_names):
                fpr, tpr, _ = roc_curve(y_true == i, y_pred_proba[:, i])
                metrics[f'auc_{class_name}'] = auc(fpr, tpr)
    
    return metrics

def plot_enhanced_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Create enhanced confusion matrix visualization"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Absolute counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    ax1.set_xlabel('Predicted Class', fontsize=12)
    ax1.set_ylabel('True Class', fontsize=12)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    
    # Percentages
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Greens', ax=ax2,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage (%)'})
    ax2.set_xlabel('Predicted Class', fontsize=12)
    ax2.set_ylabel('True Class', fontsize=12)
    ax2.set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm, cm_percent

def plot_feature_correlation_matrix(X, feature_columns, save_path):
    """Create detailed feature correlation analysis"""
    correlation_matrix = X.corr()
    
    # Mask for upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r',
                center=0, fmt='.3f', linewidths=0.5,
                cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Feature Correlation Matrix\n(Porosity vs Blowhole Classification)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return correlation_matrix

def plot_learning_curves(classifier, X, y, save_path):
    """Generate learning curves to analyze model performance"""
    train_sizes, train_scores, val_scores = learning_curve(
        classifier, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Accuracy')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.2, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Accuracy')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.2, color='red')
    
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.title('Learning Curves - Classifier Performance', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance_analysis(classifier, feature_columns, X, y, save_path):
    """Comprehensive feature importance analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Model-based feature importance (if available)
    if hasattr(classifier, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': classifier.feature_importances_
        }).sort_values('importance', ascending=True)
        
        axes[0, 0].barh(importance_df['feature'], importance_df['importance'], color='skyblue')
        axes[0, 0].set_title('Model Feature Importance', fontweight='bold')
        axes[0, 0].set_xlabel('Importance Score')
    
    # 2. Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({
        'feature': feature_columns,
        'mutual_info': mi_scores
    }).sort_values('mutual_info', ascending=True)
    
    axes[0, 1].barh(mi_df['feature'], mi_df['mutual_info'], color='lightcoral')
    axes[0, 1].set_title('Mutual Information Scores', fontweight='bold')
    axes[0, 1].set_xlabel('Mutual Information')
    
    # 3. Feature variance
    feature_var = X.var().sort_values(ascending=True)
    axes[1, 0].barh(feature_var.index, feature_var.values, color='lightgreen')
    axes[1, 0].set_title('Feature Variance', fontweight='bold')
    axes[1, 0].set_xlabel('Variance')
    
    # 4. Feature correlation with target
    correlation_with_target = X.corrwith(pd.Series(y.map({'porosity': 0, 'blowhole': 1}) 
                                                  if hasattr(y, 'map') else y)).abs().sort_values(ascending=True)
    axes[1, 1].barh(correlation_with_target.index, correlation_with_target.values, color='gold')
    axes[1, 1].set_title('Absolute Correlation with Target', fontweight='bold')
    axes[1, 1].set_xlabel('|Correlation|')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_comprehensive_report(metrics, class_names, save_path):
    """Generate comprehensive text report"""
    report_lines = [
        "="*80,
        "🔬 COMPREHENSIVE CLASSIFIER PERFORMANCE REPORT",
        "="*80,
        f"📊 Dataset Overview:",
        f"   • Total Samples: {len(metrics)}",
        f"   • Classes: {', '.join(class_names)}",
        "",
        "🎯 Overall Performance Metrics:",
        f"   • Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)",
        f"   • Macro Precision: {metrics['precision_macro']:.4f}",
        f"   • Macro Recall: {metrics['recall_macro']:.4f}",
        f"   • Macro F1-Score: {metrics['f1_macro']:.4f}",
        f"   • Matthews Correlation: {metrics['matthews_corr']:.4f}",
        f"   • Cohen's Kappa: {metrics['cohen_kappa']:.4f}",
        "",
        "📈 Per-Class Performance:",
    ]
    
    for class_name in class_names:
        report_lines.extend([
            f"   🔸 {class_name.title()}:",
            f"      - Precision: {metrics[f'precision_{class_name}']:.4f}",
            f"      - Recall: {metrics[f'recall_{class_name}']:.4f}",
            f"      - F1-Score: {metrics[f'f1_{class_name}']:.4f}",
        ])
        if f'auc_{class_name}' in metrics:
            report_lines.append(f"      - AUC-ROC: {metrics[f'auc_{class_name}']:.4f}")
        report_lines.append("")
    
    report_lines.extend([
        "🔍 Model Interpretation:",
        "   • High precision indicates low false positive rate",
        "   • High recall indicates low false negative rate", 
        "   • Matthews correlation measures overall quality",
        "   • Cohen's kappa accounts for chance agreement",
        "",
        "="*80,
    ])
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print("📄 Comprehensive report saved!")

def main():
    """Main execution function"""
    print("🚀 Starting Comprehensive Classifier Metrics Analysis...")
    print("="*60)
    
    # Load data and models
    X, X_scaled, y, y_encoded, classifier, scaler, label_encoder, feature_columns = load_data_and_models()
    class_names = label_encoder.classes_
    
    # Generate predictions
    print("🔮 Generating predictions...")
    y_pred = classifier.predict(X_scaled)
    y_pred_proba = classifier.predict_proba(X_scaled) if hasattr(classifier, 'predict_proba') else None
    
    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(y_encoded, y_pred, y_pred_proba, class_names)
    
    print("\n📊 Generated Visualizations:")
    
    # 1. Enhanced Confusion Matrix
    cm, cm_percent = plot_enhanced_confusion_matrix(
        y_encoded, y_pred, class_names,
        os.path.join(RESULTS_DIR, 'comprehensive_confusion_matrix.png')
    )
    print("   ✅ Enhanced confusion matrix")
    
    # 2. Feature Correlation Matrix
    correlation_matrix = plot_feature_correlation_matrix(
        pd.DataFrame(X, columns=feature_columns), feature_columns,
        os.path.join(RESULTS_DIR, 'feature_correlation_comprehensive.png')
    )
    print("   ✅ Feature correlation matrix")
    
    # 3. Learning Curves
    plot_learning_curves(
        classifier, X_scaled, y_encoded,
        os.path.join(RESULTS_DIR, 'learning_curves_analysis.png')
    )
    print("   ✅ Learning curves analysis")
    
    # 4. Feature Importance Analysis
    plot_feature_importance_analysis(
        classifier, feature_columns, pd.DataFrame(X, columns=feature_columns), y,
        os.path.join(RESULTS_DIR, 'comprehensive_feature_importance.png')
    )
    print("   ✅ Comprehensive feature importance")
    
    # 5. Generate comprehensive report
    generate_comprehensive_report(
        metrics, class_names,
        os.path.join(RESULTS_DIR, 'comprehensive_classifier_report.txt')
    )
    
    # Print summary
    print("\n" + "="*60)
    print("🎉 ANALYSIS COMPLETE!")
    print("="*60)
    print(f"📁 Results saved in: {RESULTS_DIR}")
    print("\n📊 Key Performance Summary:")
    print(f"   🎯 Overall Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"   📈 Macro F1-Score: {metrics['f1_macro']:.4f}")
    print(f"   🔗 Matthews Correlation: {metrics['matthews_corr']:.4f}")
    print("\n💡 Next Steps:")
    print("   • Review correlation matrix for feature engineering opportunities")
    print("   • Analyze learning curves for potential overfitting/underfitting")
    print("   • Use feature importance for model optimization")
    
if __name__ == "__main__":
    main()
