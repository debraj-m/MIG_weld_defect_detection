import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class WeldDefectClassifier:
    def __init__(self, csv_file_path):
        """
        Initialize the classifier with data from CSV file
        Expected columns: class, file, x1, y1, x2, y2, area, aspect_ratio, solidity, 
                         compactness, circularity, fill_ratio, defects_count, 
                         mean_intensity, std_intensity, lbp_uniformity
        """
        import os
        # Always resolve path relative to project root
        if not os.path.isabs(csv_file_path):
            csv_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../defect_features_balanced.csv'))
        self.data = pd.read_csv(csv_file_path)
        self.feature_columns = ['area', 'aspect_ratio', 'solidity', 'compactness',
                               'circularity', 'fill_ratio', 'defects_count',
                               'mean_intensity', 'std_intensity', 'lbp_uniformity']
        self.target_column = 'class'  # Target column for defect classification
        self.bbox_columns = ['x1', 'y1', 'x2', 'y2']  # Bounding box coordinates
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_and_preprocess_data(self):
        """Load and preprocess the data"""
        print("Data shape:", self.data.shape)
        print("\nColumns in dataset:")
        print(list(self.data.columns))
        print("\nTarget distribution:")
        print(self.data[self.target_column].value_counts())
        
        # Show file distribution
        print(f"\nNumber of unique files: {self.data['file'].nunique()}")
        print("\nFiles with most defects:")
        print(self.data['file'].value_counts().head())
        
        # Calculate bounding box width and height for additional insights
        self.data['bbox_width'] = self.data['x2'] - self.data['x1']
        self.data['bbox_height'] = self.data['y2'] - self.data['y1']
        self.data['bbox_area'] = self.data['bbox_width'] * self.data['bbox_height']
        
        # Check for missing values
        print("\nMissing values:")
        missing_values = self.data.isnull().sum()
        print(missing_values[missing_values > 0])
        
        # Handle missing values if any
        if self.data.isnull().sum().sum() > 0:
            print("Dropping rows with missing values...")
            self.data = self.data.dropna()
        
        # Prepare features and target
        X = self.data[self.feature_columns]
        y = self.label_encoder.fit_transform(self.data[self.target_column])
        
        print(f"\nFinal dataset shape: {X.shape}")
        print("Class encoding:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            print(f"  {class_name}: {i}")
        
        return X, y
    
    def exploratory_data_analysis(self, X, y):
        """Perform EDA on the features"""
        plt.figure(figsize=(15, 12))
        
        # Feature distributions by class
        for i, feature in enumerate(self.feature_columns):
            plt.subplot(3, 4, i+1)
            for class_label in np.unique(y):
                class_name = self.label_encoder.inverse_transform([class_label])[0]
                plt.hist(X[y == class_label][feature], alpha=0.7, label=class_name, bins=20)
            plt.title(f'{feature} Distribution')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Correlation matrix
        plt.figure(figsize=(12, 8))
        correlation_matrix = X.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.show()
    
    def feature_importance_analysis(self, X, y):
        """Analyze feature importance using univariate selection"""
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X, y)
        
        feature_scores = pd.DataFrame({
            'feature': self.feature_columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_scores, x='score', y='feature')
        plt.title('Feature Importance (F-score)')
        plt.xlabel('F-score')
        plt.show()
        
        return feature_scores
    
    def train_models(self, X_train, y_train):
        """Train multiple classifiers"""
        # Define models with hyperparameters
        models_config = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'SVM': {
                'model': SVC(probability=True, random_state=42),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            }
        }
        
        print("Training models with hyperparameter tuning...")
        
        for name, config in models_config.items():
            print(f"\nTraining {name}...")
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=5, 
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            self.models[name] = {
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_
            }
            
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            print(f"Best parameters: {grid_search.best_params_}")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        results = {}
        
        plt.figure(figsize=(12, 8))
        
        for i, (name, model_info) in enumerate(self.models.items()):
            model = model_info['model']
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            # Classification report
            print(f"\n{name} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"AUC Score: {auc_score:.4f}")
            print("\nClassification Report:")
            target_names = self.label_encoder.classes_
            print(classification_report(y_test, y_pred, target_names=target_names))
            
            # Confusion Matrix
            plt.subplot(2, 2, i+1)
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=target_names, yticklabels=target_names)
            plt.title(f'{name} - Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
        
        # ROC Curves
        plt.figure(figsize=(10, 8))
        for name, result in results.items():
            fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
            plt.plot(fpr, tpr, label=f"{name} (AUC = {result['auc_score']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return results
    
    def get_feature_importance(self, model_name='Random Forest'):
        """Get feature importance from tree-based models"""
        if model_name in self.models:
            model = self.models[model_name]['model']
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                plt.figure(figsize=(10, 6))
                sns.barplot(data=importance_df, x='importance', y='feature')
                plt.title(f'{model_name} - Feature Importance')
                plt.xlabel('Importance')
                plt.show()
                
                return importance_df
        return None
    
    def additional_data_analysis(self):
        """Perform additional analysis using bounding box and file information"""
        print("\n" + "="*50)
        print("ADDITIONAL DATA ANALYSIS")
        print("="*50)
        
        # Analysis by file
        plt.figure(figsize=(15, 10))
        
        # Defect count per file
        plt.subplot(2, 3, 1)
        file_defect_counts = self.data.groupby('file')[self.target_column].count().sort_values(ascending=False)
        file_defect_counts.head(10).plot(kind='bar')
        plt.title('Top 10 Files by Defect Count')
        plt.xlabel('File')
        plt.ylabel('Defect Count')
        plt.xticks(rotation=45)
        
        # Defect type distribution by file (for files with >5 defects)
        plt.subplot(2, 3, 2)
        high_defect_files = file_defect_counts[file_defect_counts > 5].index
        if len(high_defect_files) > 0:
            subset_data = self.data[self.data['file'].isin(high_defect_files)]
            defect_by_file = pd.crosstab(subset_data['file'], subset_data[self.target_column])
            defect_by_file.plot(kind='bar', stacked=True)
            plt.title('Defect Types in High-Defect Files')
            plt.xlabel('File')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.legend(title='Defect Type')
        
        # Bounding box size analysis
        plt.subplot(2, 3, 3)
        for class_label in self.data[self.target_column].unique():
            class_data = self.data[self.data[self.target_column] == class_label]
            plt.scatter(class_data['bbox_width'], class_data['bbox_height'], 
                       alpha=0.6, label=class_label, s=30)
        plt.xlabel('Bounding Box Width')
        plt.ylabel('Bounding Box Height')
        plt.title('Bounding Box Dimensions by Defect Type')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Area vs bounding box area comparison
        plt.subplot(2, 3, 4)
        plt.scatter(self.data['bbox_area'], self.data['area'], alpha=0.6, s=30)
        plt.xlabel('Bounding Box Area')
        plt.ylabel('Actual Defect Area')
        plt.title('Defect Area vs Bounding Box Area')
        plt.grid(True, alpha=0.3)
        
        # Position analysis (x1, y1 distribution)
        plt.subplot(2, 3, 5)
        for class_label in self.data[self.target_column].unique():
            class_data = self.data[self.data[self.target_column] == class_label]
            plt.scatter(class_data['x1'], class_data['y1'], 
                       alpha=0.6, label=class_label, s=30)
        plt.xlabel('X1 Position')
        plt.ylabel('Y1 Position')
        plt.title('Defect Positions by Type')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Fill ratio analysis
        plt.subplot(2, 3, 6)
        fill_ratio_analysis = self.data.groupby(self.target_column)['fill_ratio'].agg(['mean', 'std'])
        fill_ratio_analysis['mean'].plot(kind='bar', yerr=fill_ratio_analysis['std'], capsize=4)
        plt.title('Average Fill Ratio by Defect Type')
        plt.xlabel('Defect Type')
        plt.ylabel('Fill Ratio')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Statistical summary by defect type
        print("\nStatistical Summary by Defect Type:")
        print("="*40)
        for feature in self.feature_columns:
            print(f"\n{feature.upper()}:")
            summary = self.data.groupby(self.target_column)[feature].agg(['count', 'mean', 'std', 'min', 'max'])
            print(summary)
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("=" * 60)
        print("WELD DEFECT CLASSIFICATION ANALYSIS")
        print("=" * 60)

        # Load and preprocess data
        X, y = self.load_and_preprocess_data()

        # Additional analysis using bounding box and file info
        self.additional_data_analysis()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # EDA
        print("\n" + "="*40)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*40)
        self.exploratory_data_analysis(X, y)

        # Feature importance analysis
        print("\n" + "="*40)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*40)
        feature_scores = self.feature_importance_analysis(X, y)
        print("\nTop 5 most important features:")
        print(feature_scores.head())

        # Train models
        print("\n" + "="*40)
        print("MODEL TRAINING")
        print("="*40)
        self.train_models(X_train_scaled, y_train)

        # Evaluate models
        print("\n" + "="*40)
        print("MODEL EVALUATION")
        print("="*40)
        results = self.evaluate_models(X_test_scaled, y_test)

        # Feature importance for best model
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nBest performing model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")

        # Save the best model for deployment
        import joblib
        import os
        model_obj = self.models[best_model[0]]['model']
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/defect_classifier.joblib'))
        joblib.dump(model_obj, model_path)
        print(f"Best model saved as '{model_path}'.")

        # Show feature importance for tree-based models
        if best_model[0] in ['Random Forest', 'Gradient Boosting']:
            print("\n" + "="*40)
            print("FEATURE IMPORTANCE (BEST MODEL)")
            print("="*40)
            importance_df = self.get_feature_importance(best_model[0])
            if importance_df is not None:
                print(importance_df)

        return results

# Usage example:
if __name__ == "__main__":
    import os
    # Always use the correct relative path
    classifier = WeldDefectClassifier(os.path.abspath(os.path.join(os.path.dirname(__file__), '../defect_features_balanced.csv')))
    results = classifier.run_complete_analysis()