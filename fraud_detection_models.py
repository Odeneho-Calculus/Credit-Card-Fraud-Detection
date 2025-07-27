"""
Professional Credit Card Fraud Detection Models Implementation
Implements Logistic Regression, Random Forest, XGBoost, and Naive Bayes
with comprehensive evaluation and comparison framework
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Suppress Qt warnings

# Core ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, accuracy_score,
    precision_score, recall_score
)

# Model Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class FraudDetectionPipeline:
    """
    Enterprise-grade fraud detection pipeline with multiple ML algorithms
    """

    def __init__(self, dataset_path: str):
        """
        Initialize the fraud detection pipeline

        Args:
            dataset_path: Path to the credit card dataset
        """
        self.dataset_path = Path(dataset_path)
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.models = {}
        self.model_results = {}

    def load_and_explore_data(self):
        """
        Load dataset and perform comprehensive exploratory data analysis
        """
        print("🔄 Loading and exploring dataset...")

        # Load dataset
        self.df = pd.read_csv(self.dataset_path)

        print(f"📊 Dataset Shape: {self.df.shape}")
        print(f"🎯 Features: {self.df.shape[1] - 1}")
        print(f"📈 Samples: {self.df.shape[0]:,}")

        # Class distribution
        class_dist = self.df['Class'].value_counts()
        fraud_rate = self.df['Class'].mean()

        print(f"\n🏷️ Class Distribution:")
        print(f"   Legitimate (0): {class_dist[0]:,} ({(1-fraud_rate)*100:.2f}%)")
        print(f"   Fraudulent (1): {class_dist[1]:,} ({fraud_rate*100:.2f}%)")

        # Missing values check
        missing_values = self.df.isnull().sum().sum()
        print(f"\n🔍 Missing Values: {missing_values}")

        # Feature statistics
        print(f"\n💰 Amount Statistics:")
        print(self.df['Amount'].describe())

        return self.df

    def visualize_data_distribution(self):
        """
        Create comprehensive visualizations of the dataset
        """
        print("📊 Creating data visualizations...")

        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Credit Card Fraud Detection - Data Analysis', fontsize=16, fontweight='bold')

        # 1. Class Distribution
        class_counts = self.df['Class'].value_counts()
        axes[0, 0].pie(class_counts.values, labels=['Legitimate', 'Fraudulent'],
                       autopct='%1.1f%%', startangle=90, colors=['#2E8B57', '#DC143C'])
        axes[0, 0].set_title('Class Distribution', fontweight='bold')

        # 2. Amount Distribution by Class
        legitimate = self.df[self.df['Class'] == 0]['Amount']
        fraudulent = self.df[self.df['Class'] == 1]['Amount']

        axes[0, 1].hist(legitimate, bins=50, alpha=0.7, label='Legitimate', color='#2E8B57', density=True)
        axes[0, 1].hist(fraudulent, bins=50, alpha=0.7, label='Fraudulent', color='#DC143C', density=True)
        axes[0, 1].set_xlabel('Transaction Amount')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Amount Distribution by Class', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].set_xlim(0, 25000)

        # 3. Correlation Heatmap (sample of features)
        feature_cols = ['V1', 'V2', 'V3', 'V4', 'V5', 'Amount', 'Class']
        corr_matrix = self.df[feature_cols].corr()

        im = axes[1, 0].imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        axes[1, 0].set_xticks(range(len(feature_cols)))
        axes[1, 0].set_yticks(range(len(feature_cols)))
        axes[1, 0].set_xticklabels(feature_cols, rotation=45)
        axes[1, 0].set_yticklabels(feature_cols)
        axes[1, 0].set_title('Feature Correlation Matrix', fontweight='bold')

        # Add correlation values
        for i in range(len(feature_cols)):
            for j in range(len(feature_cols)):
                axes[1, 0].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha='center', va='center', fontsize=8)

        # 4. Feature Importance Preview (V1-V5)
        feature_means_legit = self.df[self.df['Class'] == 0][['V1', 'V2', 'V3', 'V4', 'V5']].mean()
        feature_means_fraud = self.df[self.df['Class'] == 1][['V1', 'V2', 'V3', 'V4', 'V5']].mean()

        x_pos = np.arange(len(feature_means_legit))
        width = 0.35

        axes[1, 1].bar(x_pos - width/2, feature_means_legit, width,
                       label='Legitimate', color='#2E8B57', alpha=0.8)
        axes[1, 1].bar(x_pos + width/2, feature_means_fraud, width,
                       label='Fraudulent', color='#DC143C', alpha=0.8)

        axes[1, 1].set_xlabel('Features')
        axes[1, 1].set_ylabel('Mean Value')
        axes[1, 1].set_title('Feature Means by Class (V1-V5)', fontweight='bold')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(['V1', 'V2', 'V3', 'V4', 'V5'])
        axes[1, 1].legend()

        plt.tight_layout()

        # Save the plot instead of showing it
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        plt.savefig(plots_dir / "data_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory

        print(f"✅ Data visualization completed and saved to: {plots_dir / 'data_analysis.png'}")

    def prepare_data(self, test_size=0.2, random_state=42):
        """
        Prepare data for machine learning with proper scaling and splitting
        """
        print("🔧 Preparing data for machine learning...")

        # Separate features and target
        X = self.df.drop(['Class', 'id'], axis=1)  # Remove ID and target
        y = self.df['Class']

        print(f"📊 Feature matrix shape: {X.shape}")
        print(f"🎯 Target vector shape: {y.shape}")

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"🔄 Training set: {self.X_train.shape[0]:,} samples")
        print(f"🔄 Testing set: {self.X_test.shape[0]:,} samples")

        # Scale the features using RobustScaler (better for outliers)
        self.scaler = RobustScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print("✅ Data preparation completed")

        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test

    def train_logistic_regression(self):
        """
        Train and optimize Logistic Regression model
        """
        print("🔄 Training Logistic Regression...")

        # Initialize with optimized parameters
        lr_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0,  # Regularization strength
            class_weight='balanced',  # Handle class imbalance
            solver='liblinear'
        )

        # Train the model
        lr_model.fit(self.X_train_scaled, self.y_train)

        # Store the model
        self.models['Logistic Regression'] = lr_model

        print("✅ Logistic Regression training completed")
        return lr_model

    def train_random_forest(self):
        """
        Train and optimize Random Forest model
        """
        print("🔄 Training Random Forest...")

        # Initialize with optimized parameters
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1  # Use all available cores
        )

        # Train the model
        rf_model.fit(self.X_train_scaled, self.y_train)

        # Store the model
        self.models['Random Forest'] = rf_model

        print("✅ Random Forest training completed")
        return rf_model

    def train_xgboost(self):
        """
        Train and optimize XGBoost model
        """
        print("🔄 Training XGBoost...")

        # Calculate scale_pos_weight for class imbalance
        scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()

        # Initialize with optimized parameters
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )

        # Train the model
        xgb_model.fit(self.X_train_scaled, self.y_train)

        # Store the model
        self.models['XGBoost'] = xgb_model

        print("✅ XGBoost training completed")
        return xgb_model

    def train_naive_bayes(self):
        """
        Train Gaussian Naive Bayes model
        """
        print("🔄 Training Naive Bayes...")

        # Initialize Gaussian Naive Bayes
        nb_model = GaussianNB()

        # Train the model
        nb_model.fit(self.X_train_scaled, self.y_train)

        # Store the model
        self.models['Naive Bayes'] = nb_model

        print("✅ Naive Bayes training completed")
        return nb_model

    def evaluate_model(self, model_name, model):
        """
        Comprehensive model evaluation with multiple metrics
        """
        print(f"📊 Evaluating {model_name}...")

        # Make predictions
        y_pred = model.predict(self.X_test_scaled)
        y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        auc_score = roc_auc_score(self.y_test, y_pred_proba)

        # Store results
        self.model_results[model_name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred)
        }

        # Print results
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   AUC Score: {auc_score:.4f}")

        return self.model_results[model_name]

    def train_all_models(self):
        """
        Train all four models and evaluate them
        """
        print("🚀 Training all models...")
        print("=" * 50)

        # Train models
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost()
        self.train_naive_bayes()

        print("\n📊 Evaluating all models...")
        print("=" * 50)

        # Evaluate all models
        for model_name, model in self.models.items():
            self.evaluate_model(model_name, model)
            print()

        print("✅ All models trained and evaluated!")

    def create_comparison_visualizations(self):
        """
        Create comprehensive comparison visualizations
        """
        print("📊 Creating model comparison visualizations...")

        # Prepare data for visualization
        model_names = list(self.model_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']

        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

        # 1. Metrics Comparison Bar Chart
        metric_data = {metric: [self.model_results[model][metric] for model in model_names]
                      for metric in metrics}

        x_pos = np.arange(len(model_names))
        width = 0.15
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

        for i, (metric, values) in enumerate(metric_data.items()):
            axes[0, 0].bar(x_pos + i * width, values, width,
                          label=metric.replace('_', ' ').title(), color=colors[i], alpha=0.8)

        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Performance Metrics Comparison', fontweight='bold')
        axes[0, 0].set_xticks(x_pos + width * 2)
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1.1)

        # 2. ROC Curves
        for model_name in model_names:
            y_pred_proba = self.model_results[model_name]['y_pred_proba']
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc_score = self.model_results[model_name]['auc_score']
            axes[0, 1].plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)

        axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curves Comparison', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Precision-Recall Curves
        for model_name in model_names:
            y_pred_proba = self.model_results[model_name]['y_pred_proba']
            precision_curve, recall_curve, _ = precision_recall_curve(self.y_test, y_pred_proba)
            axes[0, 2].plot(recall_curve, precision_curve, label=model_name, linewidth=2)

        axes[0, 2].set_xlabel('Recall')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Precision-Recall Curves', fontweight='bold')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4-6. Confusion Matrices
        for i, model_name in enumerate(model_names):
            row = 1
            col = i if i < 3 else i - 3
            if i >= 3:
                # Create additional subplot if needed
                continue

            cm = self.model_results[model_name]['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       ax=axes[row, col], cbar=False)
            axes[row, col].set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')
            axes[row, col].set_xlabel('Predicted')
            axes[row, col].set_ylabel('Actual')

        # Hide unused subplot
        if len(model_names) < 4:
            axes[1, 2].axis('off')

        plt.tight_layout()

        # Save the comparison plot
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        plt.savefig(plots_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory

        print(f"✅ Model comparison visualization saved to: {plots_dir / 'model_comparison.png'}")

    def get_model_summary(self):
        """
        Generate comprehensive model performance summary
        """
        print("📋 MODEL PERFORMANCE SUMMARY")
        print("=" * 60)

        # Create summary DataFrame
        summary_data = []
        for model_name, results in self.model_results.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': f"{results['accuracy']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}",
                'F1-Score': f"{results['f1_score']:.4f}",
                'AUC Score': f"{results['auc_score']:.4f}"
            })

        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))

        # Find best model for each metric
        print(f"\n🏆 BEST PERFORMING MODELS:")
        print("-" * 30)

        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
        for metric in metrics:
            best_model = max(self.model_results.keys(),
                           key=lambda x: self.model_results[x][metric])
            best_score = self.model_results[best_model][metric]
            print(f"{metric.replace('_', ' ').title():12}: {best_model} ({best_score:.4f})")

        # Overall recommendation
        print(f"\n💡 RECOMMENDATION:")
        print("-" * 20)

        # Calculate weighted score (F1 and AUC are most important for fraud detection)
        weighted_scores = {}
        for model_name, results in self.model_results.items():
            weighted_score = (
                results['f1_score'] * 0.4 +  # F1 is crucial for imbalanced data
                results['auc_score'] * 0.3 +  # AUC shows overall performance
                results['precision'] * 0.2 +  # Precision reduces false positives
                results['recall'] * 0.1       # Recall catches fraud cases
            )
            weighted_scores[model_name] = weighted_score

        best_overall = max(weighted_scores.keys(), key=lambda x: weighted_scores[x])
        print(f"Best Overall Model: {best_overall}")
        print(f"Weighted Score: {weighted_scores[best_overall]:.4f}")

        return summary_df, best_overall

    def save_models(self, output_dir="models"):
        """
        Save trained models for future use
        """
        import joblib

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"💾 Saving models to {output_path}...")

        for model_name, model in self.models.items():
            filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
            joblib.dump(model, output_path / filename)
            print(f"   ✅ {model_name} saved as {filename}")

        # Save scaler
        joblib.dump(self.scaler, output_path / "scaler.pkl")
        print(f"   ✅ Scaler saved as scaler.pkl")

        print("✅ All models saved successfully!")

def main():
    """
    Main execution function
    """
    # Initialize the pipeline
    dataset_path = r"c:\Users\kalculusGuy\Desktop\projectEra\ML\gabby\data\creditcard_2023.csv"
    pipeline = FraudDetectionPipeline(dataset_path)

    print("🚀 CREDIT CARD FRAUD DETECTION PIPELINE")
    print("=" * 50)

    # Execute the complete pipeline
    try:
        # 1. Load and explore data
        pipeline.load_and_explore_data()

        # 2. Visualize data
        pipeline.visualize_data_distribution()

        # 3. Prepare data
        pipeline.prepare_data()

        # 4. Train all models
        pipeline.train_all_models()

        # 5. Create visualizations
        pipeline.create_comparison_visualizations()

        # 6. Generate summary
        summary_df, best_model = pipeline.get_model_summary()

        # 7. Save models
        pipeline.save_models()

        print(f"\n🎉 Pipeline completed successfully!")
        print(f"🏆 Best performing model: {best_model}")

    except Exception as e:
        print(f"❌ Error in pipeline execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()