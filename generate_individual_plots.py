"""
Generate Individual Model Analysis Plots
Run this script to create separate detailed visualizations for each trained model
"""

from fraud_detection_models import FraudDetectionPipeline
from config import Config
import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_individual_plots():
    """
    Generate individual analysis plots for all trained models
    """
    print("🎨 GENERATING INDIVIDUAL MODEL ANALYSIS PLOTS")
    print("=" * 60)

    # Ensure required directories exist
    Config.ensure_directories()

    # Check if models exist
    if not Config.MODELS_DIR.exists():
        print("❌ Models directory not found!")
        print("Please run 'python run_complete_analysis.py' first to train the models.")
        return False

    # Validate dataset
    is_valid, dataset_path, error_message = Config.validate_dataset()
    if not is_valid:
        print(f"❌ {error_message}")
        print("Please ensure the dataset is placed in the 'data/' directory")
        return False

    try:
        print("📊 Loading dataset and preparing data...")

        # Load and prepare data (similar to the pipeline)
        df = pd.read_csv(dataset_path)

        # Prepare features and target
        X = df.drop(['Class'], axis=1)
        y = df['Class']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print("🤖 Loading trained models...")

        # Load models and create results
        model_files = Config.MODEL_FILES

        # Create a temporary pipeline instance for plotting
        pipeline = FraudDetectionPipeline(dataset_path)
        pipeline.X_test = X_test_scaled
        pipeline.y_test = y_test
        pipeline.models = {}
        pipeline.model_results = {}

        # Load each model and generate its results
        for model_name, filename in model_files.items():
            model_path = Config.MODELS_DIR / filename
            if model_path.exists():
                print(f"  📈 Processing {model_name}...")

                # Load the model
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)

                pipeline.models[model_name] = model

                # Generate predictions
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    # For models without predict_proba, use decision_function or predict
                    if hasattr(model, 'decision_function'):
                        scores = model.decision_function(X_test_scaled)
                        # Convert to probabilities using sigmoid
                        y_pred_proba = 1 / (1 + np.exp(-scores))
                    else:
                        y_pred_proba = model.predict(X_test_scaled).astype(float)

                y_pred = (y_pred_proba >= 0.5).astype(int)

                # Calculate metrics
                from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                           f1_score, roc_auc_score, confusion_matrix)

                pipeline.model_results[model_name] = {
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1_score': f1_score(y_test, y_pred, zero_division=0),
                    'auc_score': roc_auc_score(y_test, y_pred_proba),
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }

                print(f"    ✅ {model_name} processed successfully")
            else:
                print(f"    ⚠️  {model_name} model file not found: {filename}")

        if not pipeline.model_results:
            print("❌ No trained models found!")
            return False

        print(f"\n🎨 Creating individual analysis plots for {len(pipeline.model_results)} models...")
        pipeline.create_individual_model_plots()

        print(f"\n🎉 INDIVIDUAL PLOTS GENERATED SUCCESSFULLY!")
        print(f"📁 Plots saved in: plots/ directory")
        print(f"📊 Generated {len(pipeline.model_results)} detailed model analysis charts")

        # List generated files
        if Config.PLOTS_DIR.exists():
            plot_files = list(Config.PLOTS_DIR.glob("*_analysis.png"))
            if plot_files:
                print(f"\n📋 Generated plot files:")
                for plot_file in plot_files:
                    print(f"  • {plot_file.name}")

        return True

    except Exception as e:
        print(f"❌ Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🎨 Individual Model Analysis Plot Generator")
    print("This script creates detailed visualizations for each trained model.")
    print()

    success = generate_individual_plots()

    if success:
        print("\n✅ Individual model analysis plots generated successfully!")
        print("📂 Check the 'plots/' directory for detailed model visualizations.")
        print("\n🔍 Each plot includes:")
        print("  • Confusion Matrix with metrics")
        print("  • ROC and Precision-Recall curves")
        print("  • Feature importance analysis")
        print("  • Prediction distribution")
        print("  • Threshold analysis")
        print("  • Classification report heatmap")
        print("  • Performance radar chart")
        print("  • Learning curve simulation")
        print("  • Error analysis breakdown")
    else:
        print("\n❌ Failed to generate individual plots.")
        print("Please ensure models are trained first by running:")
        print("  python run_complete_analysis.py")

if __name__ == "__main__":
    main()