"""
Quick Demo of Fraud Detection Models
Runs on a smaller sample for faster demonstration
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def run_quick_demo():
    """
    Run a quick demonstration with a sample of the data
    """
    print("🚀 QUICK FRAUD DETECTION DEMO")
    print("=" * 50)

    # Load dataset
    dataset_path = r"c:\Users\kalculusGuy\Desktop\projectEra\ML\gabby\data\creditcard_2023.csv"
    print("📊 Loading dataset...")

    # Load only a sample for quick demo (10,000 rows)
    df = pd.read_csv(dataset_path, nrows=10000)
    print(f"   Sample size: {df.shape[0]:,} transactions")
    print(f"   Features: {df.shape[1] - 2} (excluding ID and Class)")

    # Check class distribution
    fraud_rate = df['Class'].mean()
    print(f"   Fraud rate: {fraud_rate:.2%}")

    # Prepare data
    print("\n🔧 Preparing data...")
    X = df.drop(['Class', 'id'], axis=1)
    y = df['Class']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"   Training samples: {X_train.shape[0]:,}")
    print(f"   Testing samples: {X_test.shape[0]:,}")

    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=50, random_state=42, eval_metric='logloss'),
        'Naive Bayes': GaussianNB()
    }

    results = {}

    print("\n🤖 Training and evaluating models...")
    print("-" * 50)

    for name, model in models.items():
        print(f"🔄 Training {name}...")

        # Train model
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC Score': auc
        }

        print(f"   ✅ {name} completed")

    # Display results
    print(f"\n📊 RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<18} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1-Score':<10} {'AUC':<8}")
    print("-" * 70)

    for model_name, metrics in results.items():
        print(f"{model_name:<18} {metrics['Accuracy']:<10.4f} {metrics['Precision']:<11.4f} "
              f"{metrics['Recall']:<8.4f} {metrics['F1-Score']:<10.4f} {metrics['AUC Score']:<8.4f}")

    # Find best model
    best_f1_model = max(results.keys(), key=lambda x: results[x]['F1-Score'])
    best_auc_model = max(results.keys(), key=lambda x: results[x]['AUC Score'])

    print(f"\n🏆 BEST PERFORMERS:")
    print(f"   Best F1-Score: {best_f1_model} ({results[best_f1_model]['F1-Score']:.4f})")
    print(f"   Best AUC Score: {best_auc_model} ({results[best_auc_model]['AUC Score']:.4f})")

    print(f"\n💡 INTERPRETATION:")
    print("   • F1-Score balances precision and recall (ideal for fraud detection)")
    print("   • AUC Score measures overall classification ability")
    print("   • High precision = fewer false fraud alerts")
    print("   • High recall = catches more actual fraud")

    print(f"\n✅ Quick demo completed! Run the full analysis for complete results.")

    return results

if __name__ == "__main__":
    results = run_quick_demo()