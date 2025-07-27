"""
Results Summary and Model Usage Guide
Run this after the main analysis to understand your results
"""

import pandas as pd
import joblib
from pathlib import Path
import numpy as np

def load_and_summarize_results():
    """
    Load saved models and provide usage instructions
    """
    print("📋 FRAUD DETECTION PROJECT RESULTS SUMMARY")
    print("=" * 60)

    models_dir = Path("models")

    if not models_dir.exists():
        print("❌ Models directory not found. Please run the main analysis first.")
        return

    # List available models
    model_files = list(models_dir.glob("*_model.pkl"))

    print(f"🤖 TRAINED MODELS ({len(model_files)} available):")
    print("-" * 40)

    for model_file in model_files:
        model_name = model_file.stem.replace('_model', '').replace('_', ' ').title()
        print(f"   ✅ {model_name}")

    print(f"\n📁 Models saved in: {models_dir.absolute()}")

    # Show how to use the models
    print(f"\n💡 HOW TO USE YOUR TRAINED MODELS:")
    print("-" * 40)

    usage_code = '''
# Example: Load and use a trained model
import joblib
import pandas as pd
import numpy as np

# Load the best model (example: Random Forest)
model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Prepare new transaction data (example)
new_transaction = {
    'V1': -1.359807, 'V2': -0.072781, 'V3': 2.536347, 'V4': 1.378155,
    'V5': -0.338321, 'V6': 0.462388, 'V7': 0.239599, 'V8': 0.098698,
    'V9': 0.363787, 'V10': 0.090794, 'V11': -0.551600, 'V12': -0.617801,
    'V13': -0.991390, 'V14': -0.311169, 'V15': 1.468177, 'V16': -0.470401,
    'V17': 0.207971, 'V18': 0.025791, 'V19': 0.403993, 'V20': 0.251412,
    'V21': -0.018307, 'V22': 0.277838, 'V23': -0.110474, 'V24': 0.066928,
    'V25': 0.128539, 'V26': -0.189115, 'V27': 0.133558, 'V28': -0.021053,
    'Amount': 149.62
}

# Convert to DataFrame
df_new = pd.DataFrame([new_transaction])

# Scale the features
df_scaled = scaler.transform(df_new)

# Make prediction
prediction = model.predict(df_scaled)[0]
probability = model.predict_proba(df_scaled)[0]

print(f"Prediction: {'FRAUD' if prediction == 1 else 'LEGITIMATE'}")
print(f"Fraud Probability: {probability[1]:.4f}")
'''

    print(usage_code)

    # Performance interpretation guide
    print(f"\n📊 UNDERSTANDING YOUR RESULTS:")
    print("-" * 40)

    interpretation = '''
KEY METRICS EXPLAINED:

🎯 ACCURACY: Overall correctness (good baseline, but can be misleading with imbalanced data)
🔍 PRECISION: Of predicted frauds, how many were actually fraud (reduces false alarms)
🎣 RECALL: Of actual frauds, how many did we catch (fraud detection rate)
⚖️ F1-SCORE: Balance between precision and recall (best overall metric)
📈 AUC SCORE: Area under ROC curve (model's ability to distinguish classes)

FOR FRAUD DETECTION:
- HIGH PRECISION = Fewer false fraud alerts (better customer experience)
- HIGH RECALL = Catch more actual fraud (better security)
- HIGH F1-SCORE = Good balance of both
- AUC > 0.9 = Excellent model performance

BUSINESS IMPACT:
- False Positives (FP): Legitimate transactions flagged as fraud → Customer frustration
- False Negatives (FN): Fraud transactions missed → Financial loss
- True Positives (TP): Fraud correctly detected → Money saved
- True Negatives (TN): Legitimate transactions correctly processed → Smooth operations
'''

    print(interpretation)

    print(f"\n🚀 NEXT STEPS:")
    print("-" * 20)
    print("1. Review the model comparison visualizations")
    print("2. Choose the best model based on your business priorities")
    print("3. Integrate the chosen model into your fraud detection system")
    print("4. Monitor model performance and retrain periodically")
    print("5. Consider ensemble methods for even better performance")

def create_prediction_template():
    """
    Create a template for making predictions on new data
    """
    template_code = '''
"""
Fraud Detection Prediction Template
Use this template to make predictions on new transactions
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

class FraudPredictor:
    def __init__(self, model_path="models/random_forest_model.pkl", scaler_path="models/scaler.pkl"):
        """Load trained model and scaler"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict_transaction(self, transaction_data):
        """
        Predict if a transaction is fraudulent

        Args:
            transaction_data: dict with V1-V28 and Amount features

        Returns:
            dict with prediction and probability
        """
        # Convert to DataFrame
        df = pd.DataFrame([transaction_data])

        # Scale features
        df_scaled = self.scaler.transform(df)

        # Make prediction
        prediction = self.model.predict(df_scaled)[0]
        probabilities = self.model.predict_proba(df_scaled)[0]

        return {
            'is_fraud': bool(prediction),
            'fraud_probability': float(probabilities[1]),
            'confidence': 'HIGH' if max(probabilities) > 0.8 else 'MEDIUM' if max(probabilities) > 0.6 else 'LOW'
        }

    def batch_predict(self, transactions_df):
        """Predict multiple transactions at once"""
        df_scaled = self.scaler.transform(transactions_df)
        predictions = self.model.predict(df_scaled)
        probabilities = self.model.predict_proba(df_scaled)[:, 1]

        results = pd.DataFrame({
            'is_fraud': predictions,
            'fraud_probability': probabilities
        })

        return results

# Example usage:
if __name__ == "__main__":
    predictor = FraudPredictor()

    # Example transaction
    sample_transaction = {
        'V1': -1.359807, 'V2': -0.072781, 'V3': 2.536347, 'V4': 1.378155,
        'V5': -0.338321, 'V6': 0.462388, 'V7': 0.239599, 'V8': 0.098698,
        'V9': 0.363787, 'V10': 0.090794, 'V11': -0.551600, 'V12': -0.617801,
        'V13': -0.991390, 'V14': -0.311169, 'V15': 1.468177, 'V16': -0.470401,
        'V17': 0.207971, 'V18': 0.025791, 'V19': 0.403993, 'V20': 0.251412,
        'V21': -0.018307, 'V22': 0.277838, 'V23': -0.110474, 'V24': 0.066928,
        'V25': 0.128539, 'V26': -0.189115, 'V27': 0.133558, 'V28': -0.021053,
        'Amount': 149.62
    }

    result = predictor.predict_transaction(sample_transaction)
    print(f"Fraud Detection Result: {result}")
'''

    # Save the template
    with open("fraud_predictor_template.py", "w") as f:
        f.write(template_code)

    print(f"\n💾 Prediction template saved as: fraud_predictor_template.py")

if __name__ == "__main__":
    load_and_summarize_results()
    create_prediction_template()