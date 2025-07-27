# Credit Card Fraud Detection Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6%2B-orange.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.0%2B-green.svg)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Project Overview

This project implements a comprehensive **Credit Card Fraud Detection System** using multiple machine learning algorithms. The system analyzes transaction patterns to identify potentially fraudulent activities in real-time, helping financial institutions protect their customers and reduce financial losses.

### 🏆 Key Features

- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, and Naive Bayes
- **Comprehensive Evaluation**: ROC curves, precision-recall analysis, confusion matrices
- **Production Ready**: Scalable architecture with model persistence and prediction templates
- **Visual Analytics**: Interactive plots and performance comparisons
- **Real-time Prediction**: Ready-to-use prediction pipeline for new transactions

## 📊 Dataset

The project uses the **Credit Card Fraud Detection Dataset 2023** containing:
- **568,630 transactions** from European cardholders
- **30 features** (V1-V28 PCA components + Amount + Class)
- **Balanced dataset** (50% legitimate, 50% fraudulent)
- **No missing values** - ready for immediate analysis

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. **Install dependencies**
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn plotly imbalanced-learn joblib
```

3. **Download the dataset**
   - Visit [Kaggle Credit Card Fraud Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)
   - Download and place `creditcard_2023.csv` in the `data/` directory

4. **Run the complete analysis**
```bash
python run_complete_analysis.py
```

### Quick Demo (10,000 samples)

```bash
python quick_demo.py
```

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── data/                          # Dataset directory
│   └── creditcard_2023.csv       # Main dataset
│
├── models/                        # Trained models (generated)
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── naive_bayes_model.pkl
│   └── scaler.pkl
│
├── fraud_detection_models.py      # Main ML pipeline
├── run_complete_analysis.py       # Complete analysis runner
├── quick_demo.py                  # Quick demonstration
├── results_summary.py             # Results interpretation
├── download_dataset.py            # Dataset downloader
├── fraud_predictor_template.py    # Prediction template (generated)
├── Group_8_MC_3B.ipynb           # Jupyter notebook
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🤖 Machine Learning Models

### 1. **Logistic Regression**
- **Use Case**: Baseline model with interpretable coefficients
- **Strengths**: Fast training, probabilistic output, feature importance
- **Best For**: Understanding feature relationships

### 2. **Random Forest**
- **Use Case**: Ensemble method with feature importance
- **Strengths**: Handles non-linear patterns, robust to outliers
- **Best For**: Balanced performance and interpretability

### 3. **XGBoost**
- **Use Case**: Gradient boosting for maximum performance
- **Strengths**: State-of-the-art accuracy, handles imbalanced data
- **Best For**: Production systems requiring highest accuracy

### 4. **Naive Bayes**
- **Use Case**: Probabilistic classifier with independence assumption
- **Strengths**: Fast prediction, works well with small datasets
- **Best For**: Real-time systems with speed requirements

## 📈 Performance Metrics

The system evaluates models using comprehensive metrics:

| Metric | Description | Importance for Fraud Detection |
|--------|-------------|-------------------------------|
| **Accuracy** | Overall correctness | Baseline performance indicator |
| **Precision** | True frauds / Predicted frauds | Reduces false alarms |
| **Recall** | True frauds / Actual frauds | Catches more fraud cases |
| **F1-Score** | Harmonic mean of precision/recall | Balanced fraud detection |
| **AUC Score** | Area under ROC curve | Overall classification ability |

## 🔍 Usage Examples

### Basic Prediction

```python
from fraud_predictor_template import FraudPredictor

# Initialize predictor
predictor = FraudPredictor()

# Sample transaction
transaction = {
    'V1': -1.359807, 'V2': -0.072781, 'V3': 2.536347,
    # ... (V4-V28)
    'Amount': 149.62
}

# Make prediction
result = predictor.predict_transaction(transaction)
print(f"Fraud Probability: {result['fraud_probability']:.4f}")
print(f"Is Fraud: {result['is_fraud']}")
```

### Batch Prediction

```python
import pandas as pd

# Load multiple transactions
transactions_df = pd.read_csv('new_transactions.csv')

# Predict all at once
results = predictor.batch_predict(transactions_df)
print(results.head())
```

## 📊 Results Interpretation

### Key Insights

- **XGBoost** typically achieves the highest AUC scores (>0.95)
- **Random Forest** provides the best balance of performance and interpretability
- **Logistic Regression** offers fastest training and clear feature importance
- **Naive Bayes** delivers fastest predictions for real-time systems

### Business Impact

- **False Positives**: Legitimate transactions flagged as fraud → Customer frustration
- **False Negatives**: Fraud transactions missed → Financial loss
- **True Positives**: Fraud correctly detected → Money saved
- **True Negatives**: Legitimate transactions processed → Smooth operations

## 🛠️ Advanced Features

### Model Persistence
```python
import joblib

# Save trained model
joblib.dump(model, 'models/my_fraud_model.pkl')

# Load for prediction
model = joblib.load('models/my_fraud_model.pkl')
```

### Custom Thresholds
```python
# Adjust prediction threshold for business needs
threshold = 0.3  # Lower = catch more fraud, higher = fewer false alarms
predictions = (probabilities > threshold).astype(int)
```

### Feature Engineering
```python
# Add custom features
df['amount_log'] = np.log1p(df['Amount'])
df['amount_normalized'] = df['Amount'] / df['Amount'].max()
```

## 📚 Documentation

### Jupyter Notebook
Open `Group_8_MC_3B.ipynb` for interactive analysis and detailed explanations.

### Results Summary
```bash
python results_summary.py
```

### Model Comparison
The system automatically generates:
- ROC curves comparison
- Precision-recall curves
- Confusion matrices
- Performance metrics table

## 🔧 Configuration

### Environment Variables
```bash
export DATASET_PATH="path/to/your/dataset.csv"
export MODEL_OUTPUT_DIR="path/to/models/"
```

### Custom Parameters
Modify `fraud_detection_models.py` to adjust:
- Train/test split ratio
- Cross-validation folds
- Model hyperparameters
- Evaluation metrics

## 🚀 Deployment

### Production Checklist
- [ ] Model validation on holdout dataset
- [ ] Performance monitoring setup
- [ ] Threshold optimization for business KPIs
- [ ] A/B testing framework
- [ ] Model retraining pipeline

### API Integration
```python
from flask import Flask, request, jsonify
from fraud_predictor_template import FraudPredictor

app = Flask(__name__)
predictor = FraudPredictor()

@app.route('/predict', methods=['POST'])
def predict_fraud():
    transaction = request.json
    result = predictor.predict_transaction(transaction)
    return jsonify(result)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Credit Card Fraud Detection Dataset 2023 from Kaggle
- **Libraries**: scikit-learn, XGBoost, pandas, numpy, matplotlib, seaborn
- **Inspiration**: Real-world fraud detection challenges in financial institutions

## 📞 Contact

- **Project Maintainer**: [Your Name]
- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn Profile]
- **GitHub**: [Your GitHub Profile]

---

⭐ **Star this repository if it helped you build better fraud detection systems!**