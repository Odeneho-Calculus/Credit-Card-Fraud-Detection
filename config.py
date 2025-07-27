"""
Configuration file for Fraud Detection System
Centralized configuration to avoid hardcoded paths
"""

import os
from pathlib import Path

class Config:
    """
    Configuration class for fraud detection system
    """

    # Project root directory
    PROJECT_ROOT = Path(__file__).parent

    # Data configuration
    DATA_DIR = PROJECT_ROOT / "data"
    DEFAULT_DATASET_NAME = "creditcard_2023.csv"
    DATASET_PATH = DATA_DIR / DEFAULT_DATASET_NAME

    # Model configuration
    MODELS_DIR = PROJECT_ROOT / "models"
    MODEL_FILES = {
        'Random Forest': 'random_forest_model.pkl',
        'XGBoost': 'xgboost_model.pkl',
        'Logistic Regression': 'logistic_regression_model.pkl',
        'Naive Bayes': 'naive_bayes_model.pkl'
    }

    # Visualization configuration
    PLOTS_DIR = PROJECT_ROOT / "plots"
    ADVANCED_PLOTS_DIR = PLOTS_DIR / "visuals"

    # API configuration
    API_HOST = os.getenv('FRAUD_API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('FRAUD_API_PORT', 5000))
    API_DEBUG = os.getenv('FRAUD_API_DEBUG', 'True').lower() == 'true'

    # Training configuration
    TEST_SIZE = float(os.getenv('FRAUD_TEST_SIZE', 0.2))
    RANDOM_STATE = int(os.getenv('FRAUD_RANDOM_STATE', 42))

    # Visualization settings
    PLOT_DPI = int(os.getenv('FRAUD_PLOT_DPI', 300))
    PLOT_STYLE = os.getenv('FRAUD_PLOT_STYLE', 'seaborn-v0_8')

    @classmethod
    def get_dataset_path(cls, custom_path=None):
        """
        Get dataset path with fallback options

        Args:
            custom_path: Custom dataset path (optional)

        Returns:
            Path: Dataset path
        """
        if custom_path:
            return Path(custom_path)

        # Check environment variable
        env_path = os.getenv('FRAUD_DATASET_PATH')
        if env_path:
            return Path(env_path)

        # Use default path
        return cls.DATASET_PATH

    @classmethod
    def ensure_directories(cls):
        """
        Ensure all required directories exist
        """
        directories = [
            cls.DATA_DIR,
            cls.MODELS_DIR,
            cls.PLOTS_DIR,
            cls.ADVANCED_PLOTS_DIR
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_dataset(cls, dataset_path=None):
        """
        Validate that dataset exists and is accessible

        Args:
            dataset_path: Path to dataset (optional)

        Returns:
            tuple: (is_valid, path, error_message)
        """
        path = cls.get_dataset_path(dataset_path)

        if not path.exists():
            return False, path, f"Dataset not found at: {path}"

        if not path.is_file():
            return False, path, f"Dataset path is not a file: {path}"

        if path.suffix.lower() != '.csv':
            return False, path, f"Dataset must be a CSV file: {path}"

        try:
            # Try to read first few lines to validate format
            import pandas as pd
            pd.read_csv(path, nrows=5)
            return True, path, None
        except Exception as e:
            return False, path, f"Cannot read dataset: {str(e)}"

    @classmethod
    def get_model_path(cls, model_name):
        """
        Get path for a specific model file

        Args:
            model_name: Name of the model

        Returns:
            Path: Model file path
        """
        if model_name not in cls.MODEL_FILES:
            raise ValueError(f"Unknown model: {model_name}")

        return cls.MODELS_DIR / cls.MODEL_FILES[model_name]

    @classmethod
    def print_config(cls):
        """
        Print current configuration
        """
        print("🔧 FRAUD DETECTION SYSTEM CONFIGURATION")
        print("=" * 50)
        print(f"📁 Project Root: {cls.PROJECT_ROOT}")
        print(f"📊 Dataset Path: {cls.get_dataset_path()}")
        print(f"🤖 Models Directory: {cls.MODELS_DIR}")
        print(f"📈 Plots Directory: {cls.PLOTS_DIR}")
        print(f"🌐 API Host: {cls.API_HOST}:{cls.API_PORT}")
        print(f"🎯 Test Size: {cls.TEST_SIZE}")
        print(f"🎲 Random State: {cls.RANDOM_STATE}")
        print("=" * 50)

# Environment variable examples for users
ENV_EXAMPLES = """
# Environment Variables for Fraud Detection System
# Set these in your shell or .env file for custom configuration

export FRAUD_DATASET_PATH="/path/to/your/creditcard_data.csv"
export FRAUD_API_HOST="localhost"
export FRAUD_API_PORT="8000"
export FRAUD_API_DEBUG="False"
export FRAUD_TEST_SIZE="0.3"
export FRAUD_RANDOM_STATE="123"
export FRAUD_PLOT_DPI="150"
"""

if __name__ == "__main__":
    # Print configuration when run directly
    Config.print_config()

    # Validate dataset
    is_valid, path, error = Config.validate_dataset()
    if is_valid:
        print(f"✅ Dataset validation successful: {path}")
    else:
        print(f"❌ Dataset validation failed: {error}")

    # Ensure directories exist
    Config.ensure_directories()
    print("✅ All required directories created/verified")