"""
Fraud Detection API Startup Script
Comprehensive startup with model verification and API launch
"""

import os
import sys
from pathlib import Path
import subprocess
import time

def check_models():
    """Check if trained models exist"""
    models_dir = Path("models")
    required_files = [
        "scaler.pkl",
        "random_forest_model.pkl",
        "xgboost_model.pkl",
        "logistic_regression_model.pkl",
        "naive_bayes_model.pkl"
    ]

    print("🔍 Checking for trained models...")

    if not models_dir.exists():
        print("❌ Models directory not found!")
        return False

    missing_files = []
    for file in required_files:
        if not (models_dir / file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"❌ Missing model files: {', '.join(missing_files)}")
        return False

    print("✅ All required models found!")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'flask', 'flask_cors', 'pandas', 'numpy',
        'scikit-learn', 'xgboost', 'joblib'
    ]

    print("📦 Checking dependencies...")

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("💡 Install with: pip install -r requirements.txt")
        return False

    print("✅ All dependencies satisfied!")
    return True

def start_api():
    """Start the Flask API"""
    print("🚀 Starting Fraud Detection API...")

    try:
        # Import and run the app
        from app import app, fraud_api

        if not fraud_api.models:
            print("❌ No models loaded. Cannot start API.")
            return False

        print(f"📊 Loaded {len(fraud_api.models)} models")
        print(f"🎯 Default model: {fraud_api.default_model}")
        print("🌐 Starting web server...")
        print("=" * 60)
        print("🔗 API Endpoints:")
        print("   • Web Interface: http://localhost:5000")
        print("   • Health Check:  http://localhost:5000/api/health")
        print("   • Predict:       http://localhost:5000/api/predict")
        print("   • Batch:         http://localhost:5000/api/predict/batch")
        print("   • Models Info:   http://localhost:5000/api/models")
        print("   • Sample Data:   http://localhost:5000/api/sample")
        print("=" * 60)
        print("💡 Press Ctrl+C to stop the server")
        print()

        # Start the Flask app
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,  # Set to False for production
            threaded=True
        )

        return True

    except Exception as e:
        print(f"❌ Failed to start API: {e}")
        return False

def main():
    """Main startup function"""
    print("🎯 FRAUD DETECTION API STARTUP")
    print("=" * 50)

    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("❌ app.py not found. Please run from the project directory.")
        sys.exit(1)

    # Check dependencies
    if not check_dependencies():
        print("\n💡 To install dependencies:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

    # Check models
    if not check_models():
        print("\n💡 To train models:")
        print("   python run_complete_analysis.py")
        sys.exit(1)

    print("\n✅ All checks passed! Starting API...")
    print("-" * 50)

    # Start the API
    if not start_api():
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 API server stopped by user")
        print("Thank you for using the Fraud Detection System!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)