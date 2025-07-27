"""
API Testing Script
Test the fraud detection API endpoints
"""

import requests
import json
import time

def test_api():
    """Test all API endpoints"""
    base_url = "http://localhost:5000/api"

    print("🧪 TESTING FRAUD DETECTION API")
    print("=" * 50)

    # Test 1: Health Check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Health check passed: {data['message']}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Health check error: {e}")

    # Test 2: Models Info
    print("\n2. Testing models endpoint...")
    try:
        response = requests.get(f"{base_url}/models")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Models info: {data['total_models']} models available")
            print(f"   📊 Available models: {', '.join(data['available_models'])}")
            print(f"   🎯 Default model: {data['default_model']}")
        else:
            print(f"   ❌ Models info failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Models info error: {e}")

    # Test 3: Sample Data
    print("\n3. Testing sample endpoint...")
    try:
        response = requests.get(f"{base_url}/sample")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Sample data retrieved")
            sample_transaction = data['sample_transaction']
        else:
            print(f"   ❌ Sample data failed: {response.status_code}")
            return
    except Exception as e:
        print(f"   ❌ Sample data error: {e}")
        return

    # Test 4: Single Prediction
    print("\n4. Testing single prediction...")
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=sample_transaction,
            headers={'Content-Type': 'application/json'}
        )
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                pred = data['prediction']
                fraud_status = "FRAUD" if pred['is_fraud'] else "LEGITIMATE"
                fraud_prob = pred['fraud_probability'] * 100
                print(f"   ✅ Prediction successful: {fraud_status}")
                print(f"   📊 Fraud probability: {fraud_prob:.2f}%")
                print(f"   🎯 Model used: {pred['model_used']}")
                print(f"   ⚡ Risk level: {pred['risk_level']}")
            else:
                print(f"   ❌ Prediction failed: {data.get('message', 'Unknown error')}")
        else:
            print(f"   ❌ Prediction request failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Prediction error: {e}")

    # Test 5: Batch Prediction
    print("\n5. Testing batch prediction...")
    try:
        batch_data = {
            "transactions": [
                sample_transaction,
                {**sample_transaction, "transaction_id": "test_002", "Amount": 50.0}
            ]
        }

        response = requests.post(
            f"{base_url}/predict/batch",
            json=batch_data,
            headers={'Content-Type': 'application/json'}
        )

        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print(f"   ✅ Batch prediction successful")
                print(f"   📊 Processed {data['total_processed']} transactions")

                for i, result in enumerate(data['results']):
                    if 'prediction' in result:
                        pred = result['prediction']
                        fraud_status = "FRAUD" if pred['is_fraud'] else "LEGITIMATE"
                        print(f"      Transaction {i+1}: {fraud_status} ({pred['fraud_probability']*100:.1f}%)")
                    else:
                        print(f"      Transaction {i+1}: ERROR - {result.get('error', 'Unknown')}")
            else:
                print(f"   ❌ Batch prediction failed: {data.get('message', 'Unknown error')}")
        else:
            print(f"   ❌ Batch prediction request failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Batch prediction error: {e}")

    # Test 6: Model Comparison
    print("\n6. Testing different models...")
    models_to_test = ['random_forest', 'xgboost', 'logistic_regression', 'naive_bayes']

    for model in models_to_test:
        try:
            test_data = {**sample_transaction, "model": model}
            response = requests.post(
                f"{base_url}/predict",
                json=test_data,
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code == 200:
                data = response.json()
                if data['success']:
                    pred = data['prediction']
                    fraud_prob = pred['fraud_probability'] * 100
                    print(f"   ✅ {model}: {fraud_prob:.2f}% fraud probability")
                else:
                    print(f"   ❌ {model}: {data.get('message', 'Failed')}")
            else:
                print(f"   ❌ {model}: HTTP {response.status_code}")
        except Exception as e:
            print(f"   ❌ {model}: {e}")

    print(f"\n🎉 API testing completed!")
    print(f"🌐 Web interface available at: http://localhost:5000")

if __name__ == "__main__":
    print("⏳ Waiting for API to start...")
    time.sleep(2)  # Give the API time to start
    test_api()