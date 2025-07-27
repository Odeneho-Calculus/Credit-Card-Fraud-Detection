"""
Complete Fraud Detection Analysis Runner
Execute this script to run the entire machine learning pipeline
"""

from fraud_detection_models import FraudDetectionPipeline
import sys
from pathlib import Path

def run_complete_analysis():
    """
    Execute the complete fraud detection analysis
    """
    print("🚀 STARTING COMPLETE FRAUD DETECTION ANALYSIS")
    print("=" * 60)

    # Dataset path
    dataset_path = r"c:\Users\kalculusGuy\Desktop\projectEra\ML\gabby\data\creditcard_2023.csv"

    # Check if dataset exists
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found at: {dataset_path}")
        print("Please ensure the dataset is downloaded and placed in the correct location.")
        return False

    try:
        # Initialize pipeline
        pipeline = FraudDetectionPipeline(dataset_path)

        print("📊 Step 1: Loading and exploring dataset...")
        pipeline.load_and_explore_data()

        print("\n📈 Step 2: Creating data visualizations...")
        pipeline.visualize_data_distribution()

        print("\n🔧 Step 3: Preparing data for machine learning...")
        pipeline.prepare_data()

        print("\n🤖 Step 4: Training all machine learning models...")
        pipeline.train_all_models()

        print("\n📊 Step 5: Creating model comparison visualizations...")
        pipeline.create_comparison_visualizations()

        print("\n📋 Step 6: Generating performance summary...")
        summary_df, best_model = pipeline.get_model_summary()

        print("\n💾 Step 7: Saving trained models...")
        pipeline.save_models()

        print(f"\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"🏆 Best performing model: {best_model}")
        print(f"📁 Models saved in: models/ directory")

        return True

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_complete_analysis()
    if success:
        print("\n✅ You can now review the results and use the trained models!")
    else:
        print("\n❌ Analysis failed. Please check the error messages above.")