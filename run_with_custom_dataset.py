"""
Run Fraud Detection Analysis with Custom Dataset Path
Example of how to use the system with different dataset locations
"""

import sys
import argparse
from pathlib import Path
from config import Config
from fraud_detection_models import FraudDetectionPipeline

def run_with_custom_dataset(dataset_path):
    """
    Run analysis with custom dataset path

    Args:
        dataset_path: Path to the dataset file
    """
    print("🚀 FRAUD DETECTION ANALYSIS WITH CUSTOM DATASET")
    print("=" * 60)
    print(f"📊 Using dataset: {dataset_path}")

    # Validate the custom dataset
    is_valid, validated_path, error_message = Config.validate_dataset(dataset_path)
    if not is_valid:
        print(f"❌ {error_message}")
        return False

    # Ensure directories exist
    Config.ensure_directories()

    try:
        # Initialize pipeline with custom dataset
        pipeline = FraudDetectionPipeline(validated_path)

        print("\n📊 Step 1: Loading and exploring dataset...")
        pipeline.load_and_explore_data()

        print("\n📈 Step 2: Creating data visualizations...")
        pipeline.visualize_data_distribution()

        print("\n🔧 Step 3: Preparing data for machine learning...")
        pipeline.prepare_data()

        print("\n🤖 Step 4: Training all machine learning models...")
        pipeline.train_all_models()

        print("\n📊 Step 5: Creating model comparison visualizations...")
        pipeline.create_comparison_visualizations()

        print("\n🎨 Step 6: Creating individual model analysis charts...")
        pipeline.create_individual_model_plots()

        print("\n📋 Step 7: Generating performance summary...")
        summary_df, best_model = pipeline.get_model_summary()

        print("\n💾 Step 8: Saving trained models...")
        pipeline.save_models()

        print(f"\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"🏆 Best performing model: {best_model}")
        print(f"📁 Models saved in: {Config.MODELS_DIR}")
        print(f"📈 Plots saved in: {Config.PLOTS_DIR}")

        return True

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Run fraud detection analysis with custom dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_with_custom_dataset.py --dataset /path/to/your/creditcard.csv
  python run_with_custom_dataset.py -d ./data/my_fraud_data.csv

Environment Variables:
  FRAUD_DATASET_PATH - Default dataset path
  FRAUD_MODELS_DIR   - Models output directory
  FRAUD_PLOTS_DIR    - Plots output directory
        """
    )

    parser.add_argument(
        '--dataset', '-d',
        type=str,
        help='Path to the dataset CSV file'
    )

    parser.add_argument(
        '--config', '-c',
        action='store_true',
        help='Show current configuration and exit'
    )

    args = parser.parse_args()

    # Show configuration if requested
    if args.config:
        Config.print_config()
        return

    # Determine dataset path
    if args.dataset:
        dataset_path = args.dataset
        print(f"📁 Using custom dataset: {dataset_path}")
    else:
        # Try to use default or environment variable
        dataset_path = Config.get_dataset_path()
        print(f"📁 Using default dataset: {dataset_path}")

    # Run analysis
    success = run_with_custom_dataset(dataset_path)

    if success:
        print("\n✅ Analysis completed successfully!")
        print("🔍 Check the output directories for results:")
        print(f"  • Models: {Config.MODELS_DIR}")
        print(f"  • Plots: {Config.PLOTS_DIR}")
    else:
        print("\n❌ Analysis failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()