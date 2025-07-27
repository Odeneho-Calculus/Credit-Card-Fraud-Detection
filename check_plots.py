"""
Check Plot Status
Quick utility to check which visualization plots exist
"""

from pathlib import Path
from config import Config
import os

def check_plot_status():
    """Check status of all expected plots"""
    print("📊 FRAUD DETECTION PLOTS STATUS CHECK")
    print("=" * 50)

    # Define all expected plots
    plots_info = {
        "Individual Model Analysis": [
            ("random_forest_analysis.png", "🌳 Random Forest Analysis"),
            ("xgboost_analysis.png", "🚀 XGBoost Analysis"),
            ("logistic_regression_analysis.png", "📈 Logistic Regression Analysis"),
            ("naive_bayes_analysis.png", "🎯 Naive Bayes Analysis")
        ],
        "Advanced Data Exploration": [
            ("visuals/dataset_overview.png", "📊 Dataset Overview"),
            ("visuals/class_distribution_analysis.png", "⚖️ Class Distribution Analysis"),
            ("visuals/correlation_analysis.png", "🔗 Correlation Analysis"),
            ("visuals/pca_analysis.png", "🎯 PCA Analysis"),
            ("visuals/amount_analysis.png", "💰 Amount Analysis"),
            ("visuals/time_analysis.png", "⏰ Time Analysis"),
            ("visuals/feature_distributions.png", "📉 Feature Distributions"),
            ("visuals/outlier_analysis.png", "🚨 Outlier Analysis")
        ],
        "General Visualizations": [
            ("model_comparison.png", "🏆 Model Comparison"),
            ("data_distribution.png", "📊 Data Distribution")
        ]
    }

    total_plots = 0
    existing_plots = 0

    for category, plots in plots_info.items():
        print(f"\n📁 {category}:")
        print("-" * (len(category) + 4))

        category_existing = 0
        for plot_file, plot_name in plots:
            total_plots += 1

            # Check if plot exists
            if plot_file.startswith("visuals/"):
                plot_path = Config.ADVANCED_PLOTS_DIR / plot_file.replace("visuals/", "")
            else:
                plot_path = Config.PLOTS_DIR / plot_file

            if plot_path.exists():
                # Get file size
                file_size = plot_path.stat().st_size
                size_mb = file_size / (1024 * 1024)

                print(f"  ✅ {plot_name}")
                print(f"     📁 {plot_path}")
                print(f"     📏 {size_mb:.2f} MB")
                existing_plots += 1
                category_existing += 1
            else:
                print(f"  ❌ {plot_name}")
                print(f"     📁 {plot_path} (missing)")

        print(f"     📊 {category_existing}/{len(plots)} plots available")

    # Summary
    print(f"\n📋 SUMMARY")
    print("=" * 20)
    print(f"✅ Available: {existing_plots}/{total_plots} plots")
    print(f"❌ Missing: {total_plots - existing_plots}/{total_plots} plots")

    if existing_plots == total_plots:
        print("\n🎉 All visualizations are available!")
    elif existing_plots == 0:
        print("\n🚀 No plots found. Run analysis to generate visualizations:")
        print("   python run_complete_analysis.py")
    else:
        print(f"\n🔧 To generate missing plots:")

        # Check which categories are missing
        missing_individual = not any((Config.PLOTS_DIR / plot[0]).exists()
                                   for plot in plots_info["Individual Model Analysis"])
        missing_advanced = not any((Config.ADVANCED_PLOTS_DIR / plot[0].replace("visuals/", "")).exists()
                                 for plot in plots_info["Advanced Data Exploration"])

        if missing_individual:
            print("   python generate_individual_plots.py  # For individual model analysis")
        if missing_advanced:
            print("   python data_visualizations.py        # For advanced data exploration")

        print("   python run_complete_analysis.py         # For complete analysis")

    # Directory info
    print(f"\n📂 DIRECTORIES")
    print("=" * 20)
    print(f"📁 Main plots: {Config.PLOTS_DIR}")
    print(f"   {'✅ Exists' if Config.PLOTS_DIR.exists() else '❌ Missing'}")

    print(f"📁 Advanced plots: {Config.ADVANCED_PLOTS_DIR}")
    print(f"   {'✅ Exists' if Config.ADVANCED_PLOTS_DIR.exists() else '❌ Missing'}")

    # Show recent plots
    if existing_plots > 0:
        print(f"\n🕒 RECENT PLOTS")
        print("=" * 20)

        all_plot_paths = []
        for category, plots in plots_info.items():
            for plot_file, plot_name in plots:
                if plot_file.startswith("visuals/"):
                    plot_path = Config.ADVANCED_PLOTS_DIR / plot_file.replace("visuals/", "")
                else:
                    plot_path = Config.PLOTS_DIR / plot_file

                if plot_path.exists():
                    all_plot_paths.append((plot_path, plot_name))

        # Sort by modification time
        all_plot_paths.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)

        # Show 5 most recent
        for plot_path, plot_name in all_plot_paths[:5]:
            import datetime
            mod_time = datetime.datetime.fromtimestamp(plot_path.stat().st_mtime)
            print(f"  📈 {plot_name}")
            print(f"     🕒 {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main function"""
    try:
        check_plot_status()
    except Exception as e:
        print(f"❌ Error checking plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()