"""
Generate Visual README with Plot Thumbnails
Creates an enhanced README with actual plot previews and descriptions
"""

import os
from pathlib import Path
from config import Config
import base64
from PIL import Image
import io

class VisualReadmeGenerator:
    """
    Generate enhanced README with visual content
    """

    def __init__(self):
        self.plots_dir = Config.PLOTS_DIR
        self.advanced_plots_dir = Config.ADVANCED_PLOTS_DIR
        self.readme_path = Config.PROJECT_ROOT / "README_VISUAL.md"

    def check_plots_exist(self):
        """Check which plots exist"""
        plots_status = {}

        # Individual model plots
        model_plots = [
            "random_forest_analysis.png",
            "xgboost_analysis.png",
            "logistic_regression_analysis.png",
            "naive_bayes_analysis.png"
        ]

        for plot in model_plots:
            plots_status[plot] = (self.plots_dir / plot).exists()

        # Advanced plots
        advanced_plots = [
            "dataset_overview.png",
            "class_distribution_analysis.png",
            "correlation_analysis.png",
            "pca_analysis.png",
            "amount_analysis.png",
            "time_analysis.png",
            "feature_distributions.png",
            "outlier_analysis.png"
        ]

        for plot in advanced_plots:
            plots_status[f"visuals/{plot}"] = (self.advanced_plots_dir / plot).exists()

        # Other plots
        other_plots = ["model_comparison.png", "data_distribution.png"]
        for plot in other_plots:
            plots_status[plot] = (self.plots_dir / plot).exists()

        return plots_status

    def get_plot_description(self, plot_name):
        """Get description for each plot"""
        descriptions = {
            "random_forest_analysis.png": {
                "title": "🌳 Random Forest - Comprehensive Analysis",
                "description": "Complete performance analysis including confusion matrix, ROC curve, feature importance, prediction distributions, threshold analysis, classification report, performance radar, learning curve, and error breakdown."
            },
            "xgboost_analysis.png": {
                "title": "🚀 XGBoost - Advanced Gradient Boosting Analysis",
                "description": "State-of-the-art gradient boosting analysis with detailed performance metrics, feature importance rankings, prediction confidence distributions, and comprehensive error analysis."
            },
            "logistic_regression_analysis.png": {
                "title": "📈 Logistic Regression - Statistical Analysis",
                "description": "Classical statistical approach analysis with probability distributions, coefficient importance, decision boundary analysis, and statistical performance metrics."
            },
            "naive_bayes_analysis.png": {
                "title": "🎯 Naive Bayes - Probabilistic Analysis",
                "description": "Probabilistic classifier analysis with likelihood distributions, feature independence assumptions, prediction confidence, and Bayesian performance metrics."
            },
            "visuals/dataset_overview.png": {
                "title": "📊 Dataset Overview - Complete Statistics",
                "description": "Comprehensive dataset statistics including transaction counts, fraud rates, missing values analysis, data types distribution, and feature value ranges."
            },
            "visuals/class_distribution_analysis.png": {
                "title": "⚖️ Class Distribution - Imbalance Analysis",
                "description": "Detailed class imbalance analysis with fraud vs legitimate ratios, amount distributions by class, statistical summaries, and imbalance impact assessment."
            },
            "visuals/correlation_analysis.png": {
                "title": "🔗 Feature Correlation - Relationship Analysis",
                "description": "Complete correlation matrix analysis, target feature correlations, highly correlated feature pairs identification, and correlation distribution patterns."
            },
            "visuals/pca_analysis.png": {
                "title": "🎯 PCA Components - V1-V28 Analysis",
                "description": "Principal Component Analysis of V1-V28 features including component distributions, variance analysis by class, top fraud-predictive components, and PCA heatmaps."
            },
            "visuals/amount_analysis.png": {
                "title": "💰 Transaction Amount - Pattern Analysis",
                "description": "Comprehensive transaction amount analysis including distributions, percentiles by class, amount ranges, statistical comparisons, and fraud amount patterns."
            },
            "visuals/time_analysis.png": {
                "title": "⏰ Time-based - Temporal Pattern Analysis",
                "description": "Time-based fraud pattern analysis including hourly transaction patterns, fraud rates by time of day, temporal distributions, and time vs amount correlations."
            },
            "visuals/feature_distributions.png": {
                "title": "📉 Feature Distributions - Class Comparison",
                "description": "Key feature distribution analysis by class with statistical annotations, mean comparisons, distribution overlaps, and feature discriminative power."
            },
            "visuals/outlier_analysis.png": {
                "title": "🚨 Outlier Analysis - Anomaly Detection",
                "description": "Comprehensive outlier detection analysis including outlier percentages by feature, fraud correlation with outliers, box plot comparisons, and anomaly patterns."
            },
            "model_comparison.png": {
                "title": "🏆 Model Comparison - Performance Benchmarking",
                "description": "Side-by-side model performance comparison with accuracy, precision, recall, F1-score, and AUC metrics across all four machine learning models."
            },
            "data_distribution.png": {
                "title": "📊 Data Distribution - Feature Overview",
                "description": "Overall data distribution analysis showing feature patterns, class distributions, and general data characteristics for fraud detection."
            }
        }

        return descriptions.get(plot_name, {
            "title": f"📈 {plot_name}",
            "description": "Detailed analysis visualization for fraud detection insights."
        })

    def create_thumbnail(self, image_path, max_size=(400, 300)):
        """Create thumbnail version of image"""
        try:
            with Image.open(image_path) as img:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)

                # Convert to base64 for embedding
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()

                return f"data:image/png;base64,{img_str}"
        except Exception as e:
            print(f"⚠️  Could not create thumbnail for {image_path}: {e}")
            return None

    def generate_visual_readme(self):
        """Generate enhanced README with visuals"""
        print("🎨 GENERATING VISUAL README")
        print("=" * 50)

        plots_status = self.check_plots_exist()
        existing_plots = [plot for plot, exists in plots_status.items() if exists]
        missing_plots = [plot for plot, exists in plots_status.items() if not exists]

        print(f"✅ Found {len(existing_plots)} existing plots")
        if missing_plots:
            print(f"⚠️  Missing {len(missing_plots)} plots:")
            for plot in missing_plots[:5]:  # Show first 5
                print(f"   • {plot}")
            if len(missing_plots) > 5:
                print(f"   • ... and {len(missing_plots) - 5} more")

        # Generate README content
        readme_content = self._generate_readme_content(plots_status)

        # Write to file
        with open(self.readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        print(f"✅ Visual README generated: {self.readme_path}")
        return True

    def _generate_readme_content(self, plots_status):
        """Generate the actual README content"""
        content = """# 🎨 **Fraud Detection System - Visual Gallery**

## 📊 **Complete Visualization Overview**

This document showcases all the visualizations generated by the fraud detection system. Each chart provides unique insights into the data and model performance.

---

"""

        # Individual Model Analysis Section
        content += """## 🤖 **Individual Model Analysis Charts**

Each model gets its own comprehensive 20×16 analysis chart with 10 detailed visualizations:

"""

        model_plots = [
            ("random_forest_analysis.png", "🌳 Random Forest"),
            ("xgboost_analysis.png", "🚀 XGBoost"),
            ("logistic_regression_analysis.png", "📈 Logistic Regression"),
            ("naive_bayes_analysis.png", "🎯 Naive Bayes")
        ]

        for plot_file, model_name in model_plots:
            plot_info = self.get_plot_description(plot_file)
            exists = plots_status.get(plot_file, False)

            content += f"""### {plot_info['title']}

{plot_info['description']}

"""

            if exists:
                content += f"""![{model_name} Analysis](plots/{plot_file})

**✅ Available** - Generated after running analysis

"""
            else:
                content += f"""*📋 Plot will be generated after running:*
```bash
python run_complete_analysis.py
# or
python generate_individual_plots.py
```

"""

        # Advanced Data Exploration Section
        content += """---

## 📈 **Advanced Data Exploration Charts**

Comprehensive EDA with 8 detailed analysis charts:

"""

        advanced_plots = [
            ("visuals/dataset_overview.png", "📊 Dataset Overview"),
            ("visuals/class_distribution_analysis.png", "⚖️ Class Distribution"),
            ("visuals/correlation_analysis.png", "🔗 Correlation Analysis"),
            ("visuals/pca_analysis.png", "🎯 PCA Analysis"),
            ("visuals/amount_analysis.png", "💰 Amount Analysis"),
            ("visuals/time_analysis.png", "⏰ Time Analysis"),
            ("visuals/feature_distributions.png", "📉 Feature Distributions"),
            ("visuals/outlier_analysis.png", "🚨 Outlier Analysis")
        ]

        for plot_file, chart_name in advanced_plots:
            plot_info = self.get_plot_description(plot_file)
            exists = plots_status.get(plot_file, False)

            content += f"""### {plot_info['title']}

{plot_info['description']}

"""

            if exists:
                content += f"""![{chart_name}](plots/{plot_file})

**✅ Available** - Generated after running analysis

"""
            else:
                content += f"""*📋 Plot will be generated after running:*
```bash
python data_visualizations.py
```

"""

        # Other Plots Section
        content += """---

## 🔄 **Additional Visualizations**

### 🏆 Model Comparison Chart
Side-by-side performance comparison of all four models.

"""

        if plots_status.get("model_comparison.png", False):
            content += """![Model Comparison](plots/model_comparison.png)

**✅ Available**

"""
        else:
            content += """*📋 Generated during complete analysis*

"""

        content += """### 📊 Data Distribution Overview
General data distribution and feature patterns.

"""

        if plots_status.get("data_distribution.png", False):
            content += """![Data Distribution](plots/data_distribution.png)

**✅ Available**

"""
        else:
            content += """*📋 Generated during data exploration*

"""

        # Generation Instructions
        content += """---

## 🚀 **How to Generate All Visualizations**

### **Complete Analysis (Recommended)**
```bash
python run_complete_analysis.py
```
*Generates all visualizations in one run*

### **Individual Components**
```bash
# Individual model analysis charts
python generate_individual_plots.py

# Advanced data exploration charts
python data_visualizations.py

# Custom dataset analysis
python run_with_custom_dataset.py --dataset /path/to/data.csv
```

### **Generated Files Structure**
```
plots/
├── random_forest_analysis.png        # Random Forest detailed analysis
├── xgboost_analysis.png              # XGBoost detailed analysis
├── logistic_regression_analysis.png  # Logistic Regression analysis
├── naive_bayes_analysis.png          # Naive Bayes analysis
├── model_comparison.png              # Model comparison chart
├── data_distribution.png             # Data distribution overview
└── visuals/
    ├── dataset_overview.png          # Dataset statistics
    ├── class_distribution_analysis.png # Class imbalance analysis
    ├── correlation_analysis.png      # Feature correlations
    ├── pca_analysis.png              # PCA components analysis
    ├── amount_analysis.png           # Transaction amounts
    ├── time_analysis.png             # Time-based patterns
    ├── feature_distributions.png     # Feature distributions
    └── outlier_analysis.png          # Outlier detection
```

---

## 📋 **Plot Status Summary**

"""

        # Add status summary
        total_plots = len(plots_status)
        existing_count = sum(plots_status.values())
        missing_count = total_plots - existing_count

        content += f"""- **Total Visualizations**: {total_plots}
- **✅ Available**: {existing_count}
- **📋 To Generate**: {missing_count}

"""

        if missing_count > 0:
            content += """### 🔧 **To Generate Missing Plots:**
```bash
# Generate all missing visualizations
python run_complete_analysis.py
```

"""

        content += """---

> **💡 Tip:** All visualizations are high-resolution (300 DPI) and suitable for presentations or publications.

> **🔄 Auto-Update:** This visual README can be regenerated anytime by running `python generate_visual_readme.py`

"""

        return content

def main():
    """Main function"""
    print("🎨 Visual README Generator")
    print("Creates an enhanced README with plot previews and descriptions")
    print()

    generator = VisualReadmeGenerator()
    success = generator.generate_visual_readme()

    if success:
        print(f"\n✅ Visual README generated successfully!")
        print(f"📂 Location: {generator.readme_path}")
        print("\n🔍 The visual README includes:")
        print("  • Plot availability status")
        print("  • Detailed descriptions for each visualization")
        print("  • Generation instructions")
        print("  • File structure overview")
    else:
        print("\n❌ Failed to generate visual README")

if __name__ == "__main__":
    main()