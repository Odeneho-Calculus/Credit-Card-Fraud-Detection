"""
Auto Generate Visual README
Automatically discovers all images and creates a comprehensive visual gallery
"""

import os
from pathlib import Path
from config import Config

def discover_all_images():
    """Discover all PNG images in plots directory and subdirectories"""
    plots_dir = Config.PLOTS_DIR

    if not plots_dir.exists():
        print("❌ Plots directory doesn't exist!")
        return {}

    images = {}

    # Search for all PNG files recursively
    for image_path in plots_dir.rglob("*.png"):
        # Get relative path from project root
        relative_path = image_path.relative_to(Config.PROJECT_ROOT)

        # Create a nice display name from filename
        image_name = image_path.stem.replace('_', ' ').title()

        # Categorize images
        if 'visuals' in str(image_path):
            category = "Advanced Data Exploration"
        elif 'analysis' in image_path.name:
            category = "Individual Model Analysis"
        else:
            category = "Performance & Comparison"

        if category not in images:
            images[category] = []

        images[category].append({
            'name': image_name,
            'path': str(relative_path).replace('\\', '/'),  # Use forward slashes for markdown
            'filename': image_path.name,
            'size_mb': image_path.stat().st_size / (1024 * 1024)
        })

    # Sort each category
    for category in images:
        images[category].sort(key=lambda x: x['filename'])

    return images

def generate_visual_gallery_section():
    """Generate the complete visual gallery section for README"""

    print("🎨 Discovering all images...")
    images = discover_all_images()

    if not images:
        print("❌ No images found!")
        return ""

    total_images = sum(len(imgs) for imgs in images.values())
    print(f"✅ Found {total_images} images in {len(images)} categories")

    # Start building the visual gallery
    gallery = []
    gallery.append("## 🖼️ **Complete Visual Gallery**")
    gallery.append("")
    gallery.append("*All visualizations generated during training and analysis process*")
    gallery.append("")
    gallery.append("---")
    gallery.append("")

    # Add each category
    for category, imgs in images.items():
        if category == "Individual Model Analysis":
            gallery.append("## 🤖 **Individual Model Analysis Charts**")
            gallery.append("*Each model gets its own comprehensive 20×16 analysis chart with 10 detailed visualizations*")
        elif category == "Advanced Data Exploration":
            gallery.append("## 📊 **Advanced Data Exploration Charts**")
            gallery.append("*Comprehensive EDA with detailed analysis visualizations*")
        elif category == "Performance & Comparison":
            gallery.append("## 🏆 **Performance & Comparison Visualizations**")

        gallery.append("")

        for img in imgs:
            gallery.append(f"### {img['name']}")
            gallery.append(f"![{img['name']}]({img['path']})")

            # Add descriptions based on image type
            if 'random_forest' in img['filename']:
                gallery.append("*Complete Random Forest performance analysis including confusion matrix, ROC curve, feature importance, prediction distributions, threshold analysis, classification report, performance radar, learning curve, and error breakdown.*")
            elif 'xgboost' in img['filename']:
                gallery.append("*State-of-the-art XGBoost gradient boosting analysis with detailed performance metrics, feature importance rankings, prediction confidence distributions, and comprehensive error analysis.*")
            elif 'logistic_regression' in img['filename']:
                gallery.append("*Classical Logistic Regression statistical approach analysis with probability distributions, coefficient importance, decision boundary analysis, and statistical performance metrics.*")
            elif 'naive_bayes' in img['filename']:
                gallery.append("*Probabilistic Naive Bayes classifier analysis with likelihood distributions, feature independence assumptions, prediction confidence, and Bayesian performance metrics.*")
            elif 'dataset_overview' in img['filename']:
                gallery.append("*Comprehensive dataset statistics including transaction counts, fraud rates, missing values analysis, data types distribution, and feature value ranges.*")
            elif 'class_distribution' in img['filename']:
                gallery.append("*Detailed class imbalance analysis with fraud vs legitimate ratios, amount distributions by class, statistical summaries, and imbalance impact assessment.*")
            elif 'correlation' in img['filename']:
                gallery.append("*Complete correlation matrix analysis, target feature correlations, highly correlated feature pairs identification, and correlation distribution patterns.*")
            elif 'pca' in img['filename']:
                gallery.append("*Principal Component Analysis of V1-V28 features including component distributions, variance analysis by class, top fraud-predictive components, and PCA heatmaps.*")
            elif 'amount' in img['filename']:
                gallery.append("*Comprehensive transaction amount analysis including distributions, percentiles by class, amount ranges, statistical comparisons, and fraud amount patterns.*")
            elif 'feature_distributions' in img['filename']:
                gallery.append("*Key feature distribution analysis by class with statistical annotations, mean comparisons, distribution overlaps, and feature discriminative power.*")
            elif 'outlier' in img['filename']:
                gallery.append("*Comprehensive outlier detection analysis including outlier percentages by feature, fraud correlation with outliers, box plot comparisons, and anomaly patterns.*")
            elif 'model_comparison' in img['filename']:
                gallery.append("*Side-by-side model performance comparison with accuracy, precision, recall, F1-score, and AUC metrics across all four machine learning models.*")
            elif 'data_analysis' in img['filename']:
                gallery.append("*Comprehensive data analysis summary showing overall patterns, distributions, and key insights for fraud detection.*")
            else:
                gallery.append(f"*{img['name']} visualization (Size: {img['size_mb']:.2f} MB)*")

            gallery.append("")

        gallery.append("---")
        gallery.append("")

    # Add summary
    gallery.append("## 📋 **Visual Gallery Summary**")
    gallery.append("")
    gallery.append("**🎨 All images above are automatically generated during the training and analysis process!**")
    gallery.append("")
    gallery.append(f"### **📊 Total Visualizations: {total_images} High-Resolution Charts**")
    gallery.append("")

    for category, imgs in images.items():
        icon = "🤖" if "Individual" in category else "📊" if "Advanced" in category else "🏆"
        gallery.append(f"- **{icon} {len(imgs)} {category} Charts** - {category.split()[-1].lower()} visualizations")

    gallery.append("")
    gallery.append("*All images are generated at 300 DPI resolution, suitable for presentations and publications.*")
    gallery.append("")

    return "\n".join(gallery)

def update_readme_with_discovered_images():
    """Update README.md with automatically discovered images"""

    readme_path = Config.PROJECT_ROOT / "README.md"

    if not readme_path.exists():
        print("❌ README.md not found!")
        return False

    print("📖 Reading current README.md...")
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the visual gallery section
    start_marker = "## 🖼️ **Complete Visual Gallery**"
    end_marker = "## 🚨 **Important Note About Visual Gallery**"

    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)

    if start_idx == -1:
        print("❌ Visual gallery section not found in README!")
        return False

    if end_idx == -1:
        print("❌ End marker not found in README!")
        return False

    # Generate new visual gallery
    print("🎨 Generating new visual gallery...")
    new_gallery = generate_visual_gallery_section()

    # Replace the section
    new_content = content[:start_idx] + new_gallery + "\n" + content[end_idx:]

    # Write back to README
    print("💾 Updating README.md...")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print("✅ README.md updated with discovered images!")
    return True

def main():
    """Main function"""
    print("🔍 AUTO VISUAL README GENERATOR")
    print("=" * 50)
    print("Automatically discovers all images and updates README")
    print()

    # Discover and show images
    images = discover_all_images()

    if not images:
        print("❌ No images found to display!")
        return

    total_images = sum(len(imgs) for imgs in images.values())
    print(f"📊 DISCOVERED IMAGES:")
    print(f"🖼️  Total Images: {total_images}")

    for category, imgs in images.items():
        print(f"\n📁 {category}: {len(imgs)} images")
        for img in imgs:
            print(f"   📄 {img['filename']} ({img['size_mb']:.2f} MB)")

    print(f"\n🎨 Updating README with all {total_images} images...")

    if update_readme_with_discovered_images():
        print(f"\n🎉 SUCCESS!")
        print(f"✅ README.md now displays all {total_images} images automatically!")
        print(f"🌐 Images will be visible on GitHub after commit/push")
        print(f"\n📋 To commit and push:")
        print(f"   git add .")
        print(f"   git commit -m 'Update README with complete visual gallery'")
        print(f"   git push")
    else:
        print(f"\n❌ Failed to update README!")

if __name__ == "__main__":
    main()