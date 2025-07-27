"""
Verify README Images
Check if all images referenced in README.md actually exist
"""

import re
from pathlib import Path
from config import Config

def extract_image_paths_from_readme():
    """Extract all image paths from README.md"""
    readme_path = Config.PROJECT_ROOT / "README.md"

    if not readme_path.exists():
        print("❌ README.md not found!")
        return []

    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all markdown image references: ![alt text](path)
    image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    matches = re.findall(image_pattern, content)

    image_paths = []
    for alt_text, path in matches:
        # Skip external URLs
        if path.startswith(('http', 'https', 'ftp')):
            continue
        image_paths.append((alt_text, path))

    return image_paths

def verify_images():
    """Verify that all README images exist"""
    print("🔍 VERIFYING README IMAGES")
    print("=" * 40)

    image_paths = extract_image_paths_from_readme()

    if not image_paths:
        print("❌ No images found in README.md")
        return False

    print(f"📋 Found {len(image_paths)} image references in README.md")
    print()

    existing_images = 0
    missing_images = 0

    for alt_text, relative_path in image_paths:
        # Convert relative path to absolute
        image_path = Config.PROJECT_ROOT / relative_path

        print(f"🖼️  {alt_text}")
        print(f"   📁 Path: {relative_path}")

        if image_path.exists():
            file_size = image_path.stat().st_size
            size_mb = file_size / (1024 * 1024)
            print(f"   ✅ Exists ({size_mb:.2f} MB)")
            existing_images += 1
        else:
            print(f"   ❌ Missing: {image_path}")
            missing_images += 1
        print()

    # Summary
    print("📊 SUMMARY")
    print("=" * 20)
    print(f"✅ Existing images: {existing_images}")
    print(f"❌ Missing images: {missing_images}")
    print(f"📊 Total references: {len(image_paths)}")

    if missing_images > 0:
        print(f"\n🔧 To generate missing images:")
        print("   python run_complete_analysis.py       # Generate all images")
        print("   python fix_corrupted_models.py        # Fix corrupted models first")
        print("   python data_visualizations.py         # Generate EDA images")
        print("   python generate_individual_plots.py   # Generate model analysis images")
        return False
    else:
        print(f"\n🎉 All images exist! Your README visual gallery is complete!")
        return True

def show_image_categories():
    """Show image categories and their status"""
    print("\n📂 IMAGE CATEGORIES")
    print("=" * 30)

    categories = {
        "Individual Model Analysis": [
            "plots/random_forest_analysis.png",
            "plots/xgboost_analysis.png",
            "plots/logistic_regression_analysis.png",
            "plots/naive_bayes_analysis.png"
        ],
        "Advanced Data Exploration": [
            "plots/visuals/dataset_overview.png",
            "plots/visuals/class_distribution_analysis.png",
            "plots/visuals/correlation_analysis.png",
            "plots/visuals/pca_analysis.png",
            "plots/visuals/amount_analysis.png",
            "plots/visuals/feature_distributions.png",
            "plots/visuals/outlier_analysis.png",
            "plots/visuals/time_analysis.png"
        ],
        "Performance & Comparison": [
            "plots/model_comparison.png",
            "plots/data_analysis.png"
        ]
    }

    for category, paths in categories.items():
        print(f"\n📁 {category}:")
        existing_count = 0

        for path in paths:
            image_path = Config.PROJECT_ROOT / path
            if image_path.exists():
                print(f"   ✅ {path}")
                existing_count += 1
            else:
                print(f"   ❌ {path}")

        print(f"   📊 {existing_count}/{len(paths)} images exist")

def main():
    """Main function"""
    print("🔍 README Image Verification Tool")
    print("Checks if all images referenced in README.md actually exist")
    print()

    success = verify_images()
    show_image_categories()

    if success:
        print("\n✅ README visual gallery is complete!")
        print("🌐 All images will display properly in GitHub, local viewers, etc.")
    else:
        print("\n⚠️  Some images are missing from your README gallery")
        print("🔧 Run the suggested commands above to generate them")

if __name__ == "__main__":
    main()