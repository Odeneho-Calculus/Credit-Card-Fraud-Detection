"""
Advanced Data Visualizations for Fraud Detection
Generate comprehensive data analysis and visualization plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from config import Config
import warnings
warnings.filterwarnings('ignore')

class AdvancedDataVisualizer:
    """
    Advanced visualization class for fraud detection data analysis
    """

    def __init__(self, dataset_path):
        """Initialize with dataset path"""
        self.dataset_path = dataset_path
        self.df = None
        self.plots_dir = Config.ADVANCED_PLOTS_DIR
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Load the dataset"""
        print("📊 Loading dataset...")
        self.df = pd.read_csv(self.dataset_path)
        print(f"✅ Dataset loaded: {self.df.shape[0]} transactions, {self.df.shape[1]} features")

    def create_comprehensive_eda(self):
        """Create comprehensive exploratory data analysis plots"""
        print("🔍 Creating comprehensive EDA visualizations...")

        # 1. Dataset Overview
        self._create_dataset_overview()

        # 2. Class Distribution Analysis
        self._create_class_distribution_analysis()

        # 3. Feature Correlation Analysis
        self._create_correlation_analysis()

        # 4. PCA Components Analysis
        self._create_pca_analysis()

        # 5. Amount Analysis
        self._create_amount_analysis()

        # 6. Time Analysis
        self._create_time_analysis()

        # 7. Feature Distribution Analysis
        self._create_feature_distributions()

        # 8. Outlier Analysis
        self._create_outlier_analysis()

        print(f"✅ Advanced visualizations saved to: {self.plots_dir}")

    def _create_dataset_overview(self):
        """Create dataset overview visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Dataset Overview Analysis', fontsize=16, fontweight='bold')

        # Basic statistics
        ax1 = axes[0, 0]
        stats_data = {
            'Total Transactions': len(self.df),
            'Fraud Cases': len(self.df[self.df['Class'] == 1]),
            'Legitimate Cases': len(self.df[self.df['Class'] == 0]),
            'Features': self.df.shape[1] - 1,
            'Fraud Rate (%)': (len(self.df[self.df['Class'] == 1]) / len(self.df)) * 100
        }

        bars = ax1.bar(range(len(stats_data)), list(stats_data.values()),
                      color=['skyblue', 'red', 'green', 'orange', 'purple'])
        ax1.set_xticks(range(len(stats_data)))
        ax1.set_xticklabels(stats_data.keys(), rotation=45, ha='right')
        ax1.set_title('Dataset Statistics', fontweight='bold')
        ax1.set_ylabel('Count / Percentage')

        # Add value labels on bars
        for bar, value in zip(bars, stats_data.values()):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.1f}' if isinstance(value, float) else f'{value:,}',
                    ha='center', va='bottom', fontweight='bold')

        # Missing values heatmap
        ax2 = axes[0, 1]
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            sns.heatmap(missing_data.values.reshape(-1, 1),
                       yticklabels=missing_data.index, ax=ax2, cmap='Reds')
            ax2.set_title('Missing Values', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No Missing Values\nDetected',
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=14, fontweight='bold')
            ax2.set_title('Missing Values Analysis', fontweight='bold')

        # Data types
        ax3 = axes[1, 0]
        dtype_counts = self.df.dtypes.value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(dtype_counts)))
        wedges, texts, autotexts = ax3.pie(dtype_counts.values, labels=dtype_counts.index,
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax3.set_title('Data Types Distribution', fontweight='bold')

        # Feature ranges
        ax4 = axes[1, 1]
        feature_ranges = []
        feature_names = []
        for col in self.df.columns:
            if col != 'Class':
                feature_ranges.append([self.df[col].min(), self.df[col].max()])
                feature_names.append(col)

        # Show ranges for first 10 features
        ranges_to_show = feature_ranges[:10]
        names_to_show = feature_names[:10]

        y_pos = np.arange(len(names_to_show))
        mins = [r[0] for r in ranges_to_show]
        maxs = [r[1] for r in ranges_to_show]

        ax4.barh(y_pos, maxs, alpha=0.7, label='Max', color='lightcoral')
        ax4.barh(y_pos, mins, alpha=0.7, label='Min', color='lightblue')
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(names_to_show)
        ax4.set_xlabel('Value Range')
        ax4.set_title('Feature Value Ranges (Top 10)', fontweight='bold')
        ax4.legend()

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'dataset_overview.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_class_distribution_analysis(self):
        """Create detailed class distribution analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Class Distribution Analysis', fontsize=16, fontweight='bold')

        # Basic class distribution
        ax1 = axes[0, 0]
        class_counts = self.df['Class'].value_counts()
        colors = ['lightgreen', 'lightcoral']
        wedges, texts, autotexts = ax1.pie(class_counts.values,
                                          labels=['Legitimate', 'Fraud'],
                                          autopct='%1.2f%%', colors=colors, startangle=90)
        ax1.set_title('Class Distribution', fontweight='bold')

        # Class distribution bar chart
        ax2 = axes[0, 1]
        bars = ax2.bar(['Legitimate', 'Fraud'], class_counts.values, color=colors)
        ax2.set_title('Transaction Counts by Class', fontweight='bold')
        ax2.set_ylabel('Number of Transactions')

        # Add value labels
        for bar, value in zip(bars, class_counts.values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:,}', ha='center', va='bottom', fontweight='bold')

        # Imbalance ratio
        ax3 = axes[0, 2]
        fraud_ratio = class_counts[1] / class_counts[0]
        ax3.bar(['Fraud to Legitimate Ratio'], [fraud_ratio], color='orange')
        ax3.set_title('Class Imbalance Ratio', fontweight='bold')
        ax3.set_ylabel('Ratio')
        ax3.text(0, fraud_ratio + fraud_ratio*0.1, f'{fraud_ratio:.4f}',
                ha='center', va='bottom', fontweight='bold')

        # Amount distribution by class
        ax4 = axes[1, 0]
        fraud_amounts = self.df[self.df['Class'] == 1]['Amount']
        legit_amounts = self.df[self.df['Class'] == 0]['Amount']

        ax4.hist(legit_amounts, bins=50, alpha=0.7, label='Legitimate',
                color='green', density=True)
        ax4.hist(fraud_amounts, bins=50, alpha=0.7, label='Fraud',
                color='red', density=True)
        ax4.set_xlabel('Transaction Amount')
        ax4.set_ylabel('Density')
        ax4.set_title('Amount Distribution by Class', fontweight='bold')
        ax4.legend()
        ax4.set_xlim(0, 1000)  # Focus on lower amounts

        # Box plot of amounts by class
        ax5 = axes[1, 1]
        data_for_box = [legit_amounts, fraud_amounts]
        box_plot = ax5.boxplot(data_for_box, labels=['Legitimate', 'Fraud'], patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightgreen')
        box_plot['boxes'][1].set_facecolor('lightcoral')
        ax5.set_title('Amount Distribution Box Plot', fontweight='bold')
        ax5.set_ylabel('Transaction Amount')
        ax5.set_ylim(0, 500)  # Focus on lower amounts

        # Statistical summary
        ax6 = axes[1, 2]
        stats_summary = pd.DataFrame({
            'Legitimate': legit_amounts.describe(),
            'Fraud': fraud_amounts.describe()
        }).round(2)

        # Create table
        table_data = []
        for idx, row in stats_summary.iterrows():
            table_data.append([idx, f"{row['Legitimate']:.2f}", f"{row['Fraud']:.2f}"])

        table = ax6.table(cellText=table_data,
                         colLabels=['Statistic', 'Legitimate', 'Fraud'],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax6.axis('off')
        ax6.set_title('Statistical Summary', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'class_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_correlation_analysis(self):
        """Create correlation analysis visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Feature Correlation Analysis', fontsize=16, fontweight='bold')

        # Full correlation matrix
        ax1 = axes[0, 0]
        corr_matrix = self.df.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, ax=ax1, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        ax1.set_title('Full Correlation Matrix', fontweight='bold')

        # Correlation with target
        ax2 = axes[0, 1]
        target_corr = self.df.corr()['Class'].abs().sort_values(ascending=False)[1:]  # Exclude Class itself
        top_features = target_corr.head(15)

        bars = ax2.barh(range(len(top_features)), top_features.values,
                       color='steelblue', alpha=0.7)
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels(top_features.index)
        ax2.set_xlabel('Absolute Correlation with Class')
        ax2.set_title('Top 15 Features Correlated with Fraud', fontweight='bold')

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, top_features.values)):
            ax2.text(value + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', ha='left', va='center', fontsize=9)

        # High correlation pairs
        ax3 = axes[1, 0]
        # Find highly correlated feature pairs (excluding target)
        feature_corr = self.df.drop('Class', axis=1).corr()
        high_corr_pairs = []

        for i in range(len(feature_corr.columns)):
            for j in range(i+1, len(feature_corr.columns)):
                corr_val = abs(feature_corr.iloc[i, j])
                if corr_val > 0.1:  # Threshold for "high" correlation
                    high_corr_pairs.append({
                        'Feature1': feature_corr.columns[i],
                        'Feature2': feature_corr.columns[j],
                        'Correlation': corr_val
                    })

        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', ascending=False).head(10)

            bars = ax3.barh(range(len(high_corr_df)), high_corr_df['Correlation'],
                           color='coral', alpha=0.7)
            ax3.set_yticks(range(len(high_corr_df)))
            ax3.set_yticklabels([f"{row['Feature1']} - {row['Feature2']}"
                               for _, row in high_corr_df.iterrows()], fontsize=8)
            ax3.set_xlabel('Absolute Correlation')
            ax3.set_title('Top 10 Highly Correlated Feature Pairs', fontweight='bold')

            # Add value labels
            for bar, value in zip(bars, high_corr_df['Correlation']):
                ax3.text(value + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{value:.3f}', ha='left', va='center', fontsize=8)
        else:
            ax3.text(0.5, 0.5, 'No highly correlated\nfeature pairs found',
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Highly Correlated Feature Pairs', fontweight='bold')

        # Correlation distribution
        ax4 = axes[1, 1]
        # Get all correlation values (excluding diagonal)
        corr_values = []
        for i in range(len(feature_corr.columns)):
            for j in range(i+1, len(feature_corr.columns)):
                corr_values.append(feature_corr.iloc[i, j])

        ax4.hist(corr_values, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax4.set_xlabel('Correlation Coefficient')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Feature Correlations', fontweight='bold')
        ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero Correlation')
        ax4.legend()

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_pca_analysis(self):
        """Create PCA components analysis"""
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('PCA Components Analysis (V1-V28)', fontsize=16, fontweight='bold')

        # Get PCA features
        pca_features = [col for col in self.df.columns if col.startswith('V')]

        # PCA components distribution
        ax1 = axes[0, 0]
        pca_data = self.df[pca_features]
        ax1.boxplot([pca_data[col] for col in pca_features[:14]],
                   labels=pca_features[:14])
        ax1.set_title('PCA Components V1-V14 Distribution', fontweight='bold')
        ax1.set_ylabel('Value')
        ax1.tick_params(axis='x', rotation=45)

        ax2 = axes[0, 1]
        ax2.boxplot([pca_data[col] for col in pca_features[14:]],
                   labels=pca_features[14:])
        ax2.set_title('PCA Components V15-V28 Distribution', fontweight='bold')
        ax2.set_ylabel('Value')
        ax2.tick_params(axis='x', rotation=45)

        # PCA variance by class
        ax3 = axes[1, 0]
        fraud_pca = self.df[self.df['Class'] == 1][pca_features]
        legit_pca = self.df[self.df['Class'] == 0][pca_features]

        fraud_var = fraud_pca.var()
        legit_var = legit_pca.var()

        x = np.arange(len(pca_features))
        width = 0.35

        ax3.bar(x - width/2, legit_var, width, label='Legitimate', alpha=0.7, color='green')
        ax3.bar(x + width/2, fraud_var, width, label='Fraud', alpha=0.7, color='red')
        ax3.set_xlabel('PCA Components')
        ax3.set_ylabel('Variance')
        ax3.set_title('PCA Component Variance by Class', fontweight='bold')
        ax3.set_xticks(x[::2])  # Show every other label
        ax3.set_xticklabels(pca_features[::2], rotation=45)
        ax3.legend()

        # PCA mean by class
        ax4 = axes[1, 1]
        fraud_mean = fraud_pca.mean()
        legit_mean = legit_pca.mean()

        ax4.bar(x - width/2, legit_mean, width, label='Legitimate', alpha=0.7, color='green')
        ax4.bar(x + width/2, fraud_mean, width, label='Fraud', alpha=0.7, color='red')
        ax4.set_xlabel('PCA Components')
        ax4.set_ylabel('Mean Value')
        ax4.set_title('PCA Component Means by Class', fontweight='bold')
        ax4.set_xticks(x[::2])
        ax4.set_xticklabels(pca_features[::2], rotation=45)
        ax4.legend()

        # Top PCA components by fraud correlation
        ax5 = axes[2, 0]
        pca_corr = self.df[pca_features + ['Class']].corr()['Class'].abs().sort_values(ascending=False)[:-1]
        top_pca = pca_corr.head(10)

        bars = ax5.barh(range(len(top_pca)), top_pca.values, color='steelblue', alpha=0.7)
        ax5.set_yticks(range(len(top_pca)))
        ax5.set_yticklabels(top_pca.index)
        ax5.set_xlabel('Absolute Correlation with Fraud')
        ax5.set_title('Top 10 PCA Components for Fraud Detection', fontweight='bold')

        # Add value labels
        for bar, value in zip(bars, top_pca.values):
            ax5.text(value + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', ha='left', va='center', fontsize=9)

        # PCA components heatmap
        ax6 = axes[2, 1]
        # Sample data for heatmap (first 1000 rows for performance)
        sample_data = self.df[pca_features].head(1000)
        sns.heatmap(sample_data.T, ax=ax6, cmap='coolwarm', center=0,
                   cbar_kws={"shrink": .8})
        ax6.set_title('PCA Components Heatmap (Sample)', fontweight='bold')
        ax6.set_xlabel('Transaction Samples')
        ax6.set_ylabel('PCA Components')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_amount_analysis(self):
        """Create comprehensive amount analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Transaction Amount Analysis', fontsize=16, fontweight='bold')

        amounts = self.df['Amount']
        fraud_amounts = self.df[self.df['Class'] == 1]['Amount']
        legit_amounts = self.df[self.df['Class'] == 0]['Amount']

        # Amount distribution
        ax1 = axes[0, 0]
        ax1.hist(amounts, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Transaction Amount')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Overall Amount Distribution', fontweight='bold')
        ax1.set_xlim(0, 1000)  # Focus on lower amounts

        # Log scale amount distribution
        ax2 = axes[0, 1]
        ax2.hist(amounts[amounts > 0], bins=100, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('Transaction Amount (Log Scale)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Amount Distribution (Log Scale)', fontweight='bold')
        ax2.set_xscale('log')

        # Amount by class comparison
        ax3 = axes[0, 2]
        ax3.hist(legit_amounts, bins=50, alpha=0.7, label='Legitimate',
                color='green', density=True)
        ax3.hist(fraud_amounts, bins=50, alpha=0.7, label='Fraud',
                color='red', density=True)
        ax3.set_xlabel('Transaction Amount')
        ax3.set_ylabel('Density')
        ax3.set_title('Amount Distribution by Class', fontweight='bold')
        ax3.legend()
        ax3.set_xlim(0, 500)

        # Amount percentiles
        ax4 = axes[1, 0]
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        legit_percentiles = [np.percentile(legit_amounts, p) for p in percentiles]
        fraud_percentiles = [np.percentile(fraud_amounts, p) for p in percentiles]

        x = np.arange(len(percentiles))
        width = 0.35

        ax4.bar(x - width/2, legit_percentiles, width, label='Legitimate',
               alpha=0.7, color='green')
        ax4.bar(x + width/2, fraud_percentiles, width, label='Fraud',
               alpha=0.7, color='red')
        ax4.set_xlabel('Percentiles')
        ax4.set_ylabel('Amount')
        ax4.set_title('Amount Percentiles by Class', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'{p}th' for p in percentiles])
        ax4.legend()

        # Amount ranges analysis
        ax5 = axes[1, 1]
        ranges = [(0, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, float('inf'))]
        range_labels = ['$0-50', '$50-100', '$100-200', '$200-500', '$500-1K', '$1K+']

        legit_counts = []
        fraud_counts = []

        for min_amt, max_amt in ranges:
            if max_amt == float('inf'):
                legit_count = len(legit_amounts[legit_amounts >= min_amt])
                fraud_count = len(fraud_amounts[fraud_amounts >= min_amt])
            else:
                legit_count = len(legit_amounts[(legit_amounts >= min_amt) & (legit_amounts < max_amt)])
                fraud_count = len(fraud_amounts[(fraud_amounts >= min_amt) & (fraud_amounts < max_amt)])

            legit_counts.append(legit_count)
            fraud_counts.append(fraud_count)

        x = np.arange(len(range_labels))
        ax5.bar(x - width/2, legit_counts, width, label='Legitimate', alpha=0.7, color='green')
        ax5.bar(x + width/2, fraud_counts, width, label='Fraud', alpha=0.7, color='red')
        ax5.set_xlabel('Amount Ranges')
        ax5.set_ylabel('Count')
        ax5.set_title('Transaction Count by Amount Range', fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(range_labels, rotation=45)
        ax5.legend()

        # Amount statistics table
        ax6 = axes[1, 2]
        stats_data = {
            'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness'],
            'Legitimate': [
                f'{legit_amounts.mean():.2f}',
                f'{legit_amounts.median():.2f}',
                f'{legit_amounts.std():.2f}',
                f'{legit_amounts.min():.2f}',
                f'{legit_amounts.max():.2f}',
                f'{legit_amounts.skew():.2f}'
            ],
            'Fraud': [
                f'{fraud_amounts.mean():.2f}',
                f'{fraud_amounts.median():.2f}',
                f'{fraud_amounts.std():.2f}',
                f'{fraud_amounts.min():.2f}',
                f'{fraud_amounts.max():.2f}',
                f'{fraud_amounts.skew():.2f}'
            ]
        }

        table_data = [[row[0], row[1], row[2]] for row in zip(stats_data['Statistic'],
                                                             stats_data['Legitimate'],
                                                             stats_data['Fraud'])]

        table = ax6.table(cellText=table_data,
                         colLabels=['Statistic', 'Legitimate', 'Fraud'],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax6.axis('off')
        ax6.set_title('Amount Statistics Comparison', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'amount_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_time_analysis(self):
        """Create time-based analysis (if Time feature exists)"""
        if 'Time' not in self.df.columns:
            print("⚠️  Time feature not found, skipping time analysis")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Time-based Transaction Analysis', fontsize=16, fontweight='bold')

        # Convert time to hours
        self.df['Hour'] = (self.df['Time'] / 3600) % 24

        # Transactions by hour
        ax1 = axes[0, 0]
        hour_counts = self.df.groupby('Hour').size()
        ax1.plot(hour_counts.index, hour_counts.values, marker='o', linewidth=2, markersize=6)
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Number of Transactions')
        ax1.set_title('Transactions by Hour of Day', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(0, 24, 2))

        # Fraud rate by hour
        ax2 = axes[0, 1]
        fraud_by_hour = self.df.groupby('Hour')['Class'].agg(['sum', 'count'])
        fraud_rate_by_hour = (fraud_by_hour['sum'] / fraud_by_hour['count']) * 100

        ax2.bar(fraud_rate_by_hour.index, fraud_rate_by_hour.values,
               alpha=0.7, color='red', edgecolor='black')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Fraud Rate (%)')
        ax2.set_title('Fraud Rate by Hour of Day', fontweight='bold')
        ax2.set_xticks(range(0, 24, 2))

        # Time distribution by class
        ax3 = axes[1, 0]
        fraud_hours = self.df[self.df['Class'] == 1]['Hour']
        legit_hours = self.df[self.df['Class'] == 0]['Hour']

        ax3.hist(legit_hours, bins=24, alpha=0.7, label='Legitimate',
                color='green', density=True)
        ax3.hist(fraud_hours, bins=24, alpha=0.7, label='Fraud',
                color='red', density=True)
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('Density')
        ax3.set_title('Time Distribution by Class', fontweight='bold')
        ax3.legend()
        ax3.set_xticks(range(0, 24, 2))

        # Time vs Amount scatter
        ax4 = axes[1, 1]
        sample_size = min(5000, len(self.df))  # Sample for performance
        sample_df = self.df.sample(n=sample_size, random_state=42)

        fraud_sample = sample_df[sample_df['Class'] == 1]
        legit_sample = sample_df[sample_df['Class'] == 0]

        ax4.scatter(legit_sample['Hour'], legit_sample['Amount'],
                   alpha=0.6, s=20, label='Legitimate', color='green')
        ax4.scatter(fraud_sample['Hour'], fraud_sample['Amount'],
                   alpha=0.8, s=30, label='Fraud', color='red')
        ax4.set_xlabel('Hour of Day')
        ax4.set_ylabel('Transaction Amount')
        ax4.set_title('Time vs Amount Scatter Plot', fontweight='bold')
        ax4.legend()
        ax4.set_ylim(0, 1000)  # Focus on lower amounts

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'time_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_feature_distributions(self):
        """Create feature distribution analysis"""
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('Feature Distribution Analysis', fontsize=16, fontweight='bold')

        # Select key features for analysis
        key_features = ['Amount'] + [col for col in self.df.columns if col.startswith('V')][:5]

        for i, feature in enumerate(key_features):
            if i >= 6:  # Only plot first 6 features
                break

            row = i // 2
            col = i % 2
            ax = axes[row, col]

            fraud_data = self.df[self.df['Class'] == 1][feature]
            legit_data = self.df[self.df['Class'] == 0][feature]

            # Create distribution plots
            ax.hist(legit_data, bins=50, alpha=0.7, label='Legitimate',
                   color='green', density=True)
            ax.hist(fraud_data, bins=50, alpha=0.7, label='Fraud',
                   color='red', density=True)

            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
            ax.set_title(f'{feature} Distribution by Class', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add statistical annotations
            legit_mean = legit_data.mean()
            fraud_mean = fraud_data.mean()
            ax.axvline(legit_mean, color='green', linestyle='--', alpha=0.8,
                      label=f'Legit Mean: {legit_mean:.2f}')
            ax.axvline(fraud_mean, color='red', linestyle='--', alpha=0.8,
                      label=f'Fraud Mean: {fraud_mean:.2f}')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_outlier_analysis(self):
        """Create outlier analysis visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Outlier Analysis', fontsize=16, fontweight='bold')

        # Amount outliers
        ax1 = axes[0, 0]
        amounts = self.df['Amount']
        Q1 = amounts.quantile(0.25)
        Q3 = amounts.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = amounts[(amounts < lower_bound) | (amounts > upper_bound)]
        normal = amounts[(amounts >= lower_bound) & (amounts <= upper_bound)]

        ax1.scatter(range(len(normal)), normal, alpha=0.6, s=10,
                   label=f'Normal ({len(normal)})', color='blue')
        ax1.scatter(range(len(normal), len(normal) + len(outliers)), outliers,
                   alpha=0.8, s=20, label=f'Outliers ({len(outliers)})', color='red')
        ax1.set_xlabel('Transaction Index')
        ax1.set_ylabel('Amount')
        ax1.set_title('Amount Outliers Detection', fontweight='bold')
        ax1.legend()

        # Outlier percentage by feature
        ax2 = axes[0, 1]
        features_to_check = ['Amount'] + [col for col in self.df.columns if col.startswith('V')][:10]
        outlier_percentages = []

        for feature in features_to_check:
            data = self.df[feature]
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers_count = len(data[(data < lower_bound) | (data > upper_bound)])
            outlier_percentage = (outliers_count / len(data)) * 100
            outlier_percentages.append(outlier_percentage)

        bars = ax2.bar(range(len(features_to_check)), outlier_percentages,
                      color='orange', alpha=0.7)
        ax2.set_xlabel('Features')
        ax2.set_ylabel('Outlier Percentage (%)')
        ax2.set_title('Outlier Percentage by Feature', fontweight='bold')
        ax2.set_xticks(range(len(features_to_check)))
        ax2.set_xticklabels(features_to_check, rotation=45)

        # Add value labels
        for bar, value in zip(bars, outlier_percentages):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=9)

        # Fraud vs outliers correlation
        ax3 = axes[1, 0]
        amount_outliers_mask = (amounts < lower_bound) | (amounts > upper_bound)
        fraud_in_outliers = self.df[amount_outliers_mask]['Class'].sum()
        fraud_in_normal = self.df[~amount_outliers_mask]['Class'].sum()

        categories = ['Amount Outliers', 'Normal Amounts']
        fraud_counts = [fraud_in_outliers, fraud_in_normal]
        total_counts = [amount_outliers_mask.sum(), (~amount_outliers_mask).sum()]
        fraud_rates = [(f/t)*100 for f, t in zip(fraud_counts, total_counts)]

        bars = ax3.bar(categories, fraud_rates, color=['red', 'green'], alpha=0.7)
        ax3.set_ylabel('Fraud Rate (%)')
        ax3.set_title('Fraud Rate: Outliers vs Normal', fontweight='bold')

        # Add value labels
        for bar, rate, count in zip(bars, fraud_rates, fraud_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{rate:.1f}%\n({count} frauds)', ha='center', va='bottom', fontweight='bold')

        # Box plot comparison
        ax4 = axes[1, 1]
        fraud_amounts = self.df[self.df['Class'] == 1]['Amount']
        legit_amounts = self.df[self.df['Class'] == 0]['Amount']

        box_data = [legit_amounts, fraud_amounts]
        box_plot = ax4.boxplot(box_data, labels=['Legitimate', 'Fraud'], patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightgreen')
        box_plot['boxes'][1].set_facecolor('lightcoral')

        ax4.set_ylabel('Transaction Amount')
        ax4.set_title('Amount Distribution Box Plot by Class', fontweight='bold')
        ax4.set_ylim(0, 1000)  # Focus on lower amounts

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'outlier_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to run advanced visualizations"""
    print("🎨 ADVANCED DATA VISUALIZATIONS FOR FRAUD DETECTION")
    print("=" * 60)

    # Ensure required directories exist
    Config.ensure_directories()

    # Validate dataset
    is_valid, dataset_path, error_message = Config.validate_dataset()
    if not is_valid:
        print(f"❌ {error_message}")
        print("Please ensure the dataset is placed in the 'data/' directory")
        return False

    try:
        # Initialize visualizer
        visualizer = AdvancedDataVisualizer(dataset_path)

        # Load data
        visualizer.load_data()

        # Create comprehensive visualizations
        visualizer.create_comprehensive_eda()

        print(f"\n🎉 ADVANCED VISUALIZATIONS COMPLETED!")
        print(f"📁 All plots saved in: {visualizer.plots_dir}")
        print(f"\n📊 Generated visualizations:")
        print("  • Dataset Overview Analysis")
        print("  • Class Distribution Analysis")
        print("  • Feature Correlation Analysis")
        print("  • PCA Components Analysis")
        print("  • Transaction Amount Analysis")
        print("  • Time-based Analysis")
        print("  • Feature Distribution Analysis")
        print("  • Outlier Analysis")

        return True

    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()