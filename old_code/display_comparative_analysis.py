"""
Script to display comparative analysis results for GHI forecasting across different cities.
This script reads the saved results and creates interactive visualizations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_results():
    """Load all results from the results directory."""
    results_dir = Path("results")
    if not results_dir.exists():
        raise FileNotFoundError("Results directory not found. Please run train.py first.")
    
    # Load Excel file with all results
    excel_path = results_dir / "comparative_analysis_results.xlsx"
    if not excel_path.exists():
        raise FileNotFoundError("Comparative analysis results not found. Please run train.py first.")
    
    # Read sheets with correct names
    comparative_metrics = pd.read_excel(excel_path, sheet_name="Comparative Metrics")
    feature_summary = pd.read_excel(excel_path, sheet_name="Feature Summary")
    
    # Clean up the data: forward fill the Metric column to replace NaN values
    comparative_metrics['Metric'] = comparative_metrics['Metric'].ffill()
    
    # Print DataFrame info for debugging
    print("\nAvailable columns in comparative_metrics:")
    print(comparative_metrics.columns.tolist())
    print("\nFirst few rows of comparative_metrics:")
    print(comparative_metrics.head())
    
    # Convert Metric column to string type
    comparative_metrics['Metric'] = comparative_metrics['Metric'].astype(str)
    
    return comparative_metrics, feature_summary

def plot_feature_comparison_by_metric(df, metric_name):
    """Create a grouped bar plot comparing feature combinations across cities for a specific metric."""
    plt.figure(figsize=(15, 8))
    
    # Filter data for the specific metric
    metric_data = df[df['Metric'] == metric_name]
    
    # Get city columns (all columns except 'Metric' and 'Feature Combination')
    city_columns = [col for col in df.columns if col not in ['Metric', 'Feature Combination']]
    
    # Set Feature Combination as index
    metric_data = metric_data.set_index('Feature Combination')[city_columns]
    
    # Create bar plot
    ax = metric_data.plot(kind='bar', width=0.8)
    plt.title(f'{metric_name} Comparison Across Cities and Feature Combinations')
    plt.xlabel('Feature Combination')
    plt.ylabel(metric_name)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='City', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', rotation=90, padding=3)
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return plt.gcf()

def plot_heatmap_by_city(df, city, metric_subset=None):
    """Create a heatmap for a specific city showing metrics vs feature combinations."""
    # Ensure we have unique combinations of Metric and Feature Combination
    city_data = df.drop_duplicates(subset=['Metric', 'Feature Combination'])
    
    # Create the pivot table
    city_data = city_data.pivot(
        index='Feature Combination',
        columns='Metric',
        values=city  # Use city name directly as the values column
    )
    
    # Select subset of metrics if specified
    if metric_subset:
        available_metrics = [m for m in metric_subset if m in city_data.columns]
        if available_metrics:
            city_data = city_data[available_metrics]
        else:
            print(f"Warning: No specified metrics found for {city}")
            return None
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(city_data, annot=True, fmt='.3f', cmap='YlOrRd', center=0)
    plt.title(f'Performance Metrics for {city}')
    plt.xlabel('Metrics')
    plt.ylabel('Feature Combination')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return plt.gcf()

def create_feature_ranking_table(df):
    """Create a table ranking feature combinations by each metric."""
    rankings = []
    
    for metric in df['Metric'].unique():
        metric_data = df[df['Metric'] == metric]
        
        # Get city columns
        city_columns = [col for col in df.columns if col not in ['Metric', 'Feature Combination']]
        
        # Calculate mean performance across cities for each feature combination
        feature_means = metric_data.groupby('Feature Combination')[city_columns].mean().mean(axis=1)
        
        # Determine if higher or lower values are better based on metric name
        if 'Error' in str(metric) or 'Loss' in str(metric):
            feature_ranks = feature_means.rank()  # Lower is better
            best_feature = feature_means.idxmin()
            best_value = feature_means.min()
        else:  # For metrics like R² Score, higher is better
            feature_ranks = feature_means.rank(ascending=False)  # Higher is better
            best_feature = feature_means.idxmax()
            best_value = feature_means.max()
        
        rankings.append({
            'Metric': metric,
            'Best Feature Combination': best_feature,
            'Best Value': best_value,
            'All Rankings': dict(feature_ranks)
        })
    
    return pd.DataFrame(rankings)

def main():
    """Main function to display comparative analysis results."""
    try:
        # Load results
        print("Loading results...")
        comparative_metrics, feature_summary = load_results()
        
        # Create output directory for plots
        output_dir = Path("comparative_analysis_plots")
        output_dir.mkdir(exist_ok=True)
        
        # Get unique metrics and subset for key metrics
        all_metrics = comparative_metrics['Metric'].unique()
        key_metrics = [m for m in all_metrics if isinstance(m, str) and 
                      any(term in m for term in ['R² Score', 'RMSE', 'MAE'])]
        
        if not key_metrics:  # If no string metrics found, use all metrics
            key_metrics = all_metrics
        
        # Get city names from columns (excluding 'Metric' and 'Feature Combination')
        cities = [col for col in comparative_metrics.columns 
                 if col not in ['Metric', 'Feature Combination']]
        
        # 1. Create feature comparison plots for each metric
        print("\nGenerating feature comparison plots...")
        for metric in all_metrics:
            fig = plot_feature_comparison_by_metric(comparative_metrics, metric)
            # Use str() to safely convert metric name to string for filename
            safe_metric_name = str(metric).lower().replace(' ', '_').replace('²', '2')
            fig.savefig(output_dir / f"feature_comparison_{safe_metric_name}.png",
                       bbox_inches='tight', dpi=300)
            plt.close(fig)
        
        # 2. Create heatmaps for each city
        print("\nGenerating city-wise heatmaps...")
        for city in cities:
            fig = plot_heatmap_by_city(comparative_metrics, city, key_metrics)
            if fig is not None:
                fig.savefig(output_dir / f"heatmap_{city.lower().replace(' ', '_')}.png",
                           bbox_inches='tight', dpi=300)
                plt.close(fig)
        
        # 3. Create and save feature ranking analysis
        print("\nGenerating feature ranking analysis...")
        rankings = create_feature_ranking_table(comparative_metrics)
        rankings.to_csv(output_dir / "feature_rankings.csv", index=False)
        
        # 4. Display summary of best performing feature combinations
        print("\nBest Performing Feature Combinations:")
        print("=" * 80)
        for _, row in rankings.iterrows():
            print(f"\nMetric: {row['Metric']}")
            print(f"Best Feature Combination: {row['Best Feature Combination']}")
            print(f"Best Value: {row['Best Value']:.4f}")
        print("=" * 80)
        
        # 5. Save detailed analysis to Excel
        print("\nSaving detailed analysis...")
        with pd.ExcelWriter(output_dir / "feature_analysis.xlsx") as writer:
            rankings.to_excel(writer, sheet_name="Feature Rankings", index=False)
            feature_summary.to_excel(writer, sheet_name="Feature Summary", index=False)
            
            # Save the data directly without pivoting since it's already in the right format
            comparative_metrics.to_excel(writer, sheet_name="Detailed Comparison", index=False)
        
        print(f"\nAll analysis results have been saved to the '{output_dir}' directory.")
        print("\nFiles generated:")
        print("1. Feature comparison plots for each metric")
        print("2. Heatmaps for each city")
        print("3. feature_rankings.csv - Ranking of feature combinations")
        print("4. feature_analysis.xlsx - Detailed analysis workbook")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 