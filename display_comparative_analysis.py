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
    
    # Read all sheets
    metrics_df = pd.read_excel(excel_path, sheet_name="Comparative Metrics")
    performance_df = pd.read_excel(excel_path, sheet_name="Performance Summary")
    summary_df = pd.read_excel(excel_path, sheet_name="Summary Statistics")
    
    return metrics_df, performance_df, summary_df

def plot_metric_comparison(metrics_df, metric_name):
    """Create a bar plot comparing a specific metric across cities."""
    plt.figure(figsize=(12, 6))
    sns.barplot(data=metrics_df, x='City', y=metric_name)
    plt.title(f'Comparison of {metric_name} Across Cities')
    plt.xlabel('City')
    plt.ylabel(metric_name)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()

def plot_radar_chart(metrics_df):
    """Create a radar chart comparing all metrics across cities."""
    # Get metrics and cities
    metrics = metrics_df.columns[1:]  # Exclude 'City' column
    cities = metrics_df['City'].values
    
    # Number of metrics
    num_metrics = len(metrics)
    
    # Compute angle for each metric
    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False)
    
    # Close the plot by appending first value
    angles = np.concatenate((angles, [angles[0]]))
    
    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Plot data for each city
    for idx, city in enumerate(cities):
        values = metrics_df[metrics_df['City'] == city].values[0][1:]
        values = np.concatenate((values, [values[0]]))  # Close the plot
        ax.plot(angles, values, 'o-', linewidth=2, label=city)
        ax.fill(angles, values, alpha=0.25)
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Performance Metrics Across Cities")
    
    plt.tight_layout()
    return plt.gcf()

def plot_summary_statistics(summary_df):
    """Create a heatmap of summary statistics."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(summary_df.set_index('Metric'), annot=True, fmt='.4f', cmap='YlOrRd')
    plt.title('Summary Statistics Heatmap')
    plt.tight_layout()
    return plt.gcf()

def display_performance_summary(performance_df):
    """Display performance summary in a formatted table."""
    print("\nPerformance Summary:")
    print("=" * 80)
    print(performance_df.to_string(index=False))
    print("=" * 80)

def main():
    """Main function to display comparative analysis results."""
    try:
        # Load results
        print("Loading results...")
        metrics_df, performance_df, summary_df = load_results()
        
        # Create output directory for plots
        output_dir = Path("comparative_analysis_plots")
        output_dir.mkdir(exist_ok=True)
        
        # 1. Display performance summary
        display_performance_summary(performance_df)
        
        # 2. Create and save metric comparison plots
        print("\nGenerating metric comparison plots...")
        for metric in metrics_df.columns[1:]:  # Exclude 'City' column
            fig = plot_metric_comparison(metrics_df, metric)
            fig.savefig(output_dir / f"comparison_{metric.lower().replace(' ', '_')}.png")
            plt.close(fig)
        
        # 3. Create and save radar chart
        print("Generating radar chart...")
        radar_fig = plot_radar_chart(metrics_df)
        radar_fig.savefig(output_dir / "radar_chart.png")
        plt.close(radar_fig)
        
        # 4. Create and save summary statistics heatmap
        print("Generating summary statistics heatmap...")
        heatmap_fig = plot_summary_statistics(summary_df)
        heatmap_fig.savefig(output_dir / "summary_statistics_heatmap.png")
        plt.close(heatmap_fig)
        
        print(f"\nAll plots have been saved to the '{output_dir}' directory.")
        print("\nComparative Analysis Summary:")
        print("-" * 50)
        print(f"Total number of cities analyzed: {len(metrics_df)}")
        print(f"Number of metrics compared: {len(metrics_df.columns) - 1}")
        print(f"Best performing city (R² Score): {performance_df[performance_df['Metric'] == 'Best R² Score']['City'].values[0]}")
        print(f"Best performing city (RMSE): {performance_df[performance_df['Metric'] == 'Best RMSE']['City'].values[0]}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 