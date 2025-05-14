"""
Compare results from individual, joint, and GNN training approaches.
This script analyzes metrics from all approaches and determines which performs better.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_metrics(file_path):
    """Load metrics from CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metrics file not found: {file_path}")
    return pd.read_csv(file_path)

def determine_better_approach(metric_name, individual_value, joint_value, gnn_value):
    """
    Determine which approach is better based on the metric.
    Returns: 'individual', 'joint', 'gnn', or 'equal'
    """
    # For error metrics, lower is better
    if any(term in metric_name.lower() for term in ['error', 'loss', 'mse', 'mae', 'rmse']):
        values = {
            'individual': individual_value,
            'joint': joint_value,
            'gnn': gnn_value
        }
        min_value = min(values.values())
        best_approaches = [k for k, v in values.items() if v == min_value]
        return best_approaches[0] if len(best_approaches) == 1 else 'equal'
    # For other metrics (like R²), higher is better
    else:
        values = {
            'individual': individual_value,
            'joint': joint_value,
            'gnn': gnn_value
        }
        max_value = max(values.values())
        best_approaches = [k for k, v in values.items() if v == max_value]
        return best_approaches[0] if len(best_approaches) == 1 else 'equal'

def create_comparison_plots(comparison_df, output_dir):
    """Create comparison plots for each metric."""
    os.makedirs(output_dir, exist_ok=True)
    
    for metric in comparison_df['Metric'].unique():
        metric_data = comparison_df[comparison_df['Metric'] == metric]
        
        # Create bar plot
        plt.figure(figsize=(15, 8))
        
        # Get unique locations and feature combinations
        locations = sorted(metric_data['Location'].unique())
        feature_combinations = sorted(metric_data['Feature Combination'].unique())
        
        # Set up the bar positions
        x = np.arange(len(locations))
        width = 0.25  # Adjusted for three bars
        
        # Create subplot for each feature combination
        fig, ax = plt.subplots(figsize=(15, 8))
        
        for i, feature_combo in enumerate(feature_combinations):
            combo_data = metric_data[metric_data['Feature Combination'] == feature_combo]
            
            # Plot bars
            individual_bars = ax.bar(x - width, combo_data['Individual Value'], width, 
                                   label=f'Individual ({feature_combo})')
            joint_bars = ax.bar(x, combo_data['Joint Value'], width, 
                               label=f'Joint ({feature_combo})')
            gnn_bars = ax.bar(x + width, combo_data['GNN Value'], width,
                             label=f'GNN ({feature_combo})')
            
            # Add value labels
            for bars in [individual_bars, joint_bars, gnn_bars]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom')
        
        # Customize plot
        ax.set_xlabel('Location')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison: Individual vs Joint vs GNN Training')
        ax.set_xticks(x)
        ax.set_xticklabels(locations, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f'comparison_{metric.lower().replace(" ", "_")}.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()

def main():
    """Main execution function."""
    # Define paths
    individual_metrics_path = "results/tables/metrics.csv"
    joint_metrics_path = "results_joint/tables/joint_metrics.csv"
    gnn_metrics_path = "results_gnn/tables/gnn_metrics.csv"
    output_dir = "results_comparison"
    
    try:
        # Load metrics
        print("Loading metrics...")
        individual_metrics = load_metrics(individual_metrics_path)
        joint_metrics = load_metrics(joint_metrics_path)
        gnn_metrics = load_metrics(gnn_metrics_path)
        
        # Create comparison DataFrame
        comparison_records = []
        
        for _, row in individual_metrics.iterrows():
            location = row['Location']
            feature_combo = row['Feature Combination']
            metric = row['Metric']
            individual_value = row['Value']
            
            # Find corresponding joint metric
            joint_row = joint_metrics[
                (joint_metrics['Location'] == location) &
                (joint_metrics['Feature Combination'] == feature_combo) &
                (joint_metrics['Metric'] == metric)
            ]
            
            # Find corresponding GNN metric
            gnn_row = gnn_metrics[
                (gnn_metrics['Location'] == location) &
                (gnn_metrics['Metric'] == metric)
            ]
            
            if not joint_row.empty and not gnn_row.empty:
                joint_value = joint_row['Value'].iloc[0]
                gnn_value = gnn_row['Value'].iloc[0]
                better_approach = determine_better_approach(metric, individual_value, joint_value, gnn_value)
                
                comparison_records.append({
                    'Location': location,
                    'Feature Combination': feature_combo,
                    'Metric': metric,
                    'Individual Value': individual_value,
                    'Joint Value': joint_value,
                    'GNN Value': gnn_value,
                    'Better Approach': better_approach
                })
        
        comparison_df = pd.DataFrame(comparison_records)
        
        # Create summary statistics
        summary = comparison_df.groupby(['Metric', 'Feature Combination'])['Better Approach'].agg([
            ('Individual', lambda x: (x == 'individual').sum()),
            ('Joint', lambda x: (x == 'joint').sum()),
            ('GNN', lambda x: (x == 'gnn').sum()),
            ('Equal', lambda x: (x == 'equal').sum())
        ])
        
        # Calculate percentages
        summary['Total'] = summary.sum(axis=1)
        summary['Individual %'] = (summary['Individual'] / summary['Total'] * 100).round(1)
        summary['Joint %'] = (summary['Joint'] / summary['Total'] * 100).round(1)
        summary['GNN %'] = (summary['GNN'] / summary['Total'] * 100).round(1)
        summary['Equal %'] = (summary['Equal'] / summary['Total'] * 100).round(1)
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        comparison_df.to_csv(os.path.join(output_dir, "detailed_comparison.csv"), index=False)
        summary.to_csv(os.path.join(output_dir, "summary_statistics.csv"))
        
        # Create comparison plots
        print("Creating comparison plots...")
        create_comparison_plots(comparison_df, os.path.join(output_dir, "plots"))
        
        # Print summary
        print("\nSummary of Results:")
        print("=" * 80)
        for metric in comparison_df['Metric'].unique():
            metric_data = comparison_df[comparison_df['Metric'] == metric]
            individual_wins = (metric_data['Better Approach'] == 'individual').sum()
            joint_wins = (metric_data['Better Approach'] == 'joint').sum()
            gnn_wins = (metric_data['Better Approach'] == 'gnn').sum()
            equal = (metric_data['Better Approach'] == 'equal').sum()
            total = len(metric_data)
            
            print(f"\nMetric: {metric}")
            print(f"Individual Training Better: {individual_wins}/{total} ({individual_wins/total*100:.1f}%)")
            print(f"Joint Training Better: {joint_wins}/{total} ({joint_wins/total*100:.1f}%)")
            print(f"GNN Training Better: {gnn_wins}/{total} ({gnn_wins/total*100:.1f}%)")
            print(f"Equal Performance: {equal}/{total} ({equal/total*100:.1f}%)")
        
        print("\nResults have been saved in the 'results_comparison' directory:")
        print("1. detailed_comparison.csv - Detailed comparison of all metrics")
        print("2. summary_statistics.csv - Summary statistics by metric and feature combination")
        print("3. plots/ - Comparison plots for each metric")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 