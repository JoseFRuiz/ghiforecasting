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
        
        # Print unique locations in each dataset
        print("\nLocations in each dataset:")
        print("Individual metrics locations:", sorted(individual_metrics['Location'].unique()))
        print("Joint metrics locations:", sorted(joint_metrics['Location'].unique()))
        print("GNN metrics locations:", sorted(gnn_metrics['Location'].unique()))
        
        # Create comparison DataFrame
        comparison_records = []
        
        # Get all unique locations
        all_locations = sorted(set(individual_metrics['Location'].unique()) | 
                             set(joint_metrics['Location'].unique()) | 
                             set(gnn_metrics['Location'].unique()))
        
        # Create data availability report
        print("\nData Availability Report:")
        print("=" * 80)
        data_availability = {
            'Location': [],
            'Metric': [],
            'Feature Combination': [],
            'Individual': [],
            'Joint': [],
            'GNN': [],
            'Complete': []
        }
        
        print("\nProcessing comparisons for each location...")
        for location in all_locations:
            print(f"\nProcessing {location}...")
            
            # Get metrics for this location
            location_individual = individual_metrics[individual_metrics['Location'] == location]
            location_joint = joint_metrics[joint_metrics['Location'] == location]
            location_gnn = gnn_metrics[gnn_metrics['Location'] == location]
            
            # Print available metrics for this location
            print(f"Individual metrics available: {len(location_individual)}")
            print(f"Joint metrics available: {len(location_joint)}")
            print(f"GNN metrics available: {len(location_gnn)}")
            
            # Get all unique metrics for this location
            all_metrics = set()
            all_metrics.update(location_individual['Metric'].unique())
            all_metrics.update(location_joint['Metric'].unique())
            all_metrics.update(location_gnn['Metric'].unique())
            
            for metric in all_metrics:
                # Get data for each approach
                ind_data = location_individual[location_individual['Metric'] == metric]
                joint_data = location_joint[location_joint['Metric'] == metric]
                gnn_data = location_gnn[location_gnn['Metric'] == metric]
                
                # Get all feature combinations
                all_feature_combos = set()
                all_feature_combos.update(ind_data['Feature Combination'].unique())
                all_feature_combos.update(joint_data['Feature Combination'].unique())
                
                for feature_combo in all_feature_combos:
                    ind_row = ind_data[ind_data['Feature Combination'] == feature_combo]
                    joint_row = joint_data[joint_data['Feature Combination'] == feature_combo]
                    gnn_row = gnn_data
                    
                    # Record data availability
                    data_availability['Location'].append(location)
                    data_availability['Metric'].append(metric)
                    data_availability['Feature Combination'].append(feature_combo)
                    data_availability['Individual'].append(not ind_row.empty)
                    data_availability['Joint'].append(not joint_row.empty)
                    data_availability['GNN'].append(not gnn_row.empty)
                    data_availability['Complete'].append(not ind_row.empty and not joint_row.empty and not gnn_row.empty)
                    
                    if not ind_row.empty and not joint_row.empty and not gnn_row.empty:
                        individual_value = ind_row['Value'].iloc[0]
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
                    else:
                        print(f"Missing data for {location} - {metric} - {feature_combo}")
                        if ind_row.empty:
                            print(f"  No individual data found")
                        if joint_row.empty:
                            print(f"  No joint data found")
                        if gnn_row.empty:
                            print(f"  No GNN data found")
        
        # Create and save data availability report
        availability_df = pd.DataFrame(data_availability)
        availability_summary = availability_df.groupby(['Metric', 'Feature Combination']).agg({
            'Individual': 'sum',
            'Joint': 'sum',
            'GNN': 'sum',
            'Complete': 'sum'
        })
        
        print("\nData Availability Summary:")
        print("=" * 80)
        print(availability_summary)
        
        # Save availability report
        availability_df.to_csv(os.path.join(output_dir, "data_availability.csv"), index=False)
        availability_summary.to_csv(os.path.join(output_dir, "data_availability_summary.csv"))
        
        comparison_df = pd.DataFrame(comparison_records)
        
        # Create summary statistics
        print("\nCreating summary statistics...")
        
        # First, get the total number of comparisons made
        total_comparisons = len(comparison_df)
        print(f"Total number of comparisons made: {total_comparisons}")
        
        # Calculate summary statistics
        summary = comparison_df.groupby(['Metric', 'Feature Combination'])['Better Approach'].agg([
            ('Individual', lambda x: (x == 'individual').sum()),
            ('Joint', lambda x: (x == 'joint').sum()),
            ('GNN', lambda x: (x == 'gnn').sum()),
            ('Equal', lambda x: (x == 'equal').sum())
        ])
        
        # Calculate percentages based on total number of locations
        total_locations = len(all_locations)
        summary['Total'] = total_locations
        summary['Comparisons Made'] = summary.sum(axis=1) - summary['Total']  # Subtract the 'Total' column
        summary['Individual %'] = (summary['Individual'] / summary['Comparisons Made'] * 100).round(1)
        summary['Joint %'] = (summary['Joint'] / summary['Comparisons Made'] * 100).round(1)
        summary['GNN %'] = (summary['GNN'] / summary['Comparisons Made'] * 100).round(1)
        summary['Equal %'] = (summary['Equal'] / summary['Comparisons Made'] * 100).round(1)
        
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
        print(f"Total number of locations: {total_locations}")
        print(f"Total number of comparisons made: {total_comparisons}")
        
        for metric in comparison_df['Metric'].unique():
            metric_data = comparison_df[comparison_df['Metric'] == metric]
            total_comparisons = len(metric_data)
            individual_wins = (metric_data['Better Approach'] == 'individual').sum()
            joint_wins = (metric_data['Better Approach'] == 'joint').sum()
            gnn_wins = (metric_data['Better Approach'] == 'gnn').sum()
            equal = (metric_data['Better Approach'] == 'equal').sum()
            
            print(f"\nMetric: {metric}")
            print(f"Total comparisons for this metric: {total_comparisons}")
            print(f"Individual Training Better: {individual_wins}/{total_comparisons} ({individual_wins/total_comparisons*100:.1f}%)")
            print(f"Joint Training Better: {joint_wins}/{total_comparisons} ({joint_wins/total_comparisons*100:.1f}%)")
            print(f"GNN Training Better: {gnn_wins}/{total_comparisons} ({gnn_wins/total_comparisons*100:.1f}%)")
            print(f"Equal Performance: {equal}/{total_comparisons} ({equal/total_comparisons*100:.1f}%)")
        
        print("\nResults have been saved in the 'results_comparison' directory:")
        print("1. detailed_comparison.csv - Detailed comparison of all metrics")
        print("2. summary_statistics.csv - Summary statistics by metric and feature combination")
        print("3. data_availability.csv - Detailed data availability report")
        print("4. data_availability_summary.csv - Summary of data availability")
        print("5. plots/ - Comparison plots for each metric")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 