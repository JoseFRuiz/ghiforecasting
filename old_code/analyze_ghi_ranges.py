"""
Analyze GHI ranges across different locations to understand scaling issues.
"""

import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from utils import CONFIG, load_data

def analyze_ghi_ranges():
    """Analyze GHI ranges for each location and the global dataset."""
    print("GHI Range Analysis")
    print("="*80)
    
    # Dictionary to store GHI statistics for each location
    location_stats = {}
    all_ghi_values = []
    
    # Analyze each location
    for city in CONFIG["data_locations"].keys():
        print(f"\n{'='*60}")
        print(f"Analyzing {city}")
        print(f"{'='*60}")
        
        # Load data
        df = load_data(CONFIG["data_locations"], city)
        
        # Get GHI statistics
        ghi_values = df['GHI'].values
        non_zero_ghi = ghi_values[ghi_values > 0]  # Exclude night time (zero values)
        
        stats = {
            'total_samples': len(ghi_values),
            'non_zero_samples': len(non_zero_ghi),
            'zero_percentage': (len(ghi_values) - len(non_zero_ghi)) / len(ghi_values) * 100,
            'ghi_min': non_zero_ghi.min() if len(non_zero_ghi) > 0 else 0,
            'ghi_max': non_zero_ghi.max() if len(non_zero_ghi) > 0 else 0,
            'ghi_mean': non_zero_ghi.mean() if len(non_zero_ghi) > 0 else 0,
            'ghi_std': non_zero_ghi.std() if len(non_zero_ghi) > 0 else 0,
            'ghi_median': np.median(non_zero_ghi) if len(non_zero_ghi) > 0 else 0,
            'ghi_95th_percentile': np.percentile(non_zero_ghi, 95) if len(non_zero_ghi) > 0 else 0,
            'ghi_99th_percentile': np.percentile(non_zero_ghi, 99) if len(non_zero_ghi) > 0 else 0
        }
        
        location_stats[city] = stats
        all_ghi_values.extend(non_zero_ghi)
        
        print(f"Total samples: {stats['total_samples']}")
        print(f"Non-zero GHI samples: {stats['non_zero_samples']}")
        print(f"Zero GHI percentage: {stats['zero_percentage']:.2f}%")
        print(f"GHI range: [{stats['ghi_min']:.2f}, {stats['ghi_max']:.2f}] W/m²")
        print(f"GHI mean: {stats['ghi_mean']:.2f} W/m²")
        print(f"GHI std: {stats['ghi_std']:.2f} W/m²")
        print(f"GHI median: {stats['ghi_median']:.2f} W/m²")
        print(f"GHI 95th percentile: {stats['ghi_95th_percentile']:.2f} W/m²")
        print(f"GHI 99th percentile: {stats['ghi_99th_percentile']:.2f} W/m²")
    
    # Analyze global dataset
    all_ghi_values = np.array(all_ghi_values)
    global_stats = {
        'total_samples': len(all_ghi_values),
        'ghi_min': all_ghi_values.min(),
        'ghi_max': all_ghi_values.max(),
        'ghi_mean': all_ghi_values.mean(),
        'ghi_std': all_ghi_values.std(),
        'ghi_median': np.median(all_ghi_values),
        'ghi_95th_percentile': np.percentile(all_ghi_values, 95),
        'ghi_99th_percentile': np.percentile(all_ghi_values, 99)
    }
    
    print(f"\n{'='*80}")
    print("GLOBAL DATASET ANALYSIS")
    print(f"{'='*80}")
    print(f"Total non-zero GHI samples: {global_stats['total_samples']}")
    print(f"Global GHI range: [{global_stats['ghi_min']:.2f}, {global_stats['ghi_max']:.2f}] W/m²")
    print(f"Global GHI mean: {global_stats['ghi_mean']:.2f} W/m²")
    print(f"Global GHI std: {global_stats['ghi_std']:.2f} W/m²")
    print(f"Global GHI median: {global_stats['ghi_median']:.2f} W/m²")
    print(f"Global GHI 95th percentile: {global_stats['ghi_95th_percentile']:.2f} W/m²")
    print(f"Global GHI 99th percentile: {global_stats['ghi_99th_percentile']:.2f} W/m²")
    
    # Create comparison table
    print(f"\n{'='*80}")
    print("LOCATION COMPARISON TABLE")
    print(f"{'='*80}")
    
    comparison_data = []
    for city, stats in location_stats.items():
        comparison_data.append({
            'Location': city,
            'GHI Min (W/m²)': stats['ghi_min'],
            'GHI Max (W/m²)': stats['ghi_max'],
            'GHI Mean (W/m²)': stats['ghi_mean'],
            'GHI Std (W/m²)': stats['ghi_std'],
            'GHI Median (W/m²)': stats['ghi_median'],
            'Zero %': stats['zero_percentage']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False, float_format='%.2f'))
    
    # Analyze scaling implications
    print(f"\n{'='*80}")
    print("SCALING IMPLICATIONS")
    print(f"{'='*80}")
    
    # Test different scaling approaches
    print("\n1. Global Scaling (Joint Model Approach):")
    global_scaler = MinMaxScaler()
    global_scaler.fit(all_ghi_values.reshape(-1, 1))
    print(f"Global scaler min: {global_scaler.data_min_[0]:.2f}")
    print(f"Global scaler max: {global_scaler.data_max_[0]:.2f}")
    
    print("\n2. Individual Scaling (Individual Model Approach):")
    for city, stats in location_stats.items():
        individual_scaler = MinMaxScaler()
        city_ghi = np.array([stats['ghi_min'], stats['ghi_max']]).reshape(-1, 1)
        individual_scaler.fit(city_ghi)
        print(f"{city}: min={individual_scaler.data_min_[0]:.2f}, max={individual_scaler.data_max_[0]:.2f}")
    
    # Test what happens when we scale and inverse scale
    print(f"\n{'='*80}")
    print("SCALING TRANSFORMATION TEST")
    print(f"{'='*80}")
    
    # Test with a sample value from each location
    test_values = [location_stats[city]['ghi_mean'] for city in location_stats.keys()]
    
    print("\nTesting scaling transformations:")
    for i, (city, test_value) in enumerate(zip(location_stats.keys(), test_values)):
        print(f"\n{city} (mean GHI: {test_value:.2f} W/m²):")
        
        # Global scaling
        global_scaled = global_scaler.transform([[test_value]])[0, 0]
        global_inverse = global_scaler.inverse_transform([[global_scaled]])[0, 0]
        print(f"  Global scaling: {test_value:.2f} → {global_scaled:.4f} → {global_inverse:.2f}")
        
        # Individual scaling
        individual_scaler = MinMaxScaler()
        city_stats = location_stats[city]
        individual_scaler.fit([[city_stats['ghi_min']], [city_stats['ghi_max']]])
        individual_scaled = individual_scaler.transform([[test_value]])[0, 0]
        individual_inverse = individual_scaler.inverse_transform([[individual_scaled]])[0, 0]
        print(f"  Individual scaling: {test_value:.2f} → {individual_scaled:.4f} → {individual_inverse:.2f}")
    
    # Save results
    results = {
        'location_stats': location_stats,
        'global_stats': global_stats,
        'global_scaler': global_scaler
    }
    
    with open('ghi_range_analysis.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nAnalysis results saved to ghi_range_analysis.pkl")
    
    return results

def check_existing_scalers():
    """Check the existing scalers to understand the current scaling."""
    print("\n" + "="*80)
    print("CHECKING EXISTING SCALERS")
    print("="*80)
    
    # Check joint model scaler
    joint_scaler_path = os.path.join("models", "target_scaler.pkl")
    if os.path.exists(joint_scaler_path):
        print(f"\nJoint model scaler found: {joint_scaler_path}")
        with open(joint_scaler_path, 'rb') as f:
            joint_scaler = pickle.load(f)
        print(f"Joint scaler data_min: {joint_scaler.data_min_[0]:.2f}")
        print(f"Joint scaler data_max: {joint_scaler.data_max_[0]:.2f}")
        print(f"Joint scaler scale_: {joint_scaler.scale_[0]:.4f}")
        print(f"Joint scaler min_: {joint_scaler.min_[0]:.4f}")
    else:
        print(f"\nJoint model scaler not found: {joint_scaler_path}")
    
    # Check individual model scalers (if they exist)
    print(f"\nChecking for individual model scalers...")
    for city in CONFIG["data_locations"].keys():
        individual_scaler_path = os.path.join("models", f"target_scaler_{city}.pkl")
        if os.path.exists(individual_scaler_path):
            print(f"Individual scaler found for {city}: {individual_scaler_path}")
            with open(individual_scaler_path, 'rb') as f:
                individual_scaler = pickle.load(f)
            print(f"  {city} scaler data_min: {individual_scaler.data_min_[0]:.2f}")
            print(f"  {city} scaler data_max: {individual_scaler.data_max_[0]:.2f}")
        else:
            print(f"No individual scaler found for {city}")

if __name__ == "__main__":
    # Analyze GHI ranges
    results = analyze_ghi_ranges()
    
    # Check existing scalers
    check_existing_scalers() 