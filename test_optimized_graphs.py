#!/usr/bin/env python3
# test_optimized_graphs.py - Test the optimized graph building functions

import sys
import time
print("Testing optimized graph building functions...")

# Import what we need
from utils import CONFIG, load_data
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.preprocessing import MinMaxScaler
from spektral.data import Graph

def create_features(df):
    """Full version of create_features with 24 lags"""
    print(f"    Creating features for dataframe with {len(df)} rows...")
    
    print(f"      Adding time-based features...")
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    print(f"      Adding GHI lag features...")
    for lag in range(1, 25):
        df[f"GHI_lag_{lag}"] = df["GHI"].shift(lag)
        if lag % 6 == 0:
            print(f"        Added {lag}/24 GHI lag features")

    print(f"      Adding meteorological lag features...")
    met_vars = ["Temperature", "Relative Humidity", "Pressure",
                "Precipitable Water", "Wind Direction", "Wind Speed"]
    for i, var in enumerate(met_vars):
        df[f"{var}_lag_24"] = df[var].shift(24)
        print(f"        Added lag feature for {var} ({i+1}/{len(met_vars)})")

    print(f"      Adding target variable...")
    df["target"] = df["GHI"].shift(-24)  # forecast 24 hours ahead
    
    print(f"      Dropping NaN values...")
    original_len = len(df)
    df = df.dropna().reset_index(drop=True)
    final_len = len(df)
    print(f"      Dropped {original_len - final_len} rows with NaN values")
    print(f"      Final shape: {df.shape}")
    
    return df

def compute_weighted_adjacency_optimized(df_all, alpha=0.5):
    print("    Computing weighted adjacency matrix (optimized)...")
    city_coords = {
        "Jaisalmer": (26.9157, 70.9083),
        "Jodhpur": (26.2389, 73.0243),
        "New Delhi": (28.6139, 77.2090),
        "Shimla": (31.1048, 77.1734),
        "Srinagar": (34.0837, 74.7973),
    }
    cities = list(city_coords.keys())
    n = len(cities)

    print(f"      Computing geodesic distances for {n} cities...")
    # Geodesic proximity (1 / distance) - vectorized
    geo_weights = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i != j:
                d = geodesic(city_coords[cities[i]], city_coords[cities[j]]).km
                geo_weights[i, j] = 1 / d

    geo_weights /= geo_weights.max()

    print(f"      Computing correlations for {n} cities...")
    # Correlation - optimized with pre-computed time series
    corr_weights = np.zeros((n, n), dtype=np.float32)
    
    # Pre-compute GHI time series for all cities
    ghi_series = {}
    for i, city in enumerate(cities):
        city_data = df_all[df_all['location'] == city].set_index('datetime')['GHI']
        ghi_series[city] = city_data
    
    # Compute correlations more efficiently
    for i, ci in enumerate(cities):
        ghi_i = ghi_series[ci]
        for j, cj in enumerate(cities):
            if i != j:  # Skip self-correlation
                ghi_j = ghi_series[cj]
                # Find common time indices
                common = ghi_i.index.intersection(ghi_j.index)
                if len(common) > 100:  # Only compute if we have enough data
                    corr = ghi_i.loc[common].corr(ghi_j.loc[common])
                    corr_weights[i, j] = corr if not np.isnan(corr) else 0

    corr_weights = np.clip(corr_weights, 0, 1)
    adj = alpha * geo_weights + (1 - alpha) * corr_weights
    np.fill_diagonal(adj, 1.0)
    print(f"      Adjacency matrix computed: {adj.shape}")
    return adj

def build_daily_graphs_optimized(df_all, adj_matrix):
    print("    Starting to build daily graphs (optimized)...")
    graphs = []
    targets = []

    print("    Fitting scaler...")
    scaler = MinMaxScaler()
    df_all['GHI_scaled'] = scaler.fit_transform(df_all[["GHI"]])

    # Pre-compute feature columns once
    print("    Pre-computing feature columns...")
    sample_data = df_all[df_all['location'] == CITIES[0]].iloc[:12]
    feature_cols = [col for col in sample_data.columns if 'lag' in col or 'sin' in col or 'cos' in col]
    print(f"    Found {len(feature_cols)} feature columns")

    # Group by date and process each day more efficiently
    df_all['date'] = pd.to_datetime(df_all["datetime"]).dt.date
    dates = df_all['date'].unique()
    
    print(f"    Processing {len(dates)} unique dates")
    
    # Pre-filter data to only include dates with sufficient data
    print("    Pre-filtering valid dates...")
    valid_dates = []
    for date in dates:
        # Check if all cities have at least 24 hours of data for this date
        all_cities_valid = True
        for city in CITIES:
            city_data = df_all[(df_all["location"] == city) & (df_all['date'] == date)]
            if len(city_data) < 24:
                all_cities_valid = False
                break
        if all_cities_valid:
            valid_dates.append(date)
    
    print(f"    Found {len(valid_dates)} valid dates out of {len(dates)} total dates")
    
    # Process valid dates in batches for better performance
    batch_size = 50
    for batch_start in range(0, len(valid_dates), batch_size):
        batch_end = min(batch_start + batch_size, len(valid_dates))
        batch_dates = valid_dates[batch_start:batch_end]
        
        print(f"      Processing batch {batch_start//batch_size + 1}/{(len(valid_dates) + batch_size - 1)//batch_size}: dates {batch_start+1}-{batch_end}")
        
        for date in batch_dates:
            node_features = []
            node_targets = []
            
            # Process all cities for this date
            for city in CITIES:
                # Get data for this city and date
                df_city = df_all[(df_all["location"] == city) & (df_all['date'] == date)].sort_values('datetime')
                
                # Use first 12 hours for sequence, next 12 hours for target
                sequence_data = df_city.iloc[:12]
                target_data = df_city.iloc[12:24]
                
                # Check if target has any non-zero values (but be less strict)
                if (target_data['GHI'] <= 0).sum() >= 11:  # Allow at least 1 non-zero value
                    continue  # Skip this city for this date
                
                # Extract features more efficiently
                X = sequence_data[feature_cols].values.flatten()
                
                # Check for NaN values in features
                if np.isnan(X).any():
                    continue  # Skip this city for this date
                
                # Target is the GHI values for the next 12 hours
                y = target_data['GHI'].values
                
                # Check for NaN values in targets
                if np.isnan(y).any():
                    continue  # Skip this city for this date
                
                node_features.append(X)
                node_targets.append(y)
            
            # Only create graph if we have data from all cities
            if len(node_features) == len(CITIES):
                x = np.stack(node_features, axis=0)
                # Create a single target per graph by averaging across cities
                y = np.mean(node_targets, axis=0)
                
                graphs.append(Graph(x=x, a=adj_matrix))
                targets.append(y)

    print(f"    Total valid daily graphs: {len(graphs)}")
    
    return graphs, np.array(targets), scaler

# Test with all cities
CITIES = list(CONFIG["data_locations"].keys())

print(f"Testing optimized functions with all cities: {CITIES}")

# Load and process all cities
all_dfs = []
for i, city in enumerate(CITIES):
    print(f"Loading data for city {i+1}/{len(CITIES)}: {city}")
    start_time = time.time()
    df = load_data(CONFIG["data_locations"], city)
    load_time = time.time() - start_time
    print(f"  Data loaded in {load_time:.2f} seconds: {df.shape}")
    
    print(f"  Creating features for {city}...")
    start_time = time.time()
    df = create_features(df)
    feature_time = time.time() - start_time
    print(f"  Features created in {feature_time:.2f} seconds: {df.shape}")
    
    df['location'] = city
    all_dfs.append(df)
    print(f"  ✓ {city} data processed")

print("Concatenating all city data...")
start_time = time.time()
df_all = pd.concat(all_dfs).sort_values("datetime")
concat_time = time.time() - start_time
print(f"Combined data shape: {df_all.shape} in {concat_time:.2f} seconds")
print(f"Total memory usage: {df_all.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

print("\nComputing adjacency matrix (optimized)...")
start_time = time.time()
adj_matrix = compute_weighted_adjacency_optimized(df_all, alpha=0.5)
adj_time = time.time() - start_time
print(f"Adjacency matrix computed in {adj_time:.2f} seconds: {adj_matrix.shape}")

print("\nBuilding daily graphs (optimized)...")
start_time = time.time()
graphs, targets, ghi_scaler = build_daily_graphs_optimized(df_all, adj_matrix)
graph_time = time.time() - start_time
print(f"Graphs built in {graph_time:.2f} seconds: {len(graphs)} graphs")

print("\n✓ Optimized graph building test completed successfully!")
print(f"Total time: {load_time + feature_time + concat_time + adj_time + graph_time:.2f} seconds")
print(f"Final dataset: {len(graphs)} graphs with {len(targets)} targets")
print(f"Graph building time: {graph_time:.2f} seconds (should be much faster than 62.61 seconds)") 