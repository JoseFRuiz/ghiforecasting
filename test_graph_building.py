#!/usr/bin/env python3
# test_graph_building.py - Test the graph building phase

import sys
import time
print("Testing graph building phase...")

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

def compute_weighted_adjacency(df_all, alpha=0.5):
    print("    Computing weighted adjacency matrix...")
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
    # Geodesic proximity (1 / distance)
    geo_weights = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i != j:
                d = geodesic(city_coords[cities[i]], city_coords[cities[j]]).km
                geo_weights[i, j] = 1 / d

    geo_weights /= geo_weights.max()

    print(f"      Computing correlations for {n} cities...")
    # Correlation
    corr_weights = np.zeros((n, n), dtype=np.float32)
    for i, ci in enumerate(cities):
        ghi_i = df_all[df_all['location'] == ci].set_index('datetime')['GHI']
        for j, cj in enumerate(cities):
            ghi_j = df_all[df_all['location'] == cj].set_index('datetime')['GHI']
            common = ghi_i.index.intersection(ghi_j.index)
            if len(common) > 0:
                corr = ghi_i.loc[common].corr(ghi_j.loc[common])
                corr_weights[i, j] = corr if not np.isnan(corr) else 0

    corr_weights = np.clip(corr_weights, 0, 1)
    adj = alpha * geo_weights + (1 - alpha) * corr_weights
    np.fill_diagonal(adj, 1.0)
    print(f"      Adjacency matrix computed: {adj.shape}")
    return adj

def build_daily_graphs(df_all, adj_matrix):
    print("    Starting to build daily graphs...")
    graphs = []
    targets = []

    print("    Fitting scaler...")
    scaler = MinMaxScaler()
    df_all['GHI_scaled'] = scaler.fit_transform(df_all[["GHI"]])

    # Group by date and process each day
    df_all['date'] = pd.to_datetime(df_all["datetime"]).dt.date
    dates = df_all['date'].unique()
    
    print(f"    Processing {len(dates)} unique dates")
    
    # Count how many dates are skipped and why
    skipped_reasons = {}
    
    CITIES = list(CONFIG["data_locations"].keys())
    
    for i, date in enumerate(dates):
        if i % 50 == 0:
            print(f"      Processing date {i+1}/{len(dates)}: {date}")
        
        node_features = []
        node_targets = []
        valid = True
        skip_reason = None
        
        for city in CITIES:
            # Get data for this city and date
            df_city = df_all[(df_all["location"] == city) & (df_all['date'] == date)]
            
            if len(df_city) < 24:  # Need at least 24 hours
                skip_reason = f"not enough data ({len(df_city)} rows, need 24)"
                valid = False
                break
            
            # Sort by datetime to ensure proper order
            df_city = df_city.sort_values('datetime')
            
            # Use first 12 hours for sequence, next 12 hours for target
            sequence_data = df_city.iloc[:12]
            target_data = df_city.iloc[12:24]
            
            if len(target_data) < 12:
                skip_reason = f"not enough target data ({len(target_data)} rows, need 12)"
                valid = False
                break
            
            # Check if target has any non-zero values (but be less strict)
            if (target_data['GHI'] <= 0).sum() >= 11:  # Allow at least 1 non-zero value
                skip_reason = f"all target GHI values are zero or negative"
                valid = False
                break
            
            # Create features from sequence data
            feature_cols = [col for col in sequence_data.columns if 'lag' in col or 'sin' in col or 'cos' in col]
            
            # Check if we have the required features
            if len(feature_cols) == 0:
                skip_reason = f"no feature columns found"
                valid = False
                break
                
            X = sequence_data[feature_cols].values.flatten()
            
            # Check for NaN values in features
            if np.isnan(X).any():
                skip_reason = f"NaN values in features"
                valid = False
                break
            
            # Target is the GHI values for the next 12 hours
            y = target_data['GHI'].values
            
            # Check for NaN values in targets
            if np.isnan(y).any():
                skip_reason = f"NaN values in targets"
                valid = False
                break
            
            node_features.append(X)
            node_targets.append(y)
        
        if valid:
            x = np.stack(node_features, axis=0)
            # Create a single target per graph by averaging across cities
            y = np.mean(node_targets, axis=0)
            
            graphs.append(Graph(x=x, a=adj_matrix))
            targets.append(y)
        else:
            if skip_reason not in skipped_reasons:
                skipped_reasons[skip_reason] = 0
            skipped_reasons[skip_reason] += 1

    print(f"    Total valid daily graphs: {len(graphs)}")
    print(f"    Skipped dates by reason:")
    for reason, count in skipped_reasons.items():
        print(f"      {reason}: {count} dates")
    
    return graphs, np.array(targets), scaler

# Test with all cities
CITIES = list(CONFIG["data_locations"].keys())

print(f"Testing with all cities: {CITIES}")

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

print("\nComputing adjacency matrix...")
start_time = time.time()
adj_matrix = compute_weighted_adjacency(df_all, alpha=0.5)
adj_time = time.time() - start_time
print(f"Adjacency matrix computed in {adj_time:.2f} seconds: {adj_matrix.shape}")

print("\nBuilding daily graphs...")
start_time = time.time()
graphs, targets, ghi_scaler = build_daily_graphs(df_all, adj_matrix)
graph_time = time.time() - start_time
print(f"Graphs built in {graph_time:.2f} seconds: {len(graphs)} graphs")

print("\n✓ Graph building test completed successfully!")
print(f"Total time: {load_time + feature_time + concat_time + adj_time + graph_time:.2f} seconds")
print(f"Final dataset: {len(graphs)} graphs with {len(targets)} targets") 