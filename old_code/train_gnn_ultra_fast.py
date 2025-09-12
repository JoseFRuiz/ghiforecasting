#!/usr/bin/env python3
# train_gnn_ultra_fast.py - Ultra-fast graph building version

import sys
import time
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from sklearn.preprocessing import MinMaxScaler
from spektral.data import Graph
from utils import CONFIG, load_data

def create_features_fast(df):
    """Fast version of create_features"""
    # Time-based features
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    # GHI lag features - vectorized
    for lag in range(1, 25):
        df[f"GHI_lag_{lag}"] = df["GHI"].shift(lag)

    # Meteorological lag features - vectorized
    met_vars = ["Temperature", "Relative Humidity", "Pressure",
                "Precipitable Water", "Wind Direction", "Wind Speed"]
    for var in met_vars:
        df[f"{var}_lag_24"] = df[var].shift(24)

    # Target
    df["target"] = df["GHI"].shift(-24)
    
    # Drop NaN values
    df = df.dropna().reset_index(drop=True)
    
    return df

def compute_weighted_adjacency_fast(df_all, alpha=0.5):
    """Ultra-fast adjacency matrix computation"""
    city_coords = {
        "Jaisalmer": (26.9157, 70.9083),
        "Jodhpur": (26.2389, 73.0243),
        "New Delhi": (28.6139, 77.2090),
        "Shimla": (31.1048, 77.1734),
        "Srinagar": (34.0837, 74.7973),
    }
    cities = list(city_coords.keys())
    n = len(cities)

    # Geodesic distances - pre-computed
    geo_weights = np.array([
        [1.0, 0.8, 0.6, 0.4, 0.3],
        [0.8, 1.0, 0.9, 0.5, 0.4],
        [0.6, 0.9, 1.0, 0.7, 0.6],
        [0.4, 0.5, 0.7, 1.0, 0.8],
        [0.3, 0.4, 0.6, 0.8, 1.0]
    ], dtype=np.float32)

    # Correlation weights - simplified
    corr_weights = np.array([
        [1.0, 0.7, 0.6, 0.5, 0.4],
        [0.7, 1.0, 0.8, 0.6, 0.5],
        [0.6, 0.8, 1.0, 0.7, 0.6],
        [0.5, 0.6, 0.7, 1.0, 0.8],
        [0.4, 0.5, 0.6, 0.8, 1.0]
    ], dtype=np.float32)

    adj = alpha * geo_weights + (1 - alpha) * corr_weights
    return adj

def build_daily_graphs_ultra_fast(df_all, adj_matrix):
    """Ultra-fast graph building using numpy operations"""
    print("    Starting ultra-fast graph building...")
    
    # Fit scaler
    scaler = MinMaxScaler()
    df_all['GHI_scaled'] = scaler.fit_transform(df_all[["GHI"]])

    # Pre-compute feature columns
    feature_cols = [col for col in df_all.columns if 'lag' in col or 'sin' in col or 'cos' in col]
    print(f"    Found {len(feature_cols)} feature columns")

    # Pre-process data efficiently
    df_all['date'] = pd.to_datetime(df_all["datetime"]).dt.date
    
    # Create efficient data structures
    city_data = {}
    for city in CITIES:
        city_df = df_all[df_all['location'] == city].copy()
        city_df = city_df.sort_values('datetime')
        city_data[city] = city_df
    
    # Get valid dates
    all_dates = set()
    for city_df in city_data.values():
        all_dates.update(city_df['date'].unique())
    dates = sorted(list(all_dates))
    
    # Pre-filter valid dates
    valid_dates = []
    for date in dates:
        all_cities_valid = True
        for city_df in city_data.values():
            if len(city_df[city_df['date'] == date]) < 24:
                all_cities_valid = False
                break
        if all_cities_valid:
            valid_dates.append(date)
    
    print(f"    Processing {len(valid_dates)} valid dates")
    
    # Process all dates at once
    graphs = []
    targets = []
    
    for date in valid_dates:
        node_features = []
        node_targets = []
        
        for city in CITIES:
            city_df = city_data[city]
            date_data = city_df[city_df['date'] == date]
            
            if len(date_data) < 24:
                continue
            
            # Extract sequence and target
            sequence_data = date_data.iloc[:12]
            target_data = date_data.iloc[12:24]
            
            # Quick validation
            if target_data['GHI'].sum() <= 0:
                continue
            
            # Extract features
            X = sequence_data[feature_cols].values.flatten()
            
            if np.isnan(X).any():
                continue
            
            y = target_data['GHI'].values
            
            if np.isnan(y).any():
                continue
            
            node_features.append(X)
            node_targets.append(y)
        
        # Create graph if all cities have data
        if len(node_features) == len(CITIES):
            x = np.stack(node_features, axis=0)
            y = np.mean(node_targets, axis=0)
            
            graphs.append(Graph(x=x, a=adj_matrix))
            targets.append(y)

    print(f"    Total graphs created: {len(graphs)}")
    return graphs, np.array(targets), scaler

# Main execution
CITIES = list(CONFIG["data_locations"].keys())

print("Ultra-fast graph building test...")

# Load and process all cities
all_dfs = []
for i, city in enumerate(CITIES):
    print(f"Loading data for city {i+1}/{len(CITIES)}: {city}")
    df = load_data(CONFIG["data_locations"], city)
    df = create_features_fast(df)
    df['location'] = city
    all_dfs.append(df)
    print(f"  ✓ {city} processed")

print("Concatenating data...")
df_all = pd.concat(all_dfs).sort_values("datetime")
print(f"Combined shape: {df_all.shape}")

print("Computing adjacency matrix...")
adj_matrix = compute_weighted_adjacency_fast(df_all)

print("Building graphs...")
start_time = time.time()
graphs, targets, scaler = build_daily_graphs_ultra_fast(df_all, adj_matrix)
graph_time = time.time() - start_time

print(f"✓ Ultra-fast graph building completed!")
print(f"Graph building time: {graph_time:.2f} seconds")
print(f"Total graphs: {len(graphs)}")
print(f"Target shape: {targets.shape}") 