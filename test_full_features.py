#!/usr/bin/env python3
# test_full_features.py - Test the full create_features function with 24 lags

import sys
import time
print("Testing full create_features function with 24 lags...")

# Import what we need
from utils import CONFIG, load_data
import pandas as pd
import numpy as np

def create_features_full(df):
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

# Test with one city
CITIES = list(CONFIG["data_locations"].keys())
test_city = CITIES[0]

print(f"Testing with city: {test_city}")

print("Loading data...")
start_time = time.time()
df = load_data(CONFIG["data_locations"], test_city)
load_time = time.time() - start_time
print(f"Data loaded in {load_time:.2f} seconds: {df.shape}")

print("Creating full features (24 lags)...")
start_time = time.time()
df_features = create_features_full(df)
feature_time = time.time() - start_time
print(f"Full features created in {feature_time:.2f} seconds: {df_features.shape}")

print("✓ Full feature creation test completed successfully!")
print(f"Total time: {load_time + feature_time:.2f} seconds")
print(f"Memory usage: {df_features.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB") 