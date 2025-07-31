#!/usr/bin/env python3
# test_features.py - Test the create_features function specifically

import sys
import time
print("Testing create_features function...")

# Import what we need
from utils import CONFIG, load_data
import pandas as pd
import numpy as np

def create_features_simple(df):
    """Simplified version of create_features to test"""
    print(f"    Creating features for dataframe with {len(df)} rows...")
    
    print(f"      Adding time-based features...")
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    print(f"      Adding GHI lag features...")
    # Test with fewer lags first
    for lag in range(1, 7):  # Only 6 lags instead of 24
        df[f"GHI_lag_{lag}"] = df["GHI"].shift(lag)
        print(f"        Added GHI lag {lag}")

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

print("Creating features...")
start_time = time.time()
df_features = create_features_simple(df)
feature_time = time.time() - start_time
print(f"Features created in {feature_time:.2f} seconds: {df_features.shape}")

print("✓ Feature creation test completed successfully!")
print(f"Total time: {load_time + feature_time:.2f} seconds") 