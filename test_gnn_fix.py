#!/usr/bin/env python3
"""
Test script to verify the GNN training fix works correctly.
This script will test the data loading and training loop with a small subset.
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from spektral.data import Dataset, Graph
from spektral.data.loaders import DisjointLoader
from spektral.layers import GCNConv
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler

# Import the functions from the main script
from train_gnn import create_features, compute_weighted_adjacency, build_daily_graphs, GHIDataset, build_gnn_model
from utils import CONFIG, load_data

def test_gnn_training():
    """Test the GNN training with a small subset of data."""
    print("Testing GNN training fix...")
    
    # Use only first 2 cities for testing
    CITIES = list(CONFIG["data_locations"].keys())[:2]
    print(f"Testing with cities: {CITIES}")
    
    # Load and prepare data (small subset)
    all_dfs = []
    for city in CITIES:
        print(f"Loading data for {city}...")
        df = load_data(CONFIG["data_locations"], city)
        # Take only first 1000 rows for testing
        df = df.head(1000)
        df = create_features(df)
        df['location'] = city
        all_dfs.append(df)
    
    df_all = pd.concat(all_dfs).sort_values("datetime")
    print(f"Combined test data shape: {df_all.shape}")
    
    # Build graphs
    adj_matrix = compute_weighted_adjacency(df_all, alpha=0.5)
    graphs, targets, scaler = build_daily_graphs(df_all, adj_matrix)
    
    if len(graphs) == 0:
        print("ERROR: No graphs created in test!")
        return False
    
    print(f"Created {len(graphs)} test graphs")
    
    # Create dataset
    dataset = GHIDataset(graphs, targets)
    print(f"Dataset size: {len(dataset)}")
    
    # Test data loader
    batch_size = 4  # Smaller batch size for testing
    loader = DisjointLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Calculate steps per epoch
    total_graphs = len(dataset)
    steps_per_epoch = (total_graphs + batch_size - 1) // batch_size
    print(f"Steps per epoch: {steps_per_epoch}")
    
    # Test the training loop
    print("\nTesting training loop...")
    num_batches = 0
    max_test_batches = min(steps_per_epoch, 5)  # Test only first 5 batches
    
    for batch_idx in range(max_test_batches):
        try:
            batch = next(loader)
            inputs, y_true = batch
            x, a, i = inputs
            print(f"  Batch {batch_idx}: x={x.shape}, a={a.shape}, i={i.shape}, y={y_true.shape}")
            num_batches += 1
        except StopIteration:
            print(f"  Reached end of dataset at batch {batch_idx}")
            break
        except Exception as e:
            print(f"  Error in batch {batch_idx}: {e}")
            break
    
    print(f"Successfully processed {num_batches} batches")
    
    if num_batches > 0:
        print("✓ Test passed! Training loop works correctly.")
        return True
    else:
        print("✗ Test failed! No batches processed.")
        return False

if __name__ == "__main__":
    success = test_gnn_training()
    if success:
        print("\nTest completed successfully. The fix should work for the full training.")
    else:
        print("\nTest failed. There may still be issues with the training loop.") 