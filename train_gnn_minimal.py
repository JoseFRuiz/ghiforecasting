#!/usr/bin/env python3
# train_gnn_minimal.py - Minimal version to test step by step

import sys
print("Starting minimal GNN script...")
print(f"Python version: {sys.version}")

# Step 1: Basic imports
print("\nStep 1: Basic imports...")
import os
import numpy as np
import pandas as pd
print("✓ Basic imports successful")

# Step 2: TensorFlow imports
print("\nStep 2: TensorFlow imports...")
import tensorflow as tf
from tensorflow.keras import layers, models
print("✓ TensorFlow imports successful")

# Step 3: Spektral imports
print("\nStep 3: Spektral imports...")
from spektral.data import Dataset, Graph
from spektral.data.loaders import DisjointLoader
from spektral.layers import GCNConv
print("✓ Spektral imports successful")

# Step 4: Other imports
print("\nStep 4: Other imports...")
from sklearn.preprocessing import MinMaxScaler
import pickle
from geopy.distance import geodesic
print("✓ Other imports successful")

# Step 5: Utils import
print("\nStep 5: Utils import...")
try:
    from utils import CONFIG, load_data
    print("✓ Utils import successful")
    print(f"  CONFIG keys: {list(CONFIG.keys())}")
except Exception as e:
    print(f"✗ Utils import failed: {e}")
    print("Stopping here due to utils import failure")
    sys.exit(1)

# Step 6: Test data loading for one city
print("\nStep 6: Testing data loading...")
try:
    CITIES = list(CONFIG["data_locations"].keys())
    print(f"  Cities: {CITIES}")
    
    # Test loading data for first city only
    test_city = CITIES[0]
    print(f"  Testing data loading for {test_city}...")
    df = load_data(CONFIG["data_locations"], test_city)
    print(f"  ✓ Data loaded for {test_city}: {df.shape}")
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    print("Stopping here due to data loading failure")
    sys.exit(1)

print("\n✓ All basic functionality working!")
print("The issue is likely in the feature creation or graph building phase.")
print("You can now run the full script with confidence that imports work.") 