#!/usr/bin/env python3
# test_imports.py - Test script to identify import issues

import sys
print("Python version:", sys.version)

print("Testing basic imports...")
try:
    import os
    print("✓ os imported")
except Exception as e:
    print(f"✗ os import failed: {e}")

try:
    import numpy as np
    print("✓ numpy imported")
except Exception as e:
    print(f"✗ numpy import failed: {e}")

try:
    import pandas as pd
    print("✓ pandas imported")
except Exception as e:
    print(f"✗ pandas import failed: {e}")

try:
    import tensorflow as tf
    print("✓ tensorflow imported")
    print(f"  TensorFlow version: {tf.__version__}")
except Exception as e:
    print(f"✗ tensorflow import failed: {e}")

try:
    from tensorflow.keras import layers, models
    print("✓ tensorflow.keras imported")
except Exception as e:
    print(f"✗ tensorflow.keras import failed: {e}")

try:
    from spektral.data import Dataset, Graph
    print("✓ spektral.data imported")
except Exception as e:
    print(f"✗ spektral.data import failed: {e}")

try:
    from spektral.data.loaders import DisjointLoader
    print("✓ spektral.data.loaders imported")
except Exception as e:
    print(f"✗ spektral.data.loaders import failed: {e}")

try:
    from spektral.layers import GCNConv
    print("✓ spektral.layers imported")
except Exception as e:
    print(f"✗ spektral.layers import failed: {e}")

try:
    from sklearn.preprocessing import MinMaxScaler
    print("✓ sklearn.preprocessing imported")
except Exception as e:
    print(f"✗ sklearn.preprocessing import failed: {e}")

try:
    import pickle
    print("✓ pickle imported")
except Exception as e:
    print(f"✗ pickle import failed: {e}")

try:
    from geopy.distance import geodesic
    print("✓ geopy.distance imported")
except Exception as e:
    print(f"✗ geopy.distance import failed: {e}")

print("\nTesting utils import...")
try:
    from utils import CONFIG, load_data
    print("✓ utils imported")
    print(f"  CONFIG keys: {list(CONFIG.keys())}")
except Exception as e:
    print(f"✗ utils import failed: {e}")

print("\nAll imports completed successfully!")
print("If you see this message, all imports are working correctly.") 