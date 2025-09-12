# test_data_loading.py
# Test script to verify data loading and fix the issue

import pandas as pd
import os

def test_data_loading():
    """Test different ways of loading the data to find the correct approach."""
    
    # Test file
    test_file = "data_Jaisalmer_2017.csv"
    
    if not os.path.exists(test_file):
        print(f"Test file {test_file} not found!")
        return
    
    print("Testing different data loading approaches...")
    print("=" * 60)
    
    # Method 1: Read without skipping rows
    print("\nMethod 1: Read without skipping rows")
    try:
        df1 = pd.read_csv(test_file)
        print(f"Shape: {df1.shape}")
        print(f"Columns: {df1.columns.tolist()}")
        print(f"First few rows:")
        print(df1.head())
        print(f"GHI column sample: {df1['GHI'].head(10).tolist()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Method 2: Skip 1 row
    print("\nMethod 2: Skip 1 row")
    try:
        df2 = pd.read_csv(test_file, skiprows=1)
        print(f"Shape: {df2.shape}")
        print(f"Columns: {df2.columns.tolist()}")
        print(f"First few rows:")
        print(df2.head())
        print(f"GHI column sample: {df2['GHI'].head(10).tolist()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Method 3: Skip 2 rows (current method)
    print("\nMethod 3: Skip 2 rows (current method)")
    try:
        df3 = pd.read_csv(test_file, skiprows=2)
        print(f"Shape: {df3.shape}")
        print(f"Columns: {df3.columns.tolist()}")
        print(f"First few rows:")
        print(df3.head())
        print(f"GHI column sample: {df3['GHI'].head(10).tolist()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Method 4: Read raw file content
    print("\nMethod 4: Read raw file content")
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"Total lines: {len(lines)}")
        print("First 10 lines:")
        for i, line in enumerate(lines[:10]):
            print(f"Line {i+1}: {line.strip()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_data_loading() 