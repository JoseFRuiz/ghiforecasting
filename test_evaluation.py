#!/usr/bin/env python3
"""
Test script to verify evaluate_configurations_fixed.py works correctly.
This script will check if the evaluation script can be imported and run basic checks.
"""

import os
import sys

def test_evaluation_script():
    """Test the evaluation script functionality."""
    print("Testing evaluate_configurations_fixed.py...")
    
    try:
        # Import the evaluation script
        from evaluate_configurations_fixed import (
            create_features,
            create_sequences_individual,
            create_sequences_joint_all_features,
            split_data_by_days,
            calculate_daily_metrics,
            evaluate_joint_models,
            evaluate_individual_models,
            main
        )
        print("✓ Successfully imported all functions from evaluate_configurations_fixed.py")
        
        # Check if utils module is available
        try:
            from utils import CONFIG, load_data
            print("✓ Successfully imported utils module")
            print(f"Available locations: {list(CONFIG['data_locations'].keys())}")
        except ImportError as e:
            print(f"× Error importing utils module: {e}")
            return False
        
        # Check if models directory exists
        if os.path.exists("models"):
            print("✓ Models directory exists")
            
            # Check for target scaler
            scaler_path = os.path.join("models", "target_scaler_fixed.pkl")
            if os.path.exists(scaler_path):
                print("✓ Target scaler found")
            else:
                print("⚠ Target scaler not found (expected if models not trained yet)")
            
            # Check for joint model
            joint_model_path = os.path.join("models", "lstm_ghi_forecast_joint_fixed.h5")
            if os.path.exists(joint_model_path):
                print("✓ Joint model found")
            else:
                print("⚠ Joint model not found (expected if not trained yet)")
            
            # Check for individual models
            individual_models = []
            for city in CONFIG["data_locations"].keys():
                model_path = os.path.join("models", f"lstm_ghi_forecast_{city}.h5")
                if os.path.exists(model_path):
                    individual_models.append(city)
            
            if individual_models:
                print(f"✓ Found {len(individual_models)} individual models: {individual_models}")
            else:
                print("⚠ No individual models found (expected if not trained yet)")
        else:
            print("⚠ Models directory not found (expected if models not trained yet)")
        
        print("\n✓ All basic checks passed!")
        print("\nTo run the evaluation:")
        print("1. First train the models:")
        print("   python train_joint_fixed.py")
        print("   python train_individual.py")
        print("2. Then run the evaluation:")
        print("   python evaluate_configurations_fixed.py")
        
        return True
        
    except ImportError as e:
        print(f"× Error importing evaluate_configurations_fixed.py: {e}")
        return False
    except Exception as e:
        print(f"× Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_evaluation_script()
    if success:
        print("\n✓ Test completed successfully!")
        sys.exit(0)
    else:
        print("\n× Test failed!")
        sys.exit(1) 