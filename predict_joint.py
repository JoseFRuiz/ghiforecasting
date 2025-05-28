"""
Script to load the trained joint LSTM model and generate predictions for all stations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# Import shared utilities and configurations
from utils import (
    CONFIG,
    load_data,
    prepare_lstm_input
)

def load_and_prepare_test_data(locations, sequence_length=24):
    """
    Load and prepare test data for all locations.
    
    Args:
        locations: Dictionary of location data from CONFIG
        sequence_length: Length of input sequences
    
    Returns:
        tuple: (test_data_dict, target_scaler)
    """
    print("\nLoading test data for all locations...")
    test_data_dict = {}
    all_ghi_values = []
    
    # Load data for each location
    for city in locations.keys():
        print(f"\nLoading data for {city}...")
        df = load_data(locations, city)
        
        # Sort by datetime
        df = df.sort_values('datetime')
        
        # Split into train/val/test (70/15/15)
        n_samples = len(df)
        test_start = int(n_samples * 0.85)
        test_data = df.iloc[test_start:].copy()
        
        # Store test data
        test_data_dict[city] = test_data
        
        # Collect GHI values for scaling
        all_ghi_values.extend(test_data['GHI'].values)
    
    # Create and fit scaler on all GHI values
    target_scaler = MinMaxScaler()
    target_scaler.fit(np.array(all_ghi_values).reshape(-1, 1))
    
    return test_data_dict, target_scaler

def create_prediction_sequences(test_data, sequence_length=24):
    """
    Create sequences for prediction.
    
    Args:
        test_data: DataFrame with test data
        sequence_length: Length of input sequences
    
    Returns:
        tuple: (X, y, dates)
    """
    # Create time-based features
    test_data["hour_sin"] = np.sin(2 * np.pi * test_data["Hour"] / 24)
    test_data["hour_cos"] = np.cos(2 * np.pi * test_data["Hour"] / 24)
    test_data["month_sin"] = np.sin(2 * np.pi * test_data["Month"] / 12)
    test_data["month_cos"] = np.cos(2 * np.pi * test_data["Month"] / 12)
    
    # Create lag features
    for lag in range(1, 25):
        test_data[f"GHI_lag_{lag}"] = test_data["GHI"].shift(lag)
    
    # Create meteorological lag features
    meteorological_features = [
        "Temperature", "Relative Humidity", "Pressure",
        "Precipitable Water", "Wind Direction", "Wind Speed"
    ]
    
    for feature in meteorological_features:
        test_data[f"{feature}_lag_24"] = test_data[feature].shift(24)
    
    # Drop rows with missing values
    test_data = test_data.dropna()
    
    # Create sequences
    X_sequences = []
    y_values = []
    dates = []
    
    for i in range(len(test_data) - sequence_length):
        # Get sequence window
        sequence_window = test_data.iloc[i:i + sequence_length]
        target_window = test_data.iloc[i + sequence_length:i + sequence_length + 1]
        
        # Get features
        ghi_lag_features = sequence_window[[f"GHI_lag_{lag}" for lag in range(1, 25)]].values
        current_ghi = sequence_window["GHI"].values.reshape(-1, 1)
        met_features = sequence_window[[
            "Temperature_lag_24", "Relative Humidity_lag_24",
            "Pressure_lag_24", "Precipitable Water_lag_24",
            "Wind Direction_lag_24", "Wind Speed_lag_24"
        ]].values
        time_features = sequence_window[[
            "hour_sin", "hour_cos",
            "month_sin", "month_cos"
        ]].values
        
        # Create feature matrix
        features = np.column_stack([
            ghi_lag_features,
            current_ghi,
            met_features,
            time_features
        ])
        
        X_sequences.append(features)
        y_values.append(target_window["GHI"].values[0])
        dates.append(target_window["datetime"].values[0])
    
    return np.array(X_sequences), np.array(y_values), dates

def generate_predictions(model, test_data_dict, target_scaler, sequence_length=24):
    """
    Generate predictions for all locations.
    
    Args:
        model: Trained model
        test_data_dict: Dictionary of test data for each location
        target_scaler: Scaler for target values
        sequence_length: Length of input sequences
    
    Returns:
        dict: Dictionary with predictions and actual values for each location
    """
    results = {}
    
    for city, test_data in test_data_dict.items():
        print(f"\nGenerating predictions for {city}...")
        
        # Create sequences
        X_test, y_test, dates = create_prediction_sequences(test_data, sequence_length)
        
        # Scale target values
        y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1))
        
        # Generate predictions
        y_pred_scaled = model.predict(X_test)
        
        # Inverse transform predictions and actual values
        y_pred = target_scaler.inverse_transform(y_pred_scaled)
        y_test = target_scaler.inverse_transform(y_test_scaled)
        
        # Store results
        results[city] = {
            'dates': dates,
            'actual': y_test.flatten(),
            'predicted': y_pred.flatten()
        }
    
    return results

def plot_predictions(results, save_dir="results_joint/predictions"):
    """
    Create plots comparing predictions vs actual values for each location.
    
    Args:
        results: Dictionary with predictions and actual values for each location
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for city, data in results.items():
        plt.figure(figsize=(15, 6))
        
        # Convert dates to datetime objects
        dates = pd.to_datetime(data['dates'])
        
        # Plot actual and predicted values
        plt.plot(dates, data['actual'], label='Actual', color='blue', alpha=0.7)
        plt.plot(dates, data['predicted'], label='Predicted', color='red', alpha=0.7)
        
        plt.title(f'GHI Predictions vs Actual Values - {city}')
        plt.xlabel('Date')
        plt.ylabel('GHI (W/m²)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(save_dir, f'predictions_{city}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main execution function."""
    # Load the trained model
    model_path = os.path.join("models", "lstm_ghi_forecast_joint.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Load and prepare test data
    test_data_dict, target_scaler = load_and_prepare_test_data(CONFIG["data_locations"])
    
    # Generate predictions
    results = generate_predictions(model, test_data_dict, target_scaler)
    
    # Create plots
    plot_predictions(results)
    
    print("\nPredictions have been generated and saved in the 'results_joint/predictions' directory")

if __name__ == "__main__":
    main() 