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

# Custom LSTM layer to handle time_major parameter
class CustomLSTM(tf.keras.layers.LSTM):
    def __init__(self, *args, **kwargs):
        # Remove time_major if present
        kwargs.pop('time_major', None)
        super().__init__(*args, **kwargs)

# Custom Lambda layer to handle output shape
class CustomLambda(tf.keras.layers.Lambda):
    def __init__(self, function, output_shape=None, **kwargs):
        if output_shape is None:
            # Try to infer output shape from the function
            if hasattr(function, 'output_shape'):
                output_shape = function.output_shape
            else:
                # Default to input shape if we can't infer
                output_shape = lambda x: x.shape
        super().__init__(function, output_shape=output_shape, **kwargs)

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
        # Skip locations with placeholder IDs
        if any(file_id == "1-2-3-4-5" for file_id in locations[city].values()):
            print(f"\nSkipping {city} - no valid data available")
            continue
            
        print(f"\nLoading data for {city}...")
        try:
            df = load_data(locations, city)
            
            # Sort by datetime
            df = df.sort_values('datetime')
            
            # Split into train/val/test by complete days
            # Get unique dates
            dates = df['datetime'].dt.date.unique()
            n_dates = len(dates)
            
            # Use last 15% of days for test
            test_start_idx = int(n_dates * 0.85)
            test_start_date = dates[test_start_idx]
            
            # Get test data starting from the beginning of the test period
            test_data = df[df['datetime'].dt.date >= test_start_date].copy()
            
            print(f"\nTest data info for {city}:")
            print(f"Total samples: {len(test_data)}")
            print(f"Date range: {test_data['datetime'].min()} to {test_data['datetime'].max()}")
            print(f"Number of complete days: {len(test_data['datetime'].dt.date.unique())}")
            print(f"Sample GHI values: {test_data['GHI'].head().values}")
            
            # Store test data
            test_data_dict[city] = test_data
            
            # Collect GHI values for scaling
            all_ghi_values.extend(test_data['GHI'].values)
        except Exception as e:
            print(f"Error loading data for {city}: {str(e)}")
            continue
    
    if not test_data_dict:
        raise ValueError("No valid data was loaded for any location")
    
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
    
    # Create lag features for GHI (24 lags)
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
        
        # Get features in the correct order to match training data
        features = []
        
        # Add GHI lag features (24 features)
        for lag in range(1, 25):
            features.append(sequence_window[f"GHI_lag_{lag}"].values)
        
        # Add current GHI (1 feature)
        features.append(sequence_window["GHI"].values)
        
        # Add meteorological features (6 features)
        for feature in meteorological_features:
            features.append(sequence_window[f"{feature}_lag_24"].values)
        
        # Add time features (4 features)
        time_features = ["hour_sin", "hour_cos", "month_sin", "month_cos"]
        for feature in time_features:
            features.append(sequence_window[feature].values)
        
        # Add location features (5 features - one-hot encoded)
        location_features = np.zeros((sequence_length, 5))  # 5 locations
        city_idx = list(CONFIG["data_locations"].keys()).index(test_data.iloc[0]["City"])
        location_features[:, city_idx] = 1
        
        # Combine all features
        features.append(location_features)
        
        # Stack features to create (24, 40) shape
        X = np.column_stack(features)
        X_sequences.append(X)
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

def plot_single_day_predictions(results, city, date_str, save_dir="results_joint/predictions"):
    """
    Create a plot comparing predictions vs actual values for a specific day.
    
    Args:
        results: Dictionary with predictions and actual values for each location
        city: City name to plot
        date_str: Date string in format 'YYYY-MM-DD'
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nDebug info for {city} on {date_str}:")
    
    # Get data for the specified city
    data = results[city]
    dates = pd.to_datetime(data['dates'])
    
    print(f"Total dates in data: {len(dates)}")
    print(f"Date range: {dates.min()} to {dates.max()}")
    
    # Filter data for the specified day
    mask = dates.strftime('%Y-%m-%d') == date_str
    day_dates = dates[mask]
    day_actual = data['actual'][mask]
    day_predicted = data['predicted'][mask]
    
    print(f"Number of points for selected day: {len(day_dates)}")
    if len(day_dates) > 0:
        print(f"First point: {day_dates[0]}")
        print(f"Last point: {day_dates[-1]}")
        print(f"Sample actual values: {day_actual[:5]}")
        print(f"Sample predicted values: {day_predicted[:5]}")
    
    if len(day_dates) == 0:
        print(f"No data available for {city} on {date_str}")
        return
    
    plt.figure(figsize=(15, 6))
    
    # Plot actual and predicted values using the actual timestamps
    plt.plot(day_dates, day_actual, label='Actual', color='blue', marker='o', alpha=0.7)
    plt.plot(day_dates, day_predicted, label='Predicted', color='red', marker='s', alpha=0.7)
    
    plt.title(f'GHI Predictions vs Actual Values - {city} ({date_str})')
    plt.xlabel('Hour')
    plt.ylabel('GHI (W/m²)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis to show hours
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    plt.xticks(day_dates[::2], rotation=45)  # Show every hour
    
    # Set x-axis limits to show full 24 hours
    plt.xlim(day_dates[0], day_dates[-1])
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(save_dir, f'predictions_{city}_{date_str}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print some statistics for the day
    mae = np.mean(np.abs(day_actual - day_predicted))
    rmse = np.sqrt(np.mean(np.square(day_actual - day_predicted)))
    print(f"\nStatistics for {city} on {date_str}:")
    print(f"Mean Absolute Error: {mae:.2f} W/m²")
    print(f"Root Mean Square Error: {rmse:.2f} W/m²")
    print(f"Maximum GHI: {np.max(day_actual):.2f} W/m²")
    print(f"Average GHI: {np.mean(day_actual):.2f} W/m²")

def find_complete_day(results, city):
    """
    Find a day that has complete data (24 points) for a given city.
    
    Args:
        results: Dictionary with predictions and actual values for each location
        city: City name to analyze
    
    Returns:
        str: Date string of a complete day, or None if no complete day is found
    """
    data = results[city]
    dates = pd.to_datetime(data['dates'])
    
    # Convert to Series and count points per day
    date_counts = pd.Series(dates).dt.strftime('%Y-%m-%d').value_counts()
    
    print(f"\nDebug info for {city}:")
    print(f"Total number of days: {len(date_counts)}")
    print(f"Points per day distribution:")
    print(date_counts.value_counts().sort_index())
    
    # Find days with exactly 24 points (hourly data)
    complete_days = date_counts[date_counts == 24].index
    
    if len(complete_days) > 0:
        selected_day = complete_days[0]
        print(f"\nSelected complete day: {selected_day}")
        print(f"Number of points on selected day: {date_counts[selected_day]}")
        
        # Print the actual timestamps for the selected day
        day_timestamps = dates[dates.strftime('%Y-%m-%d') == selected_day]
        print("\nTimestamps for selected day:")
        print(day_timestamps)
        
        return selected_day
    
    print(f"\nNo complete day found for {city}")
    return None

def main():
    """Main execution function."""
    # Load the trained model
    model_path = os.path.join("models", "lstm_ghi_forecast_joint.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    print(f"Loading model from {model_path}...")
    # Use custom_objects to handle both LSTM and Lambda layers
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'LSTM': CustomLSTM,
            'Lambda': CustomLambda
        }
    )
    
    # Load and prepare test data
    test_data_dict, target_scaler = load_and_prepare_test_data(CONFIG["data_locations"])
    
    # Generate predictions
    results = generate_predictions(model, test_data_dict, target_scaler)
    
    # Create plots for all data
    plot_predictions(results)
    
    # Create plots for a complete day for each city
    for city in results.keys():
        complete_day = find_complete_day(results, city)
        if complete_day:
            print(f"\nFound complete day for {city}: {complete_day}")
            plot_single_day_predictions(results, city, complete_day)
        else:
            print(f"\nNo complete day found for {city}")
    
    print("\nPredictions have been generated and saved in the 'results_joint/predictions' directory")

if __name__ == "__main__":
    main() 