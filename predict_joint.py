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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

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

def load_and_prepare_test_data(locations):
    """
    Load and prepare test data for prediction.
    
    Args:
        locations: Dictionary of location data
    
    Returns:
        tuple: (test_data_dict, target_scaler)
    """
    print("\nLoading and preparing test data...")
    
    # Load the target scaler
    scaler_path = os.path.join("models", "target_scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Target scaler not found at {scaler_path}")
    
    with open(scaler_path, 'rb') as f:
        target_scaler = pickle.load(f)
    
    # Load and process data for each location
    test_data_dict = {}
    
    for location, data in locations.items():
        print(f"\nProcessing {location}...")
        
        try:
            # Load data
            df = pd.read_csv(data['path'])
            print(f"Loaded {len(df)} samples")
            
            # Convert datetime column
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Sort by datetime
            df = df.sort_values('datetime')
            
            # Create features
            df = create_features(df)
            
            # Validate features
            print("\nValidating features...")
            required_features = [
                'GHI', 'Temperature', 'Relative Humidity', 'Pressure',
                'Precipitable Water', 'Wind Direction', 'Wind Speed'
            ]
            
            missing_features = [f for f in required_features if f not in df.columns]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Check for missing values
            missing_values = df[required_features].isnull().sum()
            if missing_values.any():
                print("WARNING: Missing values found:")
                print(missing_values[missing_values > 0])
                # Fill missing values with forward fill and then backward fill
                df[required_features] = df[required_features].fillna(method='ffill').fillna(method='bfill')
            
            # Create lag features
            print("\nCreating lag features...")
            for feature in required_features:
                for lag in range(1, 25):
                    df[f"{feature}_lag_{lag}"] = df[feature].shift(lag)
            
            # Create time features
            print("\nCreating time features...")
            df['hour'] = df['datetime'].dt.hour
            df['month'] = df['datetime'].dt.month
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Drop rows with NaN values (from lag creation)
            df = df.dropna()
            
            # Scale target values
            df['GHI'] = target_scaler.transform(df[['GHI']])
            
            # Store processed data
            test_data_dict[location] = df
            
            print(f"Processed {len(df)} samples for {location}")
            
        except Exception as e:
            print(f"Error processing {location}: {str(e)}")
            continue
    
    if not test_data_dict:
        raise ValueError("No valid test data was processed")
    
    return test_data_dict, target_scaler

def create_prediction_sequences(df, sequence_length, input_config='all'):
    """
    Create sequences for prediction with the specified input configuration.
    
    Args:
        df: DataFrame with processed data
        sequence_length: Length of input sequences
        input_config: One of ['ghi_only', 'ghi_met', 'met_only', 'all']
    
    Returns:
        tuple: (X, y, dates) where X is the input sequences, y is the target values,
               and dates is the corresponding datetime values
    """
    print(f"\nCreating prediction sequences with {input_config} configuration...")
    
    # Initialize lists to store sequences
    X_sequences = []
    y_values = []
    dates = []
    
    # Calculate the maximum index that allows for a complete sequence
    max_index = len(df) - sequence_length
    
    # Create sequences using rolling window approach
    for i in range(max_index):
        # Get sequence window
        sequence_window = df.iloc[i:i + sequence_length]
        target_window = df.iloc[i + sequence_length:i + sequence_length + 1]
        
        # Skip if any data is missing
        if sequence_window.isnull().any().any() or target_window.isnull().any().any():
            continue
        
        # Initialize features list
        features = []
        
        # Add features based on configuration
        if input_config in ['ghi_only', 'ghi_met', 'all']:
            # Add GHI lag features (24 features)
            for lag in range(1, 25):
                features.append(sequence_window[f"GHI_lag_{lag}"].values)
            
            # Add current GHI (1 feature)
            features.append(sequence_window["GHI"].values)
        
        if input_config in ['met_only', 'ghi_met', 'all']:
            # Add meteorological features (6 features)
            met_features = [
                "Temperature_lag_24", "Relative Humidity_lag_24",
                "Pressure_lag_24", "Precipitable Water_lag_24",
                "Wind Direction_lag_24", "Wind Speed_lag_24"
            ]
            for feature in met_features:
                features.append(sequence_window[feature].values)
        
        if input_config == 'all':
            # Add time features (4 features)
            time_features = ["hour_sin", "hour_cos", "month_sin", "month_cos"]
            for feature in time_features:
                features.append(sequence_window[feature].values)
        
        # Stack features to create input sequence
        X = np.column_stack(features)
        
        # Get target value
        y = target_window["GHI"].values[0]
        
        # Skip if target value is zero (night time)
        if y == 0:
            continue
        
        X_sequences.append(X)
        y_values.append(y)
        dates.append(target_window["datetime"].values[0])
    
    if not X_sequences:
        raise ValueError("No valid sequences were created")
    
    # Convert to numpy arrays
    X = np.array(X_sequences)
    y = np.array(y_values)
    
    print(f"\nCreated {len(X)} sequences")
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y, dates

def generate_predictions(model, test_data_dict, target_scaler, sequence_length=24, input_config='all'):
    """
    Generate predictions for all locations.
    
    Args:
        model: Trained model
        test_data_dict: Dictionary of test data for each location
        target_scaler: Scaler for target values
        sequence_length: Length of input sequences
        input_config: Input configuration used for training
    
    Returns:
        dict: Dictionary with predictions and actual values for each location
    """
    results = {}
    
    # Print model information
    print("\nModel Information:")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Number of layers: {len(model.layers)}")
    print(f"Trainable parameters: {model.count_params()}")
    
    # Print model weights statistics
    print("\nModel weights statistics:")
    for layer in model.layers:
        if hasattr(layer, 'weights'):
            weights = layer.get_weights()
            if weights:
                print(f"\nLayer: {layer.name}")
                for i, w in enumerate(weights):
                    print(f"Weight {i} stats:")
                    print(f"Shape: {w.shape}")
                    print(f"Min: {np.min(w):.4f}")
                    print(f"Max: {np.max(w):.4f}")
                    print(f"Mean: {np.mean(w):.4f}")
                    print(f"Std: {np.std(w):.4f}")
    
    for city, test_data in test_data_dict.items():
        print(f"\nGenerating predictions for {city}...")
        
        # Create sequences with the correct configuration
        X_test, y_test, dates = create_prediction_sequences(test_data, sequence_length, input_config)
        
        print(f"\nDebug info for {city}:")
        print(f"Input shape: {X_test.shape}")
        print(f"Sample input values (first sequence):")
        print(X_test[0])
        
        # Validate input data
        if np.isnan(X_test).any():
            print("WARNING: NaN values found in input data!")
            X_test = np.nan_to_num(X_test, nan=0.0)
        
        if np.isinf(X_test).any():
            print("WARNING: Inf values found in input data!")
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Get hour information for each prediction
        hours = pd.to_datetime(dates).hour
        
        # Scale target values
        y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1))
        print(f"\nSample target values before scaling: {y_test[:5]}")
        print(f"Sample target values after scaling: {y_test_scaled[:5].flatten()}")
        
        # Validate scaled target values
        if np.isnan(y_test_scaled).any() or np.isinf(y_test_scaled).any():
            print("WARNING: Invalid values in scaled target data!")
            y_test_scaled = np.nan_to_num(y_test_scaled, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Generate predictions
        y_pred_scaled = model.predict(X_test)
        print(f"\nSample predictions before inverse transform: {y_pred_scaled[:5].flatten()}")
        
        # Validate predictions
        if np.isnan(y_pred_scaled).any() or np.isinf(y_pred_scaled).any():
            print("WARNING: Invalid values in predictions!")
            y_pred_scaled = np.nan_to_num(y_pred_scaled, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Check if predictions are all zeros or constant
        if np.all(y_pred_scaled == 0) or np.std(y_pred_scaled) < 1e-6:
            print("WARNING: Predictions are constant or all zeros!")
            print("Model output shape:", y_pred_scaled.shape)
            print("Model output statistics:")
            print(f"Min: {np.min(y_pred_scaled):.4f}")
            print(f"Max: {np.max(y_pred_scaled):.4f}")
            print(f"Mean: {np.mean(y_pred_scaled):.4f}")
            print(f"Std: {np.std(y_pred_scaled):.4f}")
            
            # Check model weights
            print("\nChecking model weights for potential issues:")
            for layer in model.layers:
                if hasattr(layer, 'weights'):
                    weights = layer.get_weights()
                    if weights:
                        print(f"\nLayer: {layer.name}")
                        for i, w in enumerate(weights):
                            print(f"Weight {i} stats:")
                            print(f"Shape: {w.shape}")
                            print(f"Min: {np.min(w):.4f}")
                            print(f"Max: {np.max(w):.4f}")
                            print(f"Mean: {np.mean(w):.4f}")
                            print(f"Std: {np.std(w):.4f}")
        
        # Ensure predictions are within valid range [0, 1] for inverse transform
        y_pred_scaled = np.clip(y_pred_scaled, 0, 1)
        
        # Inverse transform predictions and actual values
        y_pred = target_scaler.inverse_transform(y_pred_scaled)
        y_test = target_scaler.inverse_transform(y_test_scaled)
        
        # Set predictions to 0 for night time hours (before sunrise and after sunset)
        # Using a simple rule: set to 0 if hour < 6 or hour > 18
        night_mask = (hours < 6) | (hours > 18)
        y_pred[night_mask] = 0
        
        print(f"\nSample predictions after inverse transform: {y_pred[:5].flatten()}")
        print(f"Sample actual values after inverse transform: {y_test[:5].flatten()}")
        
        # Validate final predictions
        if np.isnan(y_pred).any() or np.isinf(y_pred).any():
            print("WARNING: Invalid values in final predictions!")
            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1000.0, neginf=0.0)
        
        # Store results
        results[city] = {
            'dates': dates,
            'actual': y_test.flatten(),
            'predicted': y_pred.flatten()
        }
        
        # Print summary statistics
        print(f"\nSummary statistics for {city}:")
        print("Actual values:")
        print(f"Min: {np.min(y_test):.2f}")
        print(f"Max: {np.max(y_test):.2f}")
        print(f"Mean: {np.mean(y_test):.2f}")
        print(f"Std: {np.std(y_test):.2f}")
        print("\nPredicted values:")
        print(f"Min: {np.min(y_pred):.2f}")
        print(f"Max: {np.max(y_pred):.2f}")
        print(f"Mean: {np.mean(y_pred):.2f}")
        print(f"Std: {np.std(y_pred):.2f}")
        
        # Calculate and print error metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"\nError metrics:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.4f}")
        
        # Additional validation
        if mae > 1000 or rmse > 1000:
            print("WARNING: Unusually high error values detected!")
            print("This might indicate a problem with the model or data scaling.")
        
        if r2 < -1:
            print("WARNING: Very poor R² score detected!")
            print("This might indicate a problem with the model predictions.")
    
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

def calculate_performance_metrics(results):
    """
    Calculate performance metrics for all locations.
    
    Args:
        results: Dictionary with predictions and actual values for each location
    
    Returns:
        dict: Dictionary with metrics for each location and overall
    """
    all_metrics = {}
    all_actual = []
    all_predicted = []
    
    print("\nCalculating detailed performance metrics...")
    
    for city, data in results.items():
        print(f"\nAnalyzing {city}:")
        actual = data['actual']
        predicted = data['predicted']
        
        # Print raw data statistics
        print("\nRaw data statistics:")
        print(f"Number of samples: {len(actual)}")
        print(f"Actual values - Min: {np.min(actual):.2f}, Max: {np.max(actual):.2f}, Mean: {np.mean(actual):.2f}")
        print(f"Predicted values - Min: {np.min(predicted):.2f}, Max: {np.max(predicted):.2f}, Mean: {np.mean(predicted):.2f}")
        
        # Filter out zero values (night time) as done in training
        non_zero_mask = actual > 0
        actual_nonzero = actual[non_zero_mask]
        predicted_nonzero = predicted[non_zero_mask]
        
        print(f"\nNon-zero samples: {len(actual_nonzero)} ({len(actual_nonzero)/len(actual)*100:.1f}% of total)")
        print(f"Non-zero actual values - Min: {np.min(actual_nonzero):.2f}, Max: {np.max(actual_nonzero):.2f}, Mean: {np.mean(actual_nonzero):.2f}")
        print(f"Non-zero predicted values - Min: {np.min(predicted_nonzero):.2f}, Max: {np.max(predicted_nonzero):.2f}, Mean: {np.mean(predicted_nonzero):.2f}")
        
        # Store non-zero values for overall metrics
        all_actual.extend(actual_nonzero)
        all_predicted.extend(predicted_nonzero)
        
        # Calculate metrics for this city using non-zero values
        metrics = {
            'MAE': mean_absolute_error(actual_nonzero, predicted_nonzero),
            'MSE': mean_squared_error(actual_nonzero, predicted_nonzero),
            'RMSE': np.sqrt(mean_squared_error(actual_nonzero, predicted_nonzero)),
            'R2': r2_score(actual_nonzero, predicted_nonzero)
        }
        
        # Calculate percentage errors
        mape = np.mean(np.abs((actual_nonzero - predicted_nonzero) / (actual_nonzero + 1e-10))) * 100
        metrics['MAPE'] = mape
        
        # Calculate additional metrics
        metrics['Mean_Actual'] = np.mean(actual_nonzero)
        metrics['Mean_Predicted'] = np.mean(predicted_nonzero)
        metrics['Std_Actual'] = np.std(actual_nonzero)
        metrics['Std_Predicted'] = np.std(predicted_nonzero)
        
        # Calculate correlation
        correlation = np.corrcoef(actual_nonzero, predicted_nonzero)[0, 1]
        metrics['Correlation'] = correlation
        
        # Add statistics about data filtering
        metrics['Total_Samples'] = len(actual)
        metrics['Non_Zero_Samples'] = len(actual_nonzero)
        metrics['Zero_Percentage'] = (len(actual) - len(actual_nonzero)) / len(actual) * 100
        
        # Print metrics
        print("\nPerformance metrics:")
        print(f"MAE: {metrics['MAE']:.2f} W/m²")
        print(f"RMSE: {metrics['RMSE']:.2f} W/m²")
        print(f"R²: {metrics['R2']:.4f}")
        print(f"MAPE: {metrics['MAPE']:.2f}%")
        print(f"Correlation: {metrics['Correlation']:.4f}")
        
        # Validate metrics
        if metrics['MAE'] > 1000 or metrics['RMSE'] > 1000:
            print("WARNING: Unusually high error values detected!")
        if metrics['R2'] < -1:
            print("WARNING: Very poor R² score detected!")
        if metrics['MAPE'] > 1000:
            print("WARNING: Very high MAPE detected!")
        if abs(metrics['Correlation']) < 0.1:
            print("WARNING: Very low correlation detected!")
        
        all_metrics[city] = metrics
    
    # Calculate overall metrics using non-zero values
    print("\nCalculating overall metrics...")
    overall_metrics = {
        'MAE': mean_absolute_error(all_actual, all_predicted),
        'MSE': mean_squared_error(all_actual, all_predicted),
        'RMSE': np.sqrt(mean_squared_error(all_actual, all_predicted)),
        'R2': r2_score(all_actual, all_predicted),
        'MAPE': np.mean(np.abs((np.array(all_actual) - np.array(all_predicted)) / (np.array(all_actual) + 1e-10))) * 100,
        'Mean_Actual': np.mean(all_actual),
        'Mean_Predicted': np.mean(all_predicted),
        'Std_Actual': np.std(all_actual),
        'Std_Predicted': np.std(all_predicted),
        'Correlation': np.corrcoef(all_actual, all_predicted)[0, 1],
        'Total_Samples': len(all_actual),
        'Non_Zero_Samples': len(all_actual),
        'Zero_Percentage': 0  # Since we've already filtered zeros
    }
    
    print("\nOverall performance metrics:")
    print(f"MAE: {overall_metrics['MAE']:.2f} W/m²")
    print(f"RMSE: {overall_metrics['RMSE']:.2f} W/m²")
    print(f"R²: {overall_metrics['R2']:.4f}")
    print(f"MAPE: {overall_metrics['MAPE']:.2f}%")
    print(f"Correlation: {overall_metrics['Correlation']:.4f}")
    print(f"Mean Actual: {overall_metrics['Mean_Actual']:.2f} W/m²")
    print(f"Mean Predicted: {overall_metrics['Mean_Predicted']:.2f} W/m²")
    
    all_metrics['Overall'] = overall_metrics
    
    return all_metrics

def calculate_daily_correlations(results):
    """
    Calculate correlation metrics for each day in the test set.
    
    Args:
        results: Dictionary with predictions and actual values for each location
    
    Returns:
        dict: Dictionary with daily correlation metrics for each location
    """
    daily_metrics = {}
    
    for city, data in results.items():
        # Convert dates to datetime and create DataFrame
        dates = pd.to_datetime(data['dates'])
        df = pd.DataFrame({
            'date': dates,
            'actual': data['actual'],
            'predicted': data['predicted']
        })
        
        # Group by date and calculate daily metrics
        daily_stats = []
        for date, group in df.groupby(df['date'].dt.date):
            # Filter out zero values
            non_zero_mask = group['actual'] > 0
            if non_zero_mask.sum() > 0:  # Only include days with non-zero values
                actual_nonzero = group.loc[non_zero_mask, 'actual']
                predicted_nonzero = group.loc[non_zero_mask, 'predicted']
                
                # Calculate correlation using pandas corr() method which ensures [-1, 1] range
                correlation = actual_nonzero.corr(predicted_nonzero)
                
                # Add error checking for correlation
                if pd.isna(correlation) or not (-1 <= correlation <= 1):
                    print(f"Warning: Invalid correlation value {correlation} for {city} on {date}")
                    correlation = 0  # Set to 0 if invalid
                
                # Calculate other metrics
                mae = mean_absolute_error(actual_nonzero, predicted_nonzero)
                rmse = np.sqrt(mean_squared_error(actual_nonzero, predicted_nonzero))
                r2 = r2_score(actual_nonzero, predicted_nonzero)
                
                # Add error checking for R²
                if not (0 <= r2 <= 1):
                    print(f"Warning: Invalid R² value {r2} for {city} on {date}")
                    r2 = max(0, min(1, r2))  # Clamp to [0, 1] range
                
                daily_stats.append({
                    'date': date,
                    'correlation': correlation,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'num_points': len(actual_nonzero),
                    'actual_max': actual_nonzero.max(),
                    'actual_mean': actual_nonzero.mean(),
                    'actual_std': actual_nonzero.std(),
                    'predicted_max': predicted_nonzero.max(),
                    'predicted_mean': predicted_nonzero.mean(),
                    'predicted_std': predicted_nonzero.std()
                })
        
        # Convert to DataFrame and sort by date
        daily_metrics[city] = pd.DataFrame(daily_stats).sort_values('date')
        
        # Print summary of correlation values and GHI statistics
        print(f"\nSummary for {city}:")
        print("-" * 40)
        print("Correlation Statistics:")
        print(f"Min: {daily_metrics[city]['correlation'].min():.4f}")
        print(f"Max: {daily_metrics[city]['correlation'].max():.4f}")
        print(f"Mean: {daily_metrics[city]['correlation'].mean():.4f}")
        print(f"Median: {daily_metrics[city]['correlation'].median():.4f}")
        
        print("\nActual GHI Statistics:")
        print(f"Max: {daily_metrics[city]['actual_max'].max():.2f} W/m²")
        print(f"Mean: {daily_metrics[city]['actual_mean'].mean():.2f} W/m²")
        print(f"Std: {daily_metrics[city]['actual_std'].mean():.2f} W/m²")
        
        print("\nPredicted GHI Statistics:")
        print(f"Max: {daily_metrics[city]['predicted_max'].max():.2f} W/m²")
        print(f"Mean: {daily_metrics[city]['predicted_mean'].mean():.2f} W/m²")
        print(f"Std: {daily_metrics[city]['predicted_std'].mean():.2f} W/m²")
    
    return daily_metrics

def plot_daily_correlations(daily_metrics, save_dir="results_joint/correlations"):
    """
    Create plots showing daily correlation metrics for each location.
    
    Args:
        daily_metrics: Dictionary with daily correlation metrics for each location
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for city, metrics_df in daily_metrics.items():
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
        
        # Plot correlation and R²
        ax1.plot(metrics_df['date'], metrics_df['correlation'], 'b-', label='Correlation')
        ax1.plot(metrics_df['date'], metrics_df['r2'], 'r-', label='R²')
        ax1.set_title(f'Daily Correlation and R² - {city}')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot MAE and RMSE
        ax2.plot(metrics_df['date'], metrics_df['mae'], 'g-', label='MAE')
        ax2.plot(metrics_df['date'], metrics_df['rmse'], 'm-', label='RMSE')
        ax2.set_title(f'Daily MAE and RMSE - {city}')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Error (W/m²)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot GHI statistics
        ax3.plot(metrics_df['date'], metrics_df['actual_max'], 'b-', label='Actual Max')
        ax3.plot(metrics_df['date'], metrics_df['predicted_max'], 'r--', label='Predicted Max')
        ax3.plot(metrics_df['date'], metrics_df['actual_mean'], 'g-', label='Actual Mean')
        ax3.plot(metrics_df['date'], metrics_df['predicted_mean'], 'm--', label='Predicted Mean')
        ax3.set_title(f'Daily GHI Statistics - {city}')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('GHI (W/m²)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        for ax in [ax1, ax2, ax3]:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'daily_correlations_{city}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save daily metrics to CSV
        metrics_df.to_csv(os.path.join(save_dir, f'daily_correlations_{city}.csv'), index=False)

def plot_correlation_vs_r2(daily_metrics, results, save_dir="results_joint/correlations"):
    """
    Create scatter plots comparing actual vs predicted values with correlation and R² annotations.
    
    Args:
        daily_metrics: Dictionary with daily correlation metrics for each location
        results: Dictionary with predictions and actual values for each location
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for city, metrics_df in daily_metrics.items():
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Convert dates to pandas Series for proper date handling
        dates = pd.Series(results[city]['dates'])
        
        # Plot 1: Day with highest correlation
        best_day_idx = metrics_df['correlation'].idxmax()
        best_day = metrics_df.loc[best_day_idx]
        
        # Get the actual data for this day
        mask = dates.dt.date == best_day['date']
        actual_data = np.array(results[city]['actual'])[mask]
        predicted_data = np.array(results[city]['predicted'])[mask]
        
        if len(actual_data) > 0:  # Only plot if we have data
            ax1.scatter(actual_data, predicted_data, alpha=0.5)
            max_val = max(actual_data.max(), predicted_data.max())
            ax1.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
            ax1.set_title(f'Best Day (Correlation = {best_day["correlation"]:.3f})')
            ax1.set_xlabel('Actual GHI')
            ax1.set_ylabel('Predicted GHI')
            ax1.legend()
        
        # Plot 2: Day with lowest correlation
        worst_day_idx = metrics_df['correlation'].idxmin()
        worst_day = metrics_df.loc[worst_day_idx]
        
        # Get the actual data for this day
        mask = dates.dt.date == worst_day['date']
        actual_data = np.array(results[city]['actual'])[mask]
        predicted_data = np.array(results[city]['predicted'])[mask]
        
        if len(actual_data) > 0:  # Only plot if we have data
            ax2.scatter(actual_data, predicted_data, alpha=0.5)
            max_val = max(actual_data.max(), predicted_data.max())
            ax2.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
            ax2.set_title(f'Worst Day (Correlation = {worst_day["correlation"]:.3f})')
            ax2.set_xlabel('Actual GHI')
            ax2.set_ylabel('Predicted GHI')
            ax2.legend()
        
        plt.suptitle(f'Correlation Analysis for {city}\nR² = {metrics_df["r2"].mean():.3f}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'correlation_analysis_{city}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main execution function."""
    # Define the configurations to evaluate
    configs = ['ghi_only', 'ghi_met', 'met_only']
    all_results = {}
    all_metrics = {}
    all_daily_metrics = {}
    
    # Create results directory if it doesn't exist
    results_dir = "results_joint"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load and prepare test data (only once)
    test_data_dict, target_scaler = load_and_prepare_test_data(CONFIG["data_locations"])
    
    for config in configs:
        print(f"\n{'='*80}")
        print(f"Evaluating {config} configuration")
        print(f"{'='*80}")
        
        # Load the trained model for this configuration
        model_path = os.path.join("models", f"lstm_ghi_forecast_joint_{config}.h5")
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}, skipping...")
            continue
        
        print(f"Loading model from {model_path}...")
        try:
            # Use custom_objects to handle both LSTM and Lambda layers
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    'LSTM': CustomLSTM,
                    'Lambda': CustomLambda
                }
            )
            
            # Print model summary and configuration
            print("\nModel Summary:")
            model.summary()
            
            # Print model configuration
            print("\nModel Configuration:")
            print(f"Input shape: {model.input_shape}")
            print(f"Output shape: {model.output_shape}")
            print(f"Number of layers: {len(model.layers)}")
            print(f"Trainable parameters: {model.count_params()}")
            
            # Validate model architecture
            print("\nValidating model architecture...")
            expected_layers = [
                'lstm', 'batch_normalization', 'dropout',
                'lstm_1', 'batch_normalization_1', 'dropout_1',
                'lstm_2', 'batch_normalization_2', 'dropout_2',
                'dense', 'batch_normalization_3', 'dropout_3',
                'dense_1'
            ]
            
            actual_layers = [layer.name.lower() for layer in model.layers]
            missing_layers = [layer for layer in expected_layers if layer not in actual_layers]
            
            if missing_layers:
                print("WARNING: Model architecture mismatch!")
                print("Missing layers:", missing_layers)
                print("Actual layers:", actual_layers)
                continue
            
            # Generate predictions
            print("\nGenerating predictions...")
            results = generate_predictions(model, test_data_dict, target_scaler, sequence_length=24, input_config=config)
            
            # Calculate metrics
            print("\nCalculating metrics...")
            metrics = calculate_performance_metrics(results)
            all_metrics[config] = metrics
            
            # Calculate daily metrics
            print("\nCalculating daily metrics...")
            daily_metrics = calculate_daily_correlations(results)
            all_daily_metrics[config] = daily_metrics
            
            # Save results
            all_results[config] = results
            
            # Plot results
            print("\nPlotting results...")
            plot_results(results, metrics, daily_metrics, config)
            
            print(f"\nResults for {config} configuration have been saved in the 'results_joint' directory")
            
        except Exception as e:
            print(f"Error processing {config} configuration: {str(e)}")
            continue
    
    # Compare configurations
    if len(all_metrics) > 1:
        print("\nComparing configurations:")
        compare_configurations(all_metrics, all_daily_metrics)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main() 