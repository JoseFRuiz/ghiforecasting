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

def create_features(df):
    """
    Create features for prediction.
    
    Args:
        df: DataFrame with raw data
    
    Returns:
        pd.DataFrame: DataFrame with additional features
    """
    # Create time-based features
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    
    # Create target: GHI for next day at the same hour
    df["target_GHI"] = df["GHI"].shift(-24)
    
    # Create lag features for previous 24 hours of GHI
    for lag in range(1, 25):
        df[f"GHI_lag_{lag}"] = df["GHI"].shift(lag)
    
    # Create lag features for meteorological variables
    meteorological_features = [
        "Temperature", "Relative Humidity", "Pressure",
        "Precipitable Water", "Wind Direction", "Wind Speed"
    ]
    
    for feature in meteorological_features:
        df[f"{feature}_lag_24"] = df[feature].shift(24)
    
    return df

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
        raise FileNotFoundError(f"Target scaler not found at {scaler_path}. Please run train_joint.py first.")
    
    with open(scaler_path, 'rb') as f:
        target_scaler = pickle.load(f)
    
    # Load and process data for each location
    test_data_dict = {}
    
    # List of cities to process (excluding Leh and Kargil which have invalid data)
    valid_cities = ["Jaisalmer", "Jodhpur", "New Delhi", "Shimla", "Srinagar"]
    
    for city in valid_cities:
        print(f"\nProcessing {city}...")
        
        try:
            # Use the load_data function from utils
            df = load_data(locations, city)
            
            # Create features
            df = create_features(df)
            
            # Validate features
            required_features = [
                'GHI', 'Temperature', 'Relative Humidity', 'Pressure',
                'Precipitable Water', 'Wind Direction', 'Wind Speed'
            ]
            
            missing_features = [f for f in required_features if f not in df.columns]
            if missing_features:
                print(f"Error: Missing required features: {missing_features}")
                continue
            
            # Check for missing values
            missing_values = df[required_features].isnull().sum()
            if missing_values.any():
                print("Warning: Missing values found:")
                print(missing_values[missing_values > 0])
                # Fill missing values with forward fill and then backward fill
                df[required_features] = df[required_features].fillna(method='ffill').fillna(method='bfill')
            
            # Create lag features
            print("Creating lag features...")
            for feature in required_features:
                for lag in range(1, 25):
                    df[f"{feature}_lag_{lag}"] = df[feature].shift(lag)
            
            # Create time features
            print("Creating time features...")
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
            test_data_dict[city] = df
            
            print(f"Successfully processed {len(df)} samples for {city}")
            
        except Exception as e:
            print(f"Error processing {city}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            continue
    
    if not test_data_dict:
        raise ValueError("No valid test data was processed. Please check your data paths and run train_joint.py first.")
    
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
    
    # Initialize feature lists based on configuration
    feature_columns = []
    
    if input_config in ['ghi_only', 'ghi_met', 'all']:
        # Add GHI lag features (24 features)
        feature_columns.extend([f"GHI_lag_{lag}" for lag in range(1, 25)])
        # Add current GHI
        feature_columns.append("GHI")
    
    if input_config in ['met_only', 'ghi_met', 'all']:
        # Add meteorological features
        met_features = [
            "Temperature_lag_24", "Relative Humidity_lag_24",
            "Pressure_lag_24", "Precipitable Water_lag_24",
            "Wind Direction_lag_24", "Wind Speed_lag_24"
        ]
        feature_columns.extend(met_features)
    
    # Always add time features
    time_features = ["hour_sin", "hour_cos", "month_sin", "month_cos"]
    feature_columns.extend(time_features)
    
    # Get all required data at once
    data = df[feature_columns].values
    dates = df['datetime'].values
    target = df['GHI'].values
    
    # Calculate number of sequences
    n_sequences = len(df) - sequence_length
    
    # Pre-allocate arrays
    X = np.zeros((n_sequences, sequence_length, len(feature_columns)))
    y = np.zeros(n_sequences)
    sequence_dates = np.zeros(n_sequences, dtype='datetime64[ns]')
    
    # Create sequences using vectorized operations
    for i in range(n_sequences):
        X[i] = data[i:i + sequence_length]
        y[i] = target[i + sequence_length]
        sequence_dates[i] = dates[i + sequence_length]
    
    # Filter out sequences where target is zero (night time)
    mask = y > 0
    X = X[mask]
    y = y[mask]
    sequence_dates = sequence_dates[mask]
    
    print(f"\nCreated {len(X)} sequences")
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y, sequence_dates

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
    
    for city, test_data in test_data_dict.items():
        print(f"Processing {city}...")
        
        # Create sequences with the correct configuration
        X_test, y_test, dates = create_prediction_sequences(test_data, sequence_length, input_config)
        
        # Validate input data
        if np.isnan(X_test).any():
            X_test = np.nan_to_num(X_test, nan=0.0)
        
        if np.isinf(X_test).any():
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Get hour information for each prediction
        hours = pd.to_datetime(dates).hour
        
        # Scale target values
        y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1))
        
        # Validate scaled target values
        if np.isnan(y_test_scaled).any() or np.isinf(y_test_scaled).any():
            y_test_scaled = np.nan_to_num(y_test_scaled, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Generate predictions
        y_pred_scaled = model.predict(X_test)
        
        # Validate predictions
        if np.isnan(y_pred_scaled).any() or np.isinf(y_pred_scaled).any():
            y_pred_scaled = np.nan_to_num(y_pred_scaled, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Ensure predictions are within valid range [0, 1] for inverse transform
        y_pred_scaled = np.clip(y_pred_scaled, 0, 1)
        
        # Inverse transform predictions and actual values
        y_pred = target_scaler.inverse_transform(y_pred_scaled)
        y_test = target_scaler.inverse_transform(y_test_scaled)
        
        # Set predictions to 0 for night time hours (before sunrise and after sunset)
        night_mask = (hours < 6) | (hours > 18)
        y_pred[night_mask] = 0
        
        # Store results
        results[city] = {
            'dates': dates,
            'actual': y_test.flatten(),
            'predicted': y_pred.flatten()
        }
        
        # Print basic metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
    
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

def save_hourly_ghi_values(results, save_dir="results_joint/hourly_ghi"):
    """
    Save hourly GHI values (actual and predicted) to CSV files for each location.
    
    Args:
        results: Dictionary with predictions and actual values for each location
        save_dir: Directory to save CSV files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for city, data in results.items():
        # Create DataFrame with dates, actual and predicted values
        df = pd.DataFrame({
            'datetime': pd.to_datetime(data['dates']),
            'actual_ghi': data['actual'],
            'predicted_ghi': data['predicted']
        })
        
        # Sort by datetime
        df = df.sort_values('datetime')
        
        # Save to CSV
        output_file = os.path.join(save_dir, f'hourly_ghi_{city}.csv')
        df.to_csv(output_file, index=False)
        print(f"Saved hourly GHI values for {city} to {output_file}")

def compare_configurations(all_metrics, all_daily_metrics):
    """
    Compare performance metrics across different model configurations.
    
    Args:
        all_metrics: Dictionary of metrics for each configuration
        all_daily_metrics: Dictionary of daily metrics for each configuration
    """
    print("\nComparing model configurations...")
    
    # Create comparison table for overall metrics
    comparison_data = []
    
    for config, metrics in all_metrics.items():
        # Get overall metrics for this configuration
        overall = metrics['Overall']
        
        comparison_data.append({
            'Configuration': config,
            'MAE': overall['MAE'],
            'RMSE': overall['RMSE'],
            'R²': overall['R2'],
            'MAPE': overall['MAPE'],
            'Correlation': overall['Correlation'],
            'Mean Actual': overall['Mean_Actual'],
            'Mean Predicted': overall['Mean_Predicted']
        })
    
    # Create DataFrame and sort by MAE
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('MAE')
    
    # Save comparison to CSV
    comparison_df.to_csv("results_joint/configuration_comparison.csv", index=False)
    
    # Print comparison table
    print("\nConfiguration Comparison (sorted by MAE):")
    print(comparison_df.to_string(index=False))
    
    # Create comparison plots
    os.makedirs("results_joint/comparisons", exist_ok=True)
    
    # Plot MAE and RMSE comparison
    plt.figure(figsize=(12, 6))
    x = np.arange(len(comparison_df))
    width = 0.35
    
    plt.bar(x - width/2, comparison_df['MAE'], width, label='MAE')
    plt.bar(x + width/2, comparison_df['RMSE'], width, label='RMSE')
    
    plt.xlabel('Configuration')
    plt.ylabel('Error (W/m²)')
    plt.title('MAE and RMSE Comparison Across Configurations')
    plt.xticks(x, comparison_df['Configuration'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results_joint/comparisons/error_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot R² and Correlation comparison
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, comparison_df['R²'], width, label='R²')
    plt.bar(x + width/2, comparison_df['Correlation'], width, label='Correlation')
    
    plt.xlabel('Configuration')
    plt.ylabel('Score')
    plt.title('R² and Correlation Comparison Across Configurations')
    plt.xticks(x, comparison_df['Configuration'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results_joint/comparisons/score_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Compare daily metrics across configurations
    for city in all_daily_metrics[list(all_daily_metrics.keys())[0]].keys():
        plt.figure(figsize=(15, 10))
        
        # Create subplots for different metrics
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        for config, daily_metrics in all_daily_metrics.items():
            city_metrics = daily_metrics[city]
            
            # Plot correlation
            ax1.plot(city_metrics['date'], city_metrics['correlation'], 
                    label=f'{config} (mean: {city_metrics["correlation"].mean():.3f})')
            
            # Plot R²
            ax2.plot(city_metrics['date'], city_metrics['r2'],
                    label=f'{config} (mean: {city_metrics["r2"].mean():.3f})')
            
            # Plot MAE
            ax3.plot(city_metrics['date'], city_metrics['mae'],
                    label=f'{config} (mean: {city_metrics["mae"].mean():.1f})')
            
            # Plot RMSE
            ax4.plot(city_metrics['date'], city_metrics['rmse'],
                    label=f'{config} (mean: {city_metrics["rmse"].mean():.1f})')
        
        # Configure subplots
        ax1.set_title('Daily Correlation')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Correlation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Daily R²')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('R²')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.set_title('Daily MAE')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('MAE (W/m²)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4.set_title('Daily RMSE')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('RMSE (W/m²)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        for ax in [ax1, ax2, ax3, ax4]:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.suptitle(f'Daily Metrics Comparison for {city}')
        plt.tight_layout()
        plt.savefig(f"results_joint/comparisons/daily_metrics_{city}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print("\nComparison results have been saved in the 'results_joint/comparisons' directory")

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
    print("Loading and preparing test data...")
    test_data_dict, target_scaler = load_and_prepare_test_data(CONFIG["data_locations"])
    
    for config in configs:
        print(f"\nEvaluating {config} configuration...")
        
        # Load the trained model for this configuration
        model_path = os.path.join("models", f"lstm_ghi_forecast_joint_{config}.h5")
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}, skipping...")
            continue
        
        try:
            # Load model with custom objects
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    'LSTM': CustomLSTM,
                    'Lambda': CustomLambda
                }
            )
            
            # Generate predictions
            results = generate_predictions(model, test_data_dict, target_scaler, sequence_length=24, input_config=config)
            
            # Calculate metrics
            metrics = calculate_performance_metrics(results)
            all_metrics[config] = metrics
            
            # Calculate daily metrics
            daily_metrics = calculate_daily_correlations(results)
            all_daily_metrics[config] = daily_metrics
            
            # Save results
            all_results[config] = results
            
            # Save hourly GHI values
            save_hourly_ghi_values(results)
            
            # Plot results (optional, comment out if not needed)
            # plot_results(results, metrics, daily_metrics, config)
            
            print(f"Completed evaluation for {config} configuration")
            
        except Exception as e:
            print(f"Error processing {config} configuration: {str(e)}")
            continue
    
    # Compare configurations
    if len(all_metrics) > 1:
        print("\nComparing configurations...")
        compare_configurations(all_metrics, all_daily_metrics)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main() 