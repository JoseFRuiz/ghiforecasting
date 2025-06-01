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
    
    for city, data in results.items():
        actual = data['actual']
        predicted = data['predicted']
        
        # Filter out zero values (night time) as done in training
        non_zero_mask = actual > 0
        actual_nonzero = actual[non_zero_mask]
        predicted_nonzero = predicted[non_zero_mask]
        
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
        
        # Add statistics about data filtering
        metrics['Total_Samples'] = len(actual)
        metrics['Non_Zero_Samples'] = len(actual_nonzero)
        metrics['Zero_Percentage'] = (len(actual) - len(actual_nonzero)) / len(actual) * 100
        
        all_metrics[city] = metrics
    
    # Calculate overall metrics using non-zero values
    overall_metrics = {
        'MAE': mean_absolute_error(all_actual, all_predicted),
        'MSE': mean_squared_error(all_actual, all_predicted),
        'RMSE': np.sqrt(mean_squared_error(all_actual, all_predicted)),
        'R2': r2_score(all_actual, all_predicted),
        'MAPE': np.mean(np.abs((np.array(all_actual) - np.array(all_predicted)) / (np.array(all_actual) + 1e-10))) * 100,
        'Total_Samples': len(all_actual),
        'Non_Zero_Samples': len(all_actual),
        'Zero_Percentage': 0  # Since we've already filtered zeros
    }
    
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
    
    # Calculate and display performance metrics
    metrics = calculate_performance_metrics(results)
    
    # Calculate daily correlations
    print("\nCalculating daily correlations...")
    daily_metrics = calculate_daily_correlations(results)
    
    # Create results directory if it doesn't exist
    results_dir = "results_joint"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save metrics to CSV
    metrics_data = []
    for location, location_metrics in metrics.items():
        row = {'Location': location}
        row.update(location_metrics)
        metrics_data.append(row)
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = os.path.join(results_dir, "test_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to {metrics_path}")
    
    # Print metrics to console
    print("\nPerformance Metrics:")
    print("=" * 80)
    for location, location_metrics in metrics.items():
        print(f"\n{location}:")
        print("-" * 40)
        for metric_name, value in location_metrics.items():
            print(f"{metric_name}: {value:.4f}")
    
    # Print daily correlation statistics
    print("\nDaily Correlation Statistics:")
    print("=" * 80)
    for city, metrics_df in daily_metrics.items():
        print(f"\n{city}:")
        print("-" * 40)
        print(f"Number of days: {len(metrics_df)}")
        print(f"Mean correlation: {metrics_df['correlation'].mean():.4f}")
        print(f"Median correlation: {metrics_df['correlation'].median():.4f}")
        print(f"Min correlation: {metrics_df['correlation'].min():.4f}")
        print(f"Max correlation: {metrics_df['correlation'].max():.4f}")
        print(f"Days with correlation > 0.8: {len(metrics_df[metrics_df['correlation'] > 0.8])}")
        print(f"Days with correlation < 0.5: {len(metrics_df[metrics_df['correlation'] < 0.5])}")
    
    # Create plots for all data
    plot_predictions(results)
    
    # Create plots for daily correlations
    plot_daily_correlations(daily_metrics)
    
    # Create correlation analysis plots
    plot_correlation_vs_r2(daily_metrics, results)
    
    # Create plots for a complete day for each city
    for city in results.keys():
        complete_day = find_complete_day(results, city)
        if complete_day:
            print(f"\nFound complete day for {city}: {complete_day}")
            plot_single_day_predictions(results, city, complete_day)
        else:
            print(f"\nNo complete day found for {city}")
    
    print("\nPredictions and correlation analysis have been generated and saved in the 'results_joint' directory")

if __name__ == "__main__":
    main() 