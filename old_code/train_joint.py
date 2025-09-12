"""
GHI Forecasting using LSTM with Joint Training
This script trains a single LSTM model using data from all locations.
"""

import os
import numpy as np
import pandas as pd
import requests
from io import StringIO
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import datetime
import pickle

# Configure TensorFlow to use GPU if available
import tensorflow as tf

# Then import ML libraries
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import shared utilities and configurations
from utils import (
    CONFIG, 
    plot_results, 
    plot_loss_history,
    prepare_lstm_input,
    evaluate_model,
    load_data
)

def load_all_data(locations):
    """
    Load and combine data from all locations.
    
    Args:
        locations: Dictionary of location data from CONFIG
    
    Returns:
        pd.DataFrame: Combined dataset with location identifier
    """
    all_dfs = []
    
    for city in locations.keys():
        print(f"\n{'='*60}")
        print(f"Loading data for {city}")
        print(f"{'='*60}")
        try:
            # Use existing load_data function
            df = load_data(locations, city)
            
            # Add location identifier
            df['location'] = city
            
            # Print detailed information about the loaded data
            print(f"\nData loaded for {city}:")
            print(f"Number of rows: {len(df)}")
            print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"Memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")
            
            # Check for missing values
            missing_values = df.isnull().sum()
            print(f"\nMissing values per column:")
            print(missing_values[missing_values > 0])
            
            # Check for zero values in GHI
            zero_ghi = (df['GHI'] == 0).sum()
            print(f"\nZero GHI values: {zero_ghi} ({zero_ghi/len(df)*100:.2f}%)")
            
            all_dfs.append(df)
            print(f"✓ Successfully loaded {len(df)} rows for {city}")
            
        except Exception as e:
            print(f"× Error loading data for {city}:")
            print(str(e))
            import traceback
            print("\nFull traceback:")
            print(traceback.format_exc())
            continue
    
    if not all_dfs:
        raise ValueError("No data was loaded successfully for any city")
    
    # Combine all data
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n✓ Combined dataset: {len(combined_df)} total rows")
    print(f"Locations: {combined_df['location'].unique().tolist()}")
    
    # Print summary statistics for each location
    print("\nSummary statistics by location:")
    for location in combined_df['location'].unique():
        loc_df = combined_df[combined_df['location'] == location]
        print(f"\n{location}:")
        print(f"Number of rows: {len(loc_df)}")
        print(f"Date range: {loc_df['datetime'].min()} to {loc_df['datetime'].max()}")
        print(f"Zero GHI values: {(loc_df['GHI'] == 0).sum()} ({(loc_df['GHI'] == 0).sum()/len(loc_df)*100:.2f}%)")
        print(f"Missing values: {loc_df.isnull().sum().sum()}")
    
    return combined_df

def create_joint_features(df):
    """
    Create features for joint training, including location-specific features.
    
    Args:
        df: Combined DataFrame with data from all locations
    
    Returns:
        pd.DataFrame: DataFrame with additional features
    """
    print("\nCreating joint features...")
    print(f"Initial shape: {df.shape}")
    print(f"Initial locations: {df['location'].unique()}")
    print(f"Initial date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Initial rows per location:\n{df['location'].value_counts()}")
    print(f"Initial memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")
    
    # Create time-based features
    print("\nCreating time-based features...")
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    print("✓ Time-based features created")
    print(f"Shape after time features: {df.shape}")
    print(f"Rows per location after time features:\n{df['location'].value_counts()}")
    
    # Create target: GHI for next day at the same hour
    print("\nCreating target variable...")
    df["target_GHI"] = df["GHI"].shift(-24)
    print("✓ Target variable created")
    print(f"Shape after target creation: {df.shape}")
    print(f"Rows per location after target creation:\n{df['location'].value_counts()}")
    
    # Create lag features for previous 24 hours of GHI
    print("\nCreating GHI lag features...")
    for lag in range(1, 25):
        df[f"GHI_lag_{lag}"] = df["GHI"].shift(lag)
    print("✓ GHI lag features created")
    print(f"Shape after GHI lag features: {df.shape}")
    print(f"Rows per location after GHI lag features:\n{df['location'].value_counts()}")
    
    # Create lag features for meteorological variables
    print("\nCreating meteorological lag features...")
    meteorological_features = [
        "Temperature", "Relative Humidity", "Pressure",
        "Precipitable Water", "Wind Direction", "Wind Speed"
    ]
    
    for feature in meteorological_features:
        df[f"{feature}_lag_24"] = df[feature].shift(24)
    print("✓ Meteorological lag features created")
    print(f"Shape after meteorological features: {df.shape}")
    print(f"Rows per location after meteorological features:\n{df['location'].value_counts()}")
    
    # Create location-specific features
    print("\nCreating location features...")
    # One-hot encode location - using sparse=False for older scikit-learn versions
    encoder = OneHotEncoder(sparse=False)
    location_encoded = encoder.fit_transform(df[['location']])
    location_df = pd.DataFrame(location_encoded, 
                             columns=[f"location_{loc}" for loc in encoder.categories_[0]])
    
    # Combine with original data
    df = pd.concat([df, location_df], axis=1)
    print("✓ Location features created")
    print(f"Shape after location features: {df.shape}")
    print(f"Rows per location after location features:\n{df['location'].value_counts()}")
    
    # Drop rows with missing values
    print("\nCleaning data...")
    print(f"Shape before cleaning: {df.shape}")
    print(f"Missing values per column:\n{df.isnull().sum()}")
    print(f"Rows per location before cleaning:\n{df['location'].value_counts()}")
    
    df_clean = df.dropna().reset_index(drop=True)
    
    print(f"\nShape after cleaning: {df_clean.shape}")
    print(f"Rows per location after cleaning:\n{df_clean['location'].value_counts()}")
    print(f"Date range after cleaning: {df_clean['datetime'].min()} to {df_clean['datetime'].max()}")
    print(f"Missing values after cleaning:\n{df_clean.isnull().sum()}")
    print(f"Memory usage after cleaning: {df_clean.memory_usage().sum() / 1024 / 1024:.2f} MB")
    
    print(f"\nFeatures created: {df_clean.columns.tolist()}")
    return df_clean

def split_and_scale_joint_data(df, locations, sequence_length=24, target_column="GHI"):
    """
    Split and scale the data for joint model training.
    
    Args:
        df: DataFrame with all data
        locations: List of locations
        sequence_length: Length of input sequences
        target_column: Name of target column
    
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test, target_scaler)
    """
    print("\nSplitting and scaling data:")
    print(f"Total samples: {len(df)}")
    print(f"Locations: {locations}")
    
    # Create a more balanced split that ensures each location has data in all splits
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    for location in locations:
        df_loc = df[df["location"] == location].copy()
        
        # Sort by datetime to ensure chronological order
        df_loc = df_loc.sort_values("datetime")
        
        # Calculate split indices
        n_samples = len(df_loc)
        train_end = int(n_samples * 0.7)  # 70% for training
        val_end = int(n_samples * 0.85)   # 15% for validation
        
        # Split the data
        train_dfs.append(df_loc.iloc[:train_end])
        val_dfs.append(df_loc.iloc[train_end:val_end])
        test_dfs.append(df_loc.iloc[val_end:])
    
    # Combine splits
    df_train = pd.concat(train_dfs, ignore_index=True)
    df_val = pd.concat(val_dfs, ignore_index=True)
    df_test = pd.concat(test_dfs, ignore_index=True)
    
    print("\nSplit sizes:")
    print(f"Train: {len(df_train)} samples")
    print(f"Validation: {len(df_val)} samples")
    print(f"Test: {len(df_test)} samples")
    
    # Print location distribution in each split
    for split_name, split_df in [("Train", df_train), ("Validation", df_val), ("Test", df_test)]:
        print(f"\n{split_name} split location distribution:")
        for location in locations:
            count = len(split_df[split_df["location"] == location])
            print(f"{location}: {count} samples")
    
    # Scale target values
    target_scaler = MinMaxScaler()
    df_train[target_column] = target_scaler.fit_transform(df_train[[target_column]])
    df_val[target_column] = target_scaler.transform(df_val[[target_column]])
    df_test[target_column] = target_scaler.transform(df_test[[target_column]])
    
    # Save the target scaler
    scaler_path = os.path.join("models", "target_scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(target_scaler, f)
    print(f"\nTarget scaler saved to {scaler_path}")
    
    # Create sequences for each location
    X_train, y_train = create_sequences_joint(df_train, locations, sequence_length, target_column)
    X_val, y_val = create_sequences_joint(df_val, locations, sequence_length, target_column)
    X_test, y_test = create_sequences_joint(df_test, locations, sequence_length, target_column)
    
    print("\nFinal sequence shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")
    
    print("y_train min/max after scaling:", df_train[target_column].min(), df_train[target_column].max())
    
    return X_train, y_train, X_val, y_val, X_test, y_test, target_scaler

def create_joint_model(input_shape):
    """Create and compile a more complex joint LSTM model with attention mechanism."""
    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # First LSTM block with residual connection
    x = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Second LSTM block with residual connection
    lstm2 = tf.keras.layers.LSTM(128, return_sequences=True)(x)
    lstm2 = tf.keras.layers.BatchNormalization()(lstm2)
    lstm2 = tf.keras.layers.Dropout(0.3)(lstm2)
    x = tf.keras.layers.Add()([x, lstm2])  # Residual connection
    
    # Third LSTM block
    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Attention mechanism
    attention = tf.keras.layers.Dense(1, activation='tanh')(x)
    attention = tf.keras.layers.Flatten()(attention)
    attention_weights = tf.keras.layers.Activation('softmax')(attention)
    attention_weights = tf.keras.layers.RepeatVector(64)(attention_weights)
    attention_weights = tf.keras.layers.Permute([2, 1])(attention_weights)
    
    # Apply attention weights
    x = tf.keras.layers.Multiply()([x, attention_weights])
    x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(x)
    
    # Dense layers with residual connection
    dense1 = tf.keras.layers.Dense(32, activation='relu')(x)
    dense1 = tf.keras.layers.BatchNormalization()(dense1)
    dense1 = tf.keras.layers.Dropout(0.3)(dense1)
    
    dense2 = tf.keras.layers.Dense(32, activation='relu')(dense1)
    dense2 = tf.keras.layers.BatchNormalization()(dense2)
    dense2 = tf.keras.layers.Dropout(0.3)(dense2)
    
    # Final residual connection
    x = tf.keras.layers.Add()([dense1, dense2])
    
    # Output layer
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Use Adam optimizer with a lower learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
    # Compile model with Huber loss for robustness
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(),
        metrics=['mae']
    )
    
    return model

def evaluate_joint_model(model, test_data, locations, sequence_length, target_column, input_config='all'):
    """
    Evaluate the model on each location separately.
    
    Args:
        model: Trained model
        test_data: DataFrame with test data
        locations: List of locations
        sequence_length: Length of input sequences
        target_column: Name of target column
        input_config: Input configuration used for training
    
    Returns:
        dict: Dictionary with metrics for each location
    """
    print("\nEvaluating model on test data...")
    print(f"Test data shape: {test_data.shape}")
    print(f"Locations: {locations}")
    print(f"Input configuration: {input_config}")
    
    results = {}
    daily_metrics = {}
    
    for location in locations:
        print(f"\nEvaluating {location}:")
        df_loc = test_data[test_data["location"] == location].copy()
        print(f"Number of test samples: {len(df_loc)}")
        
        if len(df_loc) == 0:
            print(f"Warning: No test data for {location}")
            continue
        
        # Check for missing values
        missing_values = df_loc.isnull().sum()
        if missing_values.any():
            print(f"Warning: Missing values in {location}:")
            print(missing_values[missing_values > 0])
        
        # Check for zero values in target
        zero_target = (df_loc[target_column] == 0).sum()
        print(f"Zero {target_column} values: {zero_target} ({zero_target/len(df_loc)*100:.2f}%)")
        
        # Create sequences for evaluation with the correct configuration
        X_test, y_test = create_sequences_joint_with_config(df_loc, locations, sequence_length, target_column, input_config)
        
        if len(X_test) == 0:
            print(f"Warning: No valid sequences created for {location}")
            continue
        
        print(f"Test sequences: {len(X_test)}")
        print(f"Input shape: {X_test.shape}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Store results for this location
        results[location] = {
            'dates': df_loc['datetime'].values[-len(y_test):],  # Use the last len(y_test) dates
            'actual': y_test,
            'predicted': y_pred.flatten()
        }
        
        # Calculate daily metrics
        dates = pd.to_datetime(results[location]['dates'])
        df = pd.DataFrame({
            'date': dates,
            'actual': results[location]['actual'],
            'predicted': results[location]['predicted']
        })
        
        # Group by date and calculate daily metrics
        daily_stats = []
        for date, group in df.groupby(df['date'].dt.date):
            # Filter out zero values
            non_zero_mask = group['actual'] > 0
            if non_zero_mask.sum() > 0:  # Only include days with non-zero values
                actual_nonzero = group.loc[non_zero_mask, 'actual']
                predicted_nonzero = group.loc[non_zero_mask, 'predicted']
                
                # Calculate correlation using pandas corr() method
                correlation = actual_nonzero.corr(predicted_nonzero)
                
                # Add error checking for correlation
                if pd.isna(correlation) or not (-1 <= correlation <= 1):
                    print(f"Warning: Invalid correlation value {correlation} for {location} on {date}")
                    correlation = 0  # Set to 0 if invalid
                
                # Calculate other metrics
                mae = mean_absolute_error(actual_nonzero, predicted_nonzero)
                rmse = np.sqrt(mean_squared_error(actual_nonzero, predicted_nonzero))
                r2 = r2_score(actual_nonzero, predicted_nonzero)
                
                # Add error checking for R²
                if not (0 <= r2 <= 1):
                    print(f"Warning: Invalid R² value {r2} for {location} on {date}")
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
        daily_metrics[location] = pd.DataFrame(daily_stats).sort_values('date')
        
        # Print summary of correlation values and GHI statistics
        print(f"\nSummary for {location}:")
        print("-" * 40)
        print("Correlation Statistics:")
        print(f"Min: {daily_metrics[location]['correlation'].min():.4f}")
        print(f"Max: {daily_metrics[location]['correlation'].max():.4f}")
        print(f"Mean: {daily_metrics[location]['correlation'].mean():.4f}")
        print(f"Median: {daily_metrics[location]['correlation'].median():.4f}")
        
        print("\nActual GHI Statistics:")
        print(f"Max: {daily_metrics[location]['actual_max'].max():.2f} W/m²")
        print(f"Mean: {daily_metrics[location]['actual_mean'].mean():.2f} W/m²")
        print(f"Std: {daily_metrics[location]['actual_std'].mean():.2f} W/m²")
        
        print("\nPredicted GHI Statistics:")
        print(f"Max: {daily_metrics[location]['predicted_max'].max():.2f} W/m²")
        print(f"Mean: {daily_metrics[location]['predicted_mean'].mean():.2f} W/m²")
        print(f"Std: {daily_metrics[location]['predicted_std'].mean():.2f} W/m²")
    
    # Create plots for daily correlations
    plot_daily_correlations(daily_metrics)
    
    # Create correlation analysis plots
    plot_correlation_vs_r2(daily_metrics, results)
    
    return results, daily_metrics

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

def create_sequences_joint(df, locations, sequence_length, target_column):
    """
    Create sequences for joint model training.
    
    Args:
        df: DataFrame with data
        locations: List of locations
        sequence_length: Length of input sequences
        target_column: Name of target column
    
    Returns:
        tuple: (X, y) where X is the input sequences and y is the target values
    """
    print(f"\nCreating sequences for {len(df)} samples")
    print(f"Locations: {locations}")
    
    # Initialize lists to store sequences
    all_X_sequences = []
    all_y_sequences = []
    
    # Process each location separately
    for location in locations:
        print(f"\nProcessing {location}...")
        df_loc = df[df['location'] == location].copy()
        
        if len(df_loc) == 0:
            print(f"No data found for {location}")
            continue
            
        # Create one-hot encoded location vector for the current location
        location_vector = np.zeros(len(locations))
        location_vector[locations.index(location)] = 1
        
        # Sort by datetime to ensure chronological order
        df_loc = df_loc.sort_values("datetime")
        
        # Get all required features at once
        feature_columns = (
            [f"GHI_lag_{lag}" for lag in range(1, 25)] +  # GHI lag features
            [target_column] +  # Current GHI
            [  # Meteorological features
                "Temperature_lag_24", "Relative Humidity_lag_24",
                "Pressure_lag_24", "Precipitable Water_lag_24",
                "Wind Direction_lag_24", "Wind Speed_lag_24"
            ] +
            [  # Time features
                "hour_sin", "hour_cos",
                "month_sin", "month_cos"
            ]
        )
        
        # Get all data at once
        data = df_loc[feature_columns].values
        target = df_loc[target_column].values
        
        # Calculate number of sequences
        n_sequences = len(df_loc) - sequence_length
        
        # Pre-allocate arrays
        X = np.zeros((n_sequences, sequence_length, len(feature_columns)))
        y = np.zeros(n_sequences)
        
        # Create sequences using vectorized operations
        for i in range(n_sequences):
            X[i] = data[i:i + sequence_length]
            y[i] = target[i + sequence_length]
        
        # Filter out sequences where target is zero (night time)
        mask = y > 0
        X = X[mask]
        y = y[mask]
        
        # Add location features
        location_features = np.tile(location_vector, (len(X), sequence_length, 1))
        X = np.concatenate([X, location_features], axis=2)
        
        all_X_sequences.append(X)
        all_y_sequences.append(y)
        
        print(f"Created {len(X)} valid sequences for {location}")
    
    if not all_X_sequences:
        raise ValueError("No valid sequences were created for any location")
    
    # Combine sequences from all locations
    X = np.concatenate(all_X_sequences, axis=0)
    y = np.concatenate(all_y_sequences, axis=0)
    
    print(f"\nFinal sequence shapes:")
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")
    
    # Print sequence distribution by location
    print("\nSequence distribution by location:")
    for i, location in enumerate(locations):
        location_mask = X[:, 0, -len(locations):][:, i] == 1
        print(f"{location}: {np.sum(location_mask)} sequences")
    
    return X, y

def train_joint_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=50):
    """Train the joint model with early stopping and model checkpointing."""
    # Print training data statistics
    print("\nTraining data statistics:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_train min: {np.min(X_train):.4f}")
    print(f"X_train max: {np.max(X_train):.4f}")
    print(f"X_train mean: {np.mean(X_train):.4f}")
    print(f"X_train std: {np.std(X_train):.4f}")
    print(f"y_train min: {np.min(y_train):.4f}")
    print(f"y_train max: {np.max(y_train):.4f}")
    print(f"y_train mean: {np.mean(y_train):.4f}")
    print(f"y_train std: {np.std(y_train):.4f}")
    
    # Create callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Add learning rate reduction on plateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Add TensorBoard callback for monitoring
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True
    )
    
    # Add custom callback to monitor predictions
    class PredictionMonitor(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 5 == 0:  # Check every 5 epochs
                # Get predictions for a few samples
                sample_preds = self.model.predict(X_val[:5])
                print(f"\nEpoch {epoch} - Sample predictions:")
                print(f"Predictions: {sample_preds.flatten()}")
                print(f"Actual values: {y_val[:5]}")
                print(f"Prediction stats - Min: {np.min(sample_preds):.4f}, Max: {np.max(sample_preds):.4f}, Mean: {np.mean(sample_preds):.4f}")
    
    # Create model checkpoint callback
    checkpoint_path = "models/joint_model_checkpoint.h5"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    
    # Train the model with all callbacks
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            early_stopping,
            reduce_lr,
            tensorboard_callback,
            model_checkpoint,
            PredictionMonitor()
        ],
        verbose=1
    )
    
    # Print final model weights statistics
    print("\nFinal model weights statistics:")
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
    
    return history, model

def create_sequences_joint_with_config(df, locations, sequence_length, target_column, input_config='all'):
    """
    Create sequences for joint model training with different input configurations.
    
    Args:
        df: DataFrame with data
        locations: List of locations
        sequence_length: Length of input sequences
        target_column: Name of target column
        input_config: One of ['ghi_only', 'ghi_met', 'met_only', 'all']
    
    Returns:
        tuple: (X, y) where X is the input sequences and y is the target values
    """
    print(f"\nCreating sequences for {len(df)} samples with config: {input_config}")
    print(f"Locations: {locations}")
    
    # Initialize feature lists based on configuration
    feature_columns = []
    
    if input_config in ['ghi_only', 'ghi_met', 'all']:
        # Add GHI lag features (24 features)
        feature_columns.extend([f"GHI_lag_{lag}" for lag in range(1, 25)])
        # Add current GHI
        feature_columns.append(target_column)
    
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
    
    # Initialize lists to store sequences
    all_X_sequences = []
    all_y_sequences = []
    
    # Process each location separately
    for location in locations:
        print(f"\nProcessing {location}...")
        df_loc = df[df['location'] == location].copy()
        
        if len(df_loc) == 0:
            print(f"No data found for {location}")
            continue
            
        # Create one-hot encoded location vector for the current location
        location_vector = np.zeros(len(locations))
        location_vector[locations.index(location)] = 1
        
        # Sort by datetime to ensure chronological order
        df_loc = df_loc.sort_values("datetime")
        
        # Get all required data at once
        data = df_loc[feature_columns].values
        target = df_loc[target_column].values
        
        # Calculate number of sequences
        n_sequences = len(df_loc) - sequence_length
        
        # Pre-allocate arrays
        X = np.zeros((n_sequences, sequence_length, len(feature_columns)))
        y = np.zeros(n_sequences)
        
        # Create sequences using vectorized operations
        for i in range(n_sequences):
            X[i] = data[i:i + sequence_length]
            y[i] = target[i + sequence_length]
        
        # Filter out sequences where target is zero (night time)
        mask = y > 0
        X = X[mask]
        y = y[mask]
        
        # Add location features if using 'all' configuration
        if input_config == 'all':
            location_features = np.tile(location_vector, (len(X), sequence_length, 1))
            X = np.concatenate([X, location_features], axis=2)
        
        all_X_sequences.append(X)
        all_y_sequences.append(y)
        
        print(f"Created {len(X)} valid sequences for {location}")
    
    if not all_X_sequences:
        raise ValueError("No valid sequences were created for any location")
    
    # Combine sequences from all locations
    X = np.concatenate(all_X_sequences, axis=0)
    y = np.concatenate(all_y_sequences, axis=0)
    
    print(f"\nFinal sequence shapes:")
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")
    
    # Print target value statistics
    print("\nTarget value statistics:")
    print(f"Min: {np.min(y):.4f}")
    print(f"Max: {np.max(y):.4f}")
    print(f"Mean: {np.mean(y):.4f}")
    print(f"Std: {np.std(y):.4f}")
    
    return X, y

def create_joint_model_with_config(input_shape, input_config='all'):
    """Create and compile a joint LSTM model with different input configurations."""
    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # First LSTM block
    x = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Second LSTM block
    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Third LSTM block
    x = tf.keras.layers.LSTM(32)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Dense layers
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Output layer with linear activation
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Use Adam optimizer with a lower learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Compile model with Huber loss for robustness
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(),
        metrics=['mae', 'mse']
    )
    
    # Print model summary
    model.summary()
    
    return model

def main(skip_training=False, debug_data_loading=False):
    """
    Main execution function for joint training.
    
    Args:
        skip_training (bool): If True, load pre-trained model instead of training new one
        debug_data_loading (bool): If True, stop after data loading for debugging
    """
    # Set random seeds for reproducibility
    np.random.seed(CONFIG["random_seed"])
    tf.random.set_seed(CONFIG["random_seed"])
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results_joint", exist_ok=True)
    os.makedirs("results_joint/plots", exist_ok=True)
    os.makedirs("results_joint/tables", exist_ok=True)
    
    # Load and process data
    print("\nLoading and processing data...")
    try:
        df = load_all_data(CONFIG["data_locations"])
        print("✓ Data loaded successfully")
        
        if debug_data_loading:
            print("\nDebug mode: Stopping after data loading")
            return
            
    except Exception as e:
        print(f"× Error loading data: {str(e)}")
        return
    
    # Get unique locations
    locations = sorted(df['location'].unique().tolist())
    print(f"\nLocations: {locations}")
    
    # Define input configurations to test
    input_configs = ['ghi_only', 'ghi_met', 'met_only']
    
    for config in input_configs:
        print(f"\n{'='*80}")
        print(f"Training model with {config} configuration")
        print(f"{'='*80}")
        
        try:
            # Create features
            print("\nCreating features...")
            df_features = create_joint_features(df)
            
            # Split and scale data
            print("\nSplitting and scaling data...")
            (X_train, y_train, X_val, y_val, X_test, y_test, target_scaler) = split_and_scale_joint_data(
                df_features, locations, sequence_length=24, target_column="GHI"
            )
            
            # Create sequences with specific configuration
            X_train, y_train = create_sequences_joint_with_config(
                df_features[df_features['location'].isin(locations)], 
                locations, 24, "GHI", input_config=config
            )
            X_val, y_val = create_sequences_joint_with_config(
                df_features[df_features['location'].isin(locations)], 
                locations, 24, "GHI", input_config=config
            )
            X_test, y_test = create_sequences_joint_with_config(
                df_features[df_features['location'].isin(locations)], 
                locations, 24, "GHI", input_config=config
            )
            
            # Create and train model
            print("\nCreating model...")
            model = create_joint_model_with_config((X_train.shape[1], X_train.shape[2]), input_config=config)
            model.summary()
            
            model_path = os.path.join("models", f"lstm_ghi_forecast_joint_{config}.h5")
            
            if skip_training:
                if os.path.exists(model_path):
                    print(f"\nLoading pre-trained model...")
                    model = tf.keras.models.load_model(model_path)
                    history = None
                else:
                    print("× No pre-trained model found")
                    continue
            else:
                print("\nTraining model...")
                history, model = train_joint_model(model, X_train, y_train, X_val, y_val)
                
                # Save model
                model.save(model_path)
                print(f"✓ Model saved to {model_path}")
            
            # Evaluate model
            print("\nEvaluating model...")
            results, daily_metrics = evaluate_joint_model(model, df_features, locations, 24, "GHI", input_config=config)
            
            # Create metrics table
            records = []
            for location, metrics in daily_metrics.items():
                for metric_name in ['correlation', 'mae', 'rmse', 'r2']:
                    records.append({
                        'Location': location,
                        'Config': config,
                        'Metric': metric_name,
                        'Mean': metrics[metric_name].mean(),
                        'Median': metrics[metric_name].median(),
                        'Min': metrics[metric_name].min(),
                        'Max': metrics[metric_name].max()
                    })
            
            metrics_df = pd.DataFrame(records)
            
            # Save results
            metrics_df.to_csv(f"results_joint/tables/metrics_{config}.csv", index=False)
            
            print(f"\nResults for {config} configuration have been saved in the 'results_joint' directory")
            
            # Check model output range during training
            y_val_pred = model.predict(X_val)
            print("Validation predictions (should be in [0, 1]):", y_val_pred[:10].flatten())
            
        except Exception as e:
            print(f"× Error in {config} configuration: {str(e)}")
            import traceback
            print("\nFull traceback:")
            print(traceback.format_exc())
            continue
    
    print("\nAll configurations have been processed")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='GHI Forecasting using Joint LSTM')
    parser.add_argument('--skip-training', action='store_true',
                      help='Skip training and load pre-trained models')
    parser.add_argument('--debug-data', action='store_true',
                      help='Stop after data loading for debugging')
    args = parser.parse_args()
    
    main(skip_training=args.skip_training, debug_data_loading=args.debug_data) 