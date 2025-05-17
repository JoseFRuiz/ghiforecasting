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

# Import Comet.ml first
from comet_ml import Experiment

# Then import ML libraries
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Concatenate, Input
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import configuration and functions from train.py
from train import (
    CONFIG, 
    setup_experiment, 
    plot_results, 
    plot_loss_history,
    load_data,
    prepare_lstm_input
)

def evaluate_model(y_true, y_pred):
    """
    Calculate various metrics for model evaluation, excluding zero GHI values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Create mask for non-zero true GHI values
    non_zero_mask = y_true > 0
    
    # Filter out zero values
    y_true_nonzero = y_true[non_zero_mask]
    y_pred_nonzero = y_pred[non_zero_mask]
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    
    # Initialize metrics dictionary
    metrics = {}
    
    # Calculate metrics for all values
    metrics.update({
        "Mean Absolute Error (All)": float(mean_absolute_error(y_true, y_pred)),
        "Mean Squared Error (All)": float(mean_squared_error(y_true, y_pred)),
        "Root Mean Squared Error (All)": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R² Score (All)": float(r2_score(y_true, y_pred)),
        "Mean Absolute Percentage Error (All)": float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100),
        "Mean Squared Percentage Error (All)": float(np.mean(np.square((y_true - y_pred) / (y_true + epsilon))) * 100)
    })
    
    # Calculate percentage of non-zero values
    metrics["Non-zero Values Percentage"] = float(100 * len(y_true_nonzero) / len(y_true))
    
    # Only calculate non-zero metrics if there are non-zero values
    if len(y_true_nonzero) > 0:
        metrics.update({
            "Mean Absolute Error (Non-zero)": float(mean_absolute_error(y_true_nonzero, y_pred_nonzero)),
            "Mean Squared Error (Non-zero)": float(mean_squared_error(y_true_nonzero, y_pred_nonzero)),
            "Root Mean Squared Error (Non-zero)": float(np.sqrt(mean_squared_error(y_true_nonzero, y_pred_nonzero))),
            "R² Score (Non-zero)": float(r2_score(y_true_nonzero, y_pred_nonzero)),
            "Mean Absolute Percentage Error (Non-zero)": float(np.mean(np.abs((y_true_nonzero - y_pred_nonzero) / (y_true_nonzero + epsilon))) * 100),
            "Mean Squared Percentage Error (Non-zero)": float(np.mean(np.square((y_true_nonzero - y_pred_nonzero) / (y_true_nonzero + epsilon))) * 100)
        })
    else:
        # If no non-zero values, set these metrics to NaN
        metrics.update({
            "Mean Absolute Error (Non-zero)": float('nan'),
            "Mean Squared Error (Non-zero)": float('nan'),
            "Root Mean Squared Error (Non-zero)": float('nan'),
            "R² Score (Non-zero)": float('nan'),
            "Mean Absolute Percentage Error (Non-zero)": float('nan'),
            "Mean Squared Percentage Error (Non-zero)": float('nan')
        })
    
    return metrics

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
    # One-hot encode location
    encoder = OneHotEncoder(sparse_output=False)
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
    
    # Split data by year
    df_train = df[df["Year"].isin([2017, 2018])].copy()
    df_2019 = df[df["Year"] == 2019].copy()
    split_index = len(df_2019) // 2
    df_val = df_2019.iloc[:split_index].copy()
    df_test = df_2019.iloc[split_index:].copy()
    
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
    
    return X_train, y_train, X_val, y_val, X_test, y_test, target_scaler

def create_joint_model(input_shape):
    """Create and compile the joint LSTM model."""
    # Clear any existing models/layers in memory
    tf.keras.backend.clear_session()
    
    # Create a more complex model to handle the larger combined dataset
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(CONFIG["model_params"]["dropout_rate"]),
        LSTM(64, return_sequences=True),
        Dropout(CONFIG["model_params"]["dropout_rate"]),
        LSTM(32, return_sequences=False),
        Dropout(CONFIG["model_params"]["dropout_rate"]),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="linear")
    ])
    
    # Create loss instance with explicit name
    mse_loss = tf.keras.losses.MeanSquaredError(name='mean_squared_error')
    
    model.compile(
        optimizer='adam',
        loss=mse_loss,
        metrics=['mae', 'mse']
    )
    return model

def evaluate_joint_model(model, X_test, y_test, target_scaler, locations):
    """
    Evaluate the joint model on each location separately.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        target_scaler: Scaler for target values
        locations: List of locations
    
    Returns:
        dict: Dictionary of metrics per location
    """
    results = {}
    
    # Get the location columns (they are the last len(locations) columns in the input)
    location_columns = X_test[:, 0, -len(locations):]
    
    print("\nEvaluating model for each location:")
    print(f"Total test samples: {len(X_test)}")
    print(f"Location columns shape: {location_columns.shape}")
    print(f"Available locations: {locations}")
    
    for i, location in enumerate(locations):
        # Get test data for this location using the one-hot encoded location columns
        location_mask = location_columns[:, i] == 1
        X_loc = X_test[location_mask]
        y_loc = y_test[location_mask]
        
        print(f"\nLocation: {location}")
        print(f"Number of test samples: {len(X_loc)}")
        print(f"Location mask sum: {np.sum(location_mask)}")
        
        if len(X_loc) == 0:
            print(f"Warning: No test data for {location}")
            print(f"Location mask statistics:")
            print(f"- Total samples: {len(location_mask)}")
            print(f"- True values: {np.sum(location_mask)}")
            print(f"- False values: {len(location_mask) - np.sum(location_mask)}")
            continue
        
        try:
            # Make predictions
            y_pred = model.predict(X_loc)
            y_pred_rescaled = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
            y_test_rescaled = target_scaler.inverse_transform(y_loc.reshape(-1, 1)).ravel()
            
            # Calculate metrics
            metrics = evaluate_model(y_test_rescaled, y_pred_rescaled)
            results[location] = metrics
            
            print(f"Results for {location}:")
            for metric_name, metric_value in metrics.items():
                print(f"✅ {metric_name}: {metric_value:.4f}")
                
        except Exception as e:
            print(f"Error evaluating {location}: {str(e)}")
            print(f"X_loc shape: {X_loc.shape}")
            print(f"y_loc shape: {y_loc.shape}")
            continue
    
    return results

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
    X_sequences = []
    y_sequences = []
    
    # Group data by location
    for location in locations:
        df_loc = df[df["location"] == location].copy()
        print(f"\nProcessing {location}:")
        print(f"Number of samples: {len(df_loc)}")
        
        if len(df_loc) == 0:
            print(f"Warning: No data for {location}")
            continue
        
        # Create sequences for this location
        for i in range(len(df_loc) - sequence_length):
            # Get sequence of features
            sequence = df_loc.iloc[i:i + sequence_length]
            
            # Create one-hot encoded location vector
            location_vector = np.zeros(len(locations))
            location_vector[locations.index(location)] = 1
            
            # Combine features with location encoding
            features = sequence[target_column].values
            features = np.column_stack([features, np.tile(location_vector, (sequence_length, 1))])
            
            # Get target value
            target = df_loc.iloc[i + sequence_length][target_column]
            
            X_sequences.append(features)
            y_sequences.append(target)
    
    # Convert to numpy arrays
    X = np.array(X_sequences)
    y = np.array(y_sequences)
    
    print(f"\nFinal sequence shapes:")
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")
    
    return X, y

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
    
    # Define feature combinations to test
    feature_combinations = [
        "ghi_only",
        "Temperature",
        "Relative Humidity",
        "Pressure",
        "Precipitable Water",
        "Wind Direction",
        "Wind Speed",
        "all",
        "meteorological_only"
    ]
    
    # Try to create experiment
    try:
        experiment = setup_experiment()
        if experiment:
            experiment.set_name("GHI_Forecasting_Joint_Training")
            print("✓ Experiment created successfully")
    except Exception as e:
        print(f"× Warning: Could not create experiment: {str(e)}")
        experiment = None
    
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
    
    # Dictionary to store results
    all_results = {
        'metrics': {},
        'histories': {},
        'predictions': {}
    }
    
    for feature_combo in feature_combinations:
        print(f"\nTesting feature combination: {feature_combo}")
        
        try:
            # Create features
            print("\nCreating features...")
            df_features = create_joint_features(df)
            
            # Split and scale data
            print("\nSplitting and scaling data...")
            (X_train, y_train, X_val, y_val, X_test, y_test, target_scaler) = split_and_scale_joint_data(df_features, locations)
            
            # Create and train model
            print("\nCreating model...")
            model = create_joint_model((X_train.shape[1], X_train.shape[2]))
            model.summary()
            
            model_path = os.path.join("models", f"lstm_ghi_forecast_joint_{feature_combo}.h5")
            
            if skip_training:
                if os.path.exists(model_path):
                    print(f"\nLoading pre-trained model for {feature_combo}...")
                    model = tf.keras.models.load_model(model_path)
                    history = None
                else:
                    print(f"× No pre-trained model found for {feature_combo}")
                    continue
            else:
                print("\nTraining model...")
                history = model.fit(
                    X_train, y_train,
                    epochs=CONFIG["model_params"]["epochs"],
                    batch_size=CONFIG["model_params"]["batch_size"],
                    validation_data=(X_val, y_val),
                    verbose=1
                )
                
                # Save model
                model.save(model_path)
                print(f"✓ Model saved to {model_path}")
            
            # Evaluate model
            print("\nEvaluating model...")
            results = evaluate_joint_model(model, X_test, y_test, target_scaler, locations)
            
            # Store results
            all_results['metrics'][feature_combo] = results
            all_results['histories'][feature_combo] = history
            
            # Log results if experiment exists
            if experiment:
                try:
                    # Log metrics
                    for location, metrics in results.items():
                        for metric_name, metric_value in metrics.items():
                            experiment.log_metric(f"{location}_{feature_combo}_{metric_name}", metric_value)
                    
                    # Log plots
                    if history:
                        loss_fig = plot_loss_history(history)
                        experiment.log_figure(figure_name=f"loss_history_{feature_combo}", figure=loss_fig)
                        plt.close()
                    
                    print("✓ Results logged to Comet.ml")
                except Exception as e:
                    print(f"× Warning: Could not log results: {str(e)}")
            
        except Exception as e:
            print(f"× Error processing {feature_combo}: {str(e)}")
            continue
    
    # Create comparative analysis
    print("\nGenerating comparative analysis...")
    try:
        # Create directories for results
        os.makedirs("results_joint", exist_ok=True)
        os.makedirs("results_joint/plots", exist_ok=True)
        os.makedirs("results_joint/tables", exist_ok=True)
        
        # Create metrics table
        records = []
        for feature_combo, location_metrics in all_results['metrics'].items():
            for location, metrics in location_metrics.items():
                for metric_name, value in metrics.items():
                    records.append({
                        'Location': location,
                        'Feature Combination': feature_combo,
                        'Metric': metric_name,
                        'Value': value
                    })
        
        metrics_df = pd.DataFrame(records)
        
        # Save results
        metrics_df.to_csv("results_joint/tables/joint_metrics.csv", index=False)
        
        # Create plots for each metric
        for metric in metrics_df['Metric'].unique():
            metric_data = metrics_df[metrics_df['Metric'] == metric]
            
            plt.figure(figsize=(15, 8))
            pivot_data = metric_data.pivot(
                index='Feature Combination',
                columns='Location',
                values='Value'
            )
            ax = pivot_data.plot(kind='bar', width=0.8)
            plt.title(f'{metric} Comparison Across Locations and Feature Combinations')
            plt.xlabel('Feature Combination')
            plt.ylabel(metric)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Location', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Add value labels
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', rotation=90, padding=3)
            
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save plot
            plot_path = f"results_joint/plots/joint_comparison_{metric.lower().replace(' ', '_')}.png"
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            plt.close()
        
        print("\nResults have been saved in the 'results_joint' directory")
        
    except Exception as e:
        print(f"× Error in comparative analysis: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='GHI Forecasting using Joint LSTM')
    parser.add_argument('--skip-training', action='store_true',
                      help='Skip training and load pre-trained models')
    parser.add_argument('--debug-data', action='store_true',
                      help='Stop after data loading for debugging')
    args = parser.parse_args()
    
    main(skip_training=args.skip_training, debug_data_loading=args.debug_data) 