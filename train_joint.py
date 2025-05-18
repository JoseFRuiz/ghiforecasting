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

# Import Comet.ml first
from comet_ml import Experiment

# Configure TensorFlow to use GPU if available
import tensorflow as tf

# Check for GPU availability and configure
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Enable memory growth to avoid allocating all GPU memory at once
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"Found {len(physical_devices)} GPU(s):")
        for device in physical_devices:
            print(f"  - {device}")
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")
else:
    print("No GPU devices found. Running on CPU.")

import pdb; pdb.set_trace()

# Then import ML libraries
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
    
    # Set mixed precision policy for better GPU performance
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    # Create a more complex model to handle the larger combined dataset
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape, 
             kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal'),
        Dropout(CONFIG["model_params"]["dropout_rate"]),
        LSTM(64, return_sequences=True,
             kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal'),
        Dropout(CONFIG["model_params"]["dropout_rate"]),
        LSTM(32, return_sequences=False,
             kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal'),
        Dropout(CONFIG["model_params"]["dropout_rate"]),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="linear")
    ])
    
    # Create loss instance with explicit name
    mse_loss = tf.keras.losses.MeanSquaredError(name='mean_squared_error')
    
    # Use Adam optimizer with learning rate schedule
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(
        optimizer=optimizer,
        loss=mse_loss,
        metrics=['mae', 'mse']
    )
    
    return model

def evaluate_joint_model(model, test_data, locations, sequence_length, target_column):
    """
    Evaluate the model on each location separately.
    
    Args:
        model: Trained model
        test_data: DataFrame with test data
        locations: List of locations
        sequence_length: Length of input sequences
        target_column: Name of target column
    
    Returns:
        dict: Dictionary with metrics for each location
    """
    print("\nEvaluating model on test data...")
    print(f"Test data shape: {test_data.shape}")
    print(f"Locations: {locations}")
    
    results = {}
    
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
        
        # Create sequences for evaluation
        X_test, y_test = create_sequences_joint(df_loc, [location], sequence_length, target_column)
        
        if len(X_test) == 0:
            print(f"Warning: No valid sequences created for {location}")
            continue
        
        print(f"Test sequences: {len(X_test)}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Metrics for {location}:")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")
        
        results[location] = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2
        }
    
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
        
        # Sort by datetime to ensure chronological order
        df_loc = df_loc.sort_values("datetime")
        
        # Check for missing values
        missing_values = df_loc.isnull().sum()
        if missing_values.any():
            print(f"Warning: Missing values in {location}:")
            print(missing_values[missing_values > 0])
        
        # Check for zero values in target
        zero_target = (df_loc[target_column] == 0).sum()
        print(f"Zero {target_column} values: {zero_target} ({zero_target/len(df_loc)*100:.2f}%)")
        
        # Create sequences for this location
        valid_sequences = 0
        skipped_sequences = 0
        
        # Calculate the maximum index that allows for a complete sequence
        max_index = len(df_loc) - sequence_length
        
        for i in range(max_index):
            # Get sequence of features
            sequence = df_loc.iloc[i:i + sequence_length]
            
            # Verify chronological order
            if not sequence['datetime'].is_monotonic_increasing:
                print(f"Warning: Non-chronological sequence found at index {i}")
                skipped_sequences += 1
                continue
            
            # Check if sequence has any missing values
            if sequence.isnull().any().any():
                skipped_sequences += 1
                continue
            
            # Create one-hot encoded location vector
            location_vector = np.zeros(len(locations))
            location_vector[locations.index(location)] = 1
            
            # Combine features with location encoding
            features = sequence[target_column].values
            features = np.column_stack([features, np.tile(location_vector, (sequence_length, 1))])
            
            # Get target value
            target = df_loc.iloc[i + sequence_length][target_column]
            
            # Verify target value is not in the sequence
            if target in features[:, 0]:
                print(f"Warning: Target value found in sequence at index {i}")
                skipped_sequences += 1
                continue
            
            X_sequences.append(features)
            y_sequences.append(target)
            valid_sequences += 1
        
        print(f"Created {valid_sequences} valid sequences for {location}")
        if skipped_sequences > 0:
            print(f"Skipped {skipped_sequences} sequences due to validation failures")
    
    # Convert to numpy arrays
    X = np.array(X_sequences)
    y = np.array(y_sequences)
    
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
    
    # Create model checkpoint callback
    checkpoint_path = "models/joint_model_checkpoint.h5"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    
    # Calculate steps per epoch
    steps_per_epoch = len(X_train) // batch_size
    
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
            model_checkpoint
        ],
        verbose=1
    )
    
    return history, model

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
                history, model = train_joint_model(model, X_train, y_train, X_val, y_val)
                
                # Save model
                model.save(model_path)
                print(f"✓ Model saved to {model_path}")
            
            # Evaluate model
            print("\nEvaluating model...")
            results = evaluate_joint_model(model, X_test, locations, 24, "GHI")
            
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