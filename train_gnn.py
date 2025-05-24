"""
GHI Forecasting using Graph Neural Networks
This script trains a GNN model to forecast Global Horizontal Irradiance (GHI)
using historical weather data and GHI measurements from multiple locations.
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

# Then import ML libraries
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import shared utilities and configurations
from utils import (
    CONFIG, 
    setup_experiment, 
    plot_results, 
    plot_loss_history,
    prepare_lstm_input,
    evaluate_model
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

def create_gnn_features(df):
    """
    Create features for GNN training, including location-specific features.
    
    Args:
        df: Combined DataFrame with data from all locations
    
    Returns:
        pd.DataFrame: DataFrame with additional features
    """
    print("\nCreating GNN features...")
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
    
    # Create target: GHI for next day at the same hour
    print("\nCreating target variable...")
    df["target_GHI"] = df["GHI"].shift(-24)
    print("✓ Target variable created")
    
    # Create lag features for previous 24 hours of GHI
    print("\nCreating GHI lag features...")
    for lag in range(1, 25):
        df[f"GHI_lag_{lag}"] = df["GHI"].shift(lag)
    print("✓ GHI lag features created")
    
    # Create lag features for meteorological variables
    print("\nCreating meteorological lag features...")
    meteorological_features = [
        "Temperature", "Relative Humidity", "Pressure",
        "Precipitable Water", "Wind Direction", "Wind Speed"
    ]
    
    for feature in meteorological_features:
        df[f"{feature}_lag_24"] = df[feature].shift(24)
    print("✓ Meteorological lag features created")
    
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
    
    # Drop rows with missing values
    print("\nCleaning data...")
    df_clean = df.dropna().reset_index(drop=True)
    
    print(f"\nFinal shape: {df_clean.shape}")
    print(f"Features created: {df_clean.columns.tolist()}")
    return df_clean

def create_gnn_model(input_shape, num_locations):
    """Create and compile the GNN model with GPU optimizations."""
    # Clear any existing models/layers in memory
    tf.keras.backend.clear_session()
    
    # Configure GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")
    
    # Input layers
    ghi_input = tf.keras.layers.Input(shape=input_shape, name='ghi_input')
    met_input = tf.keras.layers.Input(shape=input_shape, name='met_input')
    location_input = tf.keras.layers.Input(shape=(num_locations,), name='location_input')
    
    # GHI processing branch
    ghi_lstm = tf.keras.layers.LSTM(128, return_sequences=True, 
                    kernel_initializer='glorot_uniform',
                    recurrent_initializer='orthogonal',
                    recurrent_activation='sigmoid',
                    time_major=False)(ghi_input)
    ghi_dropout = tf.keras.layers.Dropout(CONFIG["model_params"]["dropout_rate"])(ghi_lstm)
    ghi_lstm2 = tf.keras.layers.LSTM(64, return_sequences=False,
                     kernel_initializer='glorot_uniform',
                     recurrent_initializer='orthogonal',
                     recurrent_activation='sigmoid',
                     time_major=False)(ghi_dropout)
    
    # Meteorological processing branch
    met_lstm = tf.keras.layers.LSTM(64, return_sequences=True,
                    kernel_initializer='glorot_uniform',
                    recurrent_initializer='orthogonal',
                    recurrent_activation='sigmoid',
                    time_major=False)(met_input)
    met_dropout = tf.keras.layers.Dropout(CONFIG["model_params"]["dropout_rate"])(met_lstm)
    met_lstm2 = tf.keras.layers.LSTM(32, return_sequences=False,
                     kernel_initializer='glorot_uniform',
                     recurrent_initializer='orthogonal',
                     recurrent_activation='sigmoid',
                     time_major=False)(met_dropout)
    
    # Location processing branch
    location_dense = tf.keras.layers.Dense(32, activation='relu')(location_input)
    
    # Combine all branches
    combined = tf.keras.layers.Concatenate()([ghi_lstm2, met_lstm2, location_dense])
    dense1 = tf.keras.layers.Dense(64, activation='relu')(combined)
    dropout1 = tf.keras.layers.Dropout(CONFIG["model_params"]["dropout_rate"])(dense1)
    dense2 = tf.keras.layers.Dense(32, activation='relu')(dropout1)
    output = tf.keras.layers.Dense(1, activation='linear')(dense2)
    
    # Create model
    model = tf.keras.Model(inputs=[ghi_input, met_input, location_input], outputs=output)
    
    # Use Adam optimizer with learning rate schedule
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    return model

def split_and_scale_gnn_data(df, locations, sequence_length=24, target_column="GHI"):
    """
    Split and scale the data for GNN model training.
    
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
    
    # Scale target values
    target_scaler = MinMaxScaler()
    df_train[target_column] = target_scaler.fit_transform(df_train[[target_column]])
    df_val[target_column] = target_scaler.transform(df_val[[target_column]])
    df_test[target_column] = target_scaler.transform(df_test[[target_column]])
    
    # Create sequences for each location
    X_train, y_train = create_gnn_sequences(df_train, locations, sequence_length, target_column)
    X_val, y_val = create_gnn_sequences(df_val, locations, sequence_length, target_column)
    X_test, y_test = create_gnn_sequences(df_test, locations, sequence_length, target_column)
    
    print("\nFinal sequence shapes:")
    print(f"X_train: {[x.shape for x in X_train]}")
    print(f"y_train: {y_train.shape}")
    print(f"X_val: {[x.shape for x in X_val]}")
    print(f"y_val: {y_val.shape}")
    print(f"X_test: {[x.shape for x in X_test]}")
    print(f"y_test: {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, target_scaler

def create_gnn_sequences(df, locations, sequence_length, target_column):
    """
    Create sequences for GNN model training.
    
    Args:
        df: DataFrame with data
        locations: List of locations
        sequence_length: Length of input sequences
        target_column: Name of target column
    
    Returns:
        tuple: (X, y) where X is a list of input sequences and y is the target values
    """
    print(f"\nCreating sequences for {len(df)} samples")
    print(f"Locations: {locations}")
    
    # Initialize lists to store sequences
    ghi_sequences = []
    met_sequences = []
    location_sequences = []
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
        
        # Create one-hot encoded location vector
        location_vector = np.zeros(len(locations))
        location_vector[locations.index(location)] = 1
        
        # Calculate the maximum index that allows for a complete sequence
        max_index = len(df_loc) - sequence_length - 1  # -1 to ensure we have a target value
        
        # Create sequences using rolling window approach
        for i in range(max_index):
            # Get sequence window
            sequence_window = df_loc.iloc[i:i + sequence_length]
            target_window = df_loc.iloc[i + sequence_length:i + sequence_length + 1]
            
            # Skip if any data is missing
            if sequence_window.isnull().any().any() or target_window.isnull().any().any():
                continue
            
            # Get GHI sequence features
            ghi_features = sequence_window[target_column].values
            
            # Get meteorological features
            met_features = sequence_window[[
                "Temperature_lag_24", "Relative Humidity_lag_24",
                "Pressure_lag_24", "Precipitable Water_lag_24",
                "Wind Direction_lag_24", "Wind Speed_lag_24"
            ]].values
            
            # Get target value
            target_value = target_window[target_column].values[0]
            
            # Skip if target value is zero (night time)
            if target_value == 0:
                continue
            
            # Store sequences
            ghi_sequences.append(ghi_features)
            met_sequences.append(met_features)
            location_sequences.append(location_vector)
            y_sequences.append(target_value)
    
    # Convert to numpy arrays
    X = [
        np.array(ghi_sequences),
        np.array(met_sequences),
        np.array(location_sequences)
    ]
    y = np.array(y_sequences)
    
    print(f"\nFinal sequence shapes:")
    print(f"GHI sequences: {X[0].shape}")
    print(f"Meteorological sequences: {X[1].shape}")
    print(f"Location sequences: {X[2].shape}")
    print(f"Target values: {y.shape}")
    
    return X, y

def train_gnn_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=50):
    """Train the GNN model with early stopping and model checkpointing."""
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
    checkpoint_path = "models/gnn_model_checkpoint.h5"
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
            model_checkpoint
        ],
        verbose=1
    )
    
    return history, model

def evaluate_gnn_model(model, test_data, locations, sequence_length, target_column):
    """
    Evaluate the GNN model on each location separately.
    
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
        
        # Create sequences for evaluation
        X_test, y_test = create_gnn_sequences(df_loc, [location], sequence_length, target_column)
        
        if len(X_test[0]) == 0:
            print(f"Warning: No valid sequences created for {location}")
            continue
        
        print(f"Test sequences: {len(X_test[0])}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = evaluate_model(y_test, y_pred.ravel())
        
        print(f"Metrics for {location}:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
        
        results[location] = metrics
    
    return results

def main(skip_training=False, debug_data_loading=False):
    """
    Main execution function for GNN training.
    
    Args:
        skip_training (bool): If True, load pre-trained model instead of training new one
        debug_data_loading (bool): If True, stop after data loading for debugging
    """
    # Set random seeds for reproducibility
    np.random.seed(CONFIG["random_seed"])
    tf.random.set_seed(CONFIG["random_seed"])
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    
    # Try to create experiment
    try:
        experiment = setup_experiment()
        if experiment:
            experiment.set_name("GHI_Forecasting_GNN")
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
    
    try:
        # Create features
        print("\nCreating features...")
        df_features = create_gnn_features(df)
        
        # Split and scale data
        print("\nSplitting and scaling data...")
        (X_train, y_train, X_val, y_val, X_test, y_test, target_scaler) = split_and_scale_gnn_data(df_features, locations)
        
        # Create and train model
        print("\nCreating model...")
        model = create_gnn_model((X_train[0].shape[1], X_train[0].shape[2]), len(locations))
        model.summary()
        
        model_path = os.path.join("models", "gnn_ghi_forecast.h5")
        
        if skip_training:
            if os.path.exists(model_path):
                print(f"\nLoading pre-trained model...")
                model = tf.keras.models.load_model(model_path)
                history = None
            else:
                print("× No pre-trained model found")
                return
        else:
            print("\nTraining model...")
            history, model = train_gnn_model(model, X_train, y_train, X_val, y_val)
            
            # Save model
            model.save(model_path)
            print(f"✓ Model saved to {model_path}")
        
        # Evaluate model
        print("\nEvaluating model...")
        results = evaluate_gnn_model(model, df_features, locations, 24, "GHI")
        
        # Log results if experiment exists
        if experiment:
            try:
                # Log metrics
                for location, metrics in results.items():
                    for metric_name, metric_value in metrics.items():
                        experiment.log_metric(f"{location}_{metric_name}", metric_value)
                
                # Log plots
                if history:
                    loss_fig = plot_loss_history(history)
                    experiment.log_figure(figure_name="loss_history", figure=loss_fig)
                    plt.close()
                
                print("✓ Results logged to Comet.ml")
            except Exception as e:
                print(f"× Warning: Could not log results: {str(e)}")
        
    except Exception as e:
        print(f"× Error in main execution: {str(e)}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='GHI Forecasting using GNN')
    parser.add_argument('--skip-training', action='store_true',
                      help='Skip training and load pre-trained models')
    parser.add_argument('--debug-data', action='store_true',
                      help='Stop after data loading for debugging')
    args = parser.parse_args()
    
    main(skip_training=args.skip_training, debug_data_loading=args.debug_data) 