"""
GHI Forecasting using LSTM with Individual Training
This script trains separate LSTM models for each location.
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
    
    return combined_df

def create_features(df):
    """
    Create features for individual model training.
    
    Args:
        df: DataFrame with data
    
    Returns:
        pd.DataFrame: DataFrame with additional features
    """
    print("\nCreating features...")
    print(f"Initial shape: {df.shape}")
    print(f"Initial date range: {df['datetime'].min()} to {df['datetime'].max()}")
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
    
    # Drop rows with missing values
    print("\nCleaning data...")
    print(f"Shape before cleaning: {df.shape}")
    print(f"Missing values per column:\n{df.isnull().sum()}")
    
    df_clean = df.dropna().reset_index(drop=True)
    
    print(f"\nShape after cleaning: {df_clean.shape}")
    print(f"Date range after cleaning: {df_clean['datetime'].min()} to {df_clean['datetime'].max()}")
    print(f"Missing values after cleaning:\n{df_clean.isnull().sum()}")
    print(f"Memory usage after cleaning: {df_clean.memory_usage().sum() / 1024 / 1024:.2f} MB")
    
    print(f"\nFeatures created: {df_clean.columns.tolist()}")
    return df_clean

def split_and_scale_data(df, sequence_length=24, target_column="GHI"):
    """
    Split and scale the data for individual model training.
    
    Args:
        df: DataFrame with data
        sequence_length: Length of input sequences
        target_column: Name of target column
    
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test, target_scaler)
    """
    print("\nSplitting and scaling data:")
    print(f"Total samples: {len(df)}")
    
    # Sort by datetime to ensure chronological order
    df = df.sort_values("datetime")
    
    # Calculate split indices
    n_samples = len(df)
    train_end = int(n_samples * 0.7)  # 70% for training
    val_end = int(n_samples * 0.85)   # 15% for validation
    
    # Split the data
    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]
    
    print("\nSplit sizes:")
    print(f"Train: {len(df_train)} samples")
    print(f"Validation: {len(df_val)} samples")
    print(f"Test: {len(df_test)} samples")
    
    # Scale target values
    target_scaler = MinMaxScaler()
    df_train[target_column] = target_scaler.fit_transform(df_train[[target_column]])
    df_val[target_column] = target_scaler.transform(df_val[[target_column]])
    df_test[target_column] = target_scaler.transform(df_test[[target_column]])
    
    # Create sequences
    X_train, y_train = create_sequences(df_train, sequence_length, target_column)
    X_val, y_val = create_sequences(df_val, sequence_length, target_column)
    X_test, y_test = create_sequences(df_test, sequence_length, target_column)
    
    print("\nFinal sequence shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, target_scaler

def create_sequences(df, sequence_length, target_column):
    """
    Create sequences for individual model training.
    
    Args:
        df: DataFrame with data
        sequence_length: Length of input sequences
        target_column: Name of target column
    
    Returns:
        tuple: (X, y) where X is the input sequences and y is the target values
    """
    print(f"\nCreating sequences for {len(df)} samples")
    
    # Get all required features
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
    data = df[feature_columns].values
    target = df[target_column].values
    
    # Calculate number of sequences
    n_sequences = len(df) - sequence_length
    
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
    
    print(f"Created {len(X)} valid sequences")
    return X, y

def create_model(input_shape):
    """Create and compile an LSTM model for individual location."""
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
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, location, batch_size=32, epochs=50):
    """Train the individual model with early stopping and model checkpointing."""
    print(f"\nTraining model for {location}")
    print(f"Training data shape: {X_train.shape}")
    
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
    log_dir = f"logs/fit/{location}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True
    )
    
    # Create model checkpoint callback
    checkpoint_path = f"models/lstm_ghi_forecast_{location}_checkpoint.h5"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    
    # Train the model
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

def evaluate_model(model, X_test, y_test, target_scaler, location):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained model
        X_test: Test input sequences
        y_test: Test target values
        target_scaler: Scaler used for target values
        location: Location name
    
    Returns:
        dict: Dictionary with evaluation metrics
    """
    print(f"\nEvaluating model for {location}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_original = target_scaler.inverse_transform(y_pred)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    r2 = r2_score(y_test_original, y_pred_original)
    
    # Calculate correlation
    correlation = np.corrcoef(y_test_original.flatten(), y_pred_original.flatten())[0, 1]
    
    # Create results dictionary
    results = {
        'location': location,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'correlation': correlation,
        'actual': y_test_original.flatten(),
        'predicted': y_pred_original.flatten()
    }
    
    # Print metrics
    print(f"\nMetrics for {location}:")
    print(f"MAE: {mae:.2f} W/m²")
    print(f"RMSE: {rmse:.2f} W/m²")
    print(f"R²: {r2:.4f}")
    print(f"Correlation: {correlation:.4f}")
    
    # Create and save plots
    plot_results(results, location)
    
    return results

def plot_results(results, location):
    """
    Create and save plots for model evaluation.
    
    Args:
        results: Dictionary with evaluation results
        location: Location name
    """
    # Create directory for plots
    plot_dir = f"results_individual/{location}"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(results['actual'], results['predicted'], alpha=0.5)
    plt.plot([0, max(results['actual'])], [0, max(results['actual'])], 'r--')
    plt.xlabel('Actual GHI (W/m²)')
    plt.ylabel('Predicted GHI (W/m²)')
    plt.title(f'Actual vs Predicted GHI - {location}')
    plt.savefig(f"{plot_dir}/scatter_plot.png")
    plt.close()
    
    # Create time series plot
    plt.figure(figsize=(15, 6))
    plt.plot(results['actual'], label='Actual', alpha=0.7)
    plt.plot(results['predicted'], label='Predicted', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('GHI (W/m²)')
    plt.title(f'GHI Time Series - {location}')
    plt.legend()
    plt.savefig(f"{plot_dir}/time_series.png")
    plt.close()

def main(skip_training=False, debug_data_loading=False):
    """
    Main execution function for individual training.
    
    Args:
        skip_training (bool): If True, load pre-trained models instead of training new ones
        debug_data_loading (bool): If True, stop after data loading for debugging
    """
    # Set random seeds for reproducibility
    np.random.seed(CONFIG["random_seed"])
    tf.random.set_seed(CONFIG["random_seed"])
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results_individual", exist_ok=True)
    
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
    
    # Process each location separately
    all_results = []
    
    for location in locations:
        print(f"\n{'='*80}")
        print(f"Processing {location}")
        print(f"{'='*80}")
        
        try:
            # Filter data for current location
            df_loc = df[df['location'] == location].copy()
            
            # Create features
            print("\nCreating features...")
            df_features = create_features(df_loc)
            
            # Split and scale data
            print("\nSplitting and scaling data...")
            (X_train, y_train, X_val, y_val, X_test, y_test, target_scaler) = split_and_scale_data(
                df_features, sequence_length=24, target_column="GHI"
            )
            
            # Create and train model
            print("\nCreating model...")
            model = create_model((X_train.shape[1], X_train.shape[2]))
            model.summary()
            
            model_path = os.path.join("models", f"lstm_ghi_forecast_{location}.h5")
            
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
                history, model = train_model(model, X_train, y_train, X_val, y_val, location)
                
                # Save model
                model.save(model_path)
                print(f"✓ Model saved to {model_path}")
            
            # Evaluate model
            print("\nEvaluating model...")
            results = evaluate_model(model, X_test, y_test, target_scaler, location)
            all_results.append(results)
            
        except Exception as e:
            print(f"× Error processing {location}: {str(e)}")
            import traceback
            print("\nFull traceback:")
            print(traceback.format_exc())
            continue
    
    # Create summary of results
    if all_results:
        summary_df = pd.DataFrame([{
            'Location': r['location'],
            'MAE': r['mae'],
            'RMSE': r['rmse'],
            'R²': r['r2'],
            'Correlation': r['correlation']
        } for r in all_results])
        
        summary_df.to_csv("results_individual/summary.csv", index=False)
        print("\nSummary of results saved to results_individual/summary.csv")
    
    print("\nAll locations have been processed")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='GHI Forecasting using Individual LSTM')
    parser.add_argument('--skip-training', action='store_true',
                      help='Skip training and load pre-trained models')
    parser.add_argument('--debug-data', action='store_true',
                      help='Stop after data loading for debugging')
    args = parser.parse_args()
    
    main(skip_training=args.skip_training, debug_data_loading=args.debug_data) 