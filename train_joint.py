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
    evaluate_model, 
    plot_results, 
    plot_loss_history,
    load_data,
    prepare_lstm_input
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
        print(f"\nLoading data for {city}...")
        try:
            # Use existing load_data function
            df = load_data(locations, city)
            
            # Add location identifier
            df['location'] = city
            
            all_dfs.append(df)
            print(f"✓ Successfully loaded {len(df)} rows for {city}")
            
        except Exception as e:
            print(f"× Error loading data for {city}: {str(e)}")
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
    
    # Create location-specific features
    # One-hot encode location
    encoder = OneHotEncoder(sparse_output=False)
    location_encoded = encoder.fit_transform(df[['location']])
    location_df = pd.DataFrame(location_encoded, 
                             columns=[f"location_{loc}" for loc in encoder.categories_[0]])
    
    # Combine with original data
    df = pd.concat([df, location_df], axis=1)
    
    # Drop rows with missing values
    df_clean = df.dropna().reset_index(drop=True)
    print(f"Shape after cleaning: {df_clean.shape}")
    print(f"Features created: {df_clean.columns.tolist()}")
    
    return df_clean

def split_and_scale_joint_data(df, feature_combination="all"):
    """
    Split and scale data for joint training.
    
    Args:
        df: Combined DataFrame
        feature_combination: String specifying which features to use
    
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test), target_scaler
    """
    print("\nSplitting and scaling joint data...")
    
    # Split data by year
    df_train = df[df["Year"].isin([2017, 2018])].copy()
    df_2019 = df[df["Year"] == 2019].copy()
    split_index = len(df_2019) // 2
    df_val = df_2019.iloc[:split_index].copy()
    df_test = df_2019.iloc[split_index:].copy()
    
    print(f"Training set size: {len(df_train)}")
    print(f"Validation set size: {len(df_val)}")
    print(f"Test set size: {len(df_test)}")
    
    # Define feature groups
    ghi_features = ["GHI"] + [f"GHI_lag_{lag}" for lag in range(1, 25)]
    meteorological_features = {
        "Temperature": ["Temperature_lag_24"],
        "Relative Humidity": ["Relative Humidity_lag_24"],
        "Pressure": ["Pressure_lag_24"],
        "Precipitable Water": ["Precipitable Water_lag_24"],
        "Wind Direction": ["Wind Direction_lag_24"],
        "Wind Speed": ["Wind Speed_lag_24"]
    }
    location_features = [col for col in df.columns if col.startswith("location_")]
    
    # Select features based on combination
    if feature_combination == "ghi_only":
        selected_features = ghi_features + location_features
    elif feature_combination == "meteorological_only":
        selected_features = [f for features in meteorological_features.values() for f in features] + location_features
    elif feature_combination in meteorological_features:
        selected_features = ghi_features + meteorological_features[feature_combination] + location_features
    elif feature_combination == "all":
        selected_features = (ghi_features + 
                           [f for features in meteorological_features.values() for f in features] + 
                           location_features)
    else:
        raise ValueError(f"Invalid feature combination: {feature_combination}")
    
    print(f"Selected features: {selected_features}")
    
    try:
        # Initialize scalers
        ghi_scaler = MinMaxScaler(feature_range=CONFIG["feature_ranges"]["ghi"])
        meteorological_scaler = MinMaxScaler(feature_range=CONFIG["feature_ranges"]["meteorological"])
        target_scaler = MinMaxScaler(feature_range=CONFIG["feature_ranges"]["ghi"])
        
        # Scale features
        for df_split in [df_train, df_val, df_test]:
            if df_split is df_train:
                print("\nFitting and transforming training data...")
                # Scale GHI features
                df_split.loc[:, ghi_features] = ghi_scaler.fit_transform(df_split[ghi_features])
                
                # Scale meteorological features if needed
                if feature_combination != "ghi_only":
                    met_features = [f for f in selected_features if f not in ghi_features and not f.startswith("location_")]
                    if met_features:
                        df_split.loc[:, met_features] = meteorological_scaler.fit_transform(df_split[met_features])
                
                # Scale target
                target_values = df_split["target_GHI"].values.reshape(-1, 1)
                df_split.loc[:, "target_GHI"] = target_scaler.fit_transform(target_values).ravel()
            else:
                print("\nTransforming validation/test data...")
                # Transform using fitted scalers
                df_split.loc[:, ghi_features] = ghi_scaler.transform(df_split[ghi_features])
                
                # Transform meteorological features if needed
                if feature_combination != "ghi_only":
                    met_features = [f for f in selected_features if f not in ghi_features and not f.startswith("location_")]
                    if met_features:
                        df_split.loc[:, met_features] = meteorological_scaler.transform(df_split[met_features])
                
                # Transform target
                target_values = df_split["target_GHI"].values.reshape(-1, 1)
                df_split.loc[:, "target_GHI"] = target_scaler.transform(target_values).ravel()
        
        # Prepare features for LSTM
        feature_columns = [col for col in selected_features 
                          if col not in ["datetime", "GHI", "target_GHI", "Year", "Month", "Day", "Hour", "Minute", "location"]]
        
        print(f"\nFeature columns for LSTM: {feature_columns}")
        
        X_train, y_train = prepare_lstm_input(df_train, feature_columns, "target_GHI")
        X_val, y_val = prepare_lstm_input(df_val, feature_columns, "target_GHI")
        X_test, y_test = prepare_lstm_input(df_test, feature_columns, "target_GHI")
        
        print(f"\nFinal shapes:")
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
        print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        return (X_train, y_train, X_val, y_val, X_test, y_test), target_scaler
    
    except Exception as e:
        print(f"Error in split_and_scale_joint_data: {str(e)}")
        raise

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
    
    for location in locations:
        # Get test data for this location
        location_mask = X_test[:, 0, -len(locations):].argmax(axis=1) == locations.index(location)
        X_loc = X_test[location_mask]
        y_loc = y_test[location_mask]
        
        if len(X_loc) == 0:
            print(f"Warning: No test data for {location}")
            continue
        
        # Make predictions
        y_pred = model.predict(X_loc)
        y_pred_rescaled = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        y_test_rescaled = target_scaler.inverse_transform(y_loc.reshape(-1, 1)).ravel()
        
        # Calculate metrics
        metrics = evaluate_model(y_test_rescaled, y_pred_rescaled)
        results[location] = metrics
        
        print(f"\nResults for {location}:")
        for metric_name, metric_value in metrics.items():
            print(f"✅ {metric_name}: {metric_value:.4f}")
    
    return results

def main(skip_training=False):
    """
    Main execution function for joint training.
    
    Args:
        skip_training (bool): If True, load pre-trained model instead of training new one
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
            (X_train, y_train, X_val, y_val, X_test, y_test), target_scaler = split_and_scale_joint_data(df_features, feature_combo)
            
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
    args = parser.parse_args()
    
    main(skip_training=args.skip_training) 