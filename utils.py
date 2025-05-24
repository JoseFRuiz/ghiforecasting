"""
Shared utilities and configurations for GHI forecasting.
"""

import os
import numpy as np
import pandas as pd
import requests
from io import StringIO
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from comet_ml import Experiment

# Configuration
CONFIG = {
    "random_seed": 42,
    "comet_api_key": "qVaqH5DmdCc2UWO5If2SIm1bz",
    "project_name": "ghi-forecasting",
    "data_locations": {
        "Jaisalmer": {
            2017: "1O-dOOvSbkTwUv1Qyh33RCzDcOh5GiL2y",
            2018: "1JgIxhAck67nxXFAPrKHcX8Ql-w4wXXB_",
            2019: "1ayaT36iSigV5V7DG-haWVM8NO-kCfTv3"
        },
        "Jodhpur": {
            2017: "18s0BoJdrSlnqv7mI5jqIcD3NYvl5CNpw",
            2018: "18pspcKFTaTo94DNn8qoRbR252znQbZ04", 
            2019: "18p_xr2ewN7Sqf2kRkV0buWsXgyD4dWOS"
        },
        "New Delhi": {
            2017: "1972ErnUOXb-Z-QzYLTjRJhIdYS3Kgl6_",
            2018: "192p04Rc5F2yGAFjCdc-0G4YrECjZnwfz", 
            2019: "191jSccrBqmcQyVuwhjLnbn2iEIa6Y0hC"
        },
        "Shimla": {
            2017: "19MmMPJXihn4tQZiIIH7jC4MIJtTkQ39W",
            2018: "19KLWzCzmAE1mFAC1mOYu6ENS3Lkr-C-8", 
            2019: "19GyRRmqlUaVvLPSvCCFh7Y18FJz5k03B"
        },
        "Srinagar": {
            2017: "19Vf7YBXyK2CcFaNoY88PCoy6wl5BlbyN",
            2018: "19TIzqgh0F2bc1NxpYUEJJ_yITViPO01P", 
            2019: "19YPkh-yYam6a_G--8axdiejmKomIjnmz"
        },
        "Leh": {
            2017: "1-2-3-4-5",
            2018: "1-2-3-4-5",
            2019: "1-2-3-4-5"
        },
        "Kargil": {
            2017: "1-2-3-4-5",
            2018: "1-2-3-4-5",
            2019: "1-2-3-4-5"
        }
    },
    "model_params": {
        "lstm_units": [64, 32],
        "dense_units": [16, 1],
        "dropout_rate": 0.2,
        "batch_size": 32,
        "epochs": 20
    },
    "feature_ranges": {
        "ghi": (0, 1.2),
        "meteorological": (0, 1)
    }
}

def setup_experiment():
    """Initialize and configure Comet.ml experiment."""
    try:
        experiment = Experiment(
            api_key=CONFIG["comet_api_key"],
            project_name=CONFIG["project_name"],
            workspace=None,
            log_git=False,  # Disable git logging
            log_env_details=False,  # Disable environment logging
            log_env_host=False,  # Disable host logging
            log_env_cpu=False,  # Disable CPU logging
            log_env_gpu=False,  # Disable GPU logging
            auto_metric_logging=False,  # Disable automatic metric logging
            auto_param_logging=False,  # Disable automatic parameter logging
            auto_output_logging=False,  # Disable automatic output logging
            log_code=False  # Disable code logging
        )
        
        # Log only essential information
        experiment.log_parameter("random_seed", CONFIG["random_seed"])
        experiment.log_parameter("model_params", CONFIG["model_params"])
        experiment.log_parameter("feature_ranges", CONFIG["feature_ranges"])
        
        print("✓ Successfully created Comet.ml experiment with minimal logging")
        return experiment
    except Exception as e:
        print(f"× Error setting up Comet.ml experiment: {str(e)}")
        print("Will continue without Comet.ml logging")
        return None

def load_data(locations, city):
    """
    Load and preprocess the GHI data from multiple years for a specific city.
    Downloads data from Google Drive if local files don't exist.
    """
    try:
        print(f"\n{'='*60}")
        print(f"Loading data for {city}")
        print(f"{'='*60}")
        dfs = []
        
        for year in [2017, 2018, 2019]:
            try:
                local_file = f"data_{city}_{year}.csv"
                print(f"\nProcessing {year} data...")
                
                # Check if file exists and is not empty
                if os.path.exists(local_file) and os.path.getsize(local_file) > 0:
                    print(f"✓ Found local file: {local_file} ({os.path.getsize(local_file)/1024:.2f} KB)")
                else:
                    print(f"× File not found or empty: {local_file}")
                    file_id = locations[city][year]
                    url = f'https://drive.google.com/uc?export=download&id={file_id}'
                    
                    print(f"Downloading from Google Drive...")
                    try:
                        # First request to get the confirmation token if needed
                        session = requests.Session()
                        response = session.get(url, stream=True)
                        
                        # Check if there's a download warning (large file)
                        for key, value in response.cookies.items():
                            if key.startswith('download_warning'):
                                token = value
                                url = f"{url}&confirm={token}"
                                response = session.get(url, stream=True)
                                break
                        
                        # Save the file
                        if response.status_code == 200:
                            with open(local_file, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=1024):
                                    if chunk:
                                        f.write(chunk)
                            print(f"✓ Downloaded and saved file ({os.path.getsize(local_file)/1024:.2f} KB)")
                        else:
                            raise ValueError(f"Download failed with status {response.status_code}")
                    except Exception as e:
                        print(f"× Download error: {str(e)}")
                        raise
                
                # Try reading the file
                print("Reading data file...")
                try:
                    # First check the file content
                    with open(local_file, 'r', encoding='utf-8') as f:
                        first_lines = [next(f) for _ in range(5)]
                    print("First few lines of the file:")
                    for i, line in enumerate(first_lines):
                        print(f"Line {i+1}: {line.strip()}")
                    
                    # Now try reading with pandas
                    df = pd.read_csv(local_file, skiprows=2)
                    
                    if len(df) == 0:
                        raise ValueError("File is empty after reading")
                    
                    # Validate required columns
                    expected_cols = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'GHI', 'Temperature']
                    missing_cols = [col for col in expected_cols if col not in df.columns]
                    if missing_cols:
                        raise ValueError(f"Missing required columns: {missing_cols}")
                    
                    print(f"✓ Successfully loaded {len(df)} rows")
                    print(f"Columns: {df.columns.tolist()}")
                    dfs.append(df)
                    
                except Exception as e:
                    print(f"× Error reading file: {str(e)}")
                    if os.path.exists(local_file):
                        os.remove(local_file)
                    raise
                    
            except Exception as e:
                print(f"× Error processing {year}: {str(e)}")
                continue
        
        if not dfs:
            raise ValueError(f"No data was loaded successfully for {city}")
        
        # Combine all years
        print("\nCombining data from all years...")
        df = pd.concat(dfs, ignore_index=True)
        print(f"✓ Combined dataset: {len(df)} total rows")
        
        # Process datetime
        print("\nProcessing datetime...")
        df["datetime"] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
        df = df.sort_values("datetime").reset_index(drop=True)
        print("✓ Datetime processing complete")
        
        # Validate final dataset
        print("\nValidating final dataset:")
        print(f"Shape: {df.shape}")
        print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"Memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")
        
        return df
        
    except Exception as e:
        print(f"\n× Error loading data for {city}:")
        print(str(e))
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())
        raise

def plot_results(y_true, y_pred, title="LSTM Model: Predicted vs. True GHI"):
    """Plot comparison of predicted vs true values."""
    plt.figure(figsize=(10, 5))
    # Plot only first 100 points for clarity
    x_axis = range(100)
    plt.plot(x_axis, y_true[:100], label="True Next-Day GHI", marker="o", linestyle="-")
    plt.plot(x_axis, y_pred[:100], label="Predicted Next-Day GHI", marker="s", linestyle="--")
    plt.xlabel("Time Steps (Hours)")
    plt.ylabel("GHI (W/m²)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    return plt.gcf()

def plot_loss_history(history):
    """Plot training and validation loss history."""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    return plt.gcf()

def prepare_lstm_input(df, feature_columns, target_column):
    """Prepare data for LSTM input format."""
    X = df[feature_columns].values
    y = df[target_column].values
    X = X.reshape(X.shape[0], 1, X.shape[1])
    return X, y

def evaluate_model(y_true, y_pred):
    """Calculate various metrics for model evaluation."""
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