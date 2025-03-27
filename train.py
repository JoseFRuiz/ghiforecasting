"""
GHI Forecasting using LSTM
This script trains an LSTM model to forecast Global Horizontal Irradiance (GHI)
using historical weather data and GHI measurements.
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
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
        # Create experiment with minimal logging
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

def create_features(df):
    """
    Create time-based features and lag features for both GHI and meteorological variables.
    """
    print("\nCreating features...")
    print(f"Initial shape: {df.shape}")
    
    # Time-based features
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    
    # Create target: GHI for next day at the same hour (shift by -24)
    df["target_GHI"] = df["GHI"].shift(-24)
    print("Created target_GHI")
    
    # Create lag features for previous 24 hours of GHI
    for lag in range(1, 25):
        df[f"GHI_lag_{lag}"] = df["GHI"].shift(lag)
    print("Created GHI lag features")
    
    # Create lag features for meteorological variables
    meteorological_features = [
        "Temperature", "Relative Humidity", "Pressure",
        "Precipitable Water", "Wind Direction", "Wind Speed"
    ]
    
    # Create 24-hour lag for each meteorological feature
    for feature in meteorological_features:
        df[f"{feature}_lag_24"] = df[feature].shift(24)
    print("Created meteorological lag features")
    
    # Drop rows where we don't have complete data
    df_clean = df.dropna().reset_index(drop=True)
    print(f"Shape after cleaning: {df_clean.shape}")
    print(f"Features created: {df_clean.columns.tolist()}")
    return df_clean

def split_and_scale_data(df, feature_combination="all"):
    """
    Split data into train/val/test sets and scale features.
    
    Args:
        df: DataFrame containing the data
        feature_combination: String specifying which features to use
            - "ghi_only": Only GHI lag features
            - "ghi_temp": GHI + Temperature
            - "ghi_humidity": GHI + Relative Humidity
            - "ghi_pressure": GHI + Pressure
            - "ghi_precip": GHI + Precipitable Water
            - "ghi_wind_dir": GHI + Wind Direction
            - "ghi_wind_speed": GHI + Wind Speed
            - "all": All features (default)
    """
    print("\nSplitting and scaling data...")
    
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
    
    # Select features based on combination
    if feature_combination == "ghi_only":
        selected_features = ghi_features
        print("\nUsing only GHI features")
    elif feature_combination in meteorological_features:
        selected_features = ghi_features + meteorological_features[feature_combination]
        print(f"\nUsing GHI + {feature_combination} features")
    elif feature_combination == "all":
        selected_features = ghi_features + [f for features in meteorological_features.values() for f in features]
        print("\nUsing all features")
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
                    met_features = [f for f in selected_features if f not in ghi_features]
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
                    met_features = [f for f in selected_features if f not in ghi_features]
                    if met_features:
                        df_split.loc[:, met_features] = meteorological_scaler.transform(df_split[met_features])
                
                # Transform target
                target_values = df_split["target_GHI"].values.reshape(-1, 1)
                df_split.loc[:, "target_GHI"] = target_scaler.transform(target_values).ravel()
        
        # Prepare features for LSTM
        feature_columns = [col for col in selected_features 
                          if col not in ["datetime", "GHI", "target_GHI", "Year", "Month", "Day", "Hour", "Minute"]]
        
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
        print(f"Error in split_and_scale_data: {str(e)}")
        raise

def prepare_lstm_input(df, feature_columns, target_column):
    """
    Prepare data for LSTM input format.
    
    Args:
        df: DataFrame containing features and target
        feature_columns: List of feature column names
        target_column: Name of the target column
    """
    X = df[feature_columns].values
    y = df[target_column].values
    X = X.reshape(X.shape[0], 1, X.shape[1])
    return X, y

def create_model(input_shape):
    """Create and compile the LSTM model."""
    model = Sequential([
        LSTM(CONFIG["model_params"]["lstm_units"][0], return_sequences=True, 
             input_shape=input_shape),
        Dropout(CONFIG["model_params"]["dropout_rate"]),
        LSTM(CONFIG["model_params"]["lstm_units"][1], return_sequences=False),
        Dropout(CONFIG["model_params"]["dropout_rate"]),
        Dense(CONFIG["model_params"]["dense_units"][0], activation="relu"),
        Dense(CONFIG["model_params"]["dense_units"][1], activation="linear")
    ])
    
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def plot_results(y_true, y_pred, title="LSTM Model: Predicted vs. True GHI"):
    """
    Plot comparison of predicted vs true values.
    
    Args:
        y_true: True next-day GHI values
        y_pred: Predicted next-day GHI values
        title: Plot title
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
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
    
    # Calculate metrics only for non-zero periods
    metrics = {
        "Mean Absolute Error (Non-zero)": float(mean_absolute_error(y_true_nonzero, y_pred_nonzero)),
        "Mean Squared Error (Non-zero)": float(mean_squared_error(y_true_nonzero, y_pred_nonzero)),
        "Root Mean Squared Error (Non-zero)": float(np.sqrt(mean_squared_error(y_true_nonzero, y_pred_nonzero))),
        "R² Score (Non-zero)": float(r2_score(y_true_nonzero, y_pred_nonzero))
    }
    
    # Also calculate percentage of non-zero values
    metrics["Non-zero Values Percentage"] = float(100 * len(y_true_nonzero) / len(y_true))
    
    # Optionally, you might want to keep the original metrics for comparison
    metrics.update({
        "Mean Absolute Error (All)": float(mean_absolute_error(y_true, y_pred)),
        "Mean Squared Error (All)": float(mean_squared_error(y_true, y_pred)),
        "Root Mean Squared Error (All)": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R² Score (All)": float(r2_score(y_true, y_pred))
    })
    
    return metrics

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

def plot_comparative_metrics(all_metrics, metric_name):
    """
    Create a bar plot comparing a specific metric across cities.
    
    Args:
        all_metrics (dict): Dictionary with city-wise metrics
        metric_name (str): Name of the metric to plot
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    plt.figure(figsize=(12, 6))
    cities = list(all_metrics.keys())
    values = [metrics[metric_name] for metrics in all_metrics.values()]
    
    bars = plt.bar(cities, values)
    plt.title(f'Comparison of {metric_name} Across Cities')
    plt.xlabel('City')
    plt.ylabel(metric_name)
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    return plt.gcf()

def create_metrics_table(all_metrics):
    """
    Create a comparative table of all metrics across cities.
    
    Args:
        all_metrics (dict): Dictionary with city-wise metrics
    
    Returns:
        pd.DataFrame: Table comparing metrics across cities
    """
    # Create a list of records for better table formatting
    records = []
    for city, metrics in all_metrics.items():
        for metric_name, value in metrics.items():
            records.append({
                'City': city,
                'Metric': metric_name,
                'Value': value
            })
    
    # Create DataFrame in long format
    df = pd.DataFrame(records)
    
    # Also create a pivot table for prettier display
    pivot_df = df.pivot(index='Metric', columns='City', values='Value')
    
    return df, pivot_df

def plot_loss_comparison(all_histories):
    """
    Create a plot comparing training and validation loss across cities.
    
    Args:
        all_histories (dict): Dictionary with city-wise training histories
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    plt.figure(figsize=(12, 6))
    
    for city, history in all_histories.items():
        plt.plot(history.history['loss'], label=f'{city} (Train)')
        plt.plot(history.history['val_loss'], label=f'{city} (Val)', linestyle='--')
    
    plt.title('Training and Validation Loss Across Cities')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return plt.gcf()

def plot_radar_comparison(all_metrics):
    """
    Create a radar plot comparing cities across normalized metrics.
    
    Args:
        all_metrics (dict): Dictionary with city-wise metrics
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Get metrics and cities
    metrics = list(next(iter(all_metrics.values())).keys())
    cities = list(all_metrics.keys())
    
    # Convert metrics to numpy array and normalize
    values = np.array([[metrics_dict[m] for m in metrics] for metrics_dict in all_metrics.values()])
    
    # Invert R² score subtraction so higher is still better
    values[:, -1] = 1 - values[:, -1]
    
    # Min-max scaling for each metric
    values_normalized = (values - values.min(axis=0)) / (values.max(axis=0) - values.min(axis=0))
    
    # Convert back R² score
    values_normalized[:, -1] = 1 - values_normalized[:, -1]
    
    # Set up the angles of the plot
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    
    # Close the plot by appending first value
    values_normalized = np.c_[values_normalized, values_normalized[:, 0]]
    angles = np.concatenate((angles, [angles[0]]))
    
    # Create the plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Plot data
    for i, city in enumerate(cities):
        ax.plot(angles, values_normalized[i], 'o-', linewidth=2, label=city)
        ax.fill(angles, values_normalized[i], alpha=0.25)
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['MAE', 'MSE', 'RMSE', 'R² Score'])
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Normalized Performance Metrics Across Cities")
    
    plt.tight_layout()
    return plt.gcf()

def save_plot_to_file(fig, filename):
    """Save a matplotlib figure to a file."""
    fig.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return filename

def main():
    """Main execution function."""
    # Set random seeds for reproducibility
    np.random.seed(CONFIG["random_seed"])
    tf.random.set_seed(CONFIG["random_seed"])
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("tables", exist_ok=True)
    
    # Define feature combinations to test
    feature_combinations = [
        "ghi_only",
        "Temperature",
        "Relative Humidity",
        "Pressure",
        "Precipitable Water",
        "Wind Direction",
        "Wind Speed",
        "all"
    ]
    
    # Dictionary to store results for all cities and feature combinations
    all_results = {
        'metrics': {},
        'histories': {},
        'predictions': {}
    }
    
    # Try to create master experiment but continue if it fails
    try:
        master_experiment = setup_experiment()
        if master_experiment:
            master_experiment.set_name("GHI_Forecasting_Feature_Comparison")
            print("✓ Master experiment created successfully")
    except Exception as e:
        print(f"× Warning: Could not create master experiment: {str(e)}")
        master_experiment = None
    
    # Track progress
    total_cities = len(CONFIG["data_locations"])
    processed_cities = 0
    successful_cities = 0
    
    print(f"\nStarting analysis for {total_cities} cities...")
    
    # Iterate through each city and feature combination
    for city in CONFIG["data_locations"].keys():
        processed_cities += 1
        print(f"\n{'='*50}")
        print(f"Processing city {processed_cities}/{total_cities}: {city}")
        print(f"{'='*50}\n")
        
        city_results = {
            'metrics': {},
            'histories': {},
            'predictions': {}
        }
        
        for feature_combo in feature_combinations:
            print(f"\nTesting feature combination: {feature_combo}")
            
            city_experiment = None
            try:
                # Try to set up experiment for this city and feature combination
                try:
                    city_experiment = setup_experiment()
                    if city_experiment:
                        city_experiment.set_name(f"GHI_Forecasting_{city}_{feature_combo}")
                        print(f"✓ Created experiment for {city} with {feature_combo}")
                except Exception as e:
                    print(f"× Warning: Could not create experiment: {str(e)}")
                
                # Load and process data
                print(f"\nStep 1: Loading data for {city}...")
                try:
                    df = load_data(CONFIG["data_locations"], city)
                    print("✓ Data loaded successfully")
                except Exception as e:
                    print(f"× Error loading data: {str(e)}")
                    raise
                
                print(f"\nStep 2: Creating features...")
                try:
                    df = create_features(df)
                    print("✓ Features created successfully")
                except Exception as e:
                    print(f"× Error creating features: {str(e)}")
                    raise
                
                # Split and scale data
                print("\nStep 3: Preparing data splits...")
                try:
                    (X_train, y_train, X_val, y_val, X_test, y_test), target_scaler = split_and_scale_data(df, feature_combo)
                    print("✓ Data split and scaled successfully")
                except Exception as e:
                    print(f"× Error splitting/scaling data: {str(e)}")
                    raise
                
                # Log dataset info if experiment exists
                if city_experiment:
                    try:
                        city_experiment.log_parameters({
                            "city": city,
                            "feature_combination": feature_combo,
                            "train_size": len(X_train),
                            "val_size": len(X_val),
                            "test_size": len(X_test),
                            "feature_count": X_train.shape[2]
                        })
                    except Exception as e:
                        print(f"× Warning: Could not log parameters: {str(e)}")
                
                # Create and train model
                print(f"\nStep 4: Creating model for {city}...")
                try:
                    model = create_model((X_train.shape[1], X_train.shape[2]))
                    model.summary()
                    print("✓ Model created successfully")
                except Exception as e:
                    print(f"× Error creating model: {str(e)}")
                    raise
                
                print("\nStep 5: Training model...")
                try:
                    history = model.fit(
                        X_train, y_train,
                        epochs=CONFIG["model_params"]["epochs"],
                        batch_size=CONFIG["model_params"]["batch_size"],
                        validation_data=(X_val, y_val),
                        verbose=1
                    )
                    print("✓ Model training completed")
                except Exception as e:
                    print(f"× Error training model: {str(e)}")
                    raise
                
                # Save model
                try:
                    model_path = os.path.join("models", f"lstm_ghi_forecast_{city}_{feature_combo}.h5")
                    model.save(model_path)
                    print(f"✓ Model saved to {model_path}")
                except Exception as e:
                    print(f"× Warning: Could not save model: {str(e)}")
                
                # Evaluate on test set
                print(f"\nStep 6: Evaluating model...")
                try:
                    y_pred = model.predict(X_test)
                    y_pred_rescaled = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
                    y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
                    print("✓ Predictions generated and rescaled")
                except Exception as e:
                    print(f"× Error generating predictions: {str(e)}")
                    raise
                
                # Store results
                city_results['predictions'][feature_combo] = {
                    'true': y_test_rescaled,
                    'pred': y_pred_rescaled
                }
                
                # Calculate metrics
                print("\nStep 7: Calculating metrics...")
                try:
                    metrics = evaluate_model(y_test_rescaled, y_pred_rescaled)
                    city_results['metrics'][feature_combo] = metrics
                    city_results['histories'][feature_combo] = history
                    print("✓ Metrics calculated successfully")
                    
                    print(f"\nResults for {city} with {feature_combo}:")
                    for metric_name, metric_value in metrics.items():
                        print(f"✅ {metric_name}: {metric_value:.4f}")
                except Exception as e:
                    print(f"× Error calculating metrics: {str(e)}")
                    raise
                
                # Try to log results to Comet if available
                if city_experiment:
                    try:
                        # Create and log plots
                        loss_fig = plot_loss_history(history)
                        city_experiment.log_figure(figure_name=f"loss_history_{city}_{feature_combo}", figure=loss_fig)
                        plt.close()
                        
                        pred_fig = plot_results(y_test_rescaled, y_pred_rescaled, 
                                              title=f"LSTM Model: Predicted vs. True GHI - {city} ({feature_combo})")
                        city_experiment.log_figure(figure_name=f"predictions_{city}_{feature_combo}", figure=pred_fig)
                        plt.close()
                        
                        # Log metrics
                        metrics_df = pd.DataFrame({
                            'Metric': list(metrics.keys()),
                            'Value': list(metrics.values())
                        })
                        city_experiment.log_table(f"model_metrics_{city}_{feature_combo}.csv", metrics_df)
                        
                        for metric_name, metric_value in metrics.items():
                            city_experiment.log_metric(f"{city}_{feature_combo}_{metric_name}", metric_value)
                        
                        print("✓ Results logged to Comet.ml")
                    except Exception as e:
                        print(f"× Warning: Could not log results to Comet: {str(e)}")
                
                successful_cities += 1
                print(f"\n✅ Successfully completed processing for {city} with {feature_combo}")
                
            except Exception as e:
                print(f"\n❌ Error processing {city} with {feature_combo}:")
                print(str(e))
                import traceback
                print("\nFull traceback:")
                print(traceback.format_exc())
                if city_experiment:
                    try:
                        city_experiment.log_other("status", f"failed - {str(e)}")
                    except:
                        pass
                continue
            finally:
                if city_experiment:
                    try:
                        city_experiment.end()
                    except:
                        pass
        
        # Store results for this city
        all_results['metrics'][city] = city_results['metrics']
        all_results['histories'][city] = city_results['histories']
        all_results['predictions'][city] = city_results['predictions']
    
    # Create comparative analysis only if we have results
    if successful_cities > 0:
        print(f"\nGenerating comparative analysis for {successful_cities} successfully processed cities...")
        
        try:
            # Create directories for results if they don't exist
            os.makedirs("results", exist_ok=True)
            os.makedirs("results/plots", exist_ok=True)
            os.makedirs("results/tables", exist_ok=True)
            
            if len(all_results['metrics']) > 0:
                # Create and save comparative metrics plots for each feature combination
                for city in all_results['metrics'].keys():
                    for feature_combo in all_results['metrics'][city].keys():
                        metrics = all_results['metrics'][city][feature_combo]
                        for metric_name, metric_value in metrics.items():
                            # Create plot comparing this metric across feature combinations
                            fig = plot_comparative_metrics(
                                {fc: all_results['metrics'][city][fc] for fc in feature_combinations},
                                metric_name
                            )
                            # Save locally
                            plot_path = f"results/plots/{city}_{metric_name.lower().replace(' ', '_')}.png"
                            fig.savefig(plot_path, bbox_inches='tight', dpi=300)
                            # Log to Comet
                            if master_experiment:
                                master_experiment.log_figure(f"{city}_{metric_name}", fig)
                            plt.close(fig)
                
                # Create and save loss comparison plots
                for city in all_results['histories'].keys():
                    loss_comparison_fig = plot_loss_comparison(all_results['histories'][city])
                    # Save locally
                    loss_plot_path = f"results/plots/{city}_loss_comparison.png"
                    loss_comparison_fig.savefig(loss_plot_path, bbox_inches='tight', dpi=300)
                    # Log to Comet
                    if master_experiment:
                        master_experiment.log_figure(f"{city}_loss_comparison", loss_comparison_fig)
                    plt.close(loss_comparison_fig)
                
                # Create metrics tables
                metrics_table, metrics_pivot = create_metrics_table(all_results['metrics'])
                
                # Save tables locally and log to Comet
                # 1. Comparative metrics
                metrics_pivot.to_csv("results/tables/comparative_metrics.csv")
                metrics_pivot.to_html("results/tables/comparative_metrics.html")
                if master_experiment:
                    master_experiment.log_table("comparative_metrics.csv", metrics_pivot.reset_index())
                
                # 2. Summary statistics
                summary_stats = metrics_table.groupby('Metric')['Value'].agg(['mean', 'std', 'min', 'max']).reset_index()
                summary_stats.to_csv("results/tables/summary_statistics.csv", index=False)
                summary_stats.to_html("results/tables/summary_statistics.html")
                if master_experiment:
                    master_experiment.log_table("summary_statistics.csv", summary_stats)
                
                # Save all results to a single Excel file with multiple sheets
                with pd.ExcelWriter("results/comparative_analysis_results.xlsx") as writer:
                    metrics_pivot.to_excel(writer, sheet_name="Comparative Metrics")
                    summary_stats.to_excel(writer, sheet_name="Summary Statistics", index=False)
                
                # Print results to console
                print("\nComparative Results:")
                print("\nMetrics by City and Feature Combination:")
                print(metrics_pivot)
                print("\nSummary Statistics:")
                print(summary_stats)
                
                print("\nResults have been saved locally in the 'results' directory:")
                print("1. Plots: results/plots/")
                print("   - Comparative metric plots (.png)")
                print("   - Loss comparison plots (.png)")
                print("2. Tables: results/tables/")
                print("   - comparative_metrics (.csv, .html)")
                print("   - summary_statistics (.csv, .html)")
                print("3. Excel: results/comparative_analysis_results.xlsx")
                
                print("\nResults have also been logged to Comet.ml. You can find them in:")
                print("1. Figures: Comparative plots for each metric and loss comparison")
                print("2. Tables: comparative_metrics.csv, summary_statistics.csv")
                print("3. Metrics: Individual metrics for each city and feature combination")
            else:
                print("\n⚠️ No metrics were collected. Skipping comparative analysis.")
                if master_experiment:
                    master_experiment.log_other("status", "no metrics collected")
        except Exception as e:
            print(f"\n❌ Error in comparative analysis: {str(e)}")
            import traceback
            print("\nFull traceback:")
            print(traceback.format_exc())
            if master_experiment:
                master_experiment.log_other("status", f"comparative analysis failed - {str(e)}")
    else:
        print("\n❌ No cities were processed successfully. Skipping comparative analysis.")
        if master_experiment:
            master_experiment.log_other("status", "no successful cities")
    
    # Print final summary
    print(f"\nFinal Summary:")
    print(f"Total cities attempted: {total_cities}")
    print(f"Successfully processed: {successful_cities}")
    print(f"Failed/Skipped: {total_cities - successful_cities}")
    
    if master_experiment:
        try:
            master_experiment.log_metrics({
                "total_cities": total_cities,
                "successful_cities": successful_cities,
                "failed_cities": total_cities - successful_cities
            })
            master_experiment.end()
        except:
            pass

if __name__ == "__main__":
    main() 