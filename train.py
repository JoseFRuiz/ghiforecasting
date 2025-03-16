"""
GHI Forecasting using LSTM
This script trains an LSTM model to forecast Global Horizontal Irradiance (GHI)
using historical weather data and GHI measurements.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from comet_ml import Experiment
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

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
    return Experiment(
        api_key=CONFIG["comet_api_key"],
        project_name=CONFIG["project_name"],
        workspace=None
    )

def load_data(locations, city):
    """
    Load and preprocess the GHI data from multiple years for a specific city.
    
    Args:
        locations (dict): Dictionary containing file IDs for different cities and years
        city (str): Name of the city to load data for
    
    Returns:
        pd.DataFrame: Processed and concatenated DataFrame
    """
    try:
        file_paths = [
            f"https://drive.google.com/uc?id={locations[city][year]}"
            for year in [2017, 2018, 2019]
        ]
        
        dfs = [pd.read_csv(file, skiprows=2) for file in file_paths]
        df = pd.concat(dfs, ignore_index=True)
        
        # Convert to datetime and sort
        df["datetime"] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
        df = df.sort_values("datetime").reset_index(drop=True)
        
        return df
    except Exception as e:
        print(f"Error loading data for {city}: {str(e)}")
        raise

def create_features(df):
    """
    Create time-based features and GHI lag features.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        pd.DataFrame: DataFrame with additional features
    """
    # Time-based features
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    
    # Create lag features for last 24 hours of GHI
    for lag in range(1, 25):
        df[f"GHI_lag_{lag}"] = df["GHI"].shift(lag)
    
    return df.dropna().reset_index(drop=True)

def split_and_scale_data(df):
    """
    Split data into train/val/test sets and scale features.
    
    Args:
        df (pd.DataFrame): Processed DataFrame
    
    Returns:
        tuple: Scaled and split X, y data for train/val/test
    """
    # Split data by year
    df_train = df[df["Year"].isin([2017, 2018])].copy()
    df_2019 = df[df["Year"] == 2019].copy()
    split_index = len(df_2019) // 2
    df_val = df_2019.iloc[:split_index].copy()
    df_test = df_2019.iloc[split_index:].copy()
    
    # Define feature groups
    ghi_features = ["GHI"] + [f"GHI_lag_{lag}" for lag in range(1, 25)]
    meteorological_features = [
        "Temperature", "Relative Humidity", "Pressure",
        "Precipitable Water", "Wind Direction", "Wind Speed"
    ]
    
    # Initialize scalers
    ghi_scaler = MinMaxScaler(feature_range=CONFIG["feature_ranges"]["ghi"])
    meteorological_scaler = MinMaxScaler(feature_range=CONFIG["feature_ranges"]["meteorological"])
    
    # Scale features
    for df_split in [df_train, df_val, df_test]:
        if df_split is df_train:
            df_split.loc[:, ghi_features] = ghi_scaler.fit_transform(df_split[ghi_features])
            df_split.loc[:, meteorological_features] = meteorological_scaler.fit_transform(df_split[meteorological_features])
        else:
            df_split.loc[:, ghi_features] = ghi_scaler.transform(df_split[ghi_features])
            df_split.loc[:, meteorological_features] = meteorological_scaler.transform(df_split[meteorological_features])
    
    # Prepare features for LSTM
    feature_columns = [col for col in df_train.columns if col not in ["datetime", "GHI"]]
    
    X_train, y_train = prepare_lstm_input(df_train, feature_columns)
    X_val, y_val = prepare_lstm_input(df_val, feature_columns)
    X_test, y_test = prepare_lstm_input(df_test, feature_columns)
    
    return (X_train, y_train, X_val, y_val, X_test, y_test), ghi_scaler

def prepare_lstm_input(df, feature_columns):
    """Prepare data for LSTM input format."""
    X = df[feature_columns].values
    y = df["GHI"].values
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
        y_true: True GHI values
        y_pred: Predicted GHI values
        title: Plot title
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    plt.figure(figsize=(10, 5))
    plt.plot(y_true[:100], label="True GHI", marker="o", linestyle="-")
    plt.plot(y_pred[:100], label="Predicted GHI (LSTM)", marker="s", linestyle="--")
    plt.xlabel("Time Steps")
    plt.ylabel("GHI (W/m²)")
    plt.title(title)
    plt.legend()
    return plt.gcf()

def evaluate_model(y_true, y_pred):
    """
    Calculate various metrics for model evaluation.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    metrics = {
        "Mean Absolute Error": float(mean_absolute_error(y_true, y_pred)),
        "Mean Squared Error": float(mean_squared_error(y_true, y_pred)),
        "Root Mean Squared Error": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R² Score": float(r2_score(y_true, y_pred))
    }
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

def main():
    """Main execution function."""
    # Set random seeds for reproducibility
    np.random.seed(CONFIG["random_seed"])
    tf.random.set_seed(CONFIG["random_seed"])
    
    # Iterate through each city
    for city in CONFIG["data_locations"].keys():
        print(f"\n{'='*50}")
        print(f"Processing {city}")
        print(f"{'='*50}\n")
        
        # Initialize experiment tracking for this city
        experiment = setup_experiment()
        experiment.set_name(f"GHI_Forecasting_{city}")  # Set unique name for each city's experiment
        
        try:
            # Load and process data
            print(f"Loading and processing data for {city}...")
            df = load_data(CONFIG["data_locations"], city)
            df = create_features(df)
            
            # Split and scale data
            print("Preparing train/val/test splits...")
            (X_train, y_train, X_val, y_val, X_test, y_test), ghi_scaler = split_and_scale_data(df)
            
            # Log dataset sizes and city information
            experiment.log_parameters({
                "city": city,
                "train_size": len(X_train),
                "val_size": len(X_val),
                "test_size": len(X_test),
                "feature_count": X_train.shape[2]
            })
            
            # Create and train model
            print(f"Training model for {city}...")
            model = create_model((X_train.shape[1], X_train.shape[2]))
            
            # Log model architecture
            experiment.log_parameter("model_summary", str(model.summary()))
            
            history = model.fit(
                X_train, y_train,
                epochs=CONFIG["model_params"]["epochs"],
                batch_size=CONFIG["model_params"]["batch_size"],
                validation_data=(X_val, y_val),
                verbose=1
            )
            
            # Save model with city-specific name
            model_path = os.path.join("models", f"lstm_ghi_forecast_{city}.h5")
            os.makedirs("models", exist_ok=True)
            model.save(model_path)
            
            # Evaluate on test set
            print(f"Evaluating model for {city}...")
            y_pred = model.predict(X_test)
            
            # Rescale predictions
            y_pred_rescaled = ghi_scaler.inverse_transform(
                np.c_[y_pred, np.zeros((len(y_pred), 24))]
            )[:, 0]
            y_test_rescaled = ghi_scaler.inverse_transform(
                np.c_[y_test, np.zeros((len(y_test), 24))]
            )[:, 0]
            
            # Calculate and log metrics
            metrics = evaluate_model(y_test_rescaled, y_pred_rescaled)
            
            # Create DataFrame for metrics table
            metrics_df = pd.DataFrame({
                'Metric': list(metrics.keys()),
                'Value': list(metrics.values())
            })
            
            # Log metrics table
            experiment.log_table(
                f"model_metrics_{city}.csv",
                metrics_df
            )
            
            # Log individual metrics with city tag
            for metric_name, metric_value in metrics.items():
                experiment.log_metric(f"{city}_{metric_name}", metric_value)
            
            # Log training history
            for epoch, (loss, val_loss) in enumerate(zip(history.history['loss'], 
                                                        history.history['val_loss'])):
                experiment.log_metrics({
                    f"{city}_training_loss": loss,
                    f"{city}_validation_loss": val_loss
                }, step=epoch)
            
            # Create and log loss plot
            loss_fig = plot_loss_history(history)
            experiment.log_figure(figure_name=f"loss_history_{city}", figure=loss_fig)
            plt.close()
            
            # Create and log predictions plot
            pred_fig = plot_results(y_test_rescaled, y_pred_rescaled, 
                                  title=f"LSTM Model: Predicted vs. True GHI - {city}")
            experiment.log_figure(figure_name=f"predictions_{city}", figure=pred_fig)
            plt.close()
            
            print(f"\nResults for {city}:")
            for metric_name, metric_value in metrics.items():
                print(f"✅ {metric_name}: {metric_value:.4f}")
            
        except Exception as e:
            print(f"Error processing {city}: {str(e)}")
            continue
        finally:
            experiment.end()

if __name__ == "__main__":
    main() 