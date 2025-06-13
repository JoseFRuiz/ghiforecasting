import os
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from utils import CONFIG, load_data

# --- CONFIGURABLE ---
CITY = "Jaisalmer"  # Change to any city you want to debug
data_locations = CONFIG["data_locations"]
MODEL_CONFIG = "ghi_only"  # Change to the config you want to debug {ghi_only, met_only, ghi_met}
SEQUENCE_LENGTH = 24

# --- 1. Load scaler ---
scaler_path = os.path.join("models", "target_scaler.pkl")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Target scaler not found at {scaler_path}. Please run train_joint.py first.")
with open(scaler_path, 'rb') as f:
    target_scaler = pickle.load(f)
print("Scaler min:", target_scaler.data_min_)
print("Scaler max:", target_scaler.data_max_)

# --- 2. Load model ---
model_path = os.path.join("models", f"lstm_ghi_forecast_joint_{MODEL_CONFIG}.h5")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = tf.keras.models.load_model(model_path)
print("Model loaded.")

# --- 3. Load test data ---
df = load_data(data_locations, CITY)
print(f"Loaded {len(df)} rows for {CITY}")

# --- 4. Create features (same as in prediction pipeline) ---
def create_features(df):
    """
    Create features for prediction with detailed debugging information.
    """
    print("\nCreating features...")
    print(f"Initial shape: {df.shape}")
    
    # Create time features
    print("\nCreating time features...")
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    print("Time features created")
    
    # Create GHI lag features
    print("\nCreating GHI lag features...")
    for lag in range(1, 25):
        df[f"GHI_lag_{lag}"] = df["GHI"].shift(lag)
    print("GHI lag features created")
    
    # Create meteorological lag features
    print("\nCreating meteorological lag features...")
    met_features = [
        "Temperature", "Relative Humidity", "Pressure",
        "Precipitable Water", "Wind Direction", "Wind Speed"
    ]
    for feature in met_features:
        df[f"{feature}_lag_24"] = df[feature].shift(24)
    print("Meteorological lag features created")
    
    # Print feature statistics before dropping NA
    print("\nFeature statistics before cleaning:")
    print(df.describe())
    
    # Drop rows with missing values
    print("\nDropping rows with missing values...")
    df = df.dropna().reset_index(drop=True)
    print(f"Final shape: {df.shape}")
    
    # Print feature statistics after cleaning
    print("\nFeature statistics after cleaning:")
    print(df.describe())
    
    return df

df = create_features(df)
print(f"After feature creation: {df.shape}")

# --- 5. Prepare input features based on configuration ---
print(f"\nPreparing input features for {MODEL_CONFIG} configuration...")

# Initialize feature columns based on configuration
feature_columns = []

if MODEL_CONFIG in ['ghi_only', 'ghi_met']:
    # Add GHI lag features
    feature_columns.extend([f"GHI_lag_{lag}" for lag in range(1, 25)])
    # Add current GHI
    feature_columns.append("GHI")

if MODEL_CONFIG in ['met_only', 'ghi_met']:
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

print(f"Selected features: {feature_columns}")
print(f"Number of features: {len(feature_columns)}")

# Get data for selected features
data = df[feature_columns].values
target = df["GHI"].values

# Print data statistics
print("\nInput data statistics:")
print(f"Data shape: {data.shape}")
print(f"Data min: {np.min(data):.4f}")
print(f"Data max: {np.max(data):.4f}")
print(f"Data mean: {np.mean(data):.4f}")
print(f"Data std: {np.std(data):.4f}")

# Create sequences
n_sequences = len(df) - SEQUENCE_LENGTH
X = np.zeros((n_sequences, SEQUENCE_LENGTH, len(feature_columns)))
y = np.zeros(n_sequences)

for i in range(n_sequences):
    X[i] = data[i:i + SEQUENCE_LENGTH]
    y[i] = target[i + SEQUENCE_LENGTH]

# Only keep nonzero targets
mask = y > 0
X = X[mask]
y = y[mask]

print(f"\nFinal sequence shapes:")
print(f"X: {X.shape}")
print(f"y: {y.shape}")

# Print sequence statistics
print("\nSequence statistics:")
print(f"X min: {np.min(X):.4f}")
print(f"X max: {np.max(X):.4f}")
print(f"X mean: {np.mean(X):.4f}")
print(f"X std: {np.std(X):.4f}")
print(f"y min: {np.min(y):.4f}")
print(f"y max: {np.max(y):.4f}")
print(f"y mean: {np.mean(y):.4f}")
print(f"y std: {np.std(y):.4f}")

# --- 6. Run model prediction ---
print("\nRunning model prediction...")
y_pred_scaled = model.predict(X)
print("Raw model outputs (scaled):")
print(f"Min: {np.min(y_pred_scaled):.4f}")
print(f"Max: {np.max(y_pred_scaled):.4f}")
print(f"Mean: {np.mean(y_pred_scaled):.4f}")
print(f"Std: {np.std(y_pred_scaled):.4f}")
print("First 10 predictions:", y_pred_scaled[:10].flatten())

# --- 7. Inverse transform ---
print("\nInverse transforming predictions...")
y_pred = target_scaler.inverse_transform(y_pred_scaled)
print("Unscaled predictions:")
print(f"Min: {np.min(y_pred):.2f}")
print(f"Max: {np.max(y_pred):.2f}")
print(f"Mean: {np.mean(y_pred):.2f}")
print(f"Std: {np.std(y_pred):.2f}")
print("First 10 predictions:", y_pred[:10].flatten())

# --- 8. Print first 10 targets for comparison ---
y_true = target_scaler.inverse_transform(y.reshape(-1, 1))
print("\nTrue values:")
print(f"Min: {np.min(y_true):.2f}")
print(f"Max: {np.max(y_true):.2f}")
print(f"Mean: {np.mean(y_true):.2f}")
print(f"Std: {np.std(y_true):.2f}")
print("First 10 true values:", y_true[:10].flatten())

# --- 9. Calculate metrics ---
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("\nPrediction metrics:")
print(f"MAE: {mae:.2f} W/m²")
print(f"RMSE: {rmse:.2f} W/m²")
print(f"R²: {r2:.4f}")

# --- 10. Check model weights ---
print("\nModel weights summary:")
for layer in model.layers:
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

print("\nIf the raw model outputs are constant, or if the weights are all very small/zero, the model may not have trained properly or the data pipeline may be broken.") 