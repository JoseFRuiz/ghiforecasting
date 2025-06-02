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
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    for lag in range(1, 25):
        df[f"GHI_lag_{lag}"] = df["GHI"].shift(lag)
    met_features = [
        "Temperature", "Relative Humidity", "Pressure",
        "Precipitable Water", "Wind Direction", "Wind Speed"
    ]
    for feature in met_features:
        df[f"{feature}_lag_24"] = df[feature].shift(24)
    df = df.dropna().reset_index(drop=True)
    return df

df = create_features(df)
print(f"After feature creation: {df.shape}")

# --- 5. Prepare input features (for ghi_only config) ---
feature_columns = [f"GHI_lag_{lag}" for lag in range(1, 25)] + ["GHI"]
data = df[feature_columns].values
target = df["GHI"].values

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

print(f"Input shape for prediction: {X.shape}")

# --- 6. Run model prediction ---
y_pred_scaled = model.predict(X)
print("First 10 raw model outputs (scaled):", y_pred_scaled[:10].flatten())

# --- 7. Inverse transform ---
y_pred = target_scaler.inverse_transform(y_pred_scaled)
print("First 10 outputs after inverse scaling:", y_pred[:10].flatten())

# --- 8. Print first 10 targets for comparison ---
y_true = target_scaler.inverse_transform(y.reshape(-1, 1))
print("First 10 true values:", y_true[:10].flatten())

# --- 9. Check if model weights are non-trivial ---
print("\nModel weights summary:")
for layer in model.layers:
    weights = layer.get_weights()
    if weights:
        print(f"Layer: {layer.name}, Weight shapes: {[w.shape for w in weights]}")
        print(f"  Min: {np.min([w.min() for w in weights]):.4f}, Max: {np.max([w.max() for w in weights]):.4f}")

print("\nIf the raw model outputs are constant, or if the weights are all very small/zero, the model may not have trained properly or the data pipeline may be broken.") 