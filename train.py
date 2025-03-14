import os
from comet_ml import Experiment
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Initialize Comet.ml experiment
experiment = Experiment(
    api_key="qVaqH5DmdCc2UWO5If2SIm1bz",  # Replace with your Comet.ml API key
    project_name="ghi-forecasting",
    workspace=None  # This will use your default workspace
)

# Load data

file_id_locations = {
    "Jaisalmer": {2017: "1O-dOOvSbkTwUv1Qyh33RCzDcOh5GiL2y",
                  2018: "1JgIxhAck67nxXFAPrKHcX8Ql-w4wXXB_",
                  2019: "1ayaT36iSigV5V7DG-haWVM8NO-kCfTv3"},
}


# File paths for the three years
file_paths = [
    f"https://drive.google.com/uc?id={file_id_locations['Jaisalmer'][2017]}",
    f"https://drive.google.com/uc?id={file_id_locations['Jaisalmer'][2018]}",
    f"https://drive.google.com/uc?id={file_id_locations['Jaisalmer'][2019]}"
]

# Load and concatenate all datasets
dfs = [pd.read_csv(file, skiprows=2) for file in file_paths]
df = pd.concat(dfs, ignore_index=True)

# Convert to datetime format
df["datetime"] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
df = df.sort_values("datetime").reset_index(drop=True)

# Extract time-based features
df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

# Select relevant features
features = [
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "Temperature", "Relative Humidity", "Pressure",
    "Precipitable Water", "Wind Direction", "Wind Speed",
    "GHI"
]

# Create lag features for last 24 hours of GHI
for lag in range(1, 25):
    df[f"GHI_lag_{lag}"] = df["GHI"].shift(lag)

# Drop NaN rows caused by lags
df = df.dropna().reset_index(drop=True)

# Normalize features
scaler = MinMaxScaler(feature_range=(0, 1.5))
columns_to_normalize = ["Temperature", "Relative Humidity", "Pressure",
                         "Precipitable Water", "Wind Direction", "Wind Speed", "GHI"] + \
                        [f"GHI_lag_{lag}" for lag in range(1, 25)]

df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Print dataset shape
print(f"✅ Processed dataset shape: {df.shape}")

# Create the Train-Validation-Test Split

# Split data by year and explicitly create copies to avoid SettingWithCopyWarning
df_2017_2018 = df[df["Year"].isin([2017, 2018])].copy()
df_2019 = df[df["Year"] == 2019].copy()

# Further split 2019 data into validation (first half) and test (second half)
split_index = len(df_2019) // 2
df_val = df_2019.iloc[:split_index].copy()  # First half of 2019
df_test = df_2019.iloc[split_index:].copy()  # Second half of 2019

# Print split details
print(f"✅ Training data: {df_2017_2018.shape}")
print(f"✅ Validation data: {df_val.shape}")
print(f"✅ Test data: {df_test.shape}")

# Normalize Data Separately for Each Split

from sklearn.preprocessing import MinMaxScaler

# Define features to normalize
ghi_features = ["GHI"] + [f"GHI_lag_{lag}" for lag in range(1, 25)]
meteorological_features = ["Temperature", "Relative Humidity", "Pressure",
                           "Precipitable Water", "Wind Direction", "Wind Speed"]

# Fit scalers on training data only
ghi_scaler = MinMaxScaler(feature_range=(0, 1.2))
meteorological_scaler = MinMaxScaler()

# Normalize training data
df_2017_2018.loc[:, ghi_features] = ghi_scaler.fit_transform(df_2017_2018[ghi_features])
df_2017_2018.loc[:, meteorological_features] = meteorological_scaler.fit_transform(df_2017_2018[meteorological_features])

# Transform validation and test data using the same scaler (do NOT fit again!)
df_val.loc[:, ghi_features] = ghi_scaler.transform(df_val[ghi_features])
df_val.loc[:, meteorological_features] = meteorological_scaler.transform(df_val[meteorological_features])

df_test.loc[:, ghi_features] = ghi_scaler.transform(df_test[ghi_features])
df_test.loc[:, meteorological_features] = meteorological_scaler.transform(df_test[meteorological_features])

# Convert Data to LSTM Input Format

# Select feature columns (excluding datetime)
feature_columns = [col for col in df_2017_2018.columns if col not in ["datetime", "GHI"]]
target_column = "GHI"

# Convert to numpy arrays
X_train, y_train = df_2017_2018[feature_columns].values, df_2017_2018[target_column].values
X_val, y_val = df_val[feature_columns].values, df_val[target_column].values
X_test, y_test = df_test[feature_columns].values, df_test[target_column].values

# Reshape for LSTM (samples, timesteps, features)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Print shapes
print(f"✅ X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"✅ X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"✅ X_test: {X_test.shape}, y_test: {y_test.shape}")

# Train LSTM on 2017-2018, Validate on Half of 2019

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Define LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1, activation="linear")  # Output a single GHI value
])

# Compile model
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Train model with validation
history = model.fit(X_train, y_train, epochs=20, batch_size=32,
                    validation_data=(X_val, y_val), verbose=1)

# Save trained model
model.save(os.path.join("models","lstm_ghi_forecast.h5"))

# Evaluate on Second Half of 2019 (Unseen Data)

from tensorflow.keras.losses import MeanSquaredError

# Load trained model and define custom objects
model = tf.keras.models.load_model(
    os.path.join(pathfile, "lstm_ghi_forecast.h5"),
    custom_objects={'mse': MeanSquaredError()}
)

# Make predictions
y_pred = model.predict(X_test)

# Rescale GHI back to original values
y_pred_rescaled = ghi_scaler.inverse_transform(
    np.c_[y_pred, np.zeros((len(y_pred), len(ghi_features) - 1))]
)[:, 0]

y_test_rescaled = ghi_scaler.inverse_transform(
    np.c_[y_test, np.zeros((len(y_test), len(ghi_features) - 1))]
)[:, 0]

# Compute MAE
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
print(f"✅ Final MAE on Unseen Test Data (H2 2019): {mae:.2f} W/m²")

# Plot true vs predicted GHI
plt.figure(figsize=(10, 5))
plt.plot(y_test_rescaled[:100], label="True GHI", marker="o", linestyle="-")
plt.plot(y_pred_rescaled[:100], label="Predicted GHI (LSTM)", marker="s", linestyle="--")
plt.xlabel("Time Steps")
plt.ylabel("GHI (W/m²)")
plt.title("LSTM Model: Predicted vs. True GHI")
plt.legend()
plt.show()

# End the experiment
experiment.end() 