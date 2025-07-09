# train_gnn.py
# Joint GHI forecasting using a Graph Neural Network (GNN)

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from spektral.data import Dataset, Graph
from spektral.data.loaders import DisjointLoader
from spektral.layers import GCNConv
from sklearn.preprocessing import MinMaxScaler
import pickle
from geopy.distance import geodesic
from utils import CONFIG, load_data

# Set global parameters
SEQUENCE_LENGTH = 12  # Use 12 hours for sequence
FORECAST_HORIZON = 12  # Forecast next 12 hours

CITIES = list(CONFIG["data_locations"].keys())
CITY_IDX = {city: i for i, city in enumerate(CITIES)}

# ----------------------------------------------
# Data loading and sequence preparation
# ----------------------------------------------
def create_features(df):
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    for lag in range(1, 25):
        df[f"GHI_lag_{lag}"] = df["GHI"].shift(lag)

    met_vars = ["Temperature", "Relative Humidity", "Pressure",
                "Precipitable Water", "Wind Direction", "Wind Speed"]
    for var in met_vars:
        df[f"{var}_lag_24"] = df[var].shift(24)

    df["target"] = df["GHI"].shift(-24)  # forecast 24 hours ahead
    return df.dropna().reset_index(drop=True)

def compute_weighted_adjacency(df_all, alpha=0.5):
    city_coords = {
        "Jaisalmer": (26.9157, 70.9083),
        "Jodhpur": (26.2389, 73.0243),
        "New Delhi": (28.6139, 77.2090),
        "Shimla": (31.1048, 77.1734),
        "Srinagar": (34.0837, 74.7973),
    }
    cities = list(city_coords.keys())
    n = len(cities)

    # Geodesic proximity (1 / distance)
    geo_weights = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i != j:
                d = geodesic(city_coords[cities[i]], city_coords[cities[j]]).km
                geo_weights[i, j] = 1 / d

    geo_weights /= geo_weights.max()

    # Correlation
    corr_weights = np.zeros((n, n), dtype=np.float32)
    for i, ci in enumerate(cities):
        ghi_i = df_all[df_all['location'] == ci].set_index('datetime')['GHI']
        for j, cj in enumerate(cities):
            ghi_j = df_all[df_all['location'] == cj].set_index('datetime')['GHI']
            common = ghi_i.index.intersection(ghi_j.index)
            if len(common) > 0:
                corr = ghi_i.loc[common].corr(ghi_j.loc[common])
                corr_weights[i, j] = corr if not np.isnan(corr) else 0

    corr_weights = np.clip(corr_weights, 0, 1)
    adj = alpha * geo_weights + (1 - alpha) * corr_weights
    np.fill_diagonal(adj, 1.0)
    return adj

def build_daily_graphs(df_all, adj_matrix):
    graphs = []
    targets = []

    scaler = MinMaxScaler()
    df_all['GHI_scaled'] = scaler.fit_transform(df_all[["GHI"]])

    # Group by date and process each day
    df_all['date'] = pd.to_datetime(df_all["datetime"]).dt.date
    dates = df_all['date'].unique()
    
    print(f"Processing {len(dates)} unique dates")
    
    # Count how many dates are skipped and why
    skipped_reasons = {}
    
    for date in dates:
        node_features = []
        node_targets = []
        valid = True
        skip_reason = None
        
        for city in CITIES:
            # Get data for this city and date
            df_city = df_all[(df_all["location"] == city) & (df_all['date'] == date)]
            
            if len(df_city) < 24:  # Need at least 24 hours
                skip_reason = f"not enough data ({len(df_city)} rows, need 24)"
                valid = False
                break
            
            # Sort by datetime to ensure proper order
            df_city = df_city.sort_values('datetime')
            
            # Use first 12 hours for sequence, next 12 hours for target
            # This gives us a 12-hour forecast horizon instead of 24
            sequence_data = df_city.iloc[:12]
            target_data = df_city.iloc[12:24]
            
            if len(target_data) < 12:
                skip_reason = f"not enough target data ({len(target_data)} rows, need 12)"
                valid = False
                break
            
            # Check if target has any non-zero values (but be less strict)
            if (target_data['GHI'] <= 0).sum() >= 11:  # Allow at least 1 non-zero value
                skip_reason = f"all target GHI values are zero or negative"
                valid = False
                break
            
            # Create features from sequence data
            feature_cols = [col for col in sequence_data.columns if 'lag' in col or 'sin' in col or 'cos' in col]
            
            # Check if we have the required features
            if len(feature_cols) == 0:
                skip_reason = f"no feature columns found"
                valid = False
                break
                
            X = sequence_data[feature_cols].values.flatten()
            
            # Check for NaN values in features
            if np.isnan(X).any():
                skip_reason = f"NaN values in features"
                valid = False
                break
            
            # Target is the GHI values for the next 12 hours
            y = target_data['GHI'].values
            
            # Check for NaN values in targets
            if np.isnan(y).any():
                skip_reason = f"NaN values in targets"
                valid = False
                break
            
            node_features.append(X)
            node_targets.append(y)
        
        if valid:
            x = np.stack(node_features, axis=0)
            # Create a single target per graph by averaging across cities
            # This gives us a single 12-hour forecast for the entire region
            y = np.mean(node_targets, axis=0)
            
            graphs.append(Graph(x=x, a=adj_matrix))
            targets.append(y)
        else:
            if skip_reason not in skipped_reasons:
                skipped_reasons[skip_reason] = 0
            skipped_reasons[skip_reason] += 1

    print(f"Total valid daily graphs: {len(graphs)}")
    print(f"Skipped dates by reason:")
    for reason, count in skipped_reasons.items():
        print(f"  {reason}: {count} dates")
    
    if len(graphs) == 0:
        print("\nWARNING: No valid graphs created!")
        print("This could be due to:")
        print("1. Insufficient data per day")
        print("2. Missing feature columns")
        print("3. NaN values in data")
        print("4. All target values being zero")
        
        # Let's check a sample date to debug
        if len(dates) > 0:
            sample_date = dates[0]
            print(f"\nDebugging sample date: {sample_date}")
            for city in CITIES:
                df_city = df_all[(df_all["location"] == city) & (df_all['date'] == sample_date)]
                print(f"  {city}: {len(df_city)} rows")
                if len(df_city) > 0:
                    print(f"    Columns: {df_city.columns.tolist()}")
                    print(f"    Feature cols: {[col for col in df_city.columns if 'lag' in col or 'sin' in col or 'cos' in col]}")
    
    return graphs, np.array(targets), scaler

class GHIDataset(Dataset):
    def __init__(self, graphs, y, **kwargs):
        self.graphs = graphs
        self.y = y
        super().__init__(**kwargs)

    def read(self):
        for g, target in zip(self.graphs, self.y):
            g.y = target
        return self.graphs

# ----------------------------------------------
# GNN model
# ----------------------------------------------
def build_gnn_model(input_shape, output_units):
    # Create a model that can handle the DisjointLoader format
    # The loader provides: [x, a, i] where x is node features, a is adjacency, i is batch indices
    
    # Input layers
    x_in = layers.Input(shape=input_shape, name='x_in')
    a_in = layers.Input(shape=(None,), sparse=True, name='a_in')
    i_in = layers.Input(shape=(), dtype=tf.int64, name='i_in')
    
    # GNN layers
    x = GCNConv(64, activation='relu')([x_in, a_in])
    x = layers.Dropout(0.2)(x)
    x = GCNConv(64, activation='relu')([x, a_in])
    x = layers.Dropout(0.2)(x)
    
    # Since we have 5 cities (nodes), we can directly use the node features
    # Each node represents a city, so we can flatten and use dense layers
    x = layers.Flatten()(x)
    
    # Dense layers for final prediction
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    # Ensure output has the correct shape for 12-hour forecast
    outputs = layers.Dense(output_units, name='output')(x)

    model = models.Model(inputs=[x_in, a_in, i_in], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse', metrics=['mae'])
    return model

# ----------------------------------------------
# Main script
# ----------------------------------------------
def main():
    os.makedirs("models", exist_ok=True)
    print("\nLoading and preparing data...")
    all_dfs = []
    for city in CITIES:
        df = load_data(CONFIG["data_locations"], city)
        df = create_features(df)
        df['location'] = city
        all_dfs.append(df)
    df_all = pd.concat(all_dfs).sort_values("datetime")

    print(f"\nData statistics:")
    print(f"Total rows: {len(df_all)}")
    print(f"Date range: {df_all['datetime'].min()} to {df_all['datetime'].max()}")
    print(f"Rows per city:")
    for city in CITIES:
        city_data = df_all[df_all['location'] == city]
        print(f"  {city}: {len(city_data)} rows")
    
    # Check data per day
    df_all['date'] = pd.to_datetime(df_all["datetime"]).dt.date
    daily_counts = df_all.groupby(['date', 'location']).size().reset_index(name='count')
    print(f"\nDaily data counts (sample):")
    print(daily_counts.head(20))
    
    # Check for missing values
    print(f"\nMissing values per column:")
    missing_values = df_all.isnull().sum()
    print(missing_values[missing_values > 0])

    print("\nComputing adjacency matrix...")
    adj_matrix = compute_weighted_adjacency(df_all, alpha=0.5)

    graphs, targets, ghi_scaler = build_daily_graphs(df_all, adj_matrix)
    
    if len(graphs) == 0:
        print("ERROR: No valid graphs created. Exiting.")
        return
    
    dataset = GHIDataset(graphs, targets)
    loader = DisjointLoader(dataset, batch_size=8, epochs=1, shuffle=True)

    print("\nBuilding model...")
    num_features = dataset[0].x.shape[1]
    print(f"Number of features per node: {num_features}")
    print(f"Number of nodes per graph: {dataset[0].x.shape[0]}")
    print(f"Target shape: {targets.shape}")
    print(f"Sample target shape: {targets[0].shape}")
    
    model = build_gnn_model(input_shape=(num_features,), output_units=FORECAST_HORIZON)
    model.summary()

    print("\nTraining model...")
    # Test the model with a sample batch to check shapes
    sample_batch = next(loader.load())
    print(f"Sample batch shapes:")
    print(f"  x: {sample_batch[0].shape}")
    print(f"  a: {sample_batch[1].shape}")
    print(f"  i: {sample_batch[2].shape}")
    print(f"  y: {sample_batch[3].shape}")
    
    # Test model prediction
    sample_pred = model.predict(sample_batch[:3])
    print(f"Model output shape: {sample_pred.shape}")
    
    model.fit(loader.load(), steps_per_epoch=loader.steps_per_epoch, epochs=20)

    model.save("models/gnn_ghi_forecast.h5")
    with open("models/ghi_scaler_gnn.pkl", "wb") as f:
        pickle.dump(ghi_scaler, f)
    print("\n✓ Model and scaler saved.")

if __name__ == "__main__":
    main()
