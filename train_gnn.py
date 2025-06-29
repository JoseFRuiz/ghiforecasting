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
SEQUENCE_LENGTH = 24
FORECAST_HORIZON = 24

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

    dates = pd.to_datetime(df_all["datetime"]).dt.date.unique()
    for date in dates:
        node_features = []
        node_targets = []
        valid = True
        for city in CITIES:
            df_city = df_all[(df_all["location"] == city) &
                             (pd.to_datetime(df_all["datetime"]).dt.date == date)]
            if len(df_city) < SEQUENCE_LENGTH + FORECAST_HORIZON:
                valid = False
                break
            X = df_city.iloc[:SEQUENCE_LENGTH][[col for col in df_city.columns if 'lag' in col or 'sin' in col or 'cos' in col]].values
            y = df_city.iloc[SEQUENCE_LENGTH:SEQUENCE_LENGTH + FORECAST_HORIZON]['GHI'].values
            if (y <= 0).all():
                valid = False
                break
            node_features.append(X.flatten())
            node_targets.append(y)
        if valid:
            x = np.stack(node_features, axis=0)
            y = np.stack(node_targets, axis=0)
            graphs.append(Graph(x=x, a=adj_matrix))
            targets.append(y)

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
    inputs = layers.Input(shape=input_shape)
    a_input = layers.Input(shape=(None,), sparse=True)

    x = GCNConv(64, activation='relu')([inputs, a_input])
    x = layers.Dropout(0.2)(x)
    x = GCNConv(64, activation='relu')([x, a_input])
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(output_units)(x)

    model = models.Model(inputs=[inputs, a_input], outputs=outputs)
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

    print("\nComputing adjacency matrix...")
    adj_matrix = compute_weighted_adjacency(df_all, alpha=0.5)

    graphs, targets, ghi_scaler = build_daily_graphs(df_all, adj_matrix)
    dataset = GHIDataset(graphs, targets)
    loader = DisjointLoader(dataset, batch_size=8, epochs=1, shuffle=True)

    print("\nBuilding model...")
    num_features = dataset[0].x.shape[1]
    model = build_gnn_model(input_shape=(num_features,), output_units=FORECAST_HORIZON)
    model.summary()

    print("\nTraining model...")
    model.fit(loader.load(), steps_per_epoch=loader.steps_per_epoch, epochs=20)

    model.save("models/gnn_ghi_forecast.h5")
    with open("models/ghi_scaler_gnn.pkl", "wb") as f:
        pickle.dump(ghi_scaler, f)
    print("\n✓ Model and scaler saved.")

if __name__ == "__main__":
    main()
