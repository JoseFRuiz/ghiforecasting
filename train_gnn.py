"""
GHI Forecasting using Graph Neural Networks
This script trains a GNN model to forecast Global Horizontal Irradiance (GHI)
using historical weather data and GHI measurements from multiple locations.
"""

import os
import numpy as np
import pandas as pd
import requests
from io import StringIO
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

import pdb
pdb.set_trace()

# Import Comet.ml first
from comet_ml import Experiment

# Then import ML libraries
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout, Layer
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure GPU
print("\nChecking GPU availability...")

# Set environment variables explicitly
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

# Verify CUDA is available
print("CUDA available:", tf.test.is_built_with_cuda())
print("GPU available:", tf.test.is_gpu_available())

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"Found {len(gpus)} Physical GPUs:")
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu}")
        print(f"Found {len(logical_gpus)} Logical GPUs")
        print("\nGPU details:")
        print(tf.test.gpu_device_name())
        print("\nUsing GPU for training")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(f"Error configuring GPU: {e}")
else:
    print("No GPU available, using CPU for training")
    print("Available devices:")
    print(tf.config.list_physical_devices())
    
# Import configuration and functions from train.py
from train import (
    CONFIG, 
    setup_experiment, 
    evaluate_model, 
    plot_results, 
    plot_loss_history,
    load_data
)

class GraphAttentionLayer(Layer):
    """Graph Attention Layer for processing node features."""
    def __init__(self, units, dropout_rate=0.2):
        super().__init__()
        self.units = units
        self.dropout_rate = dropout_rate
        
        # Attention weights
        self.attention_weights = Dense(units)
        self.attention_scores = Dense(1)
        
        # Feature transformation
        self.feature_transform = Dense(units)
        self.dropout = Dropout(dropout_rate)
        
    def call(self, inputs):
        # inputs: [node_features, adjacency_matrix]
        node_features, adj_matrix = inputs
        
        # Transform features
        transformed_features = self.feature_transform(node_features)
        
        # Calculate attention scores
        # Expand dimensions for broadcasting
        features_i = tf.expand_dims(transformed_features, 1)  # [batch, 1, nodes, units]
        features_j = tf.expand_dims(transformed_features, 2)  # [batch, nodes, 1, units]
        
        # Tile features to match dimensions
        features_i = tf.tile(features_i, [1, tf.shape(transformed_features)[1], 1, 1])  # [batch, nodes, nodes, units]
        features_j = tf.tile(features_j, [1, 1, tf.shape(transformed_features)[1], 1])  # [batch, nodes, nodes, units]
        
        # Concatenate features
        attention_input = tf.concat([features_i, features_j], axis=-1)  # [batch, nodes, nodes, 2*units]
        
        # Calculate attention scores
        attention_scores = self.attention_scores(attention_input)  # [batch, nodes, nodes, 1]
        attention_scores = tf.squeeze(attention_scores, axis=-1)  # [batch, nodes, nodes]
        
        # Apply adjacency matrix and softmax
        attention_scores = tf.where(adj_matrix > 0, attention_scores, -1e9)
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        
        # Apply attention
        attended_features = tf.matmul(attention_weights, transformed_features)
        
        # Apply dropout and return
        return self.dropout(attended_features)

class GNNGHIForecaster(Model):
    """GNN model for GHI forecasting."""
    def __init__(self, num_locations, hidden_dim=64):
        super().__init__()
        self.num_locations = num_locations
        self.hidden_dim = hidden_dim
        
        # Node feature processing
        self.node_encoder = tf.keras.Sequential([
            LSTM(hidden_dim, return_sequences=True),
            Dropout(0.2),
            LSTM(hidden_dim),
            Dropout(0.2),
            Dense(hidden_dim)
        ])
        
        # Graph attention layers
        self.graph_layers = [
            GraphAttentionLayer(hidden_dim) for _ in range(2)
        ]
        
        # Prediction head
        self.prediction_head = Dense(1)
        
    def call(self, inputs):
        # inputs: (node_features, adjacency_matrix)
        node_features, adj_matrix = inputs
        
        # Reshape node features for LSTM
        batch_size = tf.shape(node_features)[0]
        num_locations = tf.shape(node_features)[1]
        num_timesteps = tf.shape(node_features)[2]
        num_features = tf.shape(node_features)[3]
        
        # Reshape to (batch_size * num_locations, timesteps, features)
        reshaped_features = tf.reshape(node_features, 
                                     [batch_size * num_locations, num_timesteps, num_features])
        
        # Process node features
        node_embeddings = self.node_encoder(reshaped_features)
        
        # Reshape back to (batch_size, num_locations, hidden_dim)
        node_embeddings = tf.reshape(node_embeddings, 
                                   [batch_size, num_locations, self.hidden_dim])
        
        # Apply graph attention
        for layer in self.graph_layers:
            node_embeddings = layer([node_embeddings, adj_matrix])
        
        # Make predictions
        predictions = self.prediction_head(node_embeddings)
        return predictions

def create_graph_structure(df):
    """
    Create graph structure based on location similarities.
    
    Args:
        df: DataFrame containing data from all locations
    
    Returns:
        tuple: (adjacency_matrix, locations)
    """
    # Get unique locations
    locations = sorted(df['location'].unique())
    num_locations = len(locations)
    
    # Calculate location similarities based on GHI patterns
    location_features = []
    for location in locations:
        location_data = df[df['location'] == location]
        # Use mean GHI values as features
        features = location_data.groupby(['Month', 'Hour'])['GHI'].mean().values
        location_features.append(features)
    
    # Calculate cosine similarity between locations
    similarity_matrix = cosine_similarity(location_features)
    
    # Create adjacency matrix (threshold similarity)
    threshold = np.percentile(similarity_matrix, 70)  # Keep top 30% of connections
    adjacency_matrix = (similarity_matrix > threshold).astype(float)
    
    # Ensure self-connections
    np.fill_diagonal(adjacency_matrix, 1)
    
    return adjacency_matrix, locations

def prepare_gnn_data(df, adjacency_matrix, locations):
    """
    Prepare data for GNN training.
    
    Args:
        df: DataFrame containing all data
        adjacency_matrix: Graph adjacency matrix
        locations: List of locations
    
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test), target_scaler, feature_columns
    """
    print("1. Splitting data by year...")
    # Split data by year
    df_train = df[df["Year"].isin([2017, 2018])].copy()
    df_2019 = df[df["Year"] == 2019].copy()
    split_index = len(df_2019) // 2
    df_val = df_2019.iloc[:split_index].copy()
    df_test = df_2019.iloc[split_index:].copy()
    
    print("2. Creating time-based features...")
    # Create time-based features
    for df_split in [df_train, df_val, df_test]:
        df_split["hour_sin"] = np.sin(2 * np.pi * df_split["Hour"] / 24)
        df_split["hour_cos"] = np.cos(2 * np.pi * df_split["Hour"] / 24)
        df_split["month_sin"] = np.sin(2 * np.pi * df_split["Month"] / 12)
        df_split["month_cos"] = np.cos(2 * np.pi * df_split["Month"] / 12)
    
    print("3. Defining feature columns...")
    # Define feature columns
    feature_columns = [
        'GHI', 'Temperature', 'Relative Humidity', 'Pressure',
        'Precipitable Water', 'Wind Direction', 'Wind Speed',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
    ]
    
    print("4. Adding lag features...")
    # Add lag features
    for lag in range(1, 25):
        print(f"   Adding lag {lag}/24...")
        df_train[f'GHI_lag_{lag}'] = df_train.groupby('location')['GHI'].shift(lag)
        df_val[f'GHI_lag_{lag}'] = df_val.groupby('location')['GHI'].shift(lag)
        df_test[f'GHI_lag_{lag}'] = df_test.groupby('location')['GHI'].shift(lag)
        feature_columns.append(f'GHI_lag_{lag}')
    
    print("5. Creating target variable...")
    # Create target
    df_train['target_GHI'] = df_train.groupby('location')['GHI'].shift(-24)
    df_val['target_GHI'] = df_val.groupby('location')['GHI'].shift(-24)
    df_test['target_GHI'] = df_test.groupby('location')['GHI'].shift(-24)
    
    print("6. Dropping rows with missing values...")
    # Drop rows with missing values
    df_train = df_train.dropna()
    df_val = df_val.dropna()
    df_test = df_test.dropna()
    
    print("7. Scaling features...")
    # Scale features
    scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    # Scale features
    for df_split in [df_train, df_val, df_test]:
        df_split[feature_columns] = scaler.fit_transform(df_split[feature_columns])
        df_split['target_GHI'] = target_scaler.fit_transform(df_split[['target_GHI']])
    
    print("8. Preparing GNN input format...")
    # Prepare GNN input format
    def prepare_gnn_input(df_split, num_locations):
        num_timesteps = 24  # Use last 24 hours
        num_features = len(feature_columns)
        
        X = np.zeros((len(df_split), num_locations, num_timesteps, num_features))
        y = np.zeros((len(df_split), num_locations))
        
        for i, location in enumerate(locations):
            print(f"   Processing location {i+1}/{len(locations)}: {location}")
            loc_data = df_split[df_split['location'] == location]
            for j in range(len(loc_data)):
                if j >= num_timesteps:
                    X[j, i] = loc_data.iloc[j-num_timesteps:j][feature_columns].values
                    y[j, i] = loc_data.iloc[j]['target_GHI']
        
        return X, y
    
    num_locations = len(locations)
    print("9. Preparing training data...")
    X_train, y_train = prepare_gnn_input(df_train, num_locations)
    print("10. Preparing validation data...")
    X_val, y_val = prepare_gnn_input(df_val, num_locations)
    print("11. Preparing test data...")
    X_test, y_test = prepare_gnn_input(df_test, num_locations)
    
    return (X_train, y_train, X_val, y_val, X_test, y_test), target_scaler, feature_columns

def main(skip_training=False):
    """
    Main execution function for GNN training.
    
    Args:
        skip_training (bool): If True, load pre-trained model instead of training new one
    """
    # Set random seeds for reproducibility
    np.random.seed(CONFIG["random_seed"])
    tf.random.set_seed(CONFIG["random_seed"])
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    
    # Try to create experiment
    try:
        experiment = setup_experiment()
        if experiment:
            experiment.set_name("GHI_Forecasting_GNN")
            print("✓ Experiment created successfully")
    except Exception as e:
        print(f"× Warning: Could not create experiment: {str(e)}")
        experiment = None
    
    # Load and process data
    print("\nLoading and processing data...")
    try:
        # Load data for all locations
        all_dfs = []
        for city in CONFIG["data_locations"].keys():
            print(f"\nLoading data for {city}...")
            df = load_data(CONFIG["data_locations"], city)
            df['location'] = city
            all_dfs.append(df)
        
        df = pd.concat(all_dfs, ignore_index=True)
        print("✓ Data loaded successfully")
    except Exception as e:
        print(f"× Error loading data: {str(e)}")
        return
    
    # Create graph structure
    print("\nCreating graph structure...")
    adjacency_matrix, locations = create_graph_structure(df)
    num_locations = len(locations)
    
    # Prepare data for GNN
    print("\nPreparing data for GNN...")
    (X_train, y_train, X_val, y_val, X_test, y_test), target_scaler, feature_columns = prepare_gnn_data(df, adjacency_matrix, locations)
    
    # Create and train model
    print("\nCreating GNN model...")
    model = GNNGHIForecaster(num_locations)
    
    # Build model with a sample batch
    sample_batch_size = 32
    sample_features = np.zeros((sample_batch_size, num_locations, 24, len(feature_columns)))
    sample_adj = np.tile(adjacency_matrix, (sample_batch_size, 1, 1))
    model([sample_features, sample_adj])  # This builds the model
    
    model.compile(
        optimizer='adam',
        loss=MeanSquaredError(),
        metrics=['mae', 'mse']
    )
    model.summary()
    
    model_path = os.path.join("models", "gnn_ghi_forecast.h5")
    
    if skip_training:
        if os.path.exists(model_path):
            print(f"\nLoading pre-trained model...")
            model = tf.keras.models.load_model(model_path)
            history = None
        else:
            print(f"× No pre-trained model found")
            return
    else:
        print("\nTraining model...")
        history = model.fit(
            [X_train, np.tile(adjacency_matrix, (X_train.shape[0], 1, 1))], y_train,
            epochs=CONFIG["model_params"]["epochs"],
            batch_size=CONFIG["model_params"]["batch_size"],
            validation_data=([X_val, np.tile(adjacency_matrix, (X_val.shape[0], 1, 1))], y_val),
            verbose=0  # Set to 0 to suppress progress output
        )
        
        # Save model
        model.save(model_path)
        print(f"✓ Model saved to {model_path}")
    
    # Evaluate model
    print("\nEvaluating model...")
    results = {}
    
    for location in locations:
        # Get test data for this location
        location_idx = locations.index(location)
        y_pred = model.predict([X_test, np.tile(adjacency_matrix, (X_test.shape[0], 1, 1))])
        y_pred_loc = y_pred[:, location_idx]
        y_test_loc = y_test[:, location_idx]
        
        # Rescale predictions
        y_pred_rescaled = target_scaler.inverse_transform(y_pred_loc.reshape(-1, 1)).ravel()
        y_test_rescaled = target_scaler.inverse_transform(y_test_loc.reshape(-1, 1)).ravel()
        
        # Calculate metrics
        metrics = evaluate_model(y_test_rescaled, y_pred_rescaled)
        results[location] = metrics
        
        print(f"\nResults for {location}:")
        for metric_name, metric_value in metrics.items():
            print(f"✅ {metric_name}: {metric_value:.4f}")
    
    # Log results if experiment exists
    if experiment:
        try:
            # Log metrics
            for location, metrics in results.items():
                for metric_name, metric_value in metrics.items():
                    experiment.log_metric(f"{location}_{metric_name}", metric_value)
            
            # Log plots
            if history:
                loss_fig = plot_loss_history(history)
                experiment.log_figure(figure_name="loss_history", figure=loss_fig)
                plt.close()
            
            print("✓ Results logged to Comet.ml")
        except Exception as e:
            print(f"× Warning: Could not log results: {str(e)}")
    
    # Save results
    print("\nSaving results...")
    try:
        os.makedirs("results_gnn", exist_ok=True)
        os.makedirs("results_gnn/plots", exist_ok=True)
        os.makedirs("results_gnn/tables", exist_ok=True)
        
        # Create metrics table
        records = []
        for location, metrics in results.items():
            for metric_name, value in metrics.items():
                records.append({
                    'Location': location,
                    'Metric': metric_name,
                    'Value': value
                })
        
        metrics_df = pd.DataFrame(records)
        metrics_df.to_csv("results_gnn/tables/gnn_metrics.csv", index=False)
        
        # Create plots for each metric
        for metric in metrics_df['Metric'].unique():
            metric_data = metrics_df[metrics_df['Metric'] == metric]
            
            plt.figure(figsize=(10, 6))
            plt.bar(metric_data['Location'], metric_data['Value'])
            plt.title(f'{metric} Across Locations')
            plt.xlabel('Location')
            plt.ylabel(metric)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plot_path = f"results_gnn/plots/{metric.lower().replace(' ', '_')}.png"
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            plt.close()
        
        print("✓ Results saved to 'results_gnn' directory")
        
    except Exception as e:
        print(f"× Error saving results: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='GHI Forecasting using GNN')
    parser.add_argument('--skip-training', action='store_true',
                      help='Skip training and load pre-trained model')
    args = parser.parse_args()
    
    main(skip_training=args.skip_training) 