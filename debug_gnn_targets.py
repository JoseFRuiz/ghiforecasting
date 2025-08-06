# debug_gnn_targets.py
# Diagnostic script to debug GNN target scaling issues

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from spektral.data import Dataset, Graph
from spektral.data.loaders import DisjointLoader
from spektral.layers import GCNConv
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt
import datetime
import gc
import time
from utils import CONFIG, load_data

# Set global parameters
SEQUENCE_LENGTH = 24  # Use 24 hours for sequence
FORECAST_HORIZON = 1  # Forecast next 1 hour (single value)

CITIES = list(CONFIG["data_locations"].keys())
CITY_IDX = {city: i for i, city in enumerate(CITIES)}

def create_features(df):
    """Create features with reduced complexity for memory efficiency."""
    print(f"    Creating features for dataframe with {len(df)} rows...")
    
    print(f"      Adding time-based features...")
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    print(f"      Adding GHI lag features...")
    # Add lag features for previous 24 hours
    for lag in range(1, 25):
        df[f"GHI_lag_{lag}"] = df["GHI"].shift(lag)

    print(f"      Adding meteorological lag features...")
    # Add meteorological features
    met_vars = ["Temperature", "Relative Humidity", "Pressure", 
                "Precipitable Water", "Wind Direction", "Wind Speed"]
    for var in met_vars:
        df[f"{var}_lag_24"] = df[var].shift(24)

    print(f"      Adding target variable...")
    df["target_GHI"] = df["GHI"].shift(-24)  # forecast 24 hours ahead (single value)
    
    # Debug: Check target values after creation
    print(f"      Target GHI stats after creation:")
    print(f"        Range: [{df['target_GHI'].min():.2f}, {df['target_GHI'].max():.2f}]")
    print(f"        Mean: {df['target_GHI'].mean():.2f}")
    print(f"        Non-zero: {(df['target_GHI'] > 0).sum()} out of {len(df)}")
    
    print(f"      Dropping NaN values...")
    original_len = len(df)
    df = df.dropna().reset_index(drop=True)
    final_len = len(df)
    print(f"      Dropped {original_len - final_len} rows with NaN values")
    print(f"      Final shape: {df.shape}")
    
    return df

def compute_weighted_adjacency(df_all, alpha=0.5):
    """Compute adjacency matrix with pre-computed weights for efficiency."""
    print("    Computing weighted adjacency matrix (optimized)...")
    
    # Get the actual cities present in the data
    actual_cities = df_all['location'].unique()
    num_cities = len(actual_cities)
    print(f"    Found {num_cities} cities in data: {actual_cities}")
    
    # Create adjacency matrix based on actual cities
    if num_cities == 5:
        # Use the pre-computed matrix for all 5 cities
        geo_weights = np.array([
            [1.0, 0.8, 0.6, 0.4, 0.3],
            [0.8, 1.0, 0.9, 0.5, 0.4],
            [0.6, 0.9, 1.0, 0.7, 0.6],
            [0.4, 0.5, 0.7, 1.0, 0.8],
            [0.3, 0.4, 0.6, 0.8, 1.0]
        ], dtype=np.float32)

        corr_weights = np.array([
            [1.0, 0.7, 0.6, 0.5, 0.4],
            [0.7, 1.0, 0.8, 0.6, 0.5],
            [0.6, 0.8, 1.0, 0.7, 0.6],
            [0.5, 0.6, 0.7, 1.0, 0.8],
            [0.4, 0.5, 0.6, 0.8, 1.0]
        ], dtype=np.float32)
    else:
        # Create a simple adjacency matrix for fewer cities
        geo_weights = np.eye(num_cities, dtype=np.float32)
        corr_weights = np.eye(num_cities, dtype=np.float32)
        
        # Add some connectivity for adjacent cities
        for i in range(num_cities - 1):
            geo_weights[i, i+1] = 0.8
            geo_weights[i+1, i] = 0.8
            corr_weights[i, i+1] = 0.7
            corr_weights[i+1, i] = 0.7
    
    # Combine weights
    weighted_adj = alpha * geo_weights + (1 - alpha) * corr_weights
    print(f"    Adjacency matrix computed: {weighted_adj.shape}")
    
    return weighted_adj

def build_daily_graphs_debug(df_all, adj_matrix, actual_cities):
    """Build daily graphs with detailed debugging."""
    print("Building daily graphs (debug version)...")
    
    # Limit the number of graphs for memory efficiency
    max_graphs = 50  # Reduced for debugging
    
    # Get unique dates
    df_all['date'] = pd.to_datetime(df_all['datetime']).dt.date
    unique_dates = sorted(df_all['date'].unique())
    
    print(f"Found {len(unique_dates)} unique dates")
    if len(unique_dates) > max_graphs:
        print(f"Limiting to {max_graphs} graphs for debugging")
        unique_dates = unique_dates[:max_graphs]
    
    graphs = []
    targets = []
    graph_dates = []
    graph_cities = []
    original_targets = []  # Store original targets for debugging
    
    # Fit scaler on a sample of the data
    print("Fitting scaler...")
    sample_data = df_all.sample(min(10000, len(df_all)), random_state=42)
    feature_columns = [col for col in sample_data.columns if col not in 
                      ['datetime', 'date', 'location', 'target_GHI', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'City']]
    print(f"Found {len(feature_columns)} feature columns")
    
    # Scale both GHI and target_GHI
    ghi_scaler = MinMaxScaler()
    ghi_scaler.fit(sample_data[['GHI']])
    
    # Use StandardScaler for targets to avoid zero issues
    target_scaler = StandardScaler()
    valid_targets = sample_data[['target_GHI']].dropna()
    print(f"Valid targets for scaler fitting: {len(valid_targets)} samples")
    print(f"Valid target range: [{valid_targets['target_GHI'].min():.2f}, {valid_targets['target_GHI'].max():.2f}]")
    target_scaler.fit(valid_targets)
    
    print(f"GHI scaler range: [{ghi_scaler.data_min_[0]:.2f}, {ghi_scaler.data_max_[0]:.2f}]")
    print(f"Target scaler mean: {target_scaler.mean_[0]:.2f}, std: {target_scaler.scale_[0]:.2f}")
    
    # Debug: Test target scaler
    test_targets = sample_data['target_GHI'].head(5).values
    test_scaled = target_scaler.transform(test_targets.reshape(-1, 1)).flatten()
    test_inverse = target_scaler.inverse_transform(test_scaled.reshape(-1, 1)).flatten()
    print(f"Target scaler test:")
    print(f"  Original: {test_targets}")
    print(f"  Scaled: {test_scaled}")
    print(f"  Inverse: {test_inverse}")
    print(f"  Difference: {np.abs(test_targets - test_inverse)}")
    
    print(f"Processing {len(unique_dates)} unique dates (limited to {max_graphs} graphs)")
    
    # Debug: Check target distribution
    print("Debug: Checking target distribution...")
    sample_targets = df_all['target_GHI'].dropna()
    print(f"  Target range: [{sample_targets.min():.2f}, {sample_targets.max():.2f}]")
    print(f"  Target mean: {sample_targets.mean():.2f}")
    print(f"  Non-zero targets: {(sample_targets > 0).sum()} out of {len(sample_targets)}")
    print(f"  Target std: {sample_targets.std():.2f}")
    
    # Check if targets are all the same
    if sample_targets.std() == 0:
        print("  WARNING: All target values are the same! This will cause training issues.")
    else:
        print(f"  Target variance: {sample_targets.var():.2f}")
    
    for i, date in enumerate(unique_dates):
        if i % 10 == 0:
            print(f"  Processing date {i+1}/{len(unique_dates)}: {date}")
        
        # Get data for this date
        date_data = df_all[df_all['date'] == date]
        
        if len(date_data) == 0:
            continue
            
        # Create node features for each city
        node_features = []
        city_targets = []
        city_original_targets = []
        
        for city in actual_cities:
            city_data = date_data[date_data['location'] == city]
            if len(city_data) == 0:
                # Create dummy features for missing city
                dummy_features = np.zeros(len(feature_columns))
                node_features.append(dummy_features)
                city_targets.append(0.0)  # Single target value
                city_original_targets.append(0.0)
            else:
                # Use the first row for this city on this date
                city_row = city_data.iloc[0]
                features = city_row[feature_columns].values.astype(np.float32)
                node_features.append(features)
                
                # Get target (single GHI value 24 hours ahead)
                target = city_row['target_GHI']
                if pd.isna(target):
                    target = 0.0
                
                # Store original target
                city_original_targets.append(target)
                
                # Scale the target value - ensure it's a 2D array
                target_scaled = target_scaler.transform(np.array([[target]]))[0, 0]
                city_targets.append(target_scaled)
        
        # Create graph if we have any valid data (relaxed condition)
        if len(node_features) > 0:
            x = np.array(node_features, dtype=np.float32)
            a = adj_matrix.astype(np.float32)
            
            # Create a single target per graph (average across cities)
            y = np.mean(city_targets, dtype=np.float32)
            y_original = np.mean(city_original_targets, dtype=np.float32)
            
            # Debug: Print target info for first few graphs
            if i < 5:
                print(f"    Date {date}:")
                print(f"      Original targets: {city_original_targets}")
                print(f"      Scaled targets: {city_targets}")
                print(f"      Original mean: {y_original:.4f}")
                print(f"      Scaled mean: {y:.4f}")
                print(f"      Inverse transform test: {target_scaler.inverse_transform(np.array([[y]]))[0, 0]:.4f}")
            
            # Create Spektral Graph
            graph = Graph(x=x, a=a, y=y)
            graphs.append(graph)
            targets.append(y)
            original_targets.append(y_original)
            graph_dates.append(date)
            graph_cities.append(actual_cities)
    
    print(f"Created {len(graphs)} daily graphs (limited to {max_graphs})")
    return graphs, targets, original_targets, ghi_scaler, target_scaler, graph_dates, graph_cities

class GHIDataset(Dataset):
    def __init__(self, graphs, y, **kwargs):
        self.graphs = graphs
        self.y = y
        super().__init__(**kwargs)

    def read(self):
        return self.graphs

def build_gnn_model(input_shape, output_units=1):
    """Build a lightweight GNN model for memory efficiency."""
    print("Building lightweight GNN model...")
    
    # Input layers
    x_in = layers.Input(shape=input_shape, name='x_in')
    a_in = layers.Input(shape=(None,), sparse=True, name='a_in')
    i_in = layers.Input(shape=(), dtype=tf.int64, name='i_in')
    
    # GNN layers with better architecture
    x = GCNConv(64, activation='relu')([x_in, a_in])
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = GCNConv(64, activation='relu')([x, a_in])
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = GCNConv(32, activation='relu')([x, a_in])
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Custom aggregation layer to group nodes by graph
    def aggregate_nodes(inputs):
        node_features, batch_indices = inputs
        # Use tf.math.unsorted_segment_mean to aggregate features per graph
        aggregated_features = tf.math.unsorted_segment_mean(
            node_features, 
            batch_indices, 
            num_segments=tf.reduce_max(batch_indices) + 1
        )
        return aggregated_features
    
    # Apply the aggregation
    x = layers.Lambda(aggregate_nodes)([x, i_in])
    
    # Dense layers for final prediction
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Use linear activation for final layer to allow negative values
    outputs = layers.Dense(output_units, activation=None, name='output')(x)

    model = models.Model(inputs=[x_in, a_in, i_in], outputs=outputs)
    return model

def main():
    """Main function with comprehensive debugging."""
    start_time = time.time()
    
    try:
        os.makedirs("models", exist_ok=True)
        print("\nLoading and preparing data...")
        
        # Load data with memory management
        all_dfs = []
        for i, city in enumerate(CITIES):
            print(f"Loading data for city {i+1}/{len(CITIES)}: {city}")
            df = load_data(CONFIG["data_locations"], city)
            print(f"  Raw data shape: {df.shape}")
            
            print(f"  Creating features for {city}...")
            df = create_features(df)
            print(f"  After feature creation: {df.shape}")
            
            df['location'] = city
            all_dfs.append(df)
            print(f"  ✓ {city} data processed")
            
            # Clear memory
            gc.collect()
        
        print("Concatenating all city data...")
        df_all = pd.concat(all_dfs).sort_values("datetime")
        print(f"Combined data shape: {df_all.shape}")
        
        # Clear individual dataframes
        del all_dfs
        gc.collect()

        # Get the actual cities present in the data
        actual_cities = df_all['location'].unique()
        print(f"\nActual cities in data: {actual_cities}")
        print(f"Number of cities: {len(actual_cities)}")

        print("\nComputing adjacency matrix...")
        adj_matrix = compute_weighted_adjacency(df_all, alpha=0.5)
        print(f"Adjacency matrix computed: {adj_matrix.shape}")

        print("\nBuilding daily graphs...")
        graphs, targets, original_targets, ghi_scaler, target_scaler, graph_dates, graph_cities = build_daily_graphs_debug(df_all, adj_matrix, actual_cities)
        
        if len(graphs) == 0:
            print("ERROR: No valid graphs created. Exiting.")
            return
        
        # Clear main dataframe
        del df_all
        gc.collect()
        
        print(f"\nCreating dataset...")
        dataset = GHIDataset(graphs, targets)
        print(f"Dataset size: {len(dataset)}")
        
        # Debug first graph
        if len(dataset) > 0:
            first_graph = dataset[0]
            print(f"First graph x shape: {first_graph.x.shape}")
            print(f"First graph a shape: {first_graph.a.shape}")
            print(f"First graph y shape: {first_graph.y.shape}")
            print(f"First graph y value: {first_graph.y}")
        
        print("\nCreating data loader...")
        loader = DisjointLoader(dataset, batch_size=4, shuffle=True)  # Small batch size
        print("✓ Data loader created")

        # Test the loader
        print("\nTesting data loader...")
        try:
            test_batch = next(iter(loader))
            print(f"✓ Loader test successful")
        except Exception as e:
            print(f"✗ Loader test failed: {e}")
            return

        print("\nBuilding model...")
        if len(dataset) > 0:
            num_features = dataset[0].x.shape[1]
            print(f"Number of features per node: {num_features}")
            print(f"Number of nodes per graph: {dataset[0].x.shape[0]}")
        else:
            print("ERROR: No graphs in dataset!")
            return
        
        model = build_gnn_model(input_shape=(num_features,), output_units=FORECAST_HORIZON)
        model.summary()
        
        # Compile model with better learning rate and loss function
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse', metrics=['mae'])
        
        # Test model compilation
        print("\nTesting model compilation...")
        try:
            # Create dummy inputs to test the model
            dummy_x = tf.random.normal((5, num_features))
            dummy_a = tf.sparse.SparseTensor(
                indices=[[0, 0], [0, 1], [1, 1], [2, 2], [3, 3], [4, 4]], 
                values=[1.0, 0.5, 1.0, 1.0, 1.0, 1.0], 
                dense_shape=[5, 5]
            )
            dummy_i = tf.constant([0, 0, 0, 0, 0])
            
            dummy_output = model([dummy_x, dummy_a, dummy_i])
            print(f"Dummy output shape: {dummy_output.shape}")
            print(f"Dummy output values: {dummy_output.numpy()}")
            print("Model compilation successful!")
        except Exception as e:
            print(f"Model compilation failed: {e}")
            return

        print("\nTraining model with early stopping...")
        
        # Training parameters
        batch_size = 8  # Increased batch size for better training
        epochs = 10  # Reduced for debugging
        patience = 5  # Early stopping patience
        
        # Calculate steps per epoch
        total_graphs = len(dataset)
        steps_per_epoch = max(1, total_graphs // batch_size)
        max_steps_per_epoch = min(steps_per_epoch, 50)  # Limit steps for debugging
        
        print(f"Total graphs: {total_graphs}")
        print(f"Batch size: {batch_size}")
        print(f"Steps per epoch: {max_steps_per_epoch}")
        
        # Training with early stopping
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            epoch_start_time = time.time()
            
            epoch_loss = 0.0
            epoch_mae = 0.0
            num_batches = 0
            
            # Create a new loader for each epoch
            epoch_loader = DisjointLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Process batches with timeout protection
            batch_timeout = 30  # 30 seconds per batch
            for batch_idx in range(max_steps_per_epoch):
                batch_start_time = time.time()
                
                try:
                    batch = next(epoch_loader)
                    
                    inputs, y_true = batch
                    x, a, i = inputs
                    y_true = tf.cast(y_true, tf.float32)
                    
                    with tf.GradientTape() as tape:
                        y_pred = model([x, a, i], training=True)
                        loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
                    
                    # Debug: Print first few predictions during training
                    if epoch == 0 and batch_idx < 3:
                        print(f"      Training batch {batch_idx}: y_true={y_true[:3].numpy()}, y_pred={y_pred[:3].numpy()}")
                    
                    gradients = tape.gradient(loss, model.trainable_variables)
                    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    
                    epoch_loss += loss.numpy()
                    epoch_mae += tf.keras.metrics.MeanAbsoluteError()(y_true, y_pred).numpy()
                    num_batches += 1
                    
                    # Check timeout
                    if time.time() - batch_start_time > batch_timeout:
                        print(f"    Batch {batch_idx} took too long, skipping...")
                        continue
                    
                    if batch_idx % 10 == 0 and batch_idx > 0:
                        progress = (batch_idx / max_steps_per_epoch) * 100
                        print(f"    Processed {batch_idx}/{max_steps_per_epoch} batches ({progress:.1f}%)")
                        
                except StopIteration:
                    print(f"    Reached end of dataset at batch {batch_idx}")
                    break
                except Exception as e:
                    print(f"    Error processing batch {batch_idx}: {e}")
                    continue
            
            # Calculate epoch metrics
            if num_batches > 0:
                epoch_loss /= num_batches
                epoch_mae /= num_batches
                epoch_time = time.time() - epoch_start_time
                
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - MAE: {epoch_mae:.4f} - Time: {epoch_time:.1f}s")
                
                # Early stopping check
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                    print(f"  ✓ New best loss: {best_loss:.4f}")
                else:
                    patience_counter += 1
                    print(f"  No improvement for {patience_counter} epochs")
                
                if patience_counter >= patience:
                    print(f"  Early stopping triggered after {epoch+1} epochs")
                    break
            else:
                print(f"Epoch {epoch+1}/{epochs} - No valid batches processed")
            
            # Clear memory after each epoch
            tf.keras.backend.clear_session()
            gc.collect()
        
        print("Training completed!")

        # Debug evaluation
        print("\nDebug evaluation...")
        
        # Test predictions on a few samples
        loader = DisjointLoader(dataset, batch_size=1, shuffle=False)
        max_eval_batches = min(10, len(dataset))
        
        print(f"Testing predictions on {max_eval_batches} samples...")
        
        for batch_idx in range(max_eval_batches):
            try:
                batch = next(loader)
                inputs, y_true = batch
                x, a, i = inputs
                y_pred = model([x, a, i], training=False)
                
                # Convert to numpy arrays
                if hasattr(y_true, 'numpy'):
                    y_true = y_true.numpy()
                if hasattr(y_pred, 'numpy'):
                    y_pred = y_pred.numpy()
                
                # Inverse transform using target scaler
                y_true_reshaped = y_true.reshape(-1, 1) if len(y_true.shape) == 1 else y_true
                y_pred_reshaped = y_pred.reshape(-1, 1) if len(y_pred.shape) == 1 else y_pred
                
                y_true_original = target_scaler.inverse_transform(y_true_reshaped).flatten()
                y_pred_original = target_scaler.inverse_transform(y_pred_reshaped).flatten()
                
                print(f"  Sample {batch_idx + 1}:")
                print(f"    Scaled - True: {y_true[0]:.4f}, Pred: {y_pred[0]:.4f}")
                print(f"    Original - True: {y_true_original[0]:.4f}, Pred: {y_pred_original[0]:.4f}")
                print(f"    Original target from dataset: {original_targets[batch_idx]:.4f}")
                
            except Exception as e:
                print(f"  Error in sample {batch_idx + 1}: {e}")
                continue
        
        total_time = time.time() - start_time
        print(f"\nDebug completed in {total_time:.1f} seconds!")

    except Exception as e:
        print(f"ERROR: Debug failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main() 