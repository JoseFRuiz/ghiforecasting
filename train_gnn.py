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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt
import datetime
import gc
import time
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
    """Create features with reduced complexity for memory efficiency."""
    print(f"    Creating features for dataframe with {len(df)} rows...")
    
    print(f"      Adding time-based features...")
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    print(f"      Adding GHI lag features (reduced)...")
    # Reduced lag features for memory efficiency
    for lag in [1, 3, 6, 12, 24]:
        df[f"GHI_lag_{lag}"] = df["GHI"].shift(lag)
        print(f"        Added GHI lag {lag}")

    print(f"      Adding meteorological lag features (reduced)...")
    # Only use key meteorological variables
    met_vars = ["Temperature", "Relative Humidity", "Wind Speed"]
    for var in met_vars:
        df[f"{var}_lag_24"] = df[var].shift(24)
        print(f"        Added lag feature for {var}")

    print(f"      Adding target variable...")
    df["target"] = df["GHI"].shift(-24)  # forecast 24 hours ahead
    
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

def build_daily_graphs(df_all, adj_matrix, actual_cities):
    """Build daily graphs with memory optimization."""
    print("Building daily graphs (optimized version)...")
    
    # Limit the number of graphs for memory efficiency
    max_graphs = 100  # Reduced from previous versions
    
    # Get unique dates
    df_all['date'] = pd.to_datetime(df_all['datetime']).dt.date
    unique_dates = sorted(df_all['date'].unique())
    
    print(f"Found {len(unique_dates)} unique dates")
    if len(unique_dates) > max_graphs:
        print(f"Limiting to {max_graphs} graphs for memory efficiency")
        unique_dates = unique_dates[:max_graphs]
    
    graphs = []
    targets = []
    graph_dates = []
    graph_cities = []
    
    # Fit scaler on a sample of the data
    print("Fitting scaler...")
    sample_data = df_all.sample(min(10000, len(df_all)), random_state=42)
    feature_columns = [col for col in sample_data.columns if col not in 
                      ['datetime', 'date', 'location', 'target', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'City']]
    print(f"Found {len(feature_columns)} feature columns")
    print(f"Feature columns: {feature_columns}")
    
    ghi_scaler = MinMaxScaler()
    ghi_scaler.fit(sample_data[['GHI']])
    
    print(f"Processing {len(unique_dates)} unique dates (limited to {max_graphs} graphs)")
    
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
        
        for city in actual_cities:
            city_data = date_data[date_data['location'] == city]
            if len(city_data) == 0:
                # Create dummy features for missing city
                dummy_features = np.zeros(len(feature_columns))
                node_features.append(dummy_features)
                city_targets.append(np.zeros(FORECAST_HORIZON))
            else:
                # Use the first row for this city on this date
                city_row = city_data.iloc[0]
                features = city_row[feature_columns].values.astype(np.float32)
                node_features.append(features)
                
                # Get target (next 12 hours of GHI)
                target = city_row['target']
                if pd.isna(target):
                    target = 0.0
                city_targets.append(np.array([target] * FORECAST_HORIZON, dtype=np.float32))
        
        # Create graph
        if len(node_features) > 0:
            x = np.array(node_features, dtype=np.float32)
            a = adj_matrix.astype(np.float32)
            y = np.array(city_targets, dtype=np.float32)
            
            # Create Spektral Graph
            graph = Graph(x=x, a=a, y=y)
            graphs.append(graph)
            targets.append(y)
            graph_dates.append(date)
            graph_cities.append(actual_cities)
    
    print(f"Created {len(graphs)} daily graphs (limited to {max_graphs})")
    return graphs, targets, ghi_scaler, graph_dates, graph_cities

class GHIDataset(Dataset):
    def __init__(self, graphs, y, **kwargs):
        self.graphs = graphs
        self.y = y
        super().__init__(**kwargs)

    def read(self):
        return self.graphs

def build_gnn_model(input_shape, output_units):
    """Build a lightweight GNN model for memory efficiency."""
    print("Building lightweight GNN model...")
    
    # Input layers
    x_in = layers.Input(shape=input_shape, name='x_in')
    a_in = layers.Input(shape=(None,), sparse=True, name='a_in')
    i_in = layers.Input(shape=(), dtype=tf.int64, name='i_in')
    
    # GNN layers (further reduced for memory efficiency)
    x = GCNConv(16, activation='relu')([x_in, a_in])
    x = layers.Dropout(0.2)(x)
    x = GCNConv(16, activation='relu')([x, a_in])
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
    
    # Dense layers for final prediction (further reduced)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(output_units, name='output')(x)

    model = models.Model(inputs=[x_in, a_in, i_in], outputs=outputs)
    return model

# ----------------------------------------------
# Evaluation functions
# ----------------------------------------------
def evaluate_gnn_model(model, dataset, ghi_scaler, graph_dates, graph_cities, actual_cities):
    """Evaluate the GNN model and generate metrics for each city."""
    print("\nEvaluating GNN model...")
    
    # Create results directory
    os.makedirs("results_gnn", exist_ok=True)
    
    # Make predictions on all graphs
    predictions = []
    actuals = []
    dates = []
    cities = []
    
    # Use a loader to get predictions
    loader = DisjointLoader(dataset, batch_size=2, shuffle=False)  # Further reduced batch size
    
    for batch in loader:
        inputs, y_true = batch
        x, a, i = inputs
        y_pred = model([x, a, i], training=False)
        
        # Convert to numpy arrays (handle both tensor and numpy cases)
        if hasattr(y_true, 'numpy'):
            y_true = y_true.numpy()
        if hasattr(y_pred, 'numpy'):
            y_pred = y_pred.numpy()
        
        # Inverse transform predictions and actual values
        y_true_original = ghi_scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(y_true.shape)
        y_pred_original = ghi_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
        
        predictions.extend(y_pred_original)
        actuals.extend(y_true_original)
    
    # Organize results by city
    city_results = {city: {'dates': [], 'actual': [], 'predicted': []} for city in actual_cities}
    
    # Process results by city
    for i, (date, city_list) in enumerate(zip(graph_dates, graph_cities)):
        if i < len(predictions):
            for j, city in enumerate(city_list):
                if j < len(predictions[i]):
                    city_results[city]['dates'].append(date)
                    city_results[city]['actual'].append(predictions[i][j])
                    city_results[city]['predicted'].append(actuals[i][j])
    
    # Calculate metrics for each city
    all_metrics = {}
    for city in actual_cities:
        if len(city_results[city]['actual']) > 0:
            actual = np.array(city_results[city]['actual'])
            predicted = np.array(city_results[city]['predicted'])
            
            # Calculate metrics
            mae = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            r2 = r2_score(actual, predicted)
            correlation = np.corrcoef(actual, predicted)[0, 1]
            
            # Calculate daily metrics
            dates = pd.to_datetime(city_results[city]['dates'])
            daily_metrics = calculate_daily_metrics_gnn(dates, actual, predicted)
            
            all_metrics[city] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'correlation': correlation,
                'daily_metrics': daily_metrics,
                'actual': actual,
                'predicted': predicted,
                'dates': dates
            }
            
            # Create plots
            plot_gnn_results(all_metrics[city], city)
    
    # Create summary table
    create_gnn_summary_table(all_metrics)
    
    return all_metrics

def calculate_daily_metrics_gnn(dates, actual, predicted):
    """Calculate daily metrics for GNN results."""
    # Group by date and calculate daily statistics
    df = pd.DataFrame({
        'date': dates,
        'actual': actual,
        'predicted': predicted
    })
    
    daily_stats = df.groupby('date').agg({
        'actual': ['mean', 'std'],
        'predicted': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    daily_stats.columns = ['date', 'actual_mean', 'actual_std', 'predicted_mean', 'predicted_std']
    
    # Calculate daily correlation
    daily_corr = df.groupby('date').apply(
        lambda x: np.corrcoef(x['actual'], x['predicted'])[0, 1] if len(x) > 1 else 0
    ).reset_index(name='correlation')
    
    # Merge statistics
    daily_metrics = pd.merge(daily_stats, daily_corr, on='date')
    
    return daily_metrics

def plot_gnn_results(results, city):
    """Create plots for GNN results."""
    city_dir = f"results_gnn/{city}"
    os.makedirs(city_dir, exist_ok=True)
    
    # Scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(results['actual'], results['predicted'], alpha=0.6)
    plt.plot([results['actual'].min(), results['actual'].max()], 
             [results['actual'].min(), results['actual'].max()], 'r--', lw=2)
    plt.xlabel('Actual GHI')
    plt.ylabel('Predicted GHI')
    plt.title(f'GNN Predictions vs Actual - {city}')
    plt.savefig(f"{city_dir}/scatter_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Time series plot
    plt.figure(figsize=(15, 6))
    plt.plot(results['dates'], results['actual'], label='Actual', alpha=0.7)
    plt.plot(results['dates'], results['predicted'], label='Predicted', alpha=0.7)
    plt.xlabel('Date')
    plt.ylabel('GHI')
    plt.title(f'GNN Time Series - {city}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{city_dir}/time_series.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save daily metrics
    results['daily_metrics'].to_csv(f"{city_dir}/daily_metrics.csv", index=False)

def create_gnn_summary_table(all_metrics):
    """Create a summary table for all GNN results."""
    summary_data = []
    
    for city, metrics in all_metrics.items():
        summary_data.append({
            'City': city,
            'MAE': metrics['mae'],
            'RMSE': metrics['rmse'],
            'R²': metrics['r2'],
            'Correlation': metrics['correlation']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("results_gnn/summary.csv", index=False)
    print(f"Summary saved to results_gnn/summary.csv")
    print(summary_df)

def main():
    """Main function with comprehensive error handling and memory management."""
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
        graphs, targets, ghi_scaler, graph_dates, graph_cities = build_daily_graphs(df_all, adj_matrix, actual_cities)
        
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
        
        print("\nCreating data loader...")
        loader = DisjointLoader(dataset, batch_size=2, shuffle=True)  # Reduced batch size
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
        
        # Compile model
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
            print("Model compilation successful!")
        except Exception as e:
            print(f"Model compilation failed: {e}")
            return

        print("\nTraining model with early stopping...")
        
        # Training parameters
        batch_size = 2  # Very small batch size for memory efficiency
        epochs = 15  # Reduced epochs
        patience = 3  # Early stopping patience
        
        # Calculate steps per epoch
        total_graphs = len(dataset)
        steps_per_epoch = max(1, total_graphs // batch_size)
        max_steps_per_epoch = min(steps_per_epoch, 50)  # Limit steps for memory safety
        
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
                    
                    gradients = tape.gradient(loss, model.trainable_variables)
                    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    
                    epoch_loss += loss.numpy()
                    epoch_mae += tf.keras.metrics.MeanAbsoluteError()(y_true, y_pred).numpy()
                    num_batches += 1
                    
                    # Check timeout
                    if time.time() - batch_start_time > batch_timeout:
                        print(f"    Batch {batch_idx} took too long, skipping...")
                        continue
                    
                    if batch_idx % 5 == 0 and batch_idx > 0:
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

        print("\nSaving model and scaler...")
        model.save("models/gnn_ghi_forecast.h5")
        with open("models/ghi_scaler_gnn.pkl", "wb") as f:
            pickle.dump(ghi_scaler, f)
        print("✓ Model and scaler saved.")

        # Evaluate the model
        print("\nEvaluating model...")
        all_metrics = evaluate_gnn_model(model, dataset, ghi_scaler, graph_dates, graph_cities, actual_cities)
        
        total_time = time.time() - start_time
        print(f"\nGNN training and evaluation completed in {total_time:.1f} seconds!")

    except Exception as e:
        print(f"ERROR: Training failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
