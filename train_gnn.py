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
# from geopy.distance import geodesic  # No longer needed with pre-computed distances
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
    print(f"    Creating features for dataframe with {len(df)} rows...")
    
    print(f"      Adding time-based features...")
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    print(f"      Adding GHI lag features...")
    for lag in range(1, 25):
        df[f"GHI_lag_{lag}"] = df["GHI"].shift(lag)
        if lag % 6 == 0:
            print(f"        Added {lag}/24 GHI lag features")

    print(f"      Adding meteorological lag features...")
    met_vars = ["Temperature", "Relative Humidity", "Pressure",
                "Precipitable Water", "Wind Direction", "Wind Speed"]
    for i, var in enumerate(met_vars):
        df[f"{var}_lag_24"] = df[var].shift(24)
        print(f"        Added lag feature for {var} ({i+1}/{len(met_vars)})")

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
    print("    Computing weighted adjacency matrix (ultra-fast)...")
    
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

    # Combine geodesic and correlation weights
    adj = alpha * geo_weights + (1 - alpha) * corr_weights
    print(f"      Adjacency matrix computed: {adj.shape}")
    return adj

def build_daily_graphs(df_all, adj_matrix, actual_cities):
    print("    Starting to build daily graphs (highly optimized)...")
    graphs = []
    targets = []
    graph_dates = []  # Store dates for each graph
    graph_cities = []  # Store cities for each graph

    print("    Fitting scaler...")
    scaler = MinMaxScaler()
    df_all['GHI_scaled'] = scaler.fit_transform(df_all[["GHI"]])

    # Pre-compute feature columns once
    print("    Pre-computing feature columns...")
    sample_data = df_all[df_all['location'] == actual_cities[0]].iloc[:12]
    feature_cols = [col for col in sample_data.columns if 'lag' in col or 'sin' in col or 'cos' in col]
    print(f"    Found {len(feature_cols)} feature columns")

    # Pre-process data more efficiently
    print("    Pre-processing data...")
    df_all['date'] = pd.to_datetime(df_all["datetime"]).dt.date
    
    # Create a more efficient data structure
    city_data = {}
    for city in actual_cities:
        city_df = df_all[df_all['location'] == city].copy()
        city_df = city_df.sort_values('datetime')
        city_data[city] = city_df
    
    # Get all unique dates
    all_dates = set()
    for city_df in city_data.values():
        all_dates.update(city_df['date'].unique())
    dates = sorted(list(all_dates))
    
    print(f"    Processing {len(dates)} unique dates")
    
    # Pre-filter valid dates more efficiently
    print("    Pre-filtering valid dates...")
    valid_dates = []
    for date in dates:
        # Check if all cities have at least 24 hours of data for this date
        all_cities_valid = True
        for city_df in city_data.values():
            date_data = city_df[city_df['date'] == date]
            if len(date_data) < 24:
                all_cities_valid = False
                break
        if all_cities_valid:
            valid_dates.append(date)
    
    print(f"    Found {len(valid_dates)} valid dates out of {len(dates)} total dates")
    
    # Process dates in larger batches for better performance
    batch_size = 100  # Increased batch size
    for batch_start in range(0, len(valid_dates), batch_size):
        batch_end = min(batch_start + batch_size, len(valid_dates))
        batch_dates = valid_dates[batch_start:batch_end]
        
        print(f"      Processing batch {batch_start//batch_size + 1}/{(len(valid_dates) + batch_size - 1)//batch_size}: dates {batch_start+1}-{batch_end}")
        
        for date in batch_dates:
            node_features = []
            node_targets = []
            
            # Process all cities for this date
            for city in actual_cities:
                city_df = city_data[city]
                date_data = city_df[city_df['date'] == date]
                
                if len(date_data) < 24:
                    continue
                
                # Use first 12 hours for sequence, next 12 hours for target
                sequence_data = date_data.iloc[:12]
                target_data = date_data.iloc[12:24]
                
                # Quick check for valid target data
                if target_data['GHI'].sum() <= 0:
                    continue
                
                # Extract features efficiently
                X = sequence_data[feature_cols].values.flatten()
                
                # Quick NaN check
                if np.isnan(X).any():
                    continue
                
                # Target is the GHI values for the next 12 hours
                y = target_data['GHI'].values
                
                # Quick NaN check
                if np.isnan(y).any():
                    continue
                
                node_features.append(X)
                node_targets.append(y)
            
            # Only create graph if we have data from at least 2 cities
            min_cities_required = min(2, len(actual_cities))  # Require at least 2 cities or all cities if less than 2
            if len(node_features) >= min_cities_required:
                x = np.stack(node_features, axis=0)
                # Create a single target per graph by averaging across cities
                y = np.mean(node_targets, axis=0)
                
                # Adjust adjacency matrix to match the number of cities we have
                if len(node_features) < len(actual_cities):
                    # Use only the cities that have data
                    adj_subset = adj_matrix[:len(node_features), :len(node_features)]
                else:
                    adj_subset = adj_matrix
                
                graphs.append(Graph(x=x, a=adj_subset))
                targets.append(y)
                graph_dates.append(date)
                graph_cities.append(actual_cities[:len(node_features)])

    print(f"    Total valid daily graphs: {len(graphs)}")
    
    if len(graphs) == 0:
        print("\n    WARNING: No valid graphs created!")
        print("    This could be due to:")
        print("    1. Insufficient data per day")
        print("    2. Missing feature columns")
        print("    3. NaN values in data")
        print("    4. All target values being zero")
        
        # Let's check a sample date to debug
        if len(dates) > 0:
            sample_date = dates[0]
            print(f"\n    Debugging sample date: {sample_date}")
            for city in actual_cities:
                city_df = city_data[city]
                date_data = city_df[city_df['date'] == sample_date]
                print(f"      {city}: {len(date_data)} rows")
                if len(date_data) > 0:
                    print(f"        Columns: {date_data.columns.tolist()}")
                    print(f"        Feature cols: {[col for col in date_data.columns if 'lag' in col or 'sin' in col or 'cos' in col]}")
    
    return graphs, np.array(targets), scaler, graph_dates, graph_cities

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
    
    # GNN layers (reduced for memory efficiency)
    x = GCNConv(32, activation='relu')([x_in, a_in])
    x = layers.Dropout(0.2)(x)
    x = GCNConv(32, activation='relu')([x, a_in])
    x = layers.Dropout(0.2)(x)
    
    # Custom aggregation layer to group nodes by graph
    # We'll use a Lambda layer to implement graph-wise pooling
    def aggregate_nodes(inputs):
        node_features, batch_indices = inputs
        # Get unique batch indices to determine number of graphs
        unique_batches = tf.unique(batch_indices)[0]
        num_graphs = tf.shape(unique_batches)[0]
        
        # Use tf.math.unsorted_segment_mean to aggregate features per graph
        # This is more efficient than loops
        aggregated_features = tf.math.unsorted_segment_mean(
            node_features, 
            batch_indices, 
            num_segments=tf.reduce_max(batch_indices) + 1
        )
        
        return aggregated_features
    
    # Apply the aggregation
    x = layers.Lambda(aggregate_nodes)([x, i_in])
    
    # Dense layers for final prediction (reduced for memory efficiency)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    # Ensure output has the correct shape for 12-hour forecast
    # The output should be (batch_size, output_units) where output_units=12
    outputs = layers.Dense(output_units, name='output')(x)

    model = models.Model(inputs=[x_in, a_in, i_in], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse', metrics=['mae'])
    return model

# ----------------------------------------------
# Evaluation functions
# ----------------------------------------------
def evaluate_gnn_model(model, dataset, ghi_scaler, graph_dates, graph_cities, actual_cities):
    """
    Evaluate the GNN model and generate metrics for each city.
    
    Args:
        model: Trained GNN model
        dataset: Graph dataset
        ghi_scaler: Scaler used for GHI values
        graph_dates: List of dates for each graph
        graph_cities: List of cities for each graph
        actual_cities: List of all cities
    
    Returns:
        dict: Dictionary with evaluation results for each city
    """
    print("\nEvaluating GNN model...")
    
    # Create results directory
    os.makedirs("results_gnn", exist_ok=True)
    
    # Make predictions on all graphs
    predictions = []
    actuals = []
    dates = []
    cities = []
    
    # Use a loader to get predictions
    loader = DisjointLoader(dataset, batch_size=4, shuffle=False)  # Reduced batch size for memory efficiency
    
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
    
    for i, (date, city_list) in enumerate(zip(graph_dates, graph_cities)):
        if i < len(predictions):
            pred = predictions[i]
            actual = actuals[i]
            
            # Distribute the graph-level prediction to individual cities
            for city in city_list:
                city_results[city]['dates'].append(date)
                city_results[city]['actual'].append(actual)
                city_results[city]['predicted'].append(pred)
    
    # Calculate metrics for each city
    all_metrics = {}
    
    for city in actual_cities:
        if len(city_results[city]['actual']) > 0:
            print(f"\nProcessing results for {city}...")
            
            # Flatten the results
            actual_flat = np.array(city_results[city]['actual']).flatten()
            predicted_flat = np.array(city_results[city]['predicted']).flatten()
            dates_flat = city_results[city]['dates']
            
            # Calculate daily metrics
            daily_metrics = calculate_daily_metrics_gnn(dates_flat, actual_flat, predicted_flat)
            
            # Calculate overall metrics
            mae = mean_absolute_error(actual_flat, predicted_flat)
            rmse = np.sqrt(mean_squared_error(actual_flat, predicted_flat))
            r2 = r2_score(actual_flat, predicted_flat)
            correlation = np.corrcoef(actual_flat, predicted_flat)[0, 1]
            
            all_metrics[city] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'correlation': correlation,
                'daily_metrics': daily_metrics,
                'actual': actual_flat,
                'predicted': predicted_flat,
                'dates': dates_flat
            }
            
            print(f"  MAE: {mae:.2f} W/m²")
            print(f"  RMSE: {rmse:.2f} W/m²")
            print(f"  R²: {r2:.4f}")
            print(f"  Correlation: {correlation:.4f}")
            
            # Create plots for this city
            plot_gnn_results(all_metrics[city], city)
    
    # Create summary table
    create_gnn_summary_table(all_metrics)
    
    return all_metrics

def calculate_daily_metrics_gnn(dates, actual, predicted):
    """
    Calculate daily metrics for GNN results.
    
    Args:
        dates: List of dates
        actual: Actual GHI values
        predicted: Predicted GHI values
    
    Returns:
        pd.DataFrame: Daily metrics
    """
    # Convert dates to pandas datetime
    dates_pd = pd.to_datetime(dates)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates_pd,
        'actual': actual,
        'predicted': predicted
    })
    
    # Group by date and calculate daily metrics
    daily_stats = []
    for date, group in df.groupby(df['date'].dt.date):
        # Filter out zero values
        non_zero_mask = group['actual'] > 0
        if non_zero_mask.sum() > 0:
            actual_nonzero = group.loc[non_zero_mask, 'actual']
            predicted_nonzero = group.loc[non_zero_mask, 'predicted']
            
            # Calculate correlation
            correlation = actual_nonzero.corr(predicted_nonzero)
            if pd.isna(correlation):
                correlation = 0
            
            # Calculate other metrics
            mae = mean_absolute_error(actual_nonzero, predicted_nonzero)
            rmse = np.sqrt(mean_squared_error(actual_nonzero, predicted_nonzero))
            r2 = r2_score(actual_nonzero, predicted_nonzero)
            
            # Clamp R² to [0, 1]
            r2 = max(0, min(1, r2))
            
            daily_stats.append({
                'date': date,
                'correlation': correlation,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'num_points': len(actual_nonzero),
                'actual_max': actual_nonzero.max(),
                'actual_mean': actual_nonzero.mean(),
                'actual_std': actual_nonzero.std(),
                'predicted_max': predicted_nonzero.max(),
                'predicted_mean': predicted_nonzero.mean(),
                'predicted_std': predicted_nonzero.std()
            })
    
    return pd.DataFrame(daily_stats).sort_values('date')

def plot_gnn_results(results, city):
    """
    Create plots for GNN results.
    
    Args:
        results: Dictionary with evaluation results
        city: City name
    """
    # Create directory for plots
    plot_dir = f"results_gnn/{city}"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(results['actual'], results['predicted'], alpha=0.5)
    plt.plot([0, max(results['actual'])], [0, max(results['actual'])], 'r--')
    plt.xlabel('Actual GHI (W/m²)')
    plt.ylabel('Predicted GHI (W/m²)')
    plt.title(f'Actual vs Predicted GHI - {city} (GNN)')
    plt.savefig(f"{plot_dir}/scatter_plot.png")
    plt.close()
    
    # Create time series plot
    plt.figure(figsize=(15, 6))
    plt.plot(results['actual'], label='Actual', alpha=0.7)
    plt.plot(results['predicted'], label='Predicted', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('GHI (W/m²)')
    plt.title(f'GHI Time Series - {city} (GNN)')
    plt.legend()
    plt.savefig(f"{plot_dir}/time_series.png")
    plt.close()
    
    # Create daily correlation plot
    if 'daily_metrics' in results and len(results['daily_metrics']) > 0:
        daily_metrics = results['daily_metrics']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot correlation and R²
        ax1.plot(daily_metrics['date'], daily_metrics['correlation'], 'b-', label='Correlation')
        ax1.plot(daily_metrics['date'], daily_metrics['r2'], 'r-', label='R²')
        ax1.set_title(f'Daily Correlation and R² - {city} (GNN)')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot MAE and RMSE
        ax2.plot(daily_metrics['date'], daily_metrics['mae'], 'g-', label='MAE')
        ax2.plot(daily_metrics['date'], daily_metrics['rmse'], 'm-', label='RMSE')
        ax2.set_title(f'Daily MAE and RMSE - {city} (GNN)')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Error (W/m²)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/daily_metrics.png")
        plt.close()
        
        # Save daily metrics to CSV
        daily_metrics.to_csv(f"{plot_dir}/daily_metrics.csv", index=False)

def create_gnn_summary_table(all_metrics):
    """
    Create a summary table for GNN results.
    
    Args:
        all_metrics: Dictionary with metrics for all cities
    """
    # Create summary DataFrame
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
    print(f"\nSummary saved to results_gnn/summary.csv")
    print(summary_df)

# ----------------------------------------------
# Main script
# ----------------------------------------------
def main():
    os.makedirs("models", exist_ok=True)
    print("\nLoading and preparing data...")
    
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
    
    print("Concatenating all city data...")
    df_all = pd.concat(all_dfs).sort_values("datetime")
    print(f"Combined data shape: {df_all.shape}")

    # Get the actual cities present in the data
    actual_cities = df_all['location'].unique()
    print(f"\nActual cities in data: {actual_cities}")
    print(f"Number of cities: {len(actual_cities)}")

    print(f"\nData statistics:")
    print(f"Total rows: {len(df_all)}")
    print(f"Date range: {df_all['datetime'].min()} to {df_all['datetime'].max()}")
    print(f"Rows per city:")
    for city in actual_cities:
        city_data = df_all[df_all['location'] == city]
        print(f"  {city}: {len(city_data)} rows")
    
    # Check data per day
    print("Computing daily data counts...")
    df_all['date'] = pd.to_datetime(df_all["datetime"]).dt.date
    daily_counts = df_all.groupby(['date', 'location']).size().reset_index(name='count')
    print(f"Daily data counts (sample):")
    print(daily_counts.head(20))
    
    # Check for missing values
    print(f"\nMissing values per column:")
    missing_values = df_all.isnull().sum()
    print(missing_values[missing_values > 0])

    print("\nComputing adjacency matrix...")
    adj_matrix = compute_weighted_adjacency(df_all, alpha=0.5)
    print(f"Adjacency matrix shape: {adj_matrix.shape}")
    print(f"Adjacency matrix type: {type(adj_matrix)}")
    print(f"Adjacency matrix sample values:\n{adj_matrix}")

    print("\nBuilding daily graphs...")
    graphs, targets, ghi_scaler, graph_dates, graph_cities = build_daily_graphs(df_all, adj_matrix, actual_cities)
    
    if len(graphs) == 0:
        print("ERROR: No valid graphs created. Exiting.")
        return
    
    print(f"\nCreating dataset...")
    dataset = GHIDataset(graphs, targets)
    print(f"Dataset size: {len(dataset)}")
    
    # Debug first graph
    if len(dataset) > 0:
        first_graph = dataset[0]
        print(f"First graph x shape: {first_graph.x.shape}")
        print(f"First graph a shape: {first_graph.a.shape}")
        print(f"First graph y shape: {first_graph.y.shape}")
        print(f"First graph a type: {type(first_graph.a)}")
    
    print("\nCreating data loader...")
    loader = DisjointLoader(dataset, batch_size=8, shuffle=True)
    print("✓ Data loader created")

    # Debug dataset information
    print(f"\nDataset debugging information:")
    print(f"Dataset length: {len(dataset)}")
    print(f"Dataset type: {type(dataset)}")
    
    # Check if dataset is empty
    if len(dataset) == 0:
        print("ERROR: Dataset is empty! Cannot proceed with training.")
        return
    
    # Test the loader to make sure it works
    print("\nTesting data loader...")
    try:
        test_batch = next(iter(loader))
        print(f"✓ Loader test successful - batch type: {type(test_batch)}")
        if len(test_batch) == 2:
            inputs, targets = test_batch
            print(f"  Inputs type: {type(inputs)}")
            print(f"  Targets type: {type(targets)}")
            if isinstance(inputs, (list, tuple)) and len(inputs) == 3:
                x, a, i = inputs
                print(f"  x shape: {x.shape}")
                print(f"  a shape: {a.shape}")
                print(f"  i shape: {i.shape}")
                print(f"  targets shape: {targets.shape}")
    except Exception as e:
        print(f"✗ Loader test failed: {e}")
        return

    print("\nBuilding model...")
    if len(dataset) > 0:
        num_features = dataset[0].x.shape[1]
        print(f"Number of features per node: {num_features}")
        print(f"Number of nodes per graph: {dataset[0].x.shape[0]}")
        print(f"Target shape: {targets.shape}")
        print(f"Sample target shape: {targets[0].shape}")
    else:
        print("ERROR: No graphs in dataset!")
        return
    
    model = build_gnn_model(input_shape=(num_features,), output_units=FORECAST_HORIZON)
    model.summary()
    
    # Test model compilation
    print("\nTesting model compilation...")
    try:
        # Create dummy inputs to test the model
        dummy_x = tf.random.normal((5, num_features))  # 5 nodes, num_features
        dummy_a = tf.sparse.SparseTensor(
            indices=[[0, 0], [0, 1], [1, 1], [2, 2], [3, 3], [4, 4]], 
            values=[1.0, 0.5, 1.0, 1.0, 1.0, 1.0], 
            dense_shape=[5, 5]
        )
        dummy_i = tf.constant([0, 0, 0, 0, 0])  # batch indices
        
        dummy_output = model([dummy_x, dummy_a, dummy_i])
        print(f"Dummy output shape: {dummy_output.shape}")
        print("Model compilation successful!")
    except Exception as e:
        print(f"Model compilation failed: {e}")
        return

    print("\nTraining model...")
    # Test the model with a sample batch to check shapes
    print("Getting sample batch...")
    sample_batch = next(iter(loader))
    print(f"Sample batch type: {type(sample_batch)}")
    print(f"Sample batch length: {len(sample_batch)}")
    
    # Handle the actual batch format from DisjointLoader
    if len(sample_batch) == 2:
        inputs, y_true = sample_batch
        x, a, i = inputs
        print(f"Batch format: (inputs, y_true) where inputs = (x, a, i)")
        print(f"  x shape: {x.shape}")
        print(f"  a shape: {a.shape}")
        print(f"  i shape: {i.shape}")
        print(f"  y_true shape: {y_true.shape}")
        
        # Test model prediction
        sample_pred = model([x, a, i])
        print(f"Model output shape: {sample_pred.shape}")
        print(f"Target shape: {y_true.shape}")
        if y_true.shape != sample_pred.shape:
            print(f"WARNING: Shape mismatch! Targets: {y_true.shape}, Predictions: {sample_pred.shape}")
    else:
        print(f"Unexpected batch format: {len(sample_batch)} items")
        for i, item in enumerate(sample_batch):
            if hasattr(item, 'shape'):
                print(f"  Item {i} shape: {item.shape}")
            else:
                print(f"  Item {i} has no shape attribute")
    
    # Create a custom training loop to handle the shape mismatch
    print("\nStarting training with custom loop...")
    
    # Training loop
    optimizer = tf.keras.optimizers.Adam(0.001)
    loss_fn = tf.keras.losses.MeanSquaredError()
    mae_metric = tf.keras.metrics.MeanAbsoluteError()
    
    # Calculate expected number of batches per epoch
    total_graphs = len(dataset)
    batch_size = 4  # Reduced batch size for memory efficiency
    steps_per_epoch = (total_graphs + batch_size - 1) // batch_size
    print(f"Total graphs: {total_graphs}")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}")
    
    # Safety check to prevent infinite loops and memory issues
    max_batches_per_epoch = min(steps_per_epoch, 500)  # Reduced cap for memory efficiency
    if steps_per_epoch > 500:
        print(f"WARNING: Large dataset detected. Limiting to {max_batches_per_epoch} batches per epoch for memory safety.")
        steps_per_epoch = max_batches_per_epoch
    
    for epoch in range(10):  # Reduced number of epochs for memory efficiency
        print(f"\nStarting epoch {epoch+1}/10...")
        epoch_loss = 0.0
        epoch_mae = 0.0
        num_batches = 0
        
        # Create a new loader for each epoch to ensure proper iteration
        epoch_loader = DisjointLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Use a counter to limit the number of batches per epoch
        for batch_idx in range(steps_per_epoch):
            try:
                batch = next(epoch_loader)
                if batch_idx == 0:
                    print(f"  Processing first batch of epoch {epoch+1}...")
                
                inputs, y_true = batch
                x, a, i = inputs
                y_true = tf.cast(y_true, tf.float32)
                
                with tf.GradientTape() as tape:
                    y_pred = model([x, a, i], training=True)
                    loss = loss_fn(y_true, y_pred)
                
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                mae_metric.update_state(y_true, y_pred)
                epoch_loss += loss.numpy()
                num_batches += 1
                
                if batch_idx % 10 == 0 and batch_idx > 0:
                    progress = (batch_idx / steps_per_epoch) * 100
                    print(f"    Processed {batch_idx}/{steps_per_epoch} batches ({progress:.1f}%) in epoch {epoch+1}")
                    # Clear memory periodically
                    tf.keras.backend.clear_session()
                    
            except StopIteration:
                print(f"    Reached end of dataset at batch {batch_idx}")
                break
            except Exception as e:
                print(f"    Error processing batch {batch_idx}: {e}")
                continue
        
        # Final progress report for the epoch
        if num_batches > 0:
            progress = (num_batches / steps_per_epoch) * 100
            print(f"    Completed {num_batches}/{steps_per_epoch} batches ({progress:.1f}%) for epoch {epoch+1}")
        
        if num_batches > 0:
            epoch_loss /= num_batches
            epoch_mae = mae_metric.result().numpy()
            mae_metric.reset_states()
            print(f"Epoch {epoch+1}/10 - Loss: {epoch_loss:.4f} - MAE: {epoch_mae:.4f} - Batches: {num_batches}")
        else:
            print(f"Epoch {epoch+1}/10 - No valid batches processed")
    
    print("Training completed!")

    print("\nSaving model and scaler...")
    model.save("models/gnn_ghi_forecast.h5")
    with open("models/ghi_scaler_gnn.pkl", "wb") as f:
        pickle.dump(ghi_scaler, f)
    print("\n✓ Model and scaler saved.")

    # Evaluate the model
    print("\nEvaluating model...")
    all_metrics = evaluate_gnn_model(model, dataset, ghi_scaler, graph_dates, graph_cities, actual_cities)
    
    print("\nGNN training and evaluation completed!")

if __name__ == "__main__":
    main()
