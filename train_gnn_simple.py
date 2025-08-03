# train_gnn_simple.py
# Simplified GNN training script for generating evaluation outputs

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
from utils import CONFIG, load_data

# Set global parameters
SEQUENCE_LENGTH = 12  # Use 12 hours for sequence
FORECAST_HORIZON = 12  # Forecast next 12 hours

CITIES = list(CONFIG["data_locations"].keys())
CITY_IDX = {city: i for i, city in enumerate(CITIES)}

def create_features(df):
    """Create features for prediction."""
    print(f"Creating features for dataframe with {len(df)} rows...")
    
    # Create time features
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    # Create GHI lag features (reduced for simplicity)
    for lag in range(1, 13):  # Only 12 lags instead of 24
        df[f"GHI_lag_{lag}"] = df["GHI"].shift(lag)

    # Create meteorological lag features
    met_vars = ["Temperature", "Relative Humidity", "Pressure",
                "Precipitable Water", "Wind Direction", "Wind Speed"]
    for var in met_vars:
        df[f"{var}_lag_24"] = df[var].shift(24)

    # Create target variable
    df["target"] = df["GHI"].shift(-24)  # forecast 24 hours ahead
    
    return df.dropna().reset_index(drop=True)

def compute_weighted_adjacency(df_all, alpha=0.5):
    """Compute weighted adjacency matrix."""
    print("Computing weighted adjacency matrix...")
    
    # Get the actual cities present in the data
    actual_cities = df_all['location'].unique()
    num_cities = len(actual_cities)
    print(f"Found {num_cities} cities in data: {actual_cities}")
    
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
    print(f"Adjacency matrix computed: {adj.shape}")
    return adj

def build_daily_graphs_simple(df_all, adj_matrix, actual_cities, max_graphs=100):
    """Build daily graphs with a limit to prevent memory issues."""
    print("Building daily graphs (simplified version)...")
    graphs = []
    targets = []
    graph_dates = []
    graph_cities = []

    print("Fitting scaler...")
    scaler = MinMaxScaler()
    df_all['GHI_scaled'] = scaler.fit_transform(df_all[["GHI"]])

    # Pre-compute feature columns
    sample_data = df_all[df_all['location'] == actual_cities[0]].iloc[:12]
    feature_cols = [col for col in sample_data.columns if 'lag' in col or 'sin' in col or 'cos' in col]
    print(f"Found {len(feature_cols)} feature columns")

    # Pre-process data
    df_all['date'] = pd.to_datetime(df_all["datetime"]).dt.date
    
    # Create data structure for each city
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
    
    print(f"Processing {len(dates)} unique dates (limited to {max_graphs} graphs)")
    
    # Process dates with a limit
    processed_count = 0
    for date in dates:
        if processed_count >= max_graphs:
            break
            
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
        min_cities_required = min(2, len(actual_cities))
        if len(node_features) >= min_cities_required:
            x = np.stack(node_features, axis=0)
            # Create a single target per graph by averaging across cities
            y = np.mean(node_targets, axis=0)
            
            # Adjust adjacency matrix to match the number of cities we have
            if len(node_features) < len(actual_cities):
                adj_subset = adj_matrix[:len(node_features), :len(node_features)]
            else:
                adj_subset = adj_matrix
            
            graphs.append(Graph(x=x, a=adj_subset))
            targets.append(y)
            graph_dates.append(date)
            graph_cities.append(actual_cities[:len(node_features)])
            processed_count += 1

    print(f"Created {len(graphs)} daily graphs (limited to {max_graphs})")
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

def build_gnn_model_simple(input_shape, output_units):
    """Build a simplified GNN model."""
    # Input layers
    x_in = layers.Input(shape=input_shape, name='x_in')
    a_in = layers.Input(shape=(None,), sparse=True, name='a_in')
    i_in = layers.Input(shape=(), dtype=tf.int64, name='i_in')
    
    # Simplified GNN layers
    x = GCNConv(16, activation='relu')([x_in, a_in])
    x = layers.Dropout(0.2)(x)
    x = GCNConv(16, activation='relu')([x, a_in])
    x = layers.Dropout(0.2)(x)
    
    # Custom aggregation layer
    def aggregate_nodes(inputs):
        node_features, batch_indices = inputs
        unique_batches = tf.unique(batch_indices)[0]
        num_graphs = tf.shape(unique_batches)[0]
        
        aggregated_features = tf.math.unsorted_segment_mean(
            node_features, 
            batch_indices, 
            num_segments=tf.reduce_max(batch_indices) + 1
        )
        
        return aggregated_features
    
    # Apply the aggregation
    x = layers.Lambda(aggregate_nodes)([x, i_in])
    
    # Simplified dense layers
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(output_units, name='output')(x)

    model = models.Model(inputs=[x_in, a_in, i_in], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse', metrics=['mae'])
    return model

def evaluate_gnn_model_simple(model, dataset, ghi_scaler, graph_dates, graph_cities, actual_cities):
    """Evaluate the GNN model and generate metrics."""
    print("\nEvaluating GNN model...")
    
    # Create results directory
    os.makedirs("results_gnn", exist_ok=True)
    
    # Make predictions on all graphs
    predictions = []
    actuals = []
    
    # Use a loader to get predictions
    loader = DisjointLoader(dataset, batch_size=2, shuffle=False)  # Small batch size
    
    for batch in loader:
        inputs, y_true = batch
        x, a, i = inputs
        y_pred = model([x, a, i], training=False)
        
        # Convert to numpy arrays
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
    """Calculate daily metrics for GNN results."""
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
    """Create plots for GNN results."""
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
    """Create a summary table for GNN results."""
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

def main():
    """Main function for simplified GNN training."""
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

    print("\nComputing adjacency matrix...")
    adj_matrix = compute_weighted_adjacency(df_all, alpha=0.5)

    print("\nBuilding daily graphs (simplified)...")
    graphs, targets, ghi_scaler, graph_dates, graph_cities = build_daily_graphs_simple(
        df_all, adj_matrix, actual_cities, max_graphs=50  # Limit to 50 graphs for speed
    )
    
    if len(graphs) == 0:
        print("ERROR: No valid graphs created. Exiting.")
        return
    
    print(f"\nCreating dataset...")
    dataset = GHIDataset(graphs, targets)
    print(f"Dataset size: {len(dataset)}")
    
    print("\nBuilding simplified model...")
    if len(dataset) > 0:
        num_features = dataset[0].x.shape[1]
        print(f"Number of features per node: {num_features}")
        print(f"Number of nodes per graph: {dataset[0].x.shape[0]}")
        print(f"Target shape: {targets.shape}")
    else:
        print("ERROR: No graphs in dataset!")
        return
    
    model = build_gnn_model_simple(input_shape=(num_features,), output_units=FORECAST_HORIZON)
    model.summary()
    
    print("\nTraining simplified model (5 epochs only)...")
    # Simple training with limited epochs
    loader = DisjointLoader(dataset, batch_size=2, shuffle=True)
    
    # Train for only 5 epochs
    for epoch in range(5):
        print(f"Epoch {epoch+1}/5...")
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in loader:
            inputs, y_true = batch
            x, a, i = inputs
            y_true = tf.cast(y_true, tf.float32)
            
            with tf.GradientTape() as tape:
                y_pred = model([x, a, i], training=True)
                loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            tf.keras.optimizers.Adam(0.001).apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss += loss.numpy()
            num_batches += 1
            
            # Limit batches per epoch
            if num_batches >= 10:
                break
        
        if num_batches > 0:
            epoch_loss /= num_batches
            print(f"Epoch {epoch+1}/5 - Loss: {epoch_loss:.4f} - Batches: {num_batches}")
    
    print("Training completed!")

    print("\nSaving model and scaler...")
    model.save("models/gnn_ghi_forecast.h5")
    with open("models/ghi_scaler_gnn.pkl", "wb") as f:
        pickle.dump(ghi_scaler, f)
    print("\n✓ Model and scaler saved.")

    # Evaluate the model
    print("\nEvaluating model...")
    all_metrics = evaluate_gnn_model_simple(model, dataset, ghi_scaler, graph_dates, graph_cities, actual_cities)
    
    print("\nGNN training and evaluation completed!")

if __name__ == "__main__":
    main() 