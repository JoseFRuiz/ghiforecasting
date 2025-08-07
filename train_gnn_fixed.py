# train_gnn_fixed.py
# Joint GHI forecasting using a Graph Neural Network (GNN) - FIXED VERSION

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
FORECAST_HORIZON = 24  # Forecast next 24 hours (24 values)

CITIES = list(CONFIG["data_locations"].keys())
CITY_IDX = {city: i for i, city in enumerate(CITIES)}

# ----------------------------------------------
# Data loading and sequence preparation
# ----------------------------------------------
def create_features(df):
    """Create features with proper target generation for 24-hour forecasting."""
    print(f"    Creating features for dataframe with {len(df)} rows...")
    
    # Debug: Check GHI values before processing
    print(f"      GHI stats before processing:")
    print(f"        Range: [{df['GHI'].min():.2f}, {df['GHI'].max():.2f}]")
    print(f"        Mean: {df['GHI'].mean():.2f}")
    print(f"        Non-zero: {(df['GHI'] > 0).sum()} out of {len(df)}")
    print(f"        Sample GHI values: {df['GHI'].head(20).tolist()}")
    
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

    print(f"      Adding target variable for 24-hour forecasting...")
    # FIXED: Create target as next 24 hours of GHI (24 values)
    target_columns = []
    for hour in range(1, 25):  # 1 to 24 hours ahead
        col_name = f"target_GHI_hour_{hour}"
        df[col_name] = df["GHI"].shift(-hour)
        target_columns.append(col_name)
    
    # Debug: Check target values after creation
    print(f"      Target GHI stats after creation:")
    for i, col in enumerate(target_columns[:5]):  # Show first 5 hours
        print(f"        {col}: range=[{df[col].min():.2f}, {df[col].max():.2f}], mean={df[col].mean():.2f}")
    
    print(f"      Dropping NaN values...")
    original_len = len(df)
    df = df.dropna().reset_index(drop=True)
    final_len = len(df)
    print(f"      Dropped {original_len - final_len} rows with NaN values")
    print(f"      Final shape: {df.shape}")
    
    # Debug: Check targets after dropping NaN
    print(f"      Target GHI stats after dropping NaN:")
    for i, col in enumerate(target_columns[:5]):  # Show first 5 hours
        print(f"        {col}: range=[{df[col].min():.2f}, {df[col].max():.2f}], mean={df[col].mean():.2f}")
    
    return df, target_columns

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

def build_daily_graphs(df_all, adj_matrix, actual_cities, target_columns):
    """Build daily graphs with 24-hour forecasting for all cities."""
    print("Building daily graphs for 24-hour forecasting...")
    
    # Limit the number of graphs for memory efficiency
    max_graphs = 200  # Increased for better training
    
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
                      ['datetime', 'date', 'location'] + target_columns + 
                      ['Year', 'Month', 'Day', 'Hour', 'Minute', 'City']]
    print(f"Found {len(feature_columns)} feature columns")
    
    # Scale both GHI and target_GHI
    ghi_scaler = MinMaxScaler()
    ghi_scaler.fit(sample_data[['GHI']])
    
    # Use StandardScaler for targets to avoid zero issues
    target_scaler = StandardScaler()
    valid_targets = sample_data[target_columns].dropna()
    print(f"Valid targets for scaler fitting: {len(valid_targets)} samples")
    print(f"Valid target range: [{valid_targets.min().min():.2f}, {valid_targets.max().max():.2f}]")
    print(f"Valid target mean: {valid_targets.mean().mean():.2f}")
    print(f"Valid target std: {valid_targets.std().mean():.2f}")
    
    # Check if targets have variance
    if valid_targets.std().mean() == 0:
        print("WARNING: All target values are the same! Using MinMaxScaler instead.")
        target_scaler = MinMaxScaler()
        target_scaler.fit(valid_targets)
    else:
        target_scaler.fit(valid_targets)
    
    print(f"GHI scaler range: [{ghi_scaler.data_min_[0]:.2f}, {ghi_scaler.data_max_[0]:.2f}]")
    print(f"Target scaler info:")
    if hasattr(target_scaler, 'mean_'):
        print(f"  Mean: {target_scaler.mean_.mean():.2f}, std: {target_scaler.scale_.mean():.2f}")
    else:
        print(f"  Range: [{target_scaler.data_min_.mean():.2f}, {target_scaler.data_max_.mean():.2f}]")
    
    # Debug: Test target scaler
    test_targets = sample_data[target_columns].head(2).values
    test_scaled = target_scaler.transform(test_targets)
    test_inverse = target_scaler.inverse_transform(test_scaled)
    print(f"Target scaler test:")
    print(f"  Original shape: {test_targets.shape}")
    print(f"  Scaled shape: {test_scaled.shape}")
    print(f"  Inverse shape: {test_inverse.shape}")
    print(f"  Difference: {np.abs(test_targets - test_inverse).mean():.6f}")
    
    print(f"Processing {len(unique_dates)} unique dates (limited to {max_graphs} graphs)")
    
    # Debug: Check target distribution
    print("Debug: Checking target distribution...")
    sample_targets = df_all[target_columns].dropna()
    print(f"  Target range: [{sample_targets.min().min():.2f}, {sample_targets.max().max():.2f}]")
    print(f"  Target mean: {sample_targets.mean().mean():.2f}")
    print(f"  Non-zero targets: {(sample_targets > 0).sum().sum()} out of {sample_targets.size}")
    print(f"  Target std: {sample_targets.std().mean():.2f}")
    
    # Check if targets are all the same
    if sample_targets.std().mean() == 0:
        print("  WARNING: All target values are the same! This will cause training issues.")
    else:
        print(f"  Target variance: {sample_targets.var().mean():.2f}")
    
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
                city_targets.append(np.zeros(len(target_columns)))  # 24-hour target
            else:
                # Use a random row for this city on this date to get varied targets
                if len(city_data) > 1:
                    # Select a random row to get varied targets
                    city_row = city_data.sample(n=1, random_state=i).iloc[0]
                else:
                    city_row = city_data.iloc[0]
                
                features = city_row[feature_columns].values.astype(np.float32)
                node_features.append(features)
                
                # Get 24-hour target (24 GHI values)
                target = city_row[target_columns].values.astype(np.float32)
                # Replace NaN with 0
                target = np.nan_to_num(target, nan=0.0)
                city_targets.append(target)
        
        # Create graph if we have any valid data
        if len(node_features) > 0:
            x = np.array(node_features, dtype=np.float32)
            a = adj_matrix.astype(np.float32)
            
            # Create 24-hour targets for all cities (5 cities × 24 hours = 120 values)
            all_targets = np.concatenate(city_targets, axis=0)  # Flatten all city targets
            
            # Debug: Print target info for first few graphs
            if i < 3:
                print(f"    Date {date}:")
                print(f"      Target shape: {all_targets.shape}")
                print(f"      Target range: [{all_targets.min():.2f}, {all_targets.max():.2f}]")
                print(f"      Sample targets (first 10): {all_targets[:10]}")
            
            # Create Spektral Graph
            graph = Graph(x=x, a=a, y=all_targets)
            graphs.append(graph)
            targets.append(all_targets)
            graph_dates.append(date)
            graph_cities.append(actual_cities)
    
    print(f"Created {len(graphs)} daily graphs (limited to {max_graphs})")
    return graphs, targets, ghi_scaler, target_scaler, graph_dates, graph_cities

class GHIDataset(Dataset):
    def __init__(self, graphs, y, **kwargs):
        self.graphs = graphs
        self.y = y
        super().__init__(**kwargs)

    def read(self):
        return self.graphs

def build_gnn_model(input_shape, output_units=120):  # 5 cities × 24 hours = 120
    """Build a state-of-the-art GNN model with attention and multi-task learning."""
    print("Building state-of-the-art GNN model with attention mechanisms...")
    
    # Input layers
    x_in = layers.Input(shape=input_shape, name='x_in')
    a_in = layers.Input(shape=(None,), sparse=True, name='a_in')
    i_in = layers.Input(shape=(), dtype=tf.int64, name='i_in')
    
    # Enhanced GNN layers with attention mechanisms
    x = GCNConv(512, activation='relu')([x_in, a_in])
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Residual connection 1
    x_res1 = x
    
    x = GCNConv(512, activation='relu')([x, a_in])
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Add()([x, x_res1])  # Residual connection
    
    # Attention mechanism for node importance
    def attention_layer(inputs):
        node_features, adj_matrix = inputs
        # Compute attention weights
        attention_weights = tf.matmul(node_features, tf.transpose(node_features))
        attention_weights = tf.nn.softmax(attention_weights / tf.math.sqrt(tf.cast(tf.shape(node_features)[-1], tf.float32)))
        # Apply attention
        attended_features = tf.matmul(attention_weights, node_features)
        return attended_features
    
    # Apply attention
    x = layers.Lambda(attention_layer)([x, a_in])
    
    x = GCNConv(256, activation='relu')([x, a_in])
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Residual connection 2
    x_res2 = x
    
    x = GCNConv(256, activation='relu')([x, a_in])
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Add()([x, x_res2])  # Residual connection
    
    x = GCNConv(128, activation='relu')([x, a_in])
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = GCNConv(128, activation='relu')([x, a_in])
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = GCNConv(64, activation='relu')([x, a_in])
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
    
    # Multi-task learning: Separate heads for each city
    city_heads = []
    for city_idx in range(5):  # 5 cities
        # City-specific dense layers
        city_x = layers.Dense(1024, activation='relu')(x)
        city_x = layers.BatchNormalization()(city_x)
        city_x = layers.Dropout(0.3)(city_x)
        
        city_x = layers.Dense(512, activation='relu')(city_x)
        city_x = layers.BatchNormalization()(city_x)
        city_x = layers.Dropout(0.2)(city_x)
        
        city_x = layers.Dense(256, activation='relu')(city_x)
        city_x = layers.BatchNormalization()(city_x)
        city_x = layers.Dropout(0.2)(city_x)
        
        # 24-hour predictions for this city
        city_output = layers.Dense(24, activation='relu', name=f'city_{city_idx}')(city_x)
        city_heads.append(city_output)
    
    # Concatenate all city predictions
    outputs = layers.Concatenate(name='output')(city_heads)

    model = models.Model(inputs=[x_in, a_in, i_in], outputs=outputs)
    return model

# ----------------------------------------------
# Evaluation functions
# ----------------------------------------------
def evaluate_gnn_model(model, dataset, ghi_scaler, target_scaler, graph_dates, graph_cities, actual_cities, target_columns):
    """Evaluate the GNN model and generate metrics for each city."""
    print("\nEvaluating GNN model for 24-hour forecasting...")
    
    # Create results directory
    os.makedirs("results_gnn", exist_ok=True)
    
    # Store predictions and actuals with their corresponding cities
    city_predictions = {city: [] for city in actual_cities}
    city_actuals = {city: [] for city in actual_cities}
    city_dates = {city: [] for city in actual_cities}
    
    # Use a smaller batch size and limit the number of predictions
    loader = DisjointLoader(dataset, batch_size=1, shuffle=False)
    max_eval_batches = min(100, len(dataset))  # Increased for better evaluation
    
    batch_count = 0
    for batch in loader:
        if batch_count >= max_eval_batches:
            break
            
        batch_count += 1
        print(f"  Processing evaluation batch {batch_count}/{max_eval_batches}...")
        
        try:
            inputs, y_true = batch
            x, a, i = inputs
            y_pred = model([x, a, i], training=False)
            
            # Convert to numpy arrays
            if hasattr(y_true, 'numpy'):
                y_true = y_true.numpy()
            if hasattr(y_pred, 'numpy'):
                y_pred = y_pred.numpy()
            
            # Reshape predictions and targets (120 values = 5 cities × 24 hours)
            y_true_reshaped = y_true.reshape(-1, len(target_columns), len(actual_cities))  # (batch, 24, 5)
            y_pred_reshaped = y_pred.reshape(-1, len(target_columns), len(actual_cities))  # (batch, 24, 5)
            
            # Inverse transform using target scaler
            y_true_original = target_scaler.inverse_transform(y_true_reshaped.reshape(-1, len(target_columns))).reshape(y_true_reshaped.shape)
            y_pred_original = target_scaler.inverse_transform(y_pred_reshaped.reshape(-1, len(target_columns))).reshape(y_pred_reshaped.shape)
            
            # Debug: Print first few predictions
            if batch_count <= 3:
                print(f"    Batch {batch_count}: y_true_shape={y_true_original.shape}, y_pred_shape={y_pred_original.shape}")
                print(f"    Batch {batch_count}: y_true_range=[{y_true_original.min():.2f}, {y_true_original.max():.2f}]")
                print(f"    Batch {batch_count}: y_pred_range=[{y_pred_original.min():.2f}, {y_pred_original.max():.2f}]")
            
            # Assign predictions to each city
            for city_idx, city in enumerate(actual_cities):
                city_pred = y_pred_original[0, :, city_idx]  # 24 hours for this city
                city_actual = y_true_original[0, :, city_idx]  # 24 hours for this city
                
                city_predictions[city].extend(city_pred)
                city_actuals[city].extend(city_actual)
                
                # Create dates for each hour
                base_date = pd.Timestamp('2020-01-01') + pd.Timedelta(days=batch_count-1)
                for hour in range(24):
                    city_dates[city].append(base_date + pd.Timedelta(hours=hour))
            
        except Exception as e:
            print(f"  Error in evaluation batch {batch_count}: {e}")
            continue
    
    print(f"Evaluation completed. Processed {batch_count} batches.")
    
    # Calculate city-specific metrics
    all_metrics = {}
    overall_predictions = []
    overall_actuals = []
    
    for city in actual_cities:
        predictions = np.array(city_predictions[city])
        actuals = np.array(city_actuals[city])
        dates = city_dates[city]
        
        # Filter out invalid values
        valid_mask = ~(np.isnan(predictions) | np.isnan(actuals) | 
                      np.isinf(predictions) | np.isinf(actuals))
        
        if np.sum(valid_mask) > 5:  # Need at least 5 valid predictions per city
            predictions = predictions[valid_mask]
            actuals = actuals[valid_mask]
            dates = [dates[i] for i in range(len(dates)) if valid_mask[i]]
            
            # Calculate city-specific metrics
            city_mae = mean_absolute_error(actuals, predictions)
            city_rmse = np.sqrt(mean_squared_error(actuals, predictions))
            city_r2 = r2_score(actuals, predictions)
            city_correlation = np.corrcoef(actuals, predictions)[0, 1] if len(actuals) > 1 else 0
            
            print(f"\n{city} Performance:")
            print(f"  Valid predictions: {len(predictions)}")
            print(f"  Prediction range: [{np.min(predictions):.2f}, {np.max(predictions):.2f}]")
            print(f"  Actual range: [{np.min(actuals):.2f}, {np.max(actuals):.2f}]")
            print(f"  MAE: {city_mae:.2f}")
            print(f"  RMSE: {city_rmse:.2f}")
            print(f"  R²: {city_r2:.4f}")
            print(f"  Correlation: {city_correlation:.4f}")
            
            # Store metrics
            all_metrics[city] = {
                'mae': city_mae,
                'rmse': city_rmse,
                'r2': city_r2,
                'correlation': city_correlation,
                'actual': actuals,
                'predicted': predictions,
                'dates': dates
            }
            
            # Add to overall metrics
            overall_predictions.extend(predictions)
            overall_actuals.extend(actuals)
            
        else:
            print(f"\n{city}: Not enough valid predictions ({np.sum(valid_mask)})")
            # Create dummy metrics for this city
            all_metrics[city] = {
                'mae': 100.0,
                'rmse': 150.0,
                'r2': 0.0,
                'correlation': 0.0,
                'actual': np.array([0.0]),
                'predicted': np.array([0.0]),
                'dates': [pd.Timestamp('2020-01-01')]
            }
    
    # Calculate overall metrics
    if len(overall_predictions) > 0:
        overall_mae = mean_absolute_error(overall_actuals, overall_predictions)
        overall_rmse = np.sqrt(mean_squared_error(overall_actuals, overall_predictions))
        overall_r2 = r2_score(overall_actuals, overall_predictions)
        overall_correlation = np.corrcoef(overall_actuals, overall_predictions)[0, 1]
        
        print(f"\nOverall GNN Performance:")
        print(f"  Total valid predictions: {len(overall_predictions)}")
        print(f"  Prediction range: [{np.min(overall_predictions):.2f}, {np.max(overall_predictions):.2f}]")
        print(f"  Actual range: [{np.min(overall_actuals):.2f}, {np.max(overall_actuals):.2f}]")
        print(f"  MAE: {overall_mae:.2f}")
        print(f"  RMSE: {overall_rmse:.2f}")
        print(f"  R²: {overall_r2:.4f}")
        print(f"  Correlation: {overall_correlation:.4f}")
    
    # Create plots for each city
    for city in actual_cities:
        plot_gnn_results(all_metrics[city], city)
    
    # Create summary table
    create_gnn_summary_table(all_metrics)
    
    return all_metrics

def plot_gnn_results(results, city):
    """Create plots for GNN results."""
    try:
        city_dir = f"results_gnn/{city}"
        os.makedirs(city_dir, exist_ok=True)
        
        # Simple scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(results['actual'], results['predicted'], alpha=0.6)
        plt.plot([0, max(results['actual'])], [0, max(results['actual'])], 'r--', lw=2)
        plt.xlabel('Actual GHI')
        plt.ylabel('Predicted GHI')
        plt.title(f'GNN 24h Predictions vs Actual - {city}')
        plt.savefig(f"{city_dir}/scatter_plot.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Simple time series plot
        plt.figure(figsize=(12, 6))
        plt.plot(results['dates'], results['actual'], label='Actual', alpha=0.7)
        plt.plot(results['dates'], results['predicted'], label='Predicted', alpha=0.7)
        plt.xlabel('Date')
        plt.ylabel('GHI')
        plt.title(f'GNN 24h Time Series - {city}')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{city_dir}/time_series.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Daily metrics plot
        daily_metrics = calculate_daily_metrics_gnn(results['dates'], results['actual'], results['predicted'])
        plt.figure(figsize=(12, 6))
        plt.plot(daily_metrics['date'], daily_metrics['correlation'], 'b-', label='Daily Correlation')
        plt.xlabel('Date')
        plt.ylabel('Correlation')
        plt.title(f'Daily Correlation - {city}')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{city_dir}/daily_metrics.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save daily metrics to CSV
        daily_metrics.to_csv(f"{city_dir}/daily_metrics.csv", index=False)
        
        print(f"  ✓ Plots created for {city}")
        
    except Exception as e:
        print(f"  Error creating plots for {city}: {e}")

def calculate_daily_metrics_gnn(dates, actual, predicted):
    """Calculate daily metrics for GNN results with 24-hour forecasting."""
    # For GNN with 24-hour forecasting, we now have multiple predictions per day
    # Group by date and calculate daily metrics
    df = pd.DataFrame({
        'date': pd.to_datetime(dates).date,
        'actual': actual,
        'predicted': predicted
    })
    
    # Group by date and calculate daily metrics
    daily_stats = []
    for date, group in df.groupby('date'):
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
        target_columns = None
        for i, city in enumerate(CITIES):
            print(f"Loading data for city {i+1}/{len(CITIES)}: {city}")
            df = load_data(CONFIG["data_locations"], city)
            print(f"  Raw data shape: {df.shape}")
            
            print(f"  Creating features for {city}...")
            df, city_target_columns = create_features(df)
            if target_columns is None:
                target_columns = city_target_columns
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

        print("\nBuilding daily graphs for 24-hour forecasting...")
        graphs, targets, ghi_scaler, target_scaler, graph_dates, graph_cities = build_daily_graphs(df_all, adj_matrix, actual_cities, target_columns)
        
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
            print(f"First graph y length: {len(first_graph.y)}")
        
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
            print(f"Output units: {len(target_columns) * len(actual_cities)}")  # 24 hours × 5 cities
        else:
            print("ERROR: No graphs in dataset!")
            return
        
        model = build_gnn_model(input_shape=(num_features,), output_units=len(target_columns) * len(actual_cities))
        model.summary()
        
        # Compile model with aggressive learning rate, gradient clipping, and custom loss
        optimizer = tf.keras.optimizers.Adam(0.001, clipnorm=1.0)  # Gradient clipping
        
        # Multi-task loss function with city-specific weighting
        def multi_task_loss(y_true, y_pred):
            # Reshape to separate cities
            y_true_reshaped = tf.reshape(y_true, [-1, 5, 24])  # [batch, cities, hours]
            y_pred_reshaped = tf.reshape(y_pred, [-1, 5, 24])  # [batch, cities, hours]
            
            # City-specific losses
            city_losses = []
            for city_idx in range(5):
                city_true = y_true_reshaped[:, city_idx, :]  # [batch, 24]
                city_pred = y_pred_reshaped[:, city_idx, :]  # [batch, 24]
                
                # Huber loss for this city
                city_loss = tf.keras.losses.Huber(delta=1.0)(city_true, city_pred)
                city_losses.append(city_loss)
            
            # Weighted combination of city losses
            city_weights = [1.0, 1.0, 1.0, 1.0, 1.0]  # Equal weights for now
            weighted_loss = tf.reduce_sum([w * l for w, l in zip(city_weights, city_losses)])
            
            # Add L2 regularization
            l2_penalty = 0.001 * tf.reduce_sum([tf.nn.l2_loss(w) for w in model.trainable_variables])
            
            return weighted_loss + l2_penalty
        
        model.compile(optimizer=optimizer, loss=multi_task_loss, metrics=['mae'])
        
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
            print(f"Dummy output values: {dummy_output.numpy()[:5]}")  # Show first 5 values
            print("Model compilation successful!")
        except Exception as e:
            print(f"Model compilation failed: {e}")
            return

        print("\nTraining model with early stopping...")
        
        # Training parameters - EXTREME AGGRESSIVE for maximum performance
        batch_size = 64  # Even larger batch size for better gradient estimates
        epochs = 300  # Much more epochs for better convergence
        patience = 35  # Much more patience for early stopping
        
        # Calculate steps per epoch
        total_graphs = len(dataset)
        steps_per_epoch = max(1, total_graphs // batch_size)
        max_steps_per_epoch = min(steps_per_epoch, 150)  # More steps per epoch
        
        print(f"Total graphs: {total_graphs}")
        print(f"Batch size: {batch_size}")
        print(f"Steps per epoch: {max_steps_per_epoch}")
        print(f"Max epochs: {epochs}")
        print(f"Early stopping patience: {patience}")
        
        # Training with early stopping and learning rate scheduling
        best_loss = float('inf')
        patience_counter = 0
        initial_lr = 0.001  # Start with higher learning rate
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            epoch_start_time = time.time()
            
            # Advanced learning rate scheduling with warmup and cosine annealing
            if epoch < 30:
                # Warmup phase
                current_lr = initial_lr * (epoch + 1) / 30
            elif epoch < 100:
                current_lr = initial_lr
            elif epoch < 200:
                current_lr = initial_lr * 0.5
            elif epoch < 250:
                current_lr = initial_lr * 0.1
            else:
                # Cosine annealing for final epochs
                progress = (epoch - 250) / (epochs - 250)
                current_lr = initial_lr * 0.05 * (1 + np.cos(np.pi * progress)) / 2
            
            # Update learning rate
            tf.keras.backend.set_value(model.optimizer.learning_rate, current_lr)
            
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
                        print(f"      Training batch {batch_idx}: y_true_shape={y_true.shape}, y_pred_shape={y_pred.shape}")
                        print(f"      Training batch {batch_idx}: y_true_range=[{y_true.numpy().min():.2f}, {y_true.numpy().max():.2f}]")
                        print(f"      Training batch {batch_idx}: y_pred_range=[{y_pred.numpy().min():.2f}, {y_pred.numpy().max():.2f}]")
                    
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
                
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - MAE: {epoch_mae:.4f} - LR: {current_lr:.6f} - Time: {epoch_time:.1f}s")
                
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

        print("\nSaving model and scalers...")
        model.save("models/gnn_ghi_forecast_24h_fixed.h5")
        with open("models/ghi_scaler_gnn_24h_fixed.pkl", "wb") as f:
            pickle.dump(ghi_scaler, f)
        with open("models/target_scaler_gnn_24h_fixed.pkl", "wb") as f:
            pickle.dump(target_scaler, f)
        print("✓ Model and scalers saved.")

        # Evaluate the model
        print("\nEvaluating model...")
        all_metrics = evaluate_gnn_model(model, dataset, ghi_scaler, target_scaler, graph_dates, graph_cities, actual_cities, target_columns)
        
        total_time = time.time() - start_time
        print(f"\nGNN 24-hour training and evaluation completed in {total_time:.1f} seconds!")

    except Exception as e:
        print(f"ERROR: Training failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main() 