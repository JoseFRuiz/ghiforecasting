import os
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from utils import CONFIG, load_data

def create_features(df):
    """Create features for prediction."""
    # Create time features
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    
    # Create GHI lag features
    for lag in range(1, 25):
        df[f"GHI_lag_{lag}"] = df["GHI"].shift(lag)
    
    # Create meteorological lag features
    met_features = [
        "Temperature", "Relative Humidity", "Pressure",
        "Precipitable Water", "Wind Direction", "Wind Speed"
    ]
    for feature in met_features:
        df[f"{feature}_lag_24"] = df[feature].shift(24)
    
    return df.dropna().reset_index(drop=True)

def create_sequences_joint(df, sequence_length, input_config):
    """Create sequences for joint models based on input configuration."""
    # Initialize feature columns based on configuration
    feature_columns = []
    
    if input_config in ['ghi_only', 'ghi_met']:
        feature_columns.extend([f"GHI_lag_{lag}" for lag in range(1, 25)])
        feature_columns.append("GHI")
    
    if input_config in ['met_only', 'ghi_met']:
        met_features = [
            "Temperature_lag_24", "Relative Humidity_lag_24",
            "Pressure_lag_24", "Precipitable Water_lag_24",
            "Wind Direction_lag_24", "Wind Speed_lag_24"
        ]
        feature_columns.extend(met_features)
    
    # Always add time features
    time_features = ["hour_sin", "hour_cos", "month_sin", "month_cos"]
    feature_columns.extend(time_features)
    
    # Get data for selected features
    data = df[feature_columns].values
    target = df["GHI"].values
    
    # Create sequences
    n_sequences = len(df) - sequence_length
    X = np.zeros((n_sequences, sequence_length, len(feature_columns)))
    y = np.zeros(n_sequences)
    
    for i in range(n_sequences):
        X[i] = data[i:i + sequence_length]
        y[i] = target[i + sequence_length]
    
    # Only keep nonzero targets
    mask = y > 0
    X = X[mask]
    y = y[mask]
    
    # Note: Joint models were trained WITHOUT location features
    # Location features are only used in the 'all' configuration, but our models use ghi_only, ghi_met, met_only
    
    return X, y, df['datetime'].values[sequence_length:][mask]

def create_sequences_individual(df, sequence_length):
    """Create sequences for individual models."""
    # Get all required features (same as in train_individual.py)
    feature_columns = (
        [f"GHI_lag_{lag}" for lag in range(1, 25)] +  # GHI lag features
        ["GHI"] +  # Current GHI
        [  # Meteorological features
            "Temperature_lag_24", "Relative Humidity_lag_24",
            "Pressure_lag_24", "Precipitable Water_lag_24",
            "Wind Direction_lag_24", "Wind Speed_lag_24"
        ] +
        [  # Time features
            "hour_sin", "hour_cos",
            "month_sin", "month_cos"
        ]
    )
    
    # Get all data at once
    data = df[feature_columns].values
    target = df["GHI"].values
    
    # Calculate number of sequences
    n_sequences = len(df) - sequence_length
    
    # Pre-allocate arrays
    X = np.zeros((n_sequences, sequence_length, len(feature_columns)))
    y = np.zeros(n_sequences)
    
    # Create sequences using vectorized operations
    for i in range(n_sequences):
        X[i] = data[i:i + sequence_length]
        y[i] = target[i + sequence_length]
    
    # Filter out sequences where target is zero (night time)
    mask = y > 0
    X = X[mask]
    y = y[mask]
    
    return X, y, df['datetime'].values[sequence_length:][mask]

def split_data_by_days(df, train_ratio=0.7, val_ratio=0.15):
    """Split data by complete days to ensure full days in each split."""
    # Get unique dates
    dates = pd.to_datetime(df['datetime']).dt.date.unique()
    dates = sorted(dates)
    
    # Calculate split indices
    n_dates = len(dates)
    train_end = int(n_dates * train_ratio)
    val_end = int(n_dates * (train_ratio + val_ratio))
    
    # Split dates
    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]
    
    # Split data
    df_train = df[pd.to_datetime(df['datetime']).dt.date.isin(train_dates)]
    df_val = df[pd.to_datetime(df['datetime']).dt.date.isin(val_dates)]
    df_test = df[pd.to_datetime(df['datetime']).dt.date.isin(test_dates)]
    
    return df_train, df_val, df_test

def calculate_daily_metrics(dates, actual, predicted):
    """Calculate daily metrics for correlation analysis with additional validation."""
    # Convert dates to pandas Series for proper date handling
    dates = pd.to_datetime(dates)
    
    # Ensure arrays are 1-dimensional
    actual = actual.flatten()
    predicted = predicted.flatten()
    
    # Validate data ranges
    print("\nData validation:")
    print(f"Actual GHI range: [{actual.min():.2f}, {actual.max():.2f}] W/m²")
    print(f"Predicted GHI range: [{predicted.min():.2f}, {predicted.max():.2f}] W/m²")
    
    # Check for unrealistic values
    if predicted.max() > 1200:
        print(f"Warning: Unrealistic predictions found (>1200 W/m²): {predicted.max():.2f} W/m²")
    if predicted.min() < 0:
        print(f"Warning: Negative predictions found: {predicted.min():.2f} W/m²")
    
    df = pd.DataFrame({
        'date': dates,
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
            
            # Calculate metrics
            correlation = actual_nonzero.corr(predicted_nonzero)
            mae = mean_absolute_error(actual_nonzero, predicted_nonzero)
            rmse = np.sqrt(mean_squared_error(actual_nonzero, predicted_nonzero))
            
            # Calculate R² properly
            ss_res = np.sum((actual_nonzero - predicted_nonzero) ** 2)
            ss_tot = np.sum((actual_nonzero - np.mean(actual_nonzero)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Ensure R² is between 0 and 1
            r2 = max(0, min(1, r2))
            
            # Calculate additional metrics
            mape = np.mean(np.abs((actual_nonzero - predicted_nonzero) / actual_nonzero)) * 100
            bias = np.mean(predicted_nonzero - actual_nonzero)
            
            daily_stats.append({
                'date': date,
                'correlation': correlation,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'bias': bias,
                'num_points': len(actual_nonzero),
                'actual_max': actual_nonzero.max(),
                'actual_mean': actual_nonzero.mean(),
                'actual_std': actual_nonzero.std(),
                'predicted_max': predicted_nonzero.max(),
                'predicted_mean': predicted_nonzero.mean(),
                'predicted_std': predicted_nonzero.std()
            })
    
    metrics_df = pd.DataFrame(daily_stats).sort_values('date')
    
    # Print overall statistics
    print("\nOverall metrics:")
    print(f"Mean MAE: {metrics_df['mae'].mean():.2f} W/m²")
    print(f"Mean RMSE: {metrics_df['rmse'].mean():.2f} W/m²")
    print(f"Mean R²: {metrics_df['r2'].mean():.4f}")
    print(f"Mean MAPE: {metrics_df['mape'].mean():.2f}%")
    print(f"Mean Bias: {metrics_df['bias'].mean():.2f} W/m²")
    
    return metrics_df

def plot_daily_correlations(daily_metrics, city, model_type, config, save_dir):
    """Create plots showing daily correlation metrics with additional validation."""
    # Create figure with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
    
    # Plot correlation and R²
    ax1.plot(daily_metrics['date'], daily_metrics['correlation'], 'b-', label='Correlation')
    ax1.plot(daily_metrics['date'], daily_metrics['r2'], 'r-', label='R²')
    ax1.set_title(f'Daily Correlation and R² - {city} ({model_type}, {config})')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot MAE and RMSE
    ax2.plot(daily_metrics['date'], daily_metrics['mae'], 'g-', label='MAE')
    ax2.plot(daily_metrics['date'], daily_metrics['rmse'], 'm-', label='RMSE')
    ax2.set_title(f'Daily MAE and RMSE - {city} ({model_type}, {config})')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Error (W/m²)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot MAPE and Bias
    ax3.plot(daily_metrics['date'], daily_metrics['mape'], 'c-', label='MAPE')
    ax3.plot(daily_metrics['date'], daily_metrics['bias'], 'y-', label='Bias')
    ax3.set_title(f'Daily MAPE and Bias - {city} ({model_type}, {config})')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot GHI statistics
    ax4.plot(daily_metrics['date'], daily_metrics['actual_max'], 'b-', label='Actual Max')
    ax4.plot(daily_metrics['date'], daily_metrics['predicted_max'], 'r--', label='Predicted Max')
    ax4.plot(daily_metrics['date'], daily_metrics['actual_mean'], 'g-', label='Actual Mean')
    ax4.plot(daily_metrics['date'], daily_metrics['predicted_mean'], 'm--', label='Predicted Mean')
    ax4.set_title(f'Daily GHI Statistics - {city} ({model_type}, {config})')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('GHI (W/m²)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    for ax in [ax1, ax2, ax3, ax4]:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'daily_correlations_{city}_{model_type}_{config}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_analysis(daily_metrics, dates, actual, predicted, city, model_type, config, save_dir):
    """Create correlation analysis plots."""
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Day with highest correlation
    best_day_idx = daily_metrics['correlation'].idxmax()
    best_day = daily_metrics.loc[best_day_idx]
    
    # Get the actual data for this day
    mask = pd.to_datetime(dates).date == best_day['date']
    actual_data = actual[mask]
    predicted_data = predicted[mask]
    
    if len(actual_data) > 0:
        ax1.scatter(actual_data, predicted_data, alpha=0.5)
        max_val = max(actual_data.max(), predicted_data.max())
        ax1.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
        ax1.set_title(f'Best Day (Correlation = {best_day["correlation"]:.3f})')
        ax1.set_xlabel('Actual GHI')
        ax1.set_ylabel('Predicted GHI')
        ax1.legend()
    
    # Plot 2: Day with lowest correlation
    worst_day_idx = daily_metrics['correlation'].idxmin()
    worst_day = daily_metrics.loc[worst_day_idx]
    
    # Get the actual data for this day
    mask = pd.to_datetime(dates).date == worst_day['date']
    actual_data = actual[mask]
    predicted_data = predicted[mask]
    
    if len(actual_data) > 0:
        ax2.scatter(actual_data, predicted_data, alpha=0.5)
        max_val = max(actual_data.max(), predicted_data.max())
        ax2.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
        ax2.set_title(f'Worst Day (Correlation = {worst_day["correlation"]:.3f})')
        ax2.set_xlabel('Actual GHI')
        ax2.set_ylabel('Predicted GHI')
        ax2.legend()
    
    plt.suptitle(f'Correlation Analysis for {city} ({model_type}, {config})\nR² = {daily_metrics["r2"].mean():.3f}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'correlation_analysis_{city}_{model_type}_{config}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(all_metrics):
    """Create a summary table of performance metrics across all locations and configurations."""
    summary_data = []
    
    for model_type, config_metrics in all_metrics.items():
        if model_type == 'joint':
            # Joint models have configurations (now just 'joint_fixed')
            for config, city_metrics in config_metrics.items():
                for city, metrics_df in city_metrics.items():
                    summary_data.append({
                        'Model Type': model_type,
                        'Configuration': config,
                        'Location': city,
                        'MAE (W/m²)': metrics_df['mae'].mean(),
                        'RMSE (W/m²)': metrics_df['rmse'].mean(),
                        'R²': metrics_df['r2'].mean(),
                        'MAPE (%)': metrics_df['mape'].mean(),
                        'Bias (W/m²)': metrics_df['bias'].mean(),
                        'Correlation': metrics_df['correlation'].mean(),
                        'Num Days': len(metrics_df),
                        'Actual Mean (W/m²)': metrics_df['actual_mean'].mean(),
                        'Predicted Mean (W/m²)': metrics_df['predicted_mean'].mean()
                    })
        else:
            # Individual models don't have configurations
            for city, metrics_df in config_metrics.items():
                summary_data.append({
                    'Model Type': model_type,
                    'Configuration': 'standard',
                    'Location': city,
                    'MAE (W/m²)': metrics_df['mae'].mean(),
                    'RMSE (W/m²)': metrics_df['rmse'].mean(),
                    'R²': metrics_df['r2'].mean(),
                    'MAPE (%)': metrics_df['mape'].mean(),
                    'Bias (W/m²)': metrics_df['bias'].mean(),
                    'Correlation': metrics_df['correlation'].mean(),
                    'Num Days': len(metrics_df),
                    'Actual Mean (W/m²)': metrics_df['actual_mean'].mean(),
                    'Predicted Mean (W/m²)': metrics_df['predicted_mean'].mean()
                })
    
    # Create DataFrame and sort by model type, configuration and location
    summary_df = pd.DataFrame(summary_data)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(['Model Type', 'Configuration', 'Location'])
        
        # Round numeric columns
        numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
        summary_df[numeric_cols] = summary_df[numeric_cols].round(4)
    
    return summary_df

def create_sequences_joint_all_features(df, sequence_length, locations):
    """
    Create sequences for joint model evaluation that matches train_joint_fixed.py.
    
    Args:
        df: DataFrame with data for a single location
        sequence_length: Length of input sequences
        locations: List of all locations (for one-hot encoding)
    
    Returns:
        tuple: (X, y, dates) where X is the input sequences and y is the target values
    """
    print(f"\nCreating sequences for joint model evaluation")
    print(f"Data shape: {df.shape}")
    print(f"Location: {df['location'].iloc[0]}")
    
    # Get all required features (same as in train_joint_fixed.py)
    feature_columns = (
        [f"GHI_lag_{lag}" for lag in range(1, 25)] +  # GHI lag features
        ["GHI"] +  # Current GHI
        [  # Meteorological features
            "Temperature_lag_24", "Relative Humidity_lag_24",
            "Pressure_lag_24", "Precipitable Water_lag_24",
            "Wind Direction_lag_24", "Wind Speed_lag_24"
        ] +
        [  # Time features
            "hour_sin", "hour_cos",
            "month_sin", "month_cos"
        ]
    )
    
    # Get all data at once
    data = df[feature_columns].values
    target = df["GHI"].values
    
    # Calculate number of sequences
    n_sequences = len(df) - sequence_length
    
    # Pre-allocate arrays
    X = np.zeros((n_sequences, sequence_length, len(feature_columns)))
    y = np.zeros(n_sequences)
    
    # Create sequences using vectorized operations
    for i in range(n_sequences):
        X[i] = data[i:i + sequence_length]
        y[i] = target[i + sequence_length]
    
    # Filter out sequences where target is zero (night time)
    mask = y > 0
    X = X[mask]
    y = y[mask]
    
    # Add location features (one-hot encoding)
    current_location = df['location'].iloc[0]
    location_vector = np.zeros(len(locations))
    location_vector[locations.index(current_location)] = 1
    
    # Add location features to each sequence
    location_features = np.tile(location_vector, (len(X), sequence_length, 1))
    X = np.concatenate([X, location_features], axis=2)
    
    print(f"Created {len(X)} valid sequences")
    print(f"Final X shape: {X.shape}")
    print(f"Final y shape: {y.shape}")
    
    return X, y, df['datetime'].values[sequence_length:][mask]

def evaluate_joint_models():
    """Evaluate joint models from train_joint_fixed.py with FIXED scaling."""
    print("\n" + "="*80)
    print("EVALUATING JOINT MODELS (FIXED SCALING)")
    print("="*80)
    
    # Create output directories
    base_dir = "results_joint_fixed"
    os.makedirs(base_dir, exist_ok=True)
    
    # Load target scaler
    scaler_path = os.path.join("models", "target_scaler_fixed.pkl")
    if not os.path.exists(scaler_path):
        print(f"Target scaler not found: {scaler_path}")
        return {}
    
    with open(scaler_path, 'rb') as f:
        target_scaler = pickle.load(f)
    
    print(f"Joint scaler info:")
    print(f"  data_min: {target_scaler.data_min_[0]:.2f}")
    print(f"  data_max: {target_scaler.data_max_[0]:.2f}")
    print(f"  scale_: {target_scaler.scale_[0]:.4f}")
    print(f"  min_: {target_scaler.min_[0]:.4f}")
    
    # Dictionary to store all metrics
    all_metrics = {}
    
    # Load the single joint model (train_joint_fixed.py saves only one model)
    model_path = os.path.join("models", "lstm_ghi_forecast_joint_fixed.h5")
    if not os.path.exists(model_path):
        print(f"Joint model not found: {model_path}")
        print("Please run train_joint_fixed.py first to train the joint model.")
        return {}
    
    print(f"\nLoading joint model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Initialize dictionary for the single joint model
    all_metrics['joint_fixed'] = {}
    
    # Get all locations for one-hot encoding
    all_locations = list(CONFIG["data_locations"].keys())
    
    # Process each city
    for city in CONFIG["data_locations"].keys():
        print(f"\n{'='*60}")
        print(f"Processing {city} with joint model")
        print(f"{'='*60}")
        
        # Load and prepare data
        df = load_data(CONFIG["data_locations"], city)
        df = create_features(df)
        
        # Add location column for joint models
        df['location'] = city
        
        # Split data by days
        df_train, df_val, df_test = split_data_by_days(df)
        
        # Create sequences for joint model (uses all features like in train_joint_fixed.py)
        X, y, dates = create_sequences_joint_all_features(df_test, sequence_length=24, locations=all_locations)
        
        if len(X) == 0:
            print(f"No valid sequences for {city}")
            continue
        
        # Make predictions
        y_pred_scaled = model.predict(X)
        
        # IMPORTANT FIX: Use the same scaling approach for both actual and predicted
        # Scale the actual values using the same global scaler
        y_scaled = target_scaler.transform(y.reshape(-1, 1))
        
        # Inverse transform both using the same scaler
        y_pred = target_scaler.inverse_transform(y_pred_scaled)
        y_true = target_scaler.inverse_transform(y_scaled)
        
        # Calculate daily metrics
        daily_metrics = calculate_daily_metrics(dates, y_true, y_pred)
        
        # Store metrics for summary
        all_metrics['joint_fixed'][city] = daily_metrics
        
        # Save daily metrics
        daily_metrics.to_csv(os.path.join(base_dir, f'daily_correlations_{city}.csv'), index=False)
        
        # Create plots
        plot_daily_correlations(daily_metrics, city, "joint", "fixed", base_dir)
        plot_correlation_analysis(daily_metrics, dates, y_true, y_pred, city, "joint", "fixed", base_dir)
        
        print(f"Completed analysis for {city}")
    
    return all_metrics

def evaluate_individual_models():
    """Evaluate individual models from train_individual.py with FIXED scaling."""
    print("\n" + "="*80)
    print("EVALUATING INDIVIDUAL MODELS (FIXED SCALING)")
    print("="*80)
    
    # Create output directories
    base_dir = "results_individual_fixed"
    os.makedirs(base_dir, exist_ok=True)
    
    # Load the same global target scaler used by joint models
    scaler_path = os.path.join("models", "target_scaler_fixed.pkl")
    if not os.path.exists(scaler_path):
        print(f"Target scaler not found: {scaler_path}")
        print("Please run train_joint_fixed.py first to create the target scaler.")
        return {}
    
    with open(scaler_path, 'rb') as f:
        target_scaler = pickle.load(f)
    
    print(f"Individual models using global scaler info:")
    print(f"  data_min: {target_scaler.data_min_[0]:.2f}")
    print(f"  data_max: {target_scaler.data_max_[0]:.2f}")
    print(f"  scale_: {target_scaler.scale_[0]:.4f}")
    print(f"  min_: {target_scaler.min_[0]:.4f}")
    
    # Dictionary to store all metrics
    all_metrics = {}
    
    # Define custom loss function (same as in train_individual.py)
    def custom_ghi_loss(y_true, y_pred):
        """Custom loss function that penalizes unrealistic GHI predictions."""
        # Standard Huber loss
        huber_loss = tf.keras.losses.Huber()(y_true, y_pred)
        
        # Penalty for predictions outside [0, 1] range (since we use sigmoid)
        range_penalty = tf.reduce_mean(tf.maximum(0.0, y_pred - 1.0) + tf.maximum(0.0, -y_pred))
        
        return huber_loss + range_penalty
    
    # Process each city
    for city in CONFIG["data_locations"].keys():
        print(f"\n{'='*60}")
        print(f"Processing individual model for {city}")
        print(f"{'='*60}")
        
        # Load model
        model_path = os.path.join("models", f"lstm_ghi_forecast_{city}.h5")
        if not os.path.exists(model_path):
            print(f"Individual model not found: {model_path}")
            print(f"Please run train_individual.py first to train the model for {city}.")
            continue
        
        try:
            model = tf.keras.models.load_model(model_path, custom_objects={'custom_ghi_loss': custom_ghi_loss})
            print(f"✓ Successfully loaded model for {city}")
        except Exception as e:
            print(f"× Error loading model for {city}: {str(e)}")
            continue
        
        # Load and prepare data
        df = load_data(CONFIG["data_locations"], city)
        df = create_features(df)
        
        # Split data by days
        df_train, df_val, df_test = split_data_by_days(df)
        
        # Create sequences
        X, y, dates = create_sequences_individual(df_test, sequence_length=24)
        
        if len(X) == 0:
            print(f"No valid sequences for {city}")
            continue
        
        # IMPORTANT FIX: Use the same global scaler as joint models
        # Scale the actual values using the same global scaler
        y_scaled = target_scaler.transform(y.reshape(-1, 1))
        
        # Make predictions
        y_pred_scaled = model.predict(X)
        y_pred = target_scaler.inverse_transform(y_pred_scaled)
        y_true = target_scaler.inverse_transform(y_scaled)
        
        # Calculate daily metrics
        daily_metrics = calculate_daily_metrics(dates, y_true, y_pred)
        
        # Store metrics for summary (individual models don't have configurations)
        all_metrics[city] = daily_metrics
        
        # Save daily metrics
        daily_metrics.to_csv(os.path.join(base_dir, f'daily_correlations_{city}.csv'), index=False)
        
        # Create plots
        plot_daily_correlations(daily_metrics, city, "individual", "standard", base_dir)
        plot_correlation_analysis(daily_metrics, dates, y_true, y_pred, city, "individual", "standard", base_dir)
        
        print(f"Completed analysis for {city}")
    
    return all_metrics

def create_model_comparison_plots(summary_df, save_dir="results_comprehensive_fixed"):
    """Create comparison plots between joint and individual models."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create comparison plots
    metrics_to_plot = ['MAE (W/m²)', 'RMSE (W/m²)', 'R²', 'MAPE (%)', 'Correlation']
    
    # Create subplots for each metric
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        
        # Get data for this metric
        joint_data = summary_df[summary_df['Model Type'] == 'joint'].groupby('Configuration')[metric].mean()
        individual_data = summary_df[summary_df['Model Type'] == 'individual'][metric].mean()
        
        # Create bar plot
        x_pos = np.arange(len(joint_data) + 1)
        values = list(joint_data.values) + [individual_data]
        labels = list(joint_data.index) + ['Individual']
        colors = ['blue', 'green', 'red', 'orange']
        
        bars = ax.bar(x_pos, values, color=colors[:len(values)])
        ax.set_title(f'{metric} Comparison')
        ax.set_ylabel(metric)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
    
    # Remove the last subplot if not needed
    if len(metrics_to_plot) < 6:
        axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create location-specific comparison
    locations = summary_df['Location'].unique()
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, location in enumerate(locations[:6]):  # Limit to 6 locations
        ax = axes[i]
        
        # Get data for this location
        loc_data = summary_df[summary_df['Location'] == location]
        
        # Create bar plot for each metric
        metrics = ['MAE (W/m²)', 'RMSE (W/m²)', 'R²']
        x_pos = np.arange(len(metrics))
        
        joint_values = []
        individual_values = []
        
        for metric in metrics:
            joint_val = loc_data[loc_data['Model Type'] == 'joint'][metric].mean()
            individual_val = loc_data[loc_data['Model Type'] == 'individual'][metric].mean()
            joint_values.append(joint_val)
            individual_values.append(individual_val)
        
        width = 0.35
        ax.bar(x_pos - width/2, joint_values, width, label='Joint (Best Config)', color='blue', alpha=0.7)
        ax.bar(x_pos + width/2, individual_values, width, label='Individual', color='orange', alpha=0.7)
        
        ax.set_title(f'{location}')
        ax.set_ylabel('Value')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics, rotation=45)
        ax.legend()
    
    # Remove unused subplots
    for i in range(len(locations), 6):
        axes[i].remove()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'location_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to evaluate both joint and individual models with fixed scaling."""
    print("GHI Forecasting Model Evaluation (FIXED SCALING)")
    print("="*80)
    
    # Check if models directory exists
    if not os.path.exists("models"):
        print("× Models directory not found!")
        print("Please run the training scripts first:")
        print("  1. python train_joint_fixed.py")
        print("  2. python train_individual.py")
        return
    
    # Check if target scaler exists
    scaler_path = os.path.join("models", "target_scaler_fixed.pkl")
    if not os.path.exists(scaler_path):
        print("× Target scaler not found!")
        print("Please run train_joint_fixed.py first to create the target scaler.")
        return
    
    # Evaluate joint models
    print("\nEvaluating joint models...")
    joint_metrics = evaluate_joint_models()
    
    # Evaluate individual models
    print("\nEvaluating individual models...")
    individual_metrics = evaluate_individual_models()
    
    # Check if we have any results
    if not joint_metrics and not individual_metrics:
        print("\n× No models were evaluated successfully!")
        print("Please ensure you have trained models available:")
        print("  - Joint model: models/lstm_ghi_forecast_joint_fixed.h5")
        print("  - Individual models: models/lstm_ghi_forecast_{city}.h5")
        return
    
    # Combine all metrics
    all_metrics = {
        'joint': joint_metrics,
        'individual': individual_metrics
    }
    
    # Create and save comprehensive summary table
    summary_df = create_summary_table(all_metrics)
    summary_path = "results_comprehensive_fixed/performance_summary.csv"
    os.makedirs("results_comprehensive_fixed", exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"\nComprehensive performance summary saved to: {summary_path}")
    
    # Create comparison plots
    create_model_comparison_plots(summary_df)
    print("Model comparison plots saved to results_comprehensive_fixed/")
    
    # Print summary statistics
    print("\nOverall Performance Summary (FIXED SCALING):")
    print("=" * 80)
    
    # Joint models summary
    if joint_metrics:
        print("\nJoint Models - Mean metrics across all locations:")
        joint_summary = summary_df[summary_df['Model Type'] == 'joint']
        if not joint_summary.empty:
            mean_metrics = joint_summary.groupby('Configuration')[['MAE (W/m²)', 'RMSE (W/m²)', 'R²', 'MAPE (%)', 'Bias (W/m²)', 'Correlation']].mean()
            print(mean_metrics.round(4))
    
    # Individual models summary
    if individual_metrics:
        print("\nIndividual Models - Mean metrics across all locations:")
        individual_summary = summary_df[summary_df['Model Type'] == 'individual']
        if not individual_summary.empty:
            mean_metrics = individual_summary[['MAE (W/m²)', 'RMSE (W/m²)', 'R²', 'MAPE (%)', 'Bias (W/m²)', 'Correlation']].mean()
            print(mean_metrics.round(4))
    
    # Best performing configuration for each metric
    if not summary_df.empty:
        print("\nBest performing configuration for each metric:")
        for metric in ['MAE (W/m²)', 'RMSE (W/m²)', 'R²', 'MAPE (%)', 'Correlation']:
            if metric in summary_df.columns:
                best_config = summary_df.groupby(['Model Type', 'Configuration'])[metric].mean()
                if metric in ['MAE (W/m²)', 'RMSE (W/m²)', 'MAPE (%)']:
                    best_idx = best_config.idxmin()
                else:
                    best_idx = best_config.idxmax()
                best_value = best_config.loc[best_idx]
                print(f"{metric}: {best_idx} = {best_value:.4f}")
        
        # Special handling for bias - use absolute values
        if 'Bias (W/m²)' in summary_df.columns:
            abs_bias = summary_df.groupby(['Model Type', 'Configuration'])['Bias (W/m²)'].mean().abs()
            best_bias_config = abs_bias.idxmin()
            print(f"Bias (W/m²) - Best configuration (lowest absolute bias): {best_bias_config}")
            print(f"Absolute bias values: {abs_bias.round(4)}")
        
        # Model comparison
        print("\nModel Type Comparison:")
        print("=" * 40)
        model_comparison = summary_df.groupby('Model Type')[['MAE (W/m²)', 'RMSE (W/m²)', 'R²', 'MAPE (%)', 'Correlation']].mean()
        print(model_comparison.round(4))
        
        # Location-specific comparison
        print("\nLocation-specific comparison (Joint vs Individual):")
        print("=" * 60)
        for location in summary_df['Location'].unique():
            loc_data = summary_df[summary_df['Location'] == location]
            joint_data = loc_data[loc_data['Model Type'] == 'joint']
            individual_data = loc_data[loc_data['Model Type'] == 'individual']
            
            if not joint_data.empty and not individual_data.empty:
                joint_best = joint_data.groupby('Configuration')['R²'].mean().max()
                individual_r2 = individual_data['R²'].mean()
                print(f"{location}: Joint (best) R² = {joint_best:.4f}, Individual R² = {individual_r2:.4f}")
                if joint_best > individual_r2:
                    print(f"  → Joint model performs better by {joint_best - individual_r2:.4f}")
                else:
                    print(f"  → Individual model performs better by {individual_r2 - joint_best:.4f}")
    
    print(f"\n✓ Evaluation completed successfully!")
    print(f"Results saved to:")
    print(f"  - Joint models: results_joint_fixed/")
    print(f"  - Individual models: results_individual_fixed/")
    print(f"  - Comprehensive comparison: results_comprehensive_fixed/")

if __name__ == "__main__":
    main() 