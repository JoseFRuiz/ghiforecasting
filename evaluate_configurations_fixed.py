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
        # Add GHI lag features (24 features)
        feature_columns.extend([f"GHI_lag_{lag}" for lag in range(1, 25)])
        # Add current GHI
        feature_columns.append("GHI")
    
    if input_config in ['met_only', 'ghi_met']:
        # Add meteorological features
        met_features = [
            "Temperature_lag_24", "Relative Humidity_lag_24",
            "Pressure_lag_24", "Precipitable Water_lag_24",
            "Wind Direction_lag_24", "Wind Speed_lag_24"
        ]
        feature_columns.extend(met_features)
    
    # Always add time features
    time_features = ["hour_sin", "hour_cos", "month_sin", "month_cos"]
    feature_columns.extend(time_features)
    
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
    
    return X, y

def create_sequences_individual(df, sequence_length):
    """Create sequences for individual models."""
    # Get all required features
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
    
    return X, y

def split_data_by_days(df, train_ratio=0.7, val_ratio=0.15):
    """Split data by days to maintain temporal consistency."""
    # Convert datetime to date for grouping
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    
    # Get unique dates
    unique_dates = sorted(df['date'].unique())
    
    # Calculate split indices
    n_dates = len(unique_dates)
    train_end = int(n_dates * train_ratio)
    val_end = int(n_dates * (train_ratio + val_ratio))
    
    # Split dates
    train_dates = unique_dates[:train_end]
    val_dates = unique_dates[train_end:val_end]
    test_dates = unique_dates[val_end:]
    
    # Split data based on dates
    train_df = df[df['date'].isin(train_dates)]
    val_df = df[df['date'].isin(val_dates)]
    test_df = df[df['date'].isin(test_dates)]
    
    return train_df, val_df, test_df

def calculate_daily_metrics(dates, actual, predicted):
    """Calculate daily metrics for evaluation."""
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

def plot_daily_correlations(daily_metrics, city, model_type, config, save_dir):
    """Create plots showing daily correlation metrics."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
    
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
    
    # Plot GHI statistics
    ax3.plot(daily_metrics['date'], daily_metrics['actual_max'], 'b-', label='Actual Max')
    ax3.plot(daily_metrics['date'], daily_metrics['predicted_max'], 'r--', label='Predicted Max')
    ax3.plot(daily_metrics['date'], daily_metrics['actual_mean'], 'g-', label='Actual Mean')
    ax3.plot(daily_metrics['date'], daily_metrics['predicted_mean'], 'm--', label='Predicted Mean')
    ax3.set_title(f'Daily GHI Statistics - {city} ({model_type}, {config})')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('GHI (W/m²)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    for ax in [ax1, ax2, ax3]:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'daily_correlations_{city}_{model_type}_{config}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save daily metrics to CSV
    daily_metrics.to_csv(os.path.join(save_dir, f'daily_correlations_{city}_{model_type}_{config}.csv'), 
                        index=False)

def plot_correlation_analysis(daily_metrics, dates, actual, predicted, city, model_type, config, save_dir):
    """Create scatter plots comparing actual vs predicted values."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Convert dates to pandas Series for proper date handling
    dates = pd.Series(dates)
    
    # Plot 1: Day with highest correlation
    best_day_idx = daily_metrics['correlation'].idxmax()
    best_day = daily_metrics.loc[best_day_idx]
    
    # Get the actual data for this day
    mask = dates.dt.date == best_day['date']
    actual_data = np.array(actual)[mask]
    predicted_data = np.array(predicted)[mask]
    
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
    mask = dates.dt.date == worst_day['date']
    actual_data = np.array(actual)[mask]
    predicted_data = np.array(predicted)[mask]
    
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
    plt.savefig(os.path.join(save_dir, f'correlation_analysis_{city}_{model_type}_{config}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(all_metrics):
    """Create a comprehensive summary table."""
    # Create summary DataFrame
    summary_data = []
    for (city, model_type, config), metrics in all_metrics.items():
                    summary_data.append({
            'City': city,
            'Model_Type': model_type,
            'Config': config,
            'MAE': metrics['mae'],
            'RMSE': metrics['rmse'],
            'R²': metrics['r2'],
            'Correlation': metrics['correlation'],
            'Mean_Correlation': metrics['daily_metrics']['correlation'].mean(),
            'Mean_R2': metrics['daily_metrics']['r2'].mean(),
            'Std_Correlation': metrics['daily_metrics']['correlation'].std(),
            'Std_R2': metrics['daily_metrics']['r2'].std()
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("results_comprehensive_fixed/performance_summary.csv", index=False)
    print(f"\nSummary saved to results_comprehensive_fixed/performance_summary.csv")
    print(summary_df)
    
    return summary_df

def create_sequences_joint_all_features(df, sequence_length, locations):
    """Create sequences for joint models with all features including location encoding."""
    # Initialize feature columns
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
    
    # Initialize lists to store sequences
    all_X_sequences = []
    all_y_sequences = []
    
    # Process each location separately
    for location in locations:
        df_loc = df[df['location'] == location].copy()
        
        if len(df_loc) == 0:
            continue
            
        # Create one-hot encoded location vector for the current location
        location_vector = np.zeros(len(locations))
        location_vector[locations.index(location)] = 1
        
        # Sort by datetime to ensure chronological order
        df_loc = df_loc.sort_values("datetime")
        
        # Get all required data at once
        data = df_loc[feature_columns].values
        target = df_loc["GHI"].values
    
    # Calculate number of sequences
        n_sequences = len(df_loc) - sequence_length
    
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
    
        # Add location features
    location_features = np.tile(location_vector, (len(X), sequence_length, 1))
    X = np.concatenate([X, location_features], axis=2)
    
    all_X_sequences.append(X)
    all_y_sequences.append(y)
    
    if not all_X_sequences:
        raise ValueError("No valid sequences were created for any location")
    
    # Combine sequences from all locations
    X = np.concatenate(all_X_sequences, axis=0)
    y = np.concatenate(all_y_sequences, axis=0)
    
    return X, y

def evaluate_joint_models():
    """Evaluate joint models with different configurations."""
    print("\nEvaluating joint models...")
    
    # Load data
    all_dfs = []
    for city in CONFIG["data_locations"].keys():
        df = load_data(CONFIG["data_locations"], city)
        df = create_features(df)
        df['location'] = city
        all_dfs.append(df)
    
    df_all = pd.concat(all_dfs, ignore_index=True)
    locations = sorted(df_all['location'].unique().tolist())
    
    # Load the fixed joint model
    model_path = "models/lstm_ghi_forecast_joint_fixed.h5"
    if not os.path.exists(model_path):
        print(f"Joint model not found: {model_path}")
        return {}
    
    print(f"Loading joint model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✓ Joint model loaded successfully")
        print(f"Model summary:")
        model.summary()
    except Exception as e:
        print(f"✗ Error loading joint model: {e}")
        return {}
    
    # Load target scaler
    scaler_path = "models/target_scaler_fixed.pkl"
    if not os.path.exists(scaler_path):
        print(f"Target scaler not found: {scaler_path}")
        return {}
    
    print(f"Loading target scaler from {scaler_path}...")
    try:
        with open(scaler_path, 'rb') as f:
            target_scaler = pickle.load(f)
        print(f"✓ Target scaler loaded successfully")
        print(f"Scaler range: [{target_scaler.data_min_[0]:.2f}, {target_scaler.data_max_[0]:.2f}]")
    except Exception as e:
        print(f"✗ Error loading target scaler: {e}")
        return {}
    
    # Evaluate the joint model
    all_metrics = {}
    
    print(f"\nEvaluating joint model...")
    
    # Create sequences using the same approach as train_joint_fixed.py
    # First, we need to create features similar to train_joint_fixed.py
    print("Creating joint features...")
    
    # Add time-based features
    df_all["hour_sin"] = np.sin(2 * np.pi * df_all["Hour"] / 24)
    df_all["hour_cos"] = np.cos(2 * np.pi * df_all["Hour"] / 24)
    df_all["month_sin"] = np.sin(2 * np.pi * df_all["Month"] / 12)
    df_all["month_cos"] = np.cos(2 * np.pi * df_all["Month"] / 12)
    
    # Create target: GHI for next day at the same hour
    df_all["target_GHI"] = df_all["GHI"].shift(-24)
    
    # Create lag features for previous 24 hours of GHI
    for lag in range(1, 25):
        df_all[f"GHI_lag_{lag}"] = df_all["GHI"].shift(lag)
    
    # Create lag features for meteorological variables
    meteorological_features = [
        "Temperature", "Relative Humidity", "Pressure",
        "Precipitable Water", "Wind Direction", "Wind Speed"
    ]
    
    for feature in meteorological_features:
        df_all[f"{feature}_lag_24"] = df_all[feature].shift(24)
    
    # Create location-specific features (one-hot encoding)
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse=False)
    location_encoded = encoder.fit_transform(df_all[['location']])
    location_df = pd.DataFrame(location_encoded, 
                             columns=[f"location_{loc}" for loc in encoder.categories_[0]])
    
    print(f"Location encoding created:")
    print(f"  Categories: {encoder.categories_[0]}")
    print(f"  Location columns: {location_df.columns.tolist()}")
    print(f"  Sample location encoding:")
    for i, loc in enumerate(locations):
        print(f"    {loc}: {location_df.iloc[i].values}")
    
    # Combine with original data
    df_all = pd.concat([df_all, location_df], axis=1)
    
    # Clean data
    df_all = df_all.dropna().reset_index(drop=True)
    
    # Split data by days (same as individual models)
    train_df, val_df, test_df = split_data_by_days(df_all)
    
    # Scale the test data using the same scaler
    test_df_scaled = test_df.copy()
    test_df_scaled["GHI"] = target_scaler.transform(test_df[["GHI"]])
    test_df_scaled["target_GHI"] = target_scaler.transform(test_df[["target_GHI"]])
    
    # Create sequences for test data
    X_test, y_test = create_sequences_joint_all_features(test_df_scaled, 24, locations)
    
    if len(X_test) == 0:
        print("No test sequences created for joint model")
        return {}
    
    print(f"Created {len(X_test)} test sequences for joint model")
    
    # Make predictions
    print(f"Making predictions on {len(X_test)} test sequences...")
    y_pred = model.predict(X_test)
    print(f"Predictions shape: {y_pred.shape}")
    print(f"Predictions range: [{np.min(y_pred):.4f}, {np.max(y_pred):.4f}]")
    
    # Inverse transform
    y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_original = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    print(f"Original test range: [{np.min(y_test_original):.2f}, {np.max(y_test_original):.2f}]")
    print(f"Original pred range: [{np.min(y_pred_original):.2f}, {np.max(y_pred_original):.2f}]")
    
    # Calculate metrics for each city
    print(f"\nEvaluating joint model for each city...")
    print(f"Total predictions: {len(y_pred_original)}")
    print(f"Location features shape: {X_test[:, 0, -len(locations):].shape}")
    
    for city in locations:
        print(f"\nEvaluating {city}...")
        
        # Filter data for this city
        city_mask = X_test[:, 0, -len(locations):][:, locations.index(city)] == 1
        print(f"  City mask for {city}: {city_mask.sum()} samples out of {len(city_mask)}")
        
        if city_mask.sum() > 0:
            city_actual = y_test_original[city_mask]
            city_predicted = y_pred_original[city_mask]
            
            print(f"  {city}: {len(city_actual)} samples")
            print(f"  Actual range: [{np.min(city_actual):.2f}, {np.max(city_actual):.2f}]")
            print(f"  Predicted range: [{np.min(city_predicted):.2f}, {np.max(city_predicted):.2f}]")
            
            # Calculate metrics
            mae = mean_absolute_error(city_actual, city_predicted)
            rmse = np.sqrt(mean_squared_error(city_actual, city_predicted))
            r2 = r2_score(city_actual, city_predicted)
            correlation = np.corrcoef(city_actual, city_predicted)[0, 1]
            
            print(f"  MAE: {mae:.2f} W/m²")
            print(f"  RMSE: {rmse:.2f} W/m²")
            print(f"  R²: {r2:.4f}")
            print(f"  Correlation: {correlation:.4f}")
            
            # Calculate daily metrics
            dates = pd.date_range(start='2020-01-01', periods=len(city_actual), freq='H')
            daily_metrics = calculate_daily_metrics(dates, city_actual, city_predicted)
            
            all_metrics[(city, 'joint', 'standard')] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'correlation': correlation,
                'daily_metrics': daily_metrics,
                'actual': city_actual,
                'predicted': city_predicted,
                'dates': dates
            }
            
            # Create plots
            plot_daily_correlations(daily_metrics, city, 'joint', 'standard', "results_joint_fixed")
            plot_correlation_analysis(daily_metrics, dates, city_actual, city_predicted, 
                                   city, 'joint', 'standard', "results_joint_fixed")
        else:
            print(f"  No data found for {city}")
            print(f"  This might indicate an issue with the location encoding")
    
    return all_metrics

def evaluate_individual_models():
    """Evaluate individual models."""
    print("\nEvaluating individual models...")
    print(f"Available cities: {list(CONFIG['data_locations'].keys())}")
    
    all_metrics = {}
    
    for city in CONFIG["data_locations"].keys():
        print(f"\nEvaluating {city}...")
        print(f"  Checking for model file: models/lstm_ghi_forecast_{city}.h5")
        
        # Load data
        df = load_data(CONFIG["data_locations"], city)
        df = create_features(df)
        
        # Load model
        model_path = f"models/lstm_ghi_forecast_{city}.h5"
        if not os.path.exists(model_path):
            print(f"  Model not found for {city}")
            continue
        
        print(f"  Loading model from {model_path}...")
        try:
            model = tf.keras.models.load_model(model_path, custom_objects={'custom_ghi_loss': custom_ghi_loss})
            print(f"  ✓ Model loaded successfully")
            print(f"  Model summary:")
            model.summary()
        except Exception as e:
            print(f"  ✗ Error loading model: {e}")
            continue
        
        # Split data (same as training)
        train_df, val_df, test_df = split_data_by_days(df)
        
        # Create scaler fitted on ALL data (same as training)
        print(f"  Creating scaler for {city} (fitted on all data)...")
        target_scaler = MinMaxScaler()
        target_scaler.fit(df[["GHI"]])
        print(f"  Scaler fitted on all data range: [{target_scaler.data_min_[0]:.2f}, {target_scaler.data_max_[0]:.2f}]")
        
        # Transform test data using the same scaler
        test_df_scaled = test_df.copy()
        test_df_scaled["GHI"] = target_scaler.transform(test_df[["GHI"]])
        
        # Create sequences from scaled test data
        X_test, y_test = create_sequences_individual(test_df_scaled, 24)
        
        if len(X_test) == 0:
            print(f"  No test sequences for {city}")
            continue
        
        # Make predictions
        print(f"  Making predictions on {len(X_test)} test sequences...")
        y_pred = model.predict(X_test)
        print(f"  Predictions shape: {y_pred.shape}")
        print(f"  Predictions range: [{np.min(y_pred):.4f}, {np.max(y_pred):.4f}]")
        
        # Inverse transform using the same scaler
        y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_original = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        print(f"  Original test range: [{np.min(y_test_original):.2f}, {np.max(y_test_original):.2f}]")
        print(f"  Original pred range: [{np.min(y_pred_original):.2f}, {np.max(y_pred_original):.2f}]")
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
        r2 = r2_score(y_test_original, y_pred_original)
        correlation = np.corrcoef(y_test_original, y_pred_original)[0, 1]
        
        # Calculate daily metrics
        dates = pd.date_range(start='2020-01-01', periods=len(y_test_original), freq='H')
        daily_metrics = calculate_daily_metrics(dates, y_test_original, y_pred_original)
        
        all_metrics[(city, 'individual', 'standard')] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'correlation': correlation,
            'daily_metrics': daily_metrics,
            'actual': y_test_original,
            'predicted': y_pred_original,
            'dates': dates
        }
        
        # Create plots
        plot_daily_correlations(daily_metrics, city, 'individual', 'standard', "results_individual_fixed")
        plot_correlation_analysis(daily_metrics, dates, y_test_original, y_pred_original, 
                               city, 'individual', 'standard', "results_individual_fixed")
    
    return all_metrics

def custom_ghi_loss(y_true, y_pred):
    """Custom loss function for GHI prediction."""
    # Standard Huber loss
    huber_loss = tf.keras.losses.Huber()(y_true, y_pred)
    
    # Penalty for predictions outside [0, 1] range
    range_penalty = tf.reduce_mean(tf.maximum(0.0, y_pred - 1.0) + tf.maximum(0.0, -y_pred))
    
    return huber_loss + range_penalty

def evaluate_gnn_models():
    """Evaluate GNN models."""
    print("\nEvaluating GNN models...")
    
    # Check if GNN results exist
    gnn_summary_path = "results_gnn/summary.csv"
    if not os.path.exists(gnn_summary_path):
        print("GNN results not found. Please run train_gnn.py first.")
        return {}
    
    # Load GNN summary
    gnn_summary = pd.read_csv(gnn_summary_path)
    
    all_metrics = {}
    
    # Load individual city results
    for city in CONFIG["data_locations"].keys():
        city_dir = f"results_gnn/{city}"
        if os.path.exists(city_dir):
            # Load daily metrics
            daily_metrics_path = f"{city_dir}/daily_metrics.csv"
            if os.path.exists(daily_metrics_path):
                daily_metrics = pd.read_csv(daily_metrics_path)
                daily_metrics['date'] = pd.to_datetime(daily_metrics['date'])
                
                # Get summary metrics from the summary file
                city_summary = gnn_summary[gnn_summary['City'] == city]
                if len(city_summary) > 0:
                    mae = city_summary['MAE'].iloc[0]
                    rmse = city_summary['RMSE'].iloc[0]
                    r2 = city_summary['R²'].iloc[0]
                    correlation = city_summary['Correlation'].iloc[0]
                    
                    # Load actual and predicted values if available
                    # For GNN, we need to reconstruct from the daily metrics
                    # This is a simplified approach
                    actual = []
                    predicted = []
                    dates = []
                    
                    for _, row in daily_metrics.iterrows():
                        # Reconstruct hourly data from daily statistics
                        # This is an approximation
                        n_points = int(row['num_points'])
                        actual.extend([row['actual_mean']] * n_points)
                        predicted.extend([row['predicted_mean']] * n_points)
                        dates.extend([row['date']] * n_points)
                    
                    all_metrics[(city, 'gnn', 'standard')] = {
                        'mae': mae,
                        'rmse': rmse,
                        'r2': r2,
                        'correlation': correlation,
                        'daily_metrics': daily_metrics,
                        'actual': np.array(actual),
                        'predicted': np.array(predicted),
                        'dates': pd.to_datetime(dates)
                    }
                    
                    # Create plots
                    plot_daily_correlations(daily_metrics, city, 'gnn', 'standard', "results_gnn_fixed")
                    plot_correlation_analysis(daily_metrics, dates, actual, predicted, 
                                           city, 'gnn', 'standard', "results_gnn_fixed")
    
    return all_metrics

def create_model_comparison_plots(summary_df, save_dir="results_comprehensive_fixed"):
    """Create comparison plots for different models."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Model comparison plot
    plt.figure(figsize=(15, 10))
    
    # Create subplots for different metrics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    
    # Get unique cities and model types
    cities = summary_df['City'].unique()
    model_types = summary_df['Model_Type'].unique()
    
    # Set up colors for different model types
    colors = {'individual': 'blue', 'joint': 'red', 'gnn': 'green'}
    
    # Plot 1: MAE comparison
    for model_type in model_types:
        model_data = summary_df[summary_df['Model_Type'] == model_type]
        ax1.bar([f"{city}_{model_type}" for city in model_data['City']], 
                model_data['MAE'], label=model_type, color=colors[model_type], alpha=0.7)
    ax1.set_title('MAE Comparison')
    ax1.set_ylabel('MAE (W/m²)')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: RMSE comparison
    for model_type in model_types:
        model_data = summary_df[summary_df['Model_Type'] == model_type]
        ax2.bar([f"{city}_{model_type}" for city in model_data['City']], 
                model_data['RMSE'], label=model_type, color=colors[model_type], alpha=0.7)
    ax2.set_title('RMSE Comparison')
    ax2.set_ylabel('RMSE (W/m²)')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: R² comparison
    for model_type in model_types:
        model_data = summary_df[summary_df['Model_Type'] == model_type]
        ax3.bar([f"{city}_{model_type}" for city in model_data['City']], 
                model_data['R²'], label=model_type, color=colors[model_type], alpha=0.7)
    ax3.set_title('R² Comparison')
    ax3.set_ylabel('R²')
    ax3.legend()
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Correlation comparison
    for model_type in model_types:
        model_data = summary_df[summary_df['Model_Type'] == model_type]
        ax4.bar([f"{city}_{model_type}" for city in model_data['City']], 
                model_data['Correlation'], label=model_type, color=colors[model_type], alpha=0.7)
    ax4.set_title('Correlation Comparison')
    ax4.set_ylabel('Correlation')
    ax4.legend()
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Location comparison plot
    plt.figure(figsize=(15, 10))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    
    # Plot by location
    for city in cities:
        city_data = summary_df[summary_df['City'] == city]
        
        # MAE by location
        ax1.bar([f"{city}_{mt}" for mt in city_data['Model_Type']], 
                city_data['MAE'], label=city, alpha=0.7)
        ax1.set_title('MAE by Location')
        ax1.set_ylabel('MAE (W/m²)')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # RMSE by location
        ax2.bar([f"{city}_{mt}" for mt in city_data['Model_Type']], 
                city_data['RMSE'], label=city, alpha=0.7)
        ax2.set_title('RMSE by Location')
        ax2.set_ylabel('RMSE (W/m²)')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        
        # R² by location
        ax3.bar([f"{city}_{mt}" for mt in city_data['Model_Type']], 
                city_data['R²'], label=city, alpha=0.7)
        ax3.set_title('R² by Location')
        ax3.set_ylabel('R²')
        ax3.legend()
        ax3.tick_params(axis='x', rotation=45)
        
        # Correlation by location
        ax4.bar([f"{city}_{mt}" for mt in city_data['Model_Type']], 
                city_data['Correlation'], label=city, alpha=0.7)
        ax4.set_title('Correlation by Location')
        ax4.set_ylabel('Correlation')
        ax4.legend()
        ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'location_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to evaluate all model configurations."""
    print("Starting comprehensive model evaluation...")
    
    # Create necessary directories
    os.makedirs("results_comprehensive_fixed", exist_ok=True)
    os.makedirs("results_individual_fixed", exist_ok=True)
    os.makedirs("results_joint_fixed", exist_ok=True)
    os.makedirs("results_gnn_fixed", exist_ok=True)
    
    # Evaluate all model types
    all_metrics = {}
    
    # Evaluate individual models
    print("\n" + "="*60)
    print("EVALUATING INDIVIDUAL MODELS")
    print("="*60)
    individual_metrics = evaluate_individual_models()
    print(f"Individual metrics returned: {len(individual_metrics)} entries")
    all_metrics.update(individual_metrics)
    
    # Evaluate joint models
    joint_metrics = evaluate_joint_models()
    all_metrics.update(joint_metrics)
    
    # Evaluate GNN models
    gnn_metrics = evaluate_gnn_models()
    all_metrics.update(gnn_metrics)
    
    # Create comprehensive summary
    if all_metrics:
        summary_df = create_summary_table(all_metrics)
        create_model_comparison_plots(summary_df)
        print("\nComprehensive evaluation completed!")
    else:
        print("\nNo metrics were calculated. Please ensure all models are trained and available.")

if __name__ == "__main__":
    main() 