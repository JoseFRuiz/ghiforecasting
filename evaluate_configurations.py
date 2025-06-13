import os
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

def create_sequences(df, sequence_length, input_config):
    """Create sequences based on input configuration."""
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
    return X[mask], y[mask], df['datetime'].values[sequence_length:][mask]

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
    
    # Validate metric ranges
    if metrics_df['r2'].mean() > 0.95:
        print("Warning: Suspiciously high R² value (>0.95)")
    if metrics_df['mape'].mean() < 5:
        print("Warning: Suspiciously low MAPE value (<5%)")
    
    # Check for any remaining negative R² values
    negative_r2 = metrics_df[metrics_df['r2'] < 0]
    if not negative_r2.empty:
        print("\nWarning: Found negative R² values:")
        print(negative_r2[['date', 'r2']].head())
        print(f"Total days with negative R²: {len(negative_r2)}")
    
    return metrics_df

def plot_daily_correlations(daily_metrics, city, config, save_dir):
    """Create plots showing daily correlation metrics with additional validation."""
    # Create figure with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
    
    # Plot correlation and R²
    ax1.plot(daily_metrics['date'], daily_metrics['correlation'], 'b-', label='Correlation')
    ax1.plot(daily_metrics['date'], daily_metrics['r2'], 'r-', label='R²')
    ax1.set_title(f'Daily Correlation and R² - {city} ({config})')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot MAE and RMSE
    ax2.plot(daily_metrics['date'], daily_metrics['mae'], 'g-', label='MAE')
    ax2.plot(daily_metrics['date'], daily_metrics['rmse'], 'm-', label='RMSE')
    ax2.set_title(f'Daily MAE and RMSE - {city} ({config})')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Error (W/m²)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot MAPE and Bias
    ax3.plot(daily_metrics['date'], daily_metrics['mape'], 'c-', label='MAPE')
    ax3.plot(daily_metrics['date'], daily_metrics['bias'], 'y-', label='Bias')
    ax3.set_title(f'Daily MAPE and Bias - {city} ({config})')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot GHI statistics
    ax4.plot(daily_metrics['date'], daily_metrics['actual_max'], 'b-', label='Actual Max')
    ax4.plot(daily_metrics['date'], daily_metrics['predicted_max'], 'r--', label='Predicted Max')
    ax4.plot(daily_metrics['date'], daily_metrics['actual_mean'], 'g-', label='Actual Mean')
    ax4.plot(daily_metrics['date'], daily_metrics['predicted_mean'], 'm--', label='Predicted Mean')
    ax4.set_title(f'Daily GHI Statistics - {city} ({config})')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('GHI (W/m²)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    for ax in [ax1, ax2, ax3, ax4]:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'daily_correlations_{city}_{config}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_analysis(daily_metrics, dates, actual, predicted, city, config, save_dir):
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
    
    plt.suptitle(f'Correlation Analysis for {city} ({config})\nR² = {daily_metrics["r2"].mean():.3f}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'correlation_analysis_{city}_{config}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(all_metrics):
    """Create a summary table of performance metrics across all locations and configurations."""
    summary_data = []
    
    for config, city_metrics in all_metrics.items():
        for city, metrics_df in city_metrics.items():
            summary_data.append({
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
    
    # Create DataFrame and sort by configuration and location
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(['Configuration', 'Location'])
    
    # Round numeric columns
    numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
    summary_df[numeric_cols] = summary_df[numeric_cols].round(4)
    
    return summary_df

def main():
    # Create output directories
    base_dir = "results_joint"
    os.makedirs(base_dir, exist_ok=True)
    
    # Load target scaler
    scaler_path = os.path.join("models", "target_scaler.pkl")
    with open(scaler_path, 'rb') as f:
        target_scaler = pickle.load(f)
    
    # Dictionary to store all metrics
    all_metrics = {}
    
    # Process each configuration
    for config in ['ghi_only', 'met_only', 'ghi_met']:
        print(f"\n{'='*80}")
        print(f"Processing {config} configuration")
        print(f"{'='*80}")
        
        # Create configuration-specific directory
        config_dir = os.path.join(base_dir, config)
        os.makedirs(config_dir, exist_ok=True)
        
        # Load model
        model_path = os.path.join("models", f"lstm_ghi_forecast_joint_{config}.h5")
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
        
        model = tf.keras.models.load_model(model_path)
        
        # Initialize dictionary for this configuration
        all_metrics[config] = {}
        
        # Process each city
        for city in CONFIG["data_locations"].keys():
            print(f"\nProcessing {city}...")
            
            # Load and prepare data
            df = load_data(CONFIG["data_locations"], city)
            df = create_features(df)
            
            # Create sequences
            X, y, dates = create_sequences(df, sequence_length=24, input_config=config)
            
            if len(X) == 0:
                print(f"No valid sequences for {city}")
                continue
            
            # Make predictions
            y_pred_scaled = model.predict(X)
            y_pred = target_scaler.inverse_transform(y_pred_scaled)
            y_true = target_scaler.inverse_transform(y.reshape(-1, 1))
            
            # Calculate daily metrics
            daily_metrics = calculate_daily_metrics(dates, y_true, y_pred)
            
            # Store metrics for summary
            all_metrics[config][city] = daily_metrics
            
            # Save daily metrics
            daily_metrics.to_csv(os.path.join(config_dir, f'daily_correlations_{city}.csv'), index=False)
            
            # Create plots
            plot_daily_correlations(daily_metrics, city, config, config_dir)
            plot_correlation_analysis(daily_metrics, dates, y_true, y_pred, city, config, config_dir)
            
            print(f"Completed analysis for {city}")
    
    # Create and save summary table
    summary_df = create_summary_table(all_metrics)
    summary_path = os.path.join(base_dir, "performance_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nPerformance summary saved to: {summary_path}")
    
    # Print summary statistics
    print("\nOverall Performance Summary:")
    print("=" * 80)
    print("\nMean metrics across all locations:")
    mean_metrics = summary_df.groupby('Configuration')[['MAE (W/m²)', 'RMSE (W/m²)', 'R²', 'MAPE (%)', 'Bias (W/m²)', 'Correlation']].mean()
    print(mean_metrics.round(4))
    
    print("\nBest performing configuration for each metric:")
    for metric in ['MAE (W/m²)', 'RMSE (W/m²)', 'R²', 'MAPE (%)', 'Correlation']:
        best_config = summary_df.groupby('Configuration')[metric].mean().idxmin() if metric in ['MAE (W/m²)', 'RMSE (W/m²)', 'MAPE (%)'] else summary_df.groupby('Configuration')[metric].mean().idxmax()
        print(f"{metric}: {best_config}")
    
    # Special handling for bias - use absolute values
    abs_bias = summary_df.groupby('Configuration')['Bias (W/m²)'].mean().abs()
    best_bias_config = abs_bias.idxmin()
    print(f"Bias (W/m²) - Best configuration (lowest absolute bias): {best_bias_config}")
    print(f"Absolute bias values: {abs_bias.round(4)}")

if __name__ == "__main__":
    main() 