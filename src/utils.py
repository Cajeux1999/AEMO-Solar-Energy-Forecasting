import pandas as pd
import numpy as np
from datetime import datetime

def add_cyclical_features(df, time_col='ds'):
    """
    Add cyclical time features (halfhour, day, month encoded in sin and cos functions)

    Args:
        df: DataFrame containing time series
        time_col: Name of the column containing the time stamps.

    Returns:
        Dataframe with added features
    """
    df[time_col] = pd.to_datetime(df[time_col])

    # Extract components with 30-minute granularity
    df['total_hours'] = (df[time_col].dt.hour +
                         df[time_col].dt.minute/60 +
                         df[time_col].dt.second/3600)

    # Cyclical encoding - now using 48 half-hour periods per day
    df['halfhour_sin'] = np.sin(2 * np.pi * (df['total_hours']*2)/48)
    df['halfhour_cos'] = np.cos(2 * np.pi * (df['total_hours']*2)/48)

    # Original day/month cycles (unchanged)
    df['day'] = df[time_col].dt.day
    df['month'] = df[time_col].dt.month
    df['day_sin'] = np.sin(2 * np.pi * df['day']/31)
    df['day_cos'] = np.cos(2 * np.pi * df['day']/31)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)

    # Drop intermediate columns (keep total_hours if you want it)
    df.drop(['day', 'month', 'total_hours'], axis=1, inplace=True)

    return df

def add_lag_features(df, target_col='y', lags=[]):
    """
    Add lag features for specified lag periods

    Args:
        df: DataFrame containing time series
        target_col: Name of target column
        lags: List of lag periods to create

    Returns:
        DataFrame with added lag features
    """
    for lag in lags:
        df[f'y_lag_{lag}'] = df[target_col].shift(lag)

    return df

def plot_pareto(front):
    front = np.array(front)  

    x = front[:, 0]
    y = front[:, 1]

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, alpha=0.6, label="Pareto Front")

    plt.xlabel("Accuracy [RMSE]")
    plt.ylabel("Time [s]")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_and_save_pareto(front, pop, experiment_name='exp'):
    # Create a unique filename using a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.npz"
    
    # Save the data
    np.savez_compressed(filename, front=front, pop=pop)
    
    print(f"Successfully saved to: {filename}")

    front = np.array(front)

    x = -front[:, 0] if np.all(front[:, 0] < 0) else front[:, 0]
    y = -front[:, 1] if np.all(front[:, 1] < 0) else front[:, 1]

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='blue', alpha=0.7, edgecolors='k', label="Pareto Front (Real)")
    
    plt.xlabel("Accuracy [Metric]")
    plt.ylabel("Time [s]")
    plt.title(f"Pareto Front: {experiment_name}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    img_filename = f"{experiment_name}_{timestamp}.png"
    
    plt.savefig(img_filename, dpi=600, bbox_inches='tight')
    print(f"Gráfico salvo como: {img_filename}")
    
    plt.show()

    return filename, img_filename