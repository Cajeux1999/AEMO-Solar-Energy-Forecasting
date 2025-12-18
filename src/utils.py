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

def plot_and_save_pareto(front, experiment_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    front = np.array(front)

    # Inverte o sinal caso o mobopt tenha retornado valores negativos (minimização)
    # Se você já tratou o sinal antes, pode remover o -
    x = -front[:, 0] if np.all(front[:, 0] < 0) else front[:, 0]
    y = -front[:, 1] if np.all(front[:, 1] < 0) else front[:, 1]

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='blue', alpha=0.7, edgecolors='k', label="Pareto Front (Real)")
    
    # Adiciona uma linha conectando os pontos para visualizar a "fronteira"
    # (Opcional: funciona melhor se os pontos estiverem ordenados por x)
    idx = np.argsort(x)
    plt.plot(x[idx], y[idx], 'r--', alpha=0.3)

    plt.xlabel("Accuracy [Metric]")
    plt.ylabel("Time [s]")
    plt.title(f"Pareto Front: {experiment_name}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Define o nome do arquivo de imagem com base no mesmo padrão do .npz
    img_filename = f"{experiment_name}_{timestamp}.png"
    
    # Salva a imagem
    plt.savefig(img_filename, dpi=600, bbox_inches='tight')
    print(f"Gráfico salvo como: {img_filename}")
    
    plt.show()
    return img_filename
