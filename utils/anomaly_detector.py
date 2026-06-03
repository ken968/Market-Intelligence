"""
Asset Anomaly Detector
Detects asset-specific price anomalies using Z-Score methodology.

This acts as a 'Micro Circuit Breaker' that fires independently of the macro VIX
level. If BTC flash-crashes while VIX is calm (e.g., exchange hack scenario),
this module catches it.

Thresholds:
  - Crypto assets (BTC): z_threshold = 2.5 (naturally more volatile)
  - Equities/Metals (SPY, Gold, AAPL, etc.): z_threshold = 3.0
"""

import numpy as np
import pandas as pd
import os
from typing import Optional


# Asset class volatility configuration
ASSET_Z_THRESHOLDS = {
    'btc': 2.5,
    'gold': 3.0,
    'default_equity': 3.0,
}

ASSET_CLASSES = {
    'btc': 'crypto',
    'gold': 'commodity',
}


def _get_z_threshold(asset_key: str) -> float:
    """Return the appropriate Z-Score threshold for an asset class."""
    key = asset_key.lower()
    if key in ASSET_Z_THRESHOLDS:
        return ASSET_Z_THRESHOLDS[key]
    # Default: equity threshold
    return ASSET_Z_THRESHOLDS['default_equity']


def compute_daily_return_zscore(series: pd.Series, window: int = 90) -> pd.Series:
    """
    Compute the rolling Z-Score of daily returns.

    For each day t, calculates:
        z_t = (r_t - mean(r_{t-window:t})) / std(r_{t-window:t})

    Where r_t is the daily percentage change (Close-to-Close return).

    Args:
        series  : Price series (Close price, daily).
        window  : Rolling lookback for mean/std calculation (default: 90 trading days).

    Returns:
        pd.Series of Z-Score values (same index as series).
    """
    daily_returns = series.pct_change()
    rolling_mean = daily_returns.rolling(window=window, min_periods=20).mean()
    rolling_std = daily_returns.rolling(window=window, min_periods=20).std()

    z_scores = (daily_returns - rolling_mean) / rolling_std.replace(0, np.nan)
    return z_scores


def detect_asset_anomaly(
    asset_key: str,
    df: pd.DataFrame,
    price_col: Optional[str] = None,
    window: int = 90,
) -> dict:
    """
    Detect if the most recent trading day is an anomalous price move.

    Uses rolling Z-Score on daily returns. If the latest day's return
    exceeds the threshold for the asset class, a Micro Circuit Breaker fires.

    Args:
        asset_key : Asset identifier (e.g., 'btc', 'gold', 'SPY').
        df        : DataFrame with a DatetimeIndex and a price column.
        price_col : Column name for Close price. If None, auto-detects.
        window    : Rolling window for Z-Score (default: 90 trading days).

    Returns:
        dict with keys:
          - 'is_anomaly'  : bool — True if micro circuit breaker fires.
          - 'z_score'     : float — Z-Score of the latest daily return.
          - 'daily_return': float — Latest daily return (pct).
          - 'threshold'   : float — Z-Score threshold used for this asset.
          - 'regime'      : str — 'NORMAL', 'CAUTION', or 'ANOMALY'.
          - 'asset_key'   : str
    """
    z_threshold = _get_z_threshold(asset_key)

    # Auto-detect price column
    if price_col is None:
        key_upper = asset_key.upper()
        if key_upper in df.columns:
            price_col = key_upper
        elif 'Close' in df.columns:
            price_col = 'Close'
        else:
            return {
                'is_anomaly': False, 'z_score': 0.0, 'daily_return': 0.0,
                'threshold': z_threshold, 'regime': 'UNKNOWN', 'asset_key': asset_key,
                'error': f"Price column not found for {asset_key}"
            }

    if price_col not in df.columns or len(df) < 30:
        return {
            'is_anomaly': False, 'z_score': 0.0, 'daily_return': 0.0,
            'threshold': z_threshold, 'regime': 'INSUFFICIENT_DATA', 'asset_key': asset_key
        }

    z_series = compute_daily_return_zscore(df[price_col], window=window)
    daily_returns = df[price_col].pct_change()

    if z_series.empty or pd.isna(z_series.iloc[-1]):
        return {
            'is_anomaly': False, 'z_score': 0.0, 'daily_return': 0.0,
            'threshold': z_threshold, 'regime': 'INSUFFICIENT_DATA', 'asset_key': asset_key
        }

    latest_z = float(z_series.iloc[-1])
    latest_return = float(daily_returns.iloc[-1]) if not pd.isna(daily_returns.iloc[-1]) else 0.0
    abs_z = abs(latest_z)

    # Determine regime
    if abs_z >= z_threshold:
        regime = 'ANOMALY'
        is_anomaly = True
    elif abs_z >= z_threshold * 0.75:   # 75% of threshold → caution
        regime = 'CAUTION'
        is_anomaly = False
    else:
        regime = 'NORMAL'
        is_anomaly = False

    direction = 'CRASH' if latest_z < 0 else 'SPIKE'

    return {
        'is_anomaly':   is_anomaly,
        'z_score':      round(latest_z, 4),
        'daily_return': round(latest_return * 100, 4),   # as % for readability
        'threshold':    z_threshold,
        'regime':       regime,
        'direction':    direction if is_anomaly else 'N/A',
        'asset_key':    asset_key,
    }


def add_zscore_feature_to_df(df: pd.DataFrame, price_col: str, window: int = 90) -> pd.DataFrame:
    """
    Add a 'return_zscore_{window}d' column to a DataFrame.

    This is the column that gets stored in the *_global_insights.csv
    so that LSTM / XGBoost models can consume the Z-Score as a continuous feature.

    Args:
        df        : DataFrame with a DatetimeIndex.
        price_col : Column name for the asset's Close price.
        window    : Rolling window (default: 90 trading days).

    Returns:
        df with a new column: f'return_zscore_{window}d'
    """
    col_name = f'return_zscore_{window}d'

    if price_col not in df.columns:
        df[col_name] = 0.0
        return df

    z = compute_daily_return_zscore(df[price_col], window=window)
    df[col_name] = z.ffill().bfill().fillna(0.0)
    return df


if __name__ == '__main__':
    # Quick validation
    import pandas as pd
    import numpy as np

    # Simulate a price series with an obvious flash crash at the end
    np.random.seed(42)
    prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.015, 300))
    prices[-1] = prices[-2] * 0.88  # Simulate -12% flash crash

    df_test = pd.DataFrame({'BTC': prices}, index=pd.date_range('2024-01-01', periods=300, freq='B'))

    result = detect_asset_anomaly('btc', df_test, price_col='BTC')
    print("\n=== Anomaly Detector Self-Test ===")
    for k, v in result.items():
        print(f"  {k:<18}: {v}")
