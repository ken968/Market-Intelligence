"""
Dynamic Feature Utilities
Computes regime-aware, rolling-window features for use as ML model inputs.

Two feature families are provided:
  1. Rolling VIX Percentile (252-day eCDF) — vix_percentile_252d
  2. Rolling Pairwise Return Correlation (90-day) — roll_corr_<asset>_spy_90d etc.

These columns are appended to each *_global_insights.csv before saving,
so that LSTM / XGBoost training pipelines automatically receive them as input features.

Design note:
  - Values are continuous floats, NOT binned categories, so ML models can learn
    their own optimal split points via gradient descent / information gain.
  - NaN handling: ffill → bfill → fill(0.0) to keep dataset complete.
"""

import numpy as np
import pandas as pd
import os
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. ROLLING VIX PERCENTILE (252-day eCDF)
# ─────────────────────────────────────────────────────────────────────────────

def _ecdf_rank(window: np.ndarray) -> float:
    """
    Empirical CDF rank: fraction of window values <= current (last) value.

    P_t = (1/M) * Σ I(VIX_i <= VIX_t)  for i in [t-M+1, t]

    Used inside a rolling .apply() call (raw=True for numpy array input).
    Returns float in [0.0, 1.0].
    """
    if len(window) < 20:
        return np.nan
    current = window[-1]
    return float(np.sum(window <= current) / len(window))


def add_vix_percentile(df: pd.DataFrame,
                       vix_col: str = 'VIX',
                       window: int = 252) -> pd.DataFrame:
    """
    Append a 'vix_percentile_252d' column to df.

    Represents, for each trading day, what fraction of the past 252 trading
    days had a lower VIX reading than today. This is a dynamic fear gauge:
      - 0.99 → today's VIX is higher than 99% of the past year → Extreme Panic
      - 0.50 → median fear level for the past year

    Args:
        df      : DataFrame with a DatetimeIndex and a VIX column.
        vix_col : Name of the VIX column (default: 'VIX').
        window  : Lookback in trading days (default: 252 ≈ 1 year).

    Returns:
        df with new column: 'vix_percentile_252d' ∈ [0.0, 1.0]
    """
    col_out = f'vix_percentile_{window}d'

    if vix_col not in df.columns:
        df[col_out] = 0.5   # neutral fallback if VIX column unavailable
        print(f"  [feature_utils] Warning: '{vix_col}' column not found. "
              f"'{col_out}' filled with 0.5 (neutral).")
        return df

    raw = df[vix_col].rolling(window=window, min_periods=30).apply(
        _ecdf_rank, raw=True
    )
    df[col_out] = raw.ffill().bfill().fillna(0.5)
    print(f"  [feature_utils] Added '{col_out}'. "
          f"Range: {df[col_out].min():.3f} – {df[col_out].max():.3f}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. ROLLING CROSS-ASSET CORRELATION (90-day sliding window)
# ─────────────────────────────────────────────────────────────────────────────

# Reference benchmarks per asset class
REFERENCE_ASSETS = {
    'btc':  'SPY',     # BTC correlated with risk-on equities
    'gold': 'DXY',     # Gold correlated (inverse) with USD index
    # Stocks and indices: SPY as default benchmark
}

# Map CSV column names → logical asset key for lookup
PRICE_COL_ASSET_MAP = {
    'BTC':  'btc',
    'Gold': 'gold',
}


def _load_reference_series(reference_asset: str) -> Optional[pd.Series]:
    """
    Load the daily close price for a reference asset (SPY or DXY) from CSV.

    Returns a pd.Series with DatetimeIndex, or None if file not available.
    """
    ref_map = {
        'SPY': 'data/SPY_global_insights.csv',
        'DXY': 'data/macro_indicators.csv',
    }
    ref_col_map = {
        'SPY': 'SPY',
        'DXY': 'DXY',
    }

    path = ref_map.get(reference_asset)
    col  = ref_col_map.get(reference_asset)

    if path is None or not os.path.exists(path):
        return None

    try:
        ref_df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
        if col not in ref_df.columns:
            return None
        return ref_df[col].ffill()
    except Exception as e:
        print(f"  [feature_utils] Warning: Could not load reference '{reference_asset}': {e}")
        return None


def add_rolling_correlation(df: pd.DataFrame,
                            price_col: str,
                            asset_key: str,
                            window: int = 90) -> pd.DataFrame:
    """
    Append a rolling 90-day return correlation column against the benchmark asset.

    Benchmark selection:
      - BTC  → correlates against SPY (risk-appetite channel)
      - Gold → correlates against DXY (inverse USD relationship)
      - Equities (SPY, QQQ, AAPL, …) → correlates against SPY (beta channel)

    Formula (Pearson on daily pct_change within the rolling window):
      r_t = Corr(Δprice_asset, Δprice_ref)  over [t-90, t]

    The output is a continuous float ∈ [-1.0, 1.0]:
      ≥ 0.8  → strong coupling   (Enforcer should be tight)
      < 0.3  → decoupling phase  (Enforcer should loosen grip)
      < 0.0  → inverse / flight-to-safety regime

    Args:
        df        : DataFrame with DatetimeIndex and price_col.
        price_col : Column name of this asset's Close price.
        asset_key : Logical key (e.g., 'btc', 'gold', 'SPY').
        window    : Rolling lookback days (default: 90).

    Returns:
        df with new column e.g. 'roll_corr_spy_90d' or 'roll_corr_dxy_90d'
    """
    # Decide benchmark — avoid self-correlation (e.g. SPY vs SPY = meaningless 1.0)
    key_lower = asset_key.lower()

    # Self-correlation overrides: if the asset IS the reference, use an alternative
    SELF_CORR_OVERRIDES = {
        'spy': 'QQQ',   # SPY uses Nasdaq 100 (tech sector) as cross-check
        'qqq': 'DIA',   # QQQ uses Dow Jones
        'dia': 'SPY',   # DIA uses S&P 500
        'dxy': 'SPY',   # DXY uses equities (risk-off inverse)
    }

    reference_asset = SELF_CORR_OVERRIDES.get(key_lower,
                      REFERENCE_ASSETS.get(key_lower, 'SPY'))
    col_suffix = reference_asset.lower()
    col_out = f'roll_corr_{col_suffix}_{window}d'

    if price_col not in df.columns:
        df[col_out] = 0.0
        return df

    # Load reference series and align to this df's index
    ref_series = _load_reference_series(reference_asset)

    if ref_series is None:
        # SPY data may not exist yet (e.g., running for SPY itself)
        # Fall back: compute vs. the macro VIX inverse as a proxy signal
        df[col_out] = 0.0
        print(f"  [feature_utils] Warning: Reference '{reference_asset}' unavailable. "
              f"'{col_out}' filled with 0.0.")
        return df

    # Align index
    ref_aligned = ref_series.reindex(df.index).ffill().bfill()

    # Daily returns
    asset_ret = df[price_col].pct_change()
    ref_ret   = ref_aligned.pct_change()

    # Rolling Pearson correlation
    roll_corr = asset_ret.rolling(window=window, min_periods=30).corr(ref_ret)
    df[col_out] = roll_corr.ffill().bfill().fillna(0.0)

    latest = df[col_out].iloc[-1]
    print(f"  [feature_utils] Added '{col_out}'. "
          f"Latest value: {latest:+.3f} | "
          f"Range: {df[col_out].min():.3f} – {df[col_out].max():.3f}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. CONVENIENCE WRAPPER — apply both features at once
# ─────────────────────────────────────────────────────────────────────────────

def add_dynamic_regime_features(df: pd.DataFrame,
                                price_col: str,
                                asset_key: str,
                                vix_col: str = 'VIX') -> pd.DataFrame:
    """
    Master convenience function.
    Appends ALL dynamic regime features to a dataset in one call:
      1. vix_percentile_252d
      2. roll_corr_<ref>_90d
      3. return_zscore_90d  (from anomaly_detector)

    Designed to be called inside data_fetcher_v2.py immediately
    before store.write_table() for each asset.

    Args:
        df        : DataFrame with DatetimeIndex.
        price_col : Asset's Close price column name.
        asset_key : Asset identifier (e.g., 'btc', 'gold', 'SPY').
        vix_col   : VIX column name in df (default: 'VIX').

    Returns:
        Enriched df.
    """
    print(f"  [feature_utils] Computing dynamic regime features for: {asset_key.upper()}")

    df = add_vix_percentile(df, vix_col=vix_col)
    df = add_rolling_correlation(df, price_col=price_col, asset_key=asset_key)

    # Z-Score anomaly feature (from anomaly_detector module)
    try:
        from utils.anomaly_detector import add_zscore_feature_to_df
        df = add_zscore_feature_to_df(df, price_col=price_col, window=90)
        print(f"  [feature_utils] Added 'return_zscore_90d'.")
    except Exception as e:
        print(f"  [feature_utils] Warning: Z-Score feature skipped: {e}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== feature_utils self-test ===")
    # Minimal synthetic test without loading real files
    idx = pd.date_range('2020-01-01', periods=400, freq='B')
    np.random.seed(0)
    prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.015, 400))
    vix    = 15 + np.abs(np.random.normal(0, 5, 400))

    df_test = pd.DataFrame({'BTC': prices, 'VIX': vix}, index=idx)

    df_test = add_vix_percentile(df_test, vix_col='VIX')
    print(f"\nvix_percentile_252d tail:\n{df_test['vix_percentile_252d'].tail()}")

    print("\n(rolling correlation test skipped in self-test — requires SPY CSV)")
    print("=== Done ===")
