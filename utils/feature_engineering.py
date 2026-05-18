"""
Feature Engineering Utilities
Centralized computation of derived features:
  - Garman-Klass Volatility (from OHLCV)
  - Lagged Macro Features (CPI_lag3, NFP_lag3, etc.)
  - Credit Spread (from FRED if available)
  - Rolling Percentile Recession Risk Score
"""

import numpy as np
import pandas as pd
import os


# ─────────────────────────────────────────────────────────────────────────────
# Garman-Klass Volatility
# ─────────────────────────────────────────────────────────────────────────────

def compute_garman_klass_vol(df: pd.DataFrame,
                              open_col: str = 'Open',
                              high_col: str  = 'High',
                              low_col: str   = 'Low',
                              close_col: str = 'Close',
                              window: int    = 21) -> pd.Series:
    """
    Garman-Klass Volatility Estimator (annualized).

    More statistically efficient than close-to-close because it uses
    the full daily range (Open, High, Low, Close).

    Formula:
        GK_daily = 0.5 * (ln(H/L))^2 - (2*ln(2)-1) * (ln(C/O))^2

    Annualized rolling vol (252 trading days):
        GK_Vol = sqrt( rolling_mean(GK_daily, window) * 252 )

    Args:
        df        : DataFrame with OHLCV columns
        open_col  : Column name for Open prices
        high_col  : Column name for High prices
        low_col   : Column name for Low prices
        close_col : Column name for Close prices
        window    : Rolling window in days (default 21 = ~1 month)

    Returns:
        pd.Series of annualized GK volatility (0.10 = 10% annualized vol)

    Reference:
        Garman, M.B. and Klass, M.J. (1980), "On the Estimation of
        Security Price Volatilities from Historical Data", Journal of Business.
    """
    required = [open_col, high_col, low_col, close_col]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Garman-Klass: Missing OHLCV columns: {missing}")

    # Avoid log(0) or log(negative) — replace zeros and negatives
    o = df[open_col].replace(0, np.nan).abs()
    h = df[high_col].replace(0, np.nan).abs()
    l = df[low_col].replace(0, np.nan).abs()
    c = df[close_col].replace(0, np.nan).abs()

    log_hl = np.log(h / l)           # High-Low range component
    log_co = np.log(c / o)           # Close-Open drift component

    # Daily GK variance
    gk_daily_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2

    # Clamp negative variances (numerical edge case) to 0
    gk_daily_var = gk_daily_var.clip(lower=0)

    # Annualized rolling volatility
    gk_vol = np.sqrt(gk_daily_var.rolling(window=window, min_periods=max(1, window // 2)).mean() * 252)

    return gk_vol


# ─────────────────────────────────────────────────────────────────────────────
# Lagged Macro Features
# ─────────────────────────────────────────────────────────────────────────────

# Indicators that are released monthly (not daily).
# These need longer lookback windows AND lagged features to capture
# monetary policy transmission lag (typically 12-18 months).
MONTHLY_INDICATORS = ['CPI_MoM', 'PPI_MoM', 'PCE_MoM', 'NFP_Change',
                       'M2_MoM', 'M2_YoY', 'YieldCurve_10Y2Y',
                       'Yield_10Y_Rate', 'Breakeven_5Y5Y']


def add_lagged_macro_features(df: pd.DataFrame,
                               lags_months: list = [3, 6]) -> pd.DataFrame:
    """
    Add lagged versions of monthly FRED indicators to capture
    monetary policy transmission lag (Fed Rate → economy = 12-18 months).

    For daily data: 1 month ≈ 21 trading days.

    Args:
        df          : DataFrame with macro columns already merged
        lags_months : List of lag periods in months (default [3, 6])

    Returns:
        DataFrame with additional columns: CPI_MoM_lag3, CPI_MoM_lag6, etc.
    """
    df = df.copy()
    trading_days_per_month = 21

    for col in MONTHLY_INDICATORS:
        if col not in df.columns:
            continue
        for lag_m in lags_months:
            lag_days = lag_m * trading_days_per_month
            lag_col  = f'{col}_lag{lag_m}'
            df[lag_col] = df[col].shift(lag_days)
            # Forward fill gaps (monthly data has many NaNs between releases)
            df[lag_col] = df[lag_col].ffill().fillna(0)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment Uncertainty (Std alongside Mean)
# ─────────────────────────────────────────────────────────────────────────────

def compute_sentiment_std(df: pd.DataFrame,
                           window: int = 5) -> pd.Series:
    """
    Rolling standard deviation of Sentiment scores.

    High Sentiment_Std = high uncertainty / conflicting news signals.
    This can be used to widen Monte Carlo uncertainty bands.

    Args:
        df     : DataFrame with 'Sentiment' column
        window : Rolling window in days

    Returns:
        pd.Series of rolling Sentiment std
    """
    if 'Sentiment' not in df.columns:
        return pd.Series(0.0, index=df.index)
    return df['Sentiment'].rolling(window=window, min_periods=1).std().fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# Rolling Percentile Recession Risk Score
# ─────────────────────────────────────────────────────────────────────────────

def compute_dynamic_recession_risk(df_fred: pd.DataFrame) -> float:
    """
    Compute recession risk score using ROLLING PERCENTILE instead of
    hard-coded thresholds. This automatically adapts to any interest rate regime
    (low rate era 2009-2021 vs high rate era 2022+).

    Factors:
        - Yield Curve (10Y-2Y): 50% weight — lower percentile = more inverted = more risk
        - 5Y5Y Breakeven      : 25% weight — higher percentile = higher inflation = more risk
        - M2 YoY              : 25% weight — lower percentile = more contraction = more risk

    Args:
        df_fred : Full history FRED DataFrame (use maximum history for robust percentiles)

    Returns:
        float in [0.0, 1.0] — 0.0 = expansion, 1.0 = deep recession risk
    """
    if df_fred.empty or len(df_fred) < 30:
        return 0.5  # Not enough data — return neutral

    def percentile_rank(series, value):
        """Where does 'value' fall in the historical distribution? (0=lowest, 1=highest)"""
        if series.std() < 1e-8:
            return 0.5
        return float((series <= value).mean())

    score = 0.0
    latest = df_fred.iloc[-1]

    # Factor 1: Yield Curve — lower percentile = more inverted = higher risk
    if 'YieldCurve_10Y2Y' in df_fred.columns:
        yc_pct = percentile_rank(df_fred['YieldCurve_10Y2Y'].dropna(),
                                  latest.get('YieldCurve_10Y2Y', 1.0))
        yc_risk = 1.0 - yc_pct   # Invert: lowest yield curve = highest risk
        score  += yc_risk * 0.50

    # Factor 2: 5Y5Y Breakeven — higher percentile = higher inflation expectations = more risk
    if 'Breakeven_5Y5Y' in df_fred.columns:
        be_pct  = percentile_rank(df_fred['Breakeven_5Y5Y'].dropna(),
                                   latest.get('Breakeven_5Y5Y', 2.3))
        score  += be_pct * 0.25

    # Factor 3: M2 YoY — lower percentile = more contraction = higher risk
    if 'M2_YoY' in df_fred.columns:
        m2_pct  = percentile_rank(df_fred['M2_YoY'].dropna(),
                                   latest.get('M2_YoY', 5.0))
        m2_risk = 1.0 - m2_pct   # Invert: lowest M2 growth = highest risk
        score  += m2_risk * 0.25

    return round(min(max(score, 0.0), 1.0), 3)


# ─────────────────────────────────────────────────────────────────────────────
# Z-Score with Frequency-Aware Window
# ─────────────────────────────────────────────────────────────────────────────

# Known release frequency for each macro indicator.
# This determines the lookback window used for "recent" in Z-score calculation.
INDICATOR_LOOKBACK = {
    # Monthly releases (21 trading days ≈ 1 month)
    'CPI_MoM':          30,
    'PPI_MoM':          30,
    'PCE_MoM':          30,
    'NFP_Change':       30,
    'M2_MoM':           30,
    'M2_YoY':           30,
    # Quarterly
    'Breakeven_5Y5Y':   63,
    # Daily market data — keep 14-day window
    'DXY':              14,
    'VIX':              14,
    'Yield_10Y':        14,
    'Oil_Price':        14,
    'YieldCurve_10Y2Y': 14,
    'Yield_10Y_Rate':   14,
}

DEFAULT_LOOKBACK = 14  # Fallback for unknown indicators


def get_indicator_lookback(feature: str) -> int:
    """Return the appropriate lookback window for a given macro indicator."""
    return INDICATOR_LOOKBACK.get(feature, DEFAULT_LOOKBACK)
