"""
Macro Insights Processor
Translates raw FRED macro data into actionable regime signals and risk scores.
These outputs feed directly into the CEO Layer (LLM Manager) and Level 2 Anchor.

Signals produced:
  - recession_risk_score  : 0.0 (expansion) → 1.0 (deep recession risk)
  - yield_regime          : 'expansion' | 'flattening' | 'inversion' | 'steepening'
  - m2_bias               : 'risk_on' | 'neutral' | 'risk_off'
  - m2_liquidity_event    : True/False — sudden MoM spike detected
  - breakeven_regime      : 'anchored' | 'rising' | 'elevated'
  - macro_summary         : Human-readable text summary for CEO Layer prompt
"""

import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()


def load_macro_data(fred_path: str = 'data/fred_indicators.csv') -> pd.DataFrame:
    """Load FRED indicator CSV and return as dated DataFrame."""
    if not os.path.exists(fred_path):
        raise FileNotFoundError(f"FRED data not found at: {fred_path}. Run fred_fetcher.py first.")
    df = pd.read_csv(fred_path, index_col='Date', parse_dates=True)
    return df


def compute_recession_risk(df: pd.DataFrame) -> float:
    """
    Compute a composite Recession Risk Score in [0.0, 1.0].

    Factors (weighted):
      - Yield Curve (T10Y2Y) inversion  : 50% weight
      - 5Y5Y Breakeven elevation        : 25% weight (high = stagflation risk)
      - M2 YoY collapse (<0%)           : 25% weight (monetary contraction)
    """
    score = 0.0
    latest = df.iloc[-1]

    # --- Factor 1: Yield Curve (50%) ---
    yc = latest.get('YieldCurve_10Y2Y', 0.0)
    if yc < -0.5:
        yc_risk = 1.0       # Deep inversion
    elif yc < 0.0:
        yc_risk = 0.7       # Mild inversion
    elif yc < 0.8:
        yc_risk = 0.3       # Flattening — early warning
    else:
        yc_risk = 0.0       # Steep/Normal
    score += yc_risk * 0.50

    # --- Factor 2: 5Y5Y Breakeven (25%) ---
    be = latest.get('Breakeven_5Y5Y', 2.3)
    if be > 2.8:
        be_risk = 1.0 
    elif be > 2.5:
        be_risk = 0.5
    elif be > 2.2:
        be_risk = 0.2       # Slight elevation
    else:
        be_risk = 0.0
       # Anchored expectations
    score += be_risk * 0.25

    # --- Factor 3: M2 YoY contraction (25%) ---
    m2_yoy = latest.get('M2_YoY', 5.0)
    if m2_yoy < -2.0:
        m2_risk = 1.0       # Sharp contraction — like 2022-2023 era
    elif m2_yoy < 0.0:
        m2_risk = 0.5       # Negative but mild
    elif m2_yoy < 3.0:
        m2_risk = 0.1       # Below trend
    else:
        m2_risk = 0.0       # Healthy growth
    score += m2_risk * 0.25

    return round(min(max(score, 0.0), 1.0), 3)


def detect_yield_regime(df: pd.DataFrame) -> str:
    """
    Identify yield curve regime from the last 30 days of movement.
    Returns: 'expansion' | 'steepening' | 'flattening' | 'inversion'
    """
    if 'YieldCurve_10Y2Y' not in df.columns or len(df) < 30:
        return 'unknown'

    recent = df['YieldCurve_10Y2Y'].dropna().iloc[-30:]
    current = recent.iloc[-1]
    prev_30d = recent.iloc[0]

    if current < 0:
        return 'inversion'
    elif current > prev_30d + 0.1:
        return 'steepening'    # Curve widening — typically early recovery
    elif current < prev_30d - 0.1:
        return 'flattening'    # Curve narrowing — typically late cycle
    else:
        return 'expansion'


def detect_m2_bias(df: pd.DataFrame) -> tuple:
    """
    Returns (bias: str, liquidity_event: bool).
    Bias is based on YoY trend; liquidity_event flags sudden MoM acceleration.
    """
    latest = df.iloc[-1]
    m2_yoy = latest.get('M2_YoY', 5.0)
    m2_mom = latest.get('M2_MoM', 0.2)
    spike_flag = int(latest.get('M2_Liquidity_Spike', 0))

    # YoY-based bias
    if m2_yoy > 5.0:
        bias = 'risk_on'       # Ample liquidity supports asset prices
    elif m2_yoy > 2.0:
        bias = 'neutral'
    else:
        bias = 'risk_off'      # Monetary contraction — assets under pressure

    # MoM spike detection (liquidity event even if YoY is bearish)
    liquidity_event = bool(spike_flag or m2_mom > 0.8)

    return bias, liquidity_event


def detect_breakeven_regime(df: pd.DataFrame) -> str:
    """
    Classify 5Y5Y breakeven inflation expectation regime.
    """
    if 'Breakeven_5Y5Y' not in df.columns:
        return 'unknown'
    be = df['Breakeven_5Y5Y'].dropna().iloc[-1]
    if be > 2.8:
        return 'elevated'      # Markets pricing in persistent inflation
    elif be > 2.3:
        return 'rising'        # Drift upward — watch carefully
    else:
        return 'anchored'      # Fed considered credible


def build_macro_context(fred_path: str = 'data/fred_indicators.csv') -> dict:
    """
    Master function: loads FRED data and computes all macro regime signals.

    Returns a dict used by CEO Layer (llm_manager.py) to ground its analysis:
    {
        'recession_risk': float,
        'yield_regime': str,
        'm2_bias': str,
        'm2_liquidity_event': bool,
        'breakeven_regime': str,
        'latest_values': dict,
        'macro_summary': str          # Human-readable for LLM prompt injection
    }
    """
    try:
        df = load_macro_data(fred_path)
    except FileNotFoundError as e:
        return {'error': str(e), 'recession_risk': 0.5, 'yield_regime': 'unknown',
                'm2_bias': 'neutral', 'm2_liquidity_event': False, 'breakeven_regime': 'unknown',
                'macro_summary': 'Macro data unavailable.'}

    recession_risk = compute_recession_risk(df)
    yield_regime = detect_yield_regime(df)
    m2_bias, m2_liquidity_event = detect_m2_bias(df)
    breakeven_regime = detect_breakeven_regime(df)

    latest = df.iloc[-1]

    # Key latest values for display and LLM prompt
    latest_values = {
        'YieldCurve_10Y2Y': round(latest.get('YieldCurve_10Y2Y', 0), 3),
        'Yield_10Y_Rate':   round(latest.get('Yield_10Y_Rate', 0), 3),
        'Breakeven_5Y5Y':   round(latest.get('Breakeven_5Y5Y', 0), 3),
        'M2_YoY':           round(latest.get('M2_YoY', 0), 2),
        'M2_MoM':           round(latest.get('M2_MoM', 0), 2),
        'CPI_MoM':          round(latest.get('CPI_MoM', 0), 2),
        'PCE_MoM':          round(latest.get('PCE_MoM', 0), 2),
        'PPI_MoM':          round(latest.get('PPI_MoM', 0), 2),
    }

    # Human-readable summary injected into LLM system prompt
    macro_summary = (
        f"MACRO CONTEXT (live):\n"
        f"- Yield Curve (10Y-2Y): {latest_values['YieldCurve_10Y2Y']:+.2f}% → Regime: {yield_regime.upper()}\n"
        f"- 10Y Treasury Rate: {latest_values['Yield_10Y_Rate']:.2f}% (Cost of Capital)\n"
        f"- 5Y5Y Inflation Breakeven: {latest_values['Breakeven_5Y5Y']:.2f}% → {breakeven_regime.upper()}\n"
        f"- M2 Money Supply: YoY {latest_values['M2_YoY']:+.1f}% / MoM {latest_values['M2_MoM']:+.2f}% → {m2_bias.upper()}"
        + (" ⚡ LIQUIDITY SPIKE DETECTED" if m2_liquidity_event else "") + "\n"
        f"- Inflation Pipeline: CPI {latest_values['CPI_MoM']:+.2f}% | PPI {latest_values['PPI_MoM']:+.2f}% | PCE {latest_values['PCE_MoM']:+.2f}% (MoM)\n"
        f"- Composite Recession Risk Score: {recession_risk:.2f}/1.00\n"
    )

    return {
        'recession_risk':      recession_risk,
        'yield_regime':        yield_regime,
        'm2_bias':             m2_bias,
        'm2_liquidity_event':  m2_liquidity_event,
        'breakeven_regime':    breakeven_regime,
        'latest_values':       latest_values,
        'macro_summary':       macro_summary,
    }


if __name__ == '__main__':
    ctx = build_macro_context()
    if 'error' in ctx:
        print(f"Error: {ctx['error']}")
    else:
        print(ctx['macro_summary'])
        print(f"Recession Risk Score: {ctx['recession_risk']}")
