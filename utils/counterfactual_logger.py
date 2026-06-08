"""
Counterfactual Logger
Saves parallel forecast records:
  - 'baseline'   : Pure LSTM output (Worker + Manager layers only)
  - 'contextual' : LSTM + LLM CEO bias injected

Purpose: Prove (or disprove) whether the CEO Layer adds real value.
After 1+ resolved records, the Dashboard shows the Hit Ratio Delta.

Storage: JSON Lines format at data/counterfactual_log.jsonl
"""

import os
import json
from datetime import datetime, timezone
import pandas as pd

LOG_PATH = 'data/counterfactual_log.jsonl'


def log_forecast(
    asset_key: str,
    forecast_date: str,
    steps: int,
    baseline_prices: list,
    contextual_prices: list,
    llm_scores: dict,
    llm_narrative: str,
    macro_regime: dict,
):
    """
    Save a parallel forecast record to the counterfactual log.
    Includes robust duplicate prevention to handle Streamlit re-renders.

    Args:
        asset_key        : e.g. 'gold', 'btc', 'aapl'
        forecast_date    : ISO/string date format (YYYY-MM-DD) when forecast was made
        steps            : Number of forecast steps
        baseline_prices  : List of float — pure LSTM predictions
        contextual_prices: List of float — LSTM + CEO bias predictions
        llm_scores       : Dict of Gemini scoring matrix output
        llm_narrative    : Raw text explanation from Gemini
        macro_regime     : Dict from macro_processor.build_macro_context()
    """
    os.makedirs('data', exist_ok=True)

    # Convert forecast_date to string if it's a pandas Timestamp
    if hasattr(forecast_date, 'strftime'):
        forecast_date = forecast_date.strftime('%Y-%m-%d')
    else:
        forecast_date = str(forecast_date)[:10]

    record = {
        'logged_at':          datetime.now(timezone.utc).isoformat(),
        'forecast_date':      forecast_date,
        'asset':              asset_key,
        'steps':              steps,
        'baseline_final':     baseline_prices[-1] if baseline_prices else None,
        'contextual_final':   contextual_prices[-1] if contextual_prices else None,
        'baseline_series':    baseline_prices,
        'contextual_series':  contextual_prices,
        'llm_scores':         llm_scores,
        'llm_narrative':      llm_narrative,
        'macro_regime':       macro_regime,
        'actual_price':       None,   # Filled later by resolve_outcome()
        'baseline_hit':       None,   # Filled later by resolve_outcome()
        'contextual_hit':     None,   # Filled later by resolve_outcome()
    }

    # Duplicate checking
    records = []
    duplicate_idx = -1
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    r = json.loads(line)
                    records.append(r)
                    if (r['asset'] == asset_key and 
                        r['forecast_date'] == forecast_date and 
                        r['steps'] == steps):
                        duplicate_idx = idx
                except Exception:
                    pass

    if duplicate_idx != -1:
        # Overwrite to prevent multiple duplicate writes on page refreshes
        records[duplicate_idx] = record
        with open(LOG_PATH, 'w', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r) + '\n')
        print(f"[CounterfactualLogger] Updated existing log for {asset_key} on {forecast_date} (steps={steps})")
    else:
        # Write new record
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
        print(f"[CounterfactualLogger] Logged new forecast for {asset_key} on {forecast_date} (steps={steps})")


def auto_resolve_all_outcomes(asset_key: str, df: pd.DataFrame, price_col: str):
    """
    Scan all unresolved forecasts in the log and automatically resolve them
    if their target date (forecast_date index + steps) has occurred in the df.
    Called automatically during daily data synchronization.
    """
    if not os.path.exists(LOG_PATH) or df is None or df.empty:
        return

    # Clean date formatting
    df = df.copy()
    if 'Date' not in df.columns:
        df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    
    # Create mapping of Date to index
    date_to_idx = {date: idx for idx, date in enumerate(df['Date'])}

    records = []
    with open(LOG_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass

    updated = False
    for r in records:
        if r['asset'] == asset_key and r.get('actual_price') is None:
            forecast_date = r['forecast_date']
            steps = r['steps']

            if forecast_date in date_to_idx:
                start_idx = date_to_idx[forecast_date]
                target_idx = start_idx + steps

                # If target index exists in our historical dataset, we can resolve!
                if target_idx < len(df):
                    actual_price = float(df[price_col].iloc[target_idx])
                    actual_date = df['Date'].iloc[target_idx]
                    
                    baseline_series = r.get('baseline_series', [])
                    ref_price = baseline_series[0] if baseline_series else r['baseline_final']

                    if ref_price and ref_price > 0:
                        actual_dir = actual_price > ref_price
                        r['actual_price'] = actual_price
                        r['resolved_at_date'] = actual_date

                        if r.get('baseline_final'):
                            r['baseline_hit'] = (r['baseline_final'] > ref_price) == actual_dir
                        if r.get('contextual_final'):
                            r['contextual_hit'] = (r['contextual_final'] > ref_price) == actual_dir
                        
                        updated = True
                        print(f"[CounterfactualLogger] Auto-resolved {asset_key} (forecast from {forecast_date}): target={actual_date}, price={actual_price:.2f}")

    if updated:
        with open(LOG_PATH, 'w', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r) + '\n')


def get_performance_summary(asset_key: str = None) -> dict:
    """
    Compute Hit Ratio comparison between Baseline and CEO-Injected forecasts.
    """
    if not os.path.exists(LOG_PATH):
        return {'error': 'No counterfactual log found. Run forecasts first.'}

    records = []
    with open(LOG_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get('actual_price') is not None:
                    if asset_key is None or r['asset'] == asset_key:
                        records.append(r)
            except Exception:
                pass

    if not records:
        return {'error': f'No resolved records yet. Waiting for outcomes.'}

    n = len(records)
    baseline_hits = sum(1 for r in records if r.get('baseline_hit') is True)
    contextual_hits = sum(1 for r in records if r.get('contextual_hit') is True)

    baseline_ratio = baseline_hits / n
    contextual_ratio = contextual_hits / n

    return {
        'total_resolved':       n,
        'baseline_hit_ratio':   round(baseline_ratio, 3),
        'contextual_hit_ratio': round(contextual_ratio, 3),
        'ceo_delta':            round(contextual_ratio - baseline_ratio, 3),
        'verdict':              '✅ CEO Layer adds value' if contextual_ratio > baseline_ratio else 'CEO Layer not improving accuracy yet',
    }
