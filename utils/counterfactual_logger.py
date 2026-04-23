"""
Counterfactual Logger
Saves parallel forecast records:
  - 'baseline'   : Pure LSTM output (Worker + Manager layers only)
  - 'contextual' : LSTM + LLM CEO bias injected

Purpose: Prove (or disprove) whether the CEO Layer adds real value.
After 30+ records, the BacktestEngine uses this log to compute Hit Ratio
and Information Coefficient for the CEO Layer.

Storage: JSON Lines format at data/counterfactual_log.jsonl
"""

import os
import json
from datetime import datetime, timezone


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

    Args:
        asset_key        : e.g. 'gold', 'btc', 'aapl'
        forecast_date    : ISO date string when forecast was made
        steps            : Number of forecast steps
        baseline_prices  : List of float — pure LSTM predictions
        contextual_prices: List of float — LSTM + CEO bias predictions
        llm_scores       : Dict of Gemini scoring matrix output
        llm_narrative    : Raw text explanation from Gemini
        macro_regime     : Dict from macro_processor.build_macro_context()
    """
    os.makedirs('data', exist_ok=True)

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

    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')


def resolve_outcome(forecast_date: str, asset_key: str, actual_price: float):
    """
    Update historical records with the actual price outcome.
    Computes directional accuracy (Hit = correct direction call).

    Call this function daily via a cron/scheduler using the previous day's actual closing price.
    """
    if not os.path.exists(LOG_PATH):
        return

    records = []
    with open(LOG_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))

    updated = False
    for r in records:
        if r['asset'] == asset_key and r['forecast_date'] == forecast_date and r['actual_price'] is None:
            # Determine reference price from baseline_series[0] as the "current" at forecast time
            baseline_series = r.get('baseline_series', [])
            ref_price = baseline_series[0] if baseline_series else r['baseline_final']

            if ref_price and ref_price > 0:
                actual_dir = actual_price > ref_price
                r['actual_price'] = actual_price

                if r['baseline_final']:
                    r['baseline_hit'] = (r['baseline_final'] > ref_price) == actual_dir
                if r['contextual_final']:
                    r['contextual_hit'] = (r['contextual_final'] > ref_price) == actual_dir

            updated = True

    if updated:
        with open(LOG_PATH, 'w', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r) + '\n')


def get_performance_summary(asset_key: str = None) -> dict:
    """
    Compute Hit Ratio comparison between Baseline and CEO-Injected forecasts.

    Returns:
        {
            'total_resolved': int,
            'baseline_hit_ratio': float,
            'contextual_hit_ratio': float,
            'ceo_delta': float          # Positive = CEO layer IMPROVES accuracy
        }
    """
    if not os.path.exists(LOG_PATH):
        return {'error': 'No counterfactual log found. Run forecasts first.'}

    records = []
    with open(LOG_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r.get('actual_price') is not None:
                if asset_key is None or r['asset'] == asset_key:
                    records.append(r)

    if not records:
        return {'error': f'No resolved records yet{"for " + asset_key if asset_key else ""}. Waiting for outcomes.'}

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


if __name__ == '__main__':
    summary = get_performance_summary()
    print(json.dumps(summary, indent=2))
