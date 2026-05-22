"""
XAUUSD Multi-Asset Terminal - Forecast Settings & Confidence
"""

import os
import json

FORECAST_RANGES = {
    '1 Day': 1,
    '1 Week': 7,
    '2 Weeks': 14,
    '1 Month': 30,
    '3 Months': 90
}

# Confidence scores — FALLBACK VALUES ONLY.
# These are conservative baselines used when no backtest data exists yet.
# Actual confidence is computed dynamically from backtest hit_ratio JSON files.
CONFIDENCE_SCORES_FALLBACK = {
    'gold': {
        '1 Day':    {'score': 0.65, 'label': 'Moderate', 'color': 'info'},
        '1 Week':   {'score': 0.60, 'label': 'Moderate', 'color': 'info'},
        '2 Weeks':  {'score': 0.55, 'label': 'Low',      'color': 'warning'},
        '1 Month':  {'score': 0.50, 'label': 'Low',      'color': 'warning'},
        '3 Months': {'score': 0.35, 'label': 'Speculative', 'color': 'error'}
    },
    'btc': {
        '1 Day':    {'score': 0.60, 'label': 'Moderate', 'color': 'info'},
        '1 Week':   {'score': 0.55, 'label': 'Moderate', 'color': 'info'},
        '2 Weeks':  {'score': 0.50, 'label': 'Low',      'color': 'warning'},
        '1 Month':  {'score': 0.45, 'label': 'Low',      'color': 'warning'},
        '3 Months': {'score': 0.35, 'label': 'Speculative', 'color': 'error'}
    },
    'stocks': {
        '1 Day':    {'score': 0.60, 'label': 'Moderate', 'color': 'info'},
        '1 Week':   {'score': 0.55, 'label': 'Moderate', 'color': 'info'},
        '2 Weeks':  {'score': 0.50, 'label': 'Low',      'color': 'warning'},
        '1 Month':  {'score': 0.45, 'label': 'Low',      'color': 'warning'},
        '3 Months': {'score': 0.35, 'label': 'Speculative', 'color': 'error'}
    }
}

# Decay factors: longer horizon = raw hit_ratio is discounted further
_TIMEFRAME_DECAY = {
    '1 Day':    1.00,
    '1 Week':   0.92,
    '2 Weeks':  0.84,
    '1 Month':  0.75,
    '3 Months': 0.55,
}

def _score_to_label(score: float) -> dict:
    """Map a numeric score [0.0-1.0] to a display label and color."""
    if score >= 0.75:
        return {'score': round(score, 3), 'label': 'High',        'color': 'success'}
    elif score >= 0.65:
        return {'score': round(score, 3), 'label': 'Good',        'color': 'success'}
    elif score >= 0.55:
        return {'score': round(score, 3), 'label': 'Moderate',    'color': 'info'}
    elif score >= 0.45:
        return {'score': round(score, 3), 'label': 'Low',         'color': 'warning'}
    else:
        return {'score': round(score, 3), 'label': 'Speculative', 'color': 'error'}

def get_dynamic_confidence(asset_key: str, timeframe: str) -> dict:
    """
    Load confidence score from the most recent backtest JSON.
    Converts hit_ratio → score:
        score = hit_ratio_combined / 100 * decay_factor
    Falls back to CONFIDENCE_SCORES_FALLBACK if no backtest JSON found.
    """
    asset_type = 'gold' if asset_key == 'gold' else ('btc' if asset_key == 'btc' else 'stocks')
    fallback = CONFIDENCE_SCORES_FALLBACK.get(asset_type, {}).get(
        timeframe, {'score': 0.50, 'label': 'Unknown', 'color': 'info'}
    ).copy()

    # 1. Try loading Dual-Head Stacker JSON first (Ensemble)
    json_path = f'reports/stacker_{asset_key}_backtest.json'
    is_stacker = True
    
    if not os.path.exists(json_path):
        # 2. Try legacy LSTM backtest JSON for this specific asset
        json_path = f'reports/backtest_{asset_key}.json'
        is_stacker = False
        
    if not os.path.exists(json_path):
        # 3. Try legacy asset_type grouping (e.g. 'stocks')
        json_path = f'reports/backtest_{asset_type}.json'
        is_stacker = False

    if not os.path.exists(json_path):
        return fallback

    try:
        with open(json_path, 'r') as fh:
            data = json.load(fh)

        if is_stacker:
            raw_hit_ratio = data.get('hit_ratio_combined', None)
        else:
            raw_hit_ratio = data.get('hit_ratio_3layer', None)
            
        if raw_hit_ratio is None:
            return fallback

        base_score   = raw_hit_ratio / 100.0
        decay_factor = _TIMEFRAME_DECAY.get(timeframe, 0.70)
        final_score  = base_score * decay_factor

        result = _score_to_label(final_score)
        if timeframe == '3 Months':
            result['label'] = 'Speculative'
            result['color'] = 'error'

        return result

    except Exception:
        return fallback

FORECAST_DISCLAIMER = """
**Forecast Limitations**: AI predictions are based on historical patterns and current market conditions.
Actual results may vary significantly due to unforeseen events, policy changes, or market shocks.
Use forecasts as directional guidance, not precise targets. Always combine with fundamental analysis.
⚠️ Forecasts beyond 2 weeks are speculative due to compounding model error — treat as scenario analysis only.
"""

CONFIDENCE_SCORES = CONFIDENCE_SCORES_FALLBACK
