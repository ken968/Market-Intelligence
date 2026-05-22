"""
Confidence Engine
Evaluates directional forecast confidence using dynamic backtest metrics and CEO Layer uplift.
"""

from typing import Dict, Any
from utils.config import get_dynamic_confidence, CONFIDENCE_SCORES

def get_confidence_score(asset_key: str, timeframe: str, ceo_confidence: float = 0.0) -> Dict[str, Any]:
    """
    Get confidence score for asset and timeframe.

    Priority order:
    1. Load real hit_ratio from backtest JSON
    2. Apply horizon decay (longer forecast = lower confidence)
    3. Apply CEO uplift if Gemini is highly confident (max +0.08 boost)
    4. Fall back to conservative static values if no backtest data exists

    Args:
        asset_key (str)       : Asset identifier (e.g., 'gold', 'btc', 'aapl')
        timeframe (str)       : Forecast timeframe (e.g., '1 Week', '1 Month')
        ceo_confidence (float): Gemini CEO confidence score [0.0 – 1.0]

    Returns:
        dict: {'score': float, 'label': str, 'color': str}
    """
    # Step 1: Load dynamic confidence from backtest JSON
    try:
        base = get_dynamic_confidence(asset_key, timeframe)
    except Exception:
        # Final fallback to static dict
        asset_type = 'gold' if asset_key == 'gold' else ('btc' if asset_key == 'btc' else 'stocks')
        base = CONFIDENCE_SCORES.get(asset_type, {}).get(timeframe, {
            'score': 0.50, 'label': 'Unknown', 'color': 'info'
        }).copy()

    # Step 2: CEO uplift — if Gemini is confident (>= 0.7), add modest boost
    # Cap at +0.08 to avoid inflating scores artificially
    if ceo_confidence >= 0.70 and timeframe != '3 Months':
        uplift = (ceo_confidence - 0.50) * 0.16   # max +0.08 at confidence=1.0
        new_score = min(0.92, base['score'] + uplift)
        # Re-label based on new score
        if new_score >= 0.75:
            base = {'score': round(new_score, 3), 'label': 'High',     'color': 'success'}
        elif new_score >= 0.65:
            base = {'score': round(new_score, 3), 'label': 'Good',     'color': 'success'}
        elif new_score >= 0.55:
            base = {'score': round(new_score, 3), 'label': 'Moderate', 'color': 'info'}
        else:
            base['score'] = round(new_score, 3)

    return base
