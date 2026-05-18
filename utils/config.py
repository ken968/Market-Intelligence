"""
XAUUSD Multi-Asset Terminal - Global Configuration
Centralized config for all assets, models, and UI settings
"""

import os

# ==================== UI THEME CONFIGURATION ====================

THEME = {
    # Core Colors
    'bg_deep': '#0E1117',
    'bg_surface': '#1E1E1E',  # Slightly lighter than deep
    'accent': '#00A8E8',      # Primary Blue
    'accent_muted': 'rgba(0, 168, 232, 0.2)',
    
    # UI Elements
    'border': '#333333',
    'text_primary': '#FFFFFF',
    'text_secondary': '#A0A0A0',
    
    # Semantic Colors
    'success': '#00CC96',
    'danger': '#EF553B',
    'warning': '#FFA15A',
    'info': '#636EFA',
    
    # Legacy/Fallback (keep for safety)
    'primary': '#00A8E8',
    'secondary': '#CCCCCC',
    'background': '#0E1117',
    'card_bg': '#262730',
    'font': 'Inter, sans-serif'
}

# ==================== ASSET CONFIGURATION ====================

ASSETS = {
    'gold': {
        'name': 'Gold (XAUUSD)',
        'ticker': 'GC=F',
        'icon': '',
        'color': '#FFD700',
        'model_file': 'models/gold_ultimate_model.keras',
        'scaler_file': 'models/scaler.pkl',
        'data_file': 'data/gold_global_insights.csv',
        'news_file': 'data/latest_news_gold.json',
        'features': ['Gold', 'DXY', 'VIX', 'Yield_10Y', 'Oil_Price',
                     'CPI_MoM', 'PPI_MoM', 'PCE_MoM', 'NFP_Change',
                     'YieldCurve_10Y2Y', 'M2_MoM', 'M2_YoY', 'Yield_10Y_Rate', 
                     'Breakeven_5Y5Y', 'M2_Liquidity_Spike', 'MacroEvent_Flag',
                     'Sentiment', 'EMA_90'],
        'sequence_length': 60,
        'description': 'Precious Metal & Safe Haven Asset'
    },
    'btc': {
        'name': 'Bitcoin',
        'ticker': 'BTC-USD',
        'icon': '₿',
        'color': '#F7931A',
        'model_file': 'models/btc_ultimate_model.keras',
        'scaler_file': 'models/btc_scaler.pkl',
        'data_file': 'data/btc_global_insights.csv',
        'news_file': 'data/latest_news_btc.json',
        'features': ['BTC', 'DXY', 'VIX', 'Yield_10Y', 'Oil_Price',
                     'CPI_MoM', 'PPI_MoM', 'PCE_MoM', 'NFP_Change',
                     'YieldCurve_10Y2Y', 'M2_MoM', 'M2_YoY', 'Yield_10Y_Rate', 
                     'Breakeven_5Y5Y', 'M2_Liquidity_Spike', 'MacroEvent_Flag',
                     'Sentiment', 'Halving_Cycle', 'EMA_90'],
        'sequence_length': 90,
        'description': 'Digital Gold & Cryptocurrency Leader'
    }
}

# Stock configurations
STOCK_TICKERS = {
    # Indices
    'SPY': {'name': 'S&P 500 ETF', 'sector': 'Index', 'color': '#4169E1'},
    'QQQ': {'name': 'Nasdaq 100 ETF', 'sector': 'Index', 'color': '#00CED1'},
    'DIA': {'name': 'Dow Jones ETF', 'sector': 'Index', 'color': '#4682B4'},
    
    # Magnificent 7
    'AAPL': {'name': 'Apple Inc.', 'sector': 'Technology', 'color': '#A2AAAD'},
    'MSFT': {'name': 'Microsoft Corp.', 'sector': 'Technology', 'color': '#00A4EF'},
    'GOOGL': {'name': 'Alphabet Inc.', 'sector': 'Technology', 'color': '#4285F4'},
    'AMZN': {'name': 'Amazon.com Inc.', 'sector': 'Consumer', 'color': '#FF9900'},
    'NVDA': {'name': 'NVIDIA Corp.', 'sector': 'Technology', 'color': '#76B900'},
    'META': {'name': 'Meta Platforms', 'sector': 'Technology', 'color': '#0668E1'},
    'TSLA': {'name': 'Tesla Inc.', 'sector': 'Automotive', 'color': '#CC0000'},
    
    # TSMC
    'TSM': {'name': 'Taiwan Semiconductor', 'sector': 'Technology', 'color': '#E60012'}
}

# Add stock configs to ASSETS
for ticker, info in STOCK_TICKERS.items():
    ASSETS[ticker.lower()] = {
        'name': info['name'],
        'ticker': ticker,
        'icon': '',
        'color': info['color'],
        'model_file': f'models/{ticker}_ultimate_model.keras',
        'scaler_file': f'models/{ticker}_scaler.pkl',
        'data_file': f'data/{ticker}_global_insights.csv',
        'news_file': f'data/latest_news_{ticker.lower()}.json',
        'features': [ticker, 'DXY', 'VIX', 'Yield_10Y', 'Oil_Price',
                     'CPI_MoM', 'PPI_MoM', 'PCE_MoM', 'NFP_Change',
                     'YieldCurve_10Y2Y', 'M2_MoM', 'M2_YoY', 'Yield_10Y_Rate', 
                     'Breakeven_5Y5Y', 'M2_Liquidity_Spike', 'MacroEvent_Flag',
                     'Sentiment', 'EMA_90'],
        'sequence_length': 60,
        'description': f"{info['sector']} - {info['name']}"
    }

# ==================== FORECAST SETTINGS ====================

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
# Run: python scripts/backtest_engine.py [asset] to generate real scores.
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
# because compounding error makes long-range confidence inherently lower
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
    The backtest engine (scripts/backtest_engine.py) saves:
        reports/backtest_{asset_key}.json
    with field 'hit_ratio_3layer' (e.g., 63.5 means 63.5% directional accuracy).

    Converts hit_ratio → score:
        score = hit_ratio_3layer / 100 * decay_factor

    Falls back to CONFIDENCE_SCORES_FALLBACK if no backtest JSON found.

    Args:
        asset_key : e.g. 'gold', 'btc', 'spy'
        timeframe : e.g. '1 Day', '1 Week', '3 Months'

    Returns:
        dict: {'score': float, 'label': str, 'color': str}
    """
    import json as _json
    import os as _os

    asset_type = 'gold' if asset_key == 'gold' else ('btc' if asset_key == 'btc' else 'stocks')
    fallback = CONFIDENCE_SCORES_FALLBACK.get(asset_type, {}).get(
        timeframe, {'score': 0.50, 'label': 'Unknown', 'color': 'info'}
    ).copy()

    # Try loading backtest JSON for this specific asset first
    json_path = f'reports/backtest_{asset_key}.json'
    if not _os.path.exists(json_path):
        # Try asset_type grouping (e.g. 'stocks' covers all tickers)
        json_path = f'reports/backtest_{asset_type}.json'

    if not _os.path.exists(json_path):
        return fallback

    try:
        with open(json_path, 'r') as fh:
            data = _json.load(fh)

        raw_hit_ratio = data.get('hit_ratio_3layer', None)
        if raw_hit_ratio is None:
            return fallback

        # Convert percentage to [0, 1] and apply horizon decay
        base_score   = raw_hit_ratio / 100.0
        decay_factor = _TIMEFRAME_DECAY.get(timeframe, 0.70)
        final_score  = base_score * decay_factor

        result = _score_to_label(final_score)
        # 3 Months is always Speculative regardless of score
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

# Legacy alias — kept for backward compatibility.
# New code should call get_dynamic_confidence() directly.
CONFIDENCE_SCORES = CONFIDENCE_SCORES_FALLBACK

# ==================== HELPER FUNCTIONS ====================

def get_asset_config(asset_key):
    """Retrieve config for a specific asset"""
    return ASSETS.get(asset_key.lower())

def get_all_stock_tickers():
    """Return list of all supported stock tickers"""
    return list(STOCK_TICKERS.keys())

def check_model_exists(asset_key):
    """Check if model exists for asset"""
    config = get_asset_config(asset_key)
    if not config: return False
    return os.path.exists(config['model_file'])

def check_data_exists(asset_key):
    """Check if data exists for asset"""
    config = get_asset_config(asset_key)
    if not config: return False
    return os.path.exists(config['data_file'])

def get_asset_status(asset_key=None):
    """
    Get sync and training status.
    If asset_key is provided, returns string status for that asset.
    If asset_key is None, returns dictionary with full system status.
    """
    if asset_key:
        # Single asset check
        has_model = check_model_exists(asset_key)
        has_data = check_data_exists(asset_key)
        
        if has_model and has_data:
            return "READY"
        elif has_data:
            return "NEEDS TRAINING"
        else:
            return "NEEDS SYNC"
    else:
        # Full system status (for Dashboard/Settings)
        status = {}
        # Gold & BTC
        for asset in ['gold', 'btc']:
            status[asset] = {
                'data': check_data_exists(asset),
                'model': check_model_exists(asset)
            }
        # Stocks
        for ticker in get_all_stock_tickers():
            key = ticker.lower()
            status[key] = {
                'data': check_data_exists(key),
                'model': check_model_exists(key)
            }
        return status
