"""
XAUUSD Multi-Asset Terminal - Global Configuration
Centralized config for all assets, models, and UI settings
"""

import os

# ==================== ASSET CONFIGURATION ====================

ASSETS = {
    'gold': {
        'name': 'Gold (XAUUSD)',
        'ticker': 'GC=F',
        'icon': '🏆',
        'color': '#FFD700',
        'model_file': 'gold_ultimate_model.h5',
        'scaler_file': 'scaler.pkl',
        'data_file': 'gold_global_insights.csv',
        'news_file': 'latest_news.json',
        'features': ['Gold', 'DXY', 'VIX', 'Yield_10Y', 'Sentiment'],
        'sequence_length': 60,
        'description': 'Precious Metal & Safe Haven Asset'
    },
    'btc': {
        'name': 'Bitcoin',
        'ticker': 'BTC-USD',
        'icon': '₿',
        'color': '#F7931A',
        'model_file': 'btc_ultimate_model.h5',
        'scaler_file': 'btc_scaler.pkl',
        'data_file': 'btc_global_insights.csv',
        'news_file': 'latest_news_btc.json',
        'features': ['BTC', 'DXY', 'VIX', 'Yield_10Y', 'Sentiment', 'Halving_Cycle'],
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
        'icon': '📈',
        'color': info['color'],
        'sector': info['sector'],
        'model_file': f'{ticker}_ultimate_model.h5',
        'scaler_file': f'{ticker}_scaler.pkl',
        'data_file': f'{ticker}_global_insights.csv',
        'news_file': f'latest_news_{ticker.lower()}.json',
        'features': [ticker, 'DXY', 'VIX', 'Yield_10Y', 'Sentiment'],
        'sequence_length': 60,
        'description': f'{info["sector"]} - {info["name"]}'
    }

# ==================== UI THEME ====================

THEME = {
    'bg_deep': '#0B101B',
    'bg_surface': '#151B28',
    'accent': '#C5A059',
    'accent_muted': 'rgba(197, 160, 89, 0.2)',
    'border': '#232D3F',
    'text_primary': '#E2E8F0',
    'text_secondary': '#94A3B8',
    'success': '#00C076',
    'danger': '#FF4D4D',
    'warning': '#FFB020'
}

# ==================== PREDICTION RANGES ====================

FORECAST_RANGES = {
    "1 Day": 1,
    "1 Week": 5,
    "2 Weeks": 10,
    "1 Month": 21,
    "3 Months": 63,
    "6 Months": 126,
    "1 Year": 252
}

# ==================== TRAINING CONFIG ====================

TRAINING_PARAMS = {
    'gold': {
        'epochs': 30,
        'batch_size': 32,
        'lstm_units': [100, 50, 25],
        'dropout': 0.2
    },
    'btc': {
        'epochs': 50,
        'batch_size': 32,
        'lstm_units': [128, 64, 32],
        'dropout': 0.3
    },
    'stocks': {
        'epochs': 30,
        'batch_size': 32,
        'lstm_units': [100, 50, 25],
        'dropout': 0.2
    }
}

# ==================== HELPER FUNCTIONS ====================

def get_asset_config(asset_key):
    """Get configuration for specific asset"""
    return ASSETS.get(asset_key.lower(), None)

def get_all_stock_tickers():
    """Get list of all stock ticker symbols"""
    return list(STOCK_TICKERS.keys())

def check_model_exists(asset_key):
    """Check if trained model exists for asset"""
    config = get_asset_config(asset_key)
    if not config:
        return False
    return os.path.exists(config['model_file']) and os.path.exists(config['scaler_file'])

def check_data_exists(asset_key):
    """Check if data file exists for asset"""
    config = get_asset_config(asset_key)
    if not config:
        return False
    return os.path.exists(config['data_file'])

def get_asset_status():
    """Get status of all assets (data & model availability)"""
    status = {}
    for key in ['gold', 'btc'] + [t.lower() for t in STOCK_TICKERS.keys()]:
        status[key] = {
            'data': check_data_exists(key),
            'model': check_model_exists(key)
        }
    return status

# ==================== NEWS API CONFIG ====================

NEWS_API_KEY = 'cb548b26fc6542c0a6bb871ef3786eba'
TRUSTED_DOMAINS = (
    "bloomberg.com,reuters.com,cnbc.com,wsj.com,finance.yahoo.com,"
    "investing.com,marketwatch.com,economist.com,ft.com,coindesk.com,cointelegraph.com"
)

# ==================== MACRO INDICATORS ====================

MACRO_INDICATORS = {
    'DXY': {'name': 'US Dollar Index', 'ticker': 'DX-Y.NYB'},
    'VIX': {'name': 'Volatility Index', 'ticker': '^VIX'},
    'Yield_10Y': {'name': 'US 10Y Treasury', 'ticker': '^TNX'}
}
