"""
XAUUSD Multi-Asset Terminal - Global Configuration
Centralized config for all assets, models, and UI settings
"""

import os

# ==================== UI THEME CONFIGURATION ====================

THEME = {
    'primary': '#00A8E8',
    'secondary': '#CCCCCC',
    'background': '#0E1117',
    'card_bg': '#262730',
    'success': '#00CC96',
    'danger': '#EF553B',
    'warning': '#FFA15A',
    'info': '#636EFA',
    'font': 'Inter, sans-serif'
}

# ==================== ASSET CONFIGURATION ====================

ASSETS = {
    'gold': {
        'name': 'Gold (XAUUSD)',
        'ticker': 'GC=F',
        'icon': '🏆',
        'color': '#FFD700',
        'model_file': 'models/gold_ultimate_model.keras',
        'scaler_file': 'models/scaler.pkl',
        'data_file': 'data/gold_global_insights.csv',
        'news_file': 'data/latest_news_gold.json',
        'features': ['Gold', 'DXY', 'VIX', 'Yield_10Y', 'Sentiment', 'EMA_90'],
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
        'features': ['BTC', 'DXY', 'VIX', 'Yield_10Y', 'Sentiment', 'Halving_Cycle', 'EMA_90'],
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
        'model_file': f'models/{ticker}_ultimate_model.keras',
        'scaler_file': f'models/{ticker}_scaler.pkl',
        'data_file': f'data/{ticker}_global_insights.csv',
        'news_file': f'data/latest_news_{ticker.lower()}.json',
        'features': [ticker, 'DXY', 'VIX', 'Yield_10Y', 'Sentiment', 'EMA_90'],
        'sequence_length': 60,
        'description': f"{info['sector']} - {info['name']}"
    }

# ==================== FORECAST SETTINGS ====================

FORECAST_RANGES = {
    '1 Day': 1,
    '1 Week': 7,
    '2 Weeks': 14,
    '1 Month': 30,
    '3 Months': 90,
    '6 Months': 180,
    '1 Year': 365
}

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

def get_asset_status(asset_key):
    """Get sync and training status"""
    has_model = check_model_exists(asset_key)
    has_data = check_data_exists(asset_key)
    
    if has_model and has_data:
        return "READY"
    elif has_data:
        return "NEEDS TRAINING"
    else:
        return "NEEDS SYNC"
