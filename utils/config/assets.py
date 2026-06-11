"""
XAUUSD Multi-Asset Terminal - Asset Configuration
"""

import os

ASSETS = {
    'gold': {
        'name': 'Gold (XAUUSD)',
        'ticker': 'GC=F',
        'icon': '',
        'color': '#FFD700',
        'model_file': 'models/gold_ultimate_model.keras',
        'scaler_file': 'models/gold_scaler.pkl',
        'data_file': 'data/gold_global_insights.csv',
        'news_file': 'data/latest_news_gold.json',
        'features': ['Gold', 'DXY', 'VIX', 'Yield_10Y', 'Oil_Price',
                     'CPI_MoM', 'PPI_MoM', 'PCE_MoM', 'NFP_Change',
                     'YieldCurve_10Y2Y', 'M2_MoM', 'M2_YoY', 'Yield_10Y_Rate',
                     'Breakeven_5Y5Y', 'M2_Liquidity_Spike', 'MacroEvent_Flag',
                     'Credit_Spread',
                     'Sentiment', 'EMA_90', 
                     # Phase 3: Dynamic Regime Features
                     'vix_percentile_252d',   # VIX rolling 252d eCDF (0.0-1.0)
                     'roll_corr_dxy_90d',     # Gold vs DXY rolling 90d corr
                     'return_zscore_90d'],    # Micro circuit-breaker Z-Score
        'sequence_length': 90,
        # Phase 7: Reduced from [128,64,32]+attention to [64,32] — model was overparameterized
        # ~155k params with ~449 samples/window = ratio 0.003 (too complex)
        # ~18k params with ~449 samples/window = ratio 0.025 (acceptable)
        'model_arch': {'units': [64, 32], 'dropout': 0.2, 'attention': False},
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
                     'Credit_Spread',
                     'Sentiment', 'Halving_Cycle', 'EMA_90', 
                     # Phase 3: Dynamic Regime Features
                     'vix_percentile_252d',   # VIX rolling 252d eCDF (0.0-1.0)
                     'roll_corr_spy_90d',     # BTC vs SPY rolling 90d corr (decoupling detector)
                     'return_zscore_90d'],    # Micro circuit-breaker Z-Score
        'sequence_length': 90,
        # Phase 7: Reduced from [128,64,32]+attention to [64,32] — same reason as gold
        # BTC has more data (4283 rows) so slightly more dropout to prevent overfitting
        'model_arch': {'units': [64, 32], 'dropout': 0.25, 'attention': False},
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

# Volatile stocks need deeper LSTM + higher dropout + attention
# Stable indices need smaller, less prone to overfitting
VOLATILE_STOCKS = {'NVDA', 'TSLA', 'META', 'AMZN'}  # High β, sensitive to macro
STABLE_INDICES  = {'SPY', 'DIA', 'QQQ'}              # Low β, broad market

# Add stock configs to ASSETS
for ticker, info in STOCK_TICKERS.items():
    is_volatile = ticker in VOLATILE_STOCKS
    is_index    = ticker in STABLE_INDICES
    if is_volatile:
        arch = {'units': [128, 64], 'dropout': 0.35, 'attention': True}
    elif is_index:
        arch = {'units': [64, 32],  'dropout': 0.20, 'attention': False}
    else:
        arch = {'units': [100, 50], 'dropout': 0.25, 'attention': False}  # default Mag7

    if ticker.lower() == 'spy':
        roll_corr_feat = 'roll_corr_qqq_90d'
    elif ticker.lower() == 'qqq':
        roll_corr_feat = 'roll_corr_dia_90d'
    elif ticker.lower() == 'dia':
        roll_corr_feat = 'roll_corr_spy_90d'
    else:
        roll_corr_feat = 'roll_corr_spy_90d'

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
                     'Credit_Spread',
                     'Sentiment', 'EMA_90', 
                     # Phase 3: Dynamic Regime Features
                     'vix_percentile_252d',   # VIX rolling 252d eCDF (0.0-1.0)
                     roll_corr_feat,          # Dynamic correlation feature
                     'return_zscore_90d'],    # Micro circuit-breaker Z-Score
        'sequence_length': 90,
        'model_arch': arch,
        'description': f"{info['sector']} - {info['name']}"
    }

def get_asset_config(asset_key: str) -> dict:
    """Retrieve config for a specific asset"""
    return ASSETS.get(asset_key.lower())

def get_all_stock_tickers() -> list:
    """Return list of all supported stock tickers"""
    return list(STOCK_TICKERS.keys())

def check_model_exists(asset_key: str) -> bool:
    """Check if model exists for asset"""
    config = get_asset_config(asset_key)
    if not config: return False
    return os.path.exists(config['model_file'])

def check_data_exists(asset_key: str) -> bool:
    """Check if data exists for asset"""
    config = get_asset_config(asset_key)
    if not config: return False
    return os.path.exists(config['data_file'])

def get_asset_status(asset_key: str = None) -> dict | str:
    """
    Get sync and training status.
    If asset_key is provided, returns string status for that asset.
    If asset_key is None, returns dictionary with full system status.
    """
    if asset_key:
        has_model = check_model_exists(asset_key)
        has_data = check_data_exists(asset_key)
        
        if has_model and has_data:
            return "READY"
        elif has_data:
            return "NEEDS TRAINING"
        else:
            return "NEEDS SYNC"
    else:
        status = {}
        for asset in ['gold', 'btc']:
            status[asset] = {
                'data': check_data_exists(asset),
                'model': check_model_exists(asset)
            }
        for ticker in get_all_stock_tickers():
            key = ticker.lower()
            status[key] = {
                'data': check_data_exists(key),
                'model': check_model_exists(key)
            }
        return status
