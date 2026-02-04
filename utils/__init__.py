"""
XAUUSD Multi-Asset Terminal - Utilities Package
Shared components for prediction, UI, and configuration
"""

__version__ = "2.0.0"
__author__ = "Ken968"

from .config import (
    ASSETS,
    STOCK_TICKERS,
    THEME,
    FORECAST_RANGES,
    get_asset_config,
    get_all_stock_tickers,
    check_model_exists,
    check_data_exists,
    get_asset_status
)

from .predictor import AssetPredictor, batch_predict_tomorrow, get_forecast_dataframe

from .ui_components import (
    inject_custom_css,
    render_metric_card,
    render_news_section,
    render_status_badge,
    create_price_chart,
    create_multi_asset_comparison,
    create_forecast_chart,
    render_prediction_table,
    render_page_header
)

__all__ = [
    # Config
    'ASSETS',
    'STOCK_TICKERS',
    'THEME',
    'FORECAST_RANGES',
    'get_asset_config',
    'get_all_stock_tickers',
    'check_model_exists',
    'check_data_exists',
    'get_asset_status',
    
    # Predictor
    'AssetPredictor',
    'batch_predict_tomorrow',
    'get_forecast_dataframe',
    
    # UI Components
    'inject_custom_css',
    'render_metric_card',
    'render_news_section',
    'render_status_badge',
    'create_price_chart',
    'create_multi_asset_comparison',
    'create_forecast_chart',
    'render_prediction_table',
    'render_page_header'
]
