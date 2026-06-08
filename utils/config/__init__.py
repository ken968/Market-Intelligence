"""
Centralized Configuration Package for Market Intelligence
"""

import os
import logging
# Suppress TensorFlow C++ Info/Warning messages and oneDNN warnings globally
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Suppress Python-level TensorFlow warnings (like tf.function retracing)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

from .theme import THEME
from .assets import (
    ASSETS, 
    STOCK_TICKERS, 
    VOLATILE_STOCKS, 
    STABLE_INDICES,
    get_asset_config, 
    get_all_stock_tickers, 
    check_model_exists, 
    check_data_exists, 
    get_asset_status
)
from .forecast import (
    FORECAST_RANGES, 
    CONFIDENCE_SCORES_FALLBACK, 
    CONFIDENCE_SCORES,
    FORECAST_DISCLAIMER, 
    get_dynamic_confidence
)

# Export all explicitly
__all__ = [
    'THEME',
    'ASSETS',
    'STOCK_TICKERS',
    'VOLATILE_STOCKS',
    'STABLE_INDICES',
    'get_asset_config',
    'get_all_stock_tickers',
    'check_model_exists',
    'check_data_exists',
    'get_asset_status',
    'FORECAST_RANGES',
    'CONFIDENCE_SCORES_FALLBACK',
    'CONFIDENCE_SCORES',
    'FORECAST_DISCLAIMER',
    'get_dynamic_confidence'
]
