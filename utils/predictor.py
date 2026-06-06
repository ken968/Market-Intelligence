"""
Unified Prediction Engine for Multi-Asset Terminal
Handles forecasting for Gold, Bitcoin, and all US Stocks
This is now a Facade that delegates to predictor_data.py and predictor_engine.py
"""

import pandas as pd
from typing import Dict, Any, List, Optional, Union

from utils.config import get_asset_config
from utils.predictor_data import DataHandler
from utils.predictor_engine import ForecastEngine

class AssetPredictor:
    """Universal predictor for all asset types (Facade)"""
    
    def __init__(self, asset_key: str):
        """
        Initialize predictor for specific asset
        
        Args:
            asset_key (str): 'gold', 'btc', or stock ticker (e.g., 'aapl')
        """
        self.asset_key = asset_key.lower()
        self.config = get_asset_config(self.asset_key)
        
        if not self.config:
            raise ValueError(f"Unknown asset: {asset_key}")
        
        self.data_handler = DataHandler(self.asset_key, self.config)
        self.engine = ForecastEngine(self.asset_key, self.config, self.data_handler)
    
    # ── Property Delegation (for compatibility with scripts that directly access .data or .model) ──
    @property
    def data(self):
        return self.data_handler.data
        
    @data.setter
    def data(self, value):
        self.data_handler.data = value
        
    @property
    def scaler(self):
        return self.engine.scaler

    @scaler.setter
    def scaler(self, value):
        self.engine.scaler = value
        
    @property
    def model(self):
        return self.engine.model

    @model.setter
    def model(self, value):
        self.engine.model = value
        
    @property
    def is_loaded(self):
        return self.engine.is_loaded

    @is_loaded.setter
    def is_loaded(self, value):
        self.engine.is_loaded = value

    # ── Method Delegation ──
    def load_model(self) -> bool:
        """Load trained model and scaler"""
        return self.engine.load_model()
    
    def load_data(self) -> pd.DataFrame:
        """Load historical data"""
        return self.data_handler.load_data()
    
    def predict_next_step(self, sequence) -> float:
        """Predict next timestep given a sequence."""
        return self.engine.predict_next_step(sequence)
    
    def recursive_forecast(self, steps: int, ceo_drift_multiplier: float = 1.0) -> List[float]:
        """Generate multi-step forecast using recursive prediction."""
        return self.engine.recursive_forecast(steps, ceo_drift_multiplier)
    
    def get_multi_range_forecast(self, headlines: Optional[List[Dict]] = None, published_at_list: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate forecasts for all predefined ranges with confidence scores."""
        return self.engine.get_multi_range_forecast(headlines, published_at_list)
    
    def get_latest_price(self) -> float:
        """Get the most recent actual price"""
        return self.data_handler.get_latest_price()

    def ensemble_forecast(self) -> Dict[str, Any]:
        """Generate 7-day % change forecast using Dual-Head Stacker."""
        return self.engine.ensemble_forecast()

    def pct_chain_forecast(self, steps: int, ceo_drift_multiplier: float = 1.0, ensemble_7d_pct: float = None) -> list:
        """Build a realistic multi-step price path blending Stacker and LSTM."""
        return self.engine.pct_chain_forecast(steps, ceo_drift_multiplier, ensemble_7d_pct)

    def predict_tomorrow(self) -> Dict[str, Union[float, str]]:
        """Quick prediction for next day only."""
        return self.engine.predict_tomorrow()
    
    def predict_week(self) -> Dict[str, Union[float, str, bool]]:
        """Quick prediction for 1 week (7 days) ahead."""
        return self.engine.predict_week()


def batch_predict_tomorrow(asset_keys):
    """
    Predict tomorrow's price for multiple assets
    """
    results = {}
    for key in asset_keys:
        try:
            predictor = AssetPredictor(key)
            pred = predictor.predict_tomorrow()
            if isinstance(pred, dict):
                results[key] = pred
            else:
                results[key] = {'error': 'Prediction returned non-dict value'}
        except Exception as e:
            results[key] = {'error': str(e)}
    return results


def batch_predict_week(asset_keys):
    """
    Predict 1 week ahead price for multiple assets
    """
    results = {}
    for key in asset_keys:
        try:
            predictor = AssetPredictor(key)
            results[key] = predictor.predict_week()
        except Exception as e:
            results[key] = {'error': str(e)}
    return results


def batch_multi_range_forecast(asset_keys):
    """
    Generate multi-range forecasts for multiple assets
    """
    results = {}
    for key in asset_keys:
        try:
            predictor = AssetPredictor(key)
            fetched_forecasts = predictor.get_multi_range_forecast()
            
            if isinstance(fetched_forecasts, dict):
                results[key] = fetched_forecasts
            else:
                results[key] = {'Current': 0, 'error': 'Invalid data format'}
            
            results[key]['Current'] = predictor.get_latest_price()
        except Exception as e:
            results[key] = {'error': str(e)}
    return results


def get_forecast_dataframe(asset_key):
    """
    Generate forecast as a formatted DataFrame for display
    """
    predictor = AssetPredictor(asset_key)
    forecasts = predictor.get_multi_range_forecast()
    
    df = pd.DataFrame({
        'Timeframe': list(forecasts.keys()),
        'Predicted Price': list(forecasts.values())
    })
    return df


if __name__ == "__main__":
    # Test prediction for Gold
    print("Testing Gold Predictor...")
    gold = AssetPredictor('gold')
    
    try:
        tomorrow = gold.predict_tomorrow()
        print(f"Current: ${tomorrow['current']:,.2f}")
        print(f"Tomorrow: ${tomorrow['predicted']:,.2f}")
        print(f"Change: ${tomorrow['change']:+,.2f} ({tomorrow['pct_change']:+.2f}%)")
        
        print("\nMulti-range forecast:")
        forecasts = gold.get_multi_range_forecast()
        for range_name, price in forecasts.items():
            if isinstance(price, dict) and 'price' in price:
                print(f"{range_name:12s}: ${price['price']:,.2f}")
            else:
                print(f"{range_name:12s}: ${price:,.2f} (raw)")
            
    except Exception as e:
        print(f"Error: {e}")
