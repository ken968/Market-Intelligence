"""
Unified Prediction Engine for Multi-Asset Terminal
Handles forecasting for Gold, Bitcoin, and all US Stocks
"""

import numpy as np
import pandas as pd
import pickle
import os
from tensorflow.keras.models import load_model
from utils.config import get_asset_config, FORECAST_RANGES

class AssetPredictor:
    """Universal predictor for all asset types"""
    
    def __init__(self, asset_key):
        """
        Initialize predictor for specific asset
        
        Args:
            asset_key (str): 'gold', 'btc', or stock ticker (e.g., 'aapl')
        """
        self.asset_key = asset_key.lower()
        self.config = get_asset_config(self.asset_key)
        
        if not self.config:
            raise ValueError(f"Unknown asset: {asset_key}")
        
        self.model = None
        self.scaler = None
        self.data = None
        self.is_loaded = False
    
    def load_model(self):
        """Load trained model and scaler"""
        model_path = self.config['model_file']
        scaler_path = self.config['scaler_file']
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        self.model = load_model(model_path)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.is_loaded = True
        return True
    
    def load_data(self):
        """Load historical data"""
        data_path = self.config['data_file']
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data not found: {data_path}")
        
        df = pd.read_csv(data_path)
        
        # Ensure all required features exist
        features = self.config['features']
        missing = [f for f in features if f not in df.columns]
        
        if missing:
            # Try to add missing features with defaults
            for feat in missing:
                if feat == 'Sentiment':
                    df['Sentiment'] = 0
                elif feat == 'Halving_Cycle':
                    df['Halving_Cycle'] = 0
                else:
                    raise ValueError(f"Missing required feature: {feat}")
        
        self.data = df[features].values
        return df
    
    def predict_next_step(self, sequence):
        """
        Predict next timestep given a sequence
        
        Args:
            sequence (np.array): Shape (1, seq_len, n_features)
        
        Returns:
            float: Predicted value (scaled)
        """
        if not self.is_loaded:
            self.load_model()
        
        prediction = self.model.predict(sequence, verbose=0)
        return prediction[0, 0]
    
    def recursive_forecast(self, steps):
        """
        Generate multi-step forecast using recursive prediction
        
        Args:
            steps (int): Number of steps to predict
        
        Returns:
            list: Predicted values (in original scale)
        """
        if not self.is_loaded:
            self.load_model()
        
        if self.data is None:
            self.load_data()
        
        # Normalize data
        scaled_data = self.scaler.transform(self.data)
        
        # Get initial sequence
        seq_len = self.config['sequence_length']
        current_batch = scaled_data[-seq_len:].reshape(1, seq_len, len(self.config['features']))
        
        predictions = []
        temp_data = current_batch.copy()
        
        for _ in range(steps):
            # Predict next step
            pred_scaled = self.predict_next_step(temp_data)
            
            # Create new frame (assume other features stay constant)
            last_frame = temp_data[0, -1, :].copy()
            last_frame[0] = pred_scaled  # Update only price (first feature)
            
            # Shift window and add new prediction
            new_batch = np.append(temp_data[:, 1:, :], [[last_frame]], axis=1)
            temp_data = new_batch
            
            predictions.append(pred_scaled)
        
        # Inverse transform predictions
        n_features = len(self.config['features'])
        dummy = np.zeros((len(predictions), n_features))
        dummy[:, 0] = predictions
        
        predictions_original = self.scaler.inverse_transform(dummy)[:, 0]
        
        return predictions_original.tolist()
    
    def get_multi_range_forecast(self):
        """
        Generate forecasts for all predefined ranges
        
        Returns:
            dict: {range_name: predicted_price}
        """
        results = {}
        
        for label, steps in FORECAST_RANGES.items():
            forecast = self.recursive_forecast(steps)
            results[label] = forecast[-1]  # Take last prediction for this range
        
        return results
    
    def get_latest_price(self):
        """Get the most recent actual price"""
        if self.data is None:
            df = self.load_data()
        else:
            df = pd.read_csv(self.config['data_file'])
        
        # First feature is always the price
        return df[self.config['features'][0]].iloc[-1]
    
    def predict_tomorrow(self):
        """
        Quick prediction for next day only
        
        Returns:
            dict: {'current': float, 'predicted': float, 'change': float, 'pct_change': float}
        """
        current_price = self.get_latest_price()
        forecast = self.recursive_forecast(1)
        predicted_price = forecast[0]
        
        change = predicted_price - current_price
        pct_change = (change / current_price) * 100
        
        return {
            'current': current_price,
            'predicted': predicted_price,
            'change': change,
            'pct_change': pct_change,
            'direction': 'up' if change > 0 else 'down'
        }


def batch_predict_tomorrow(asset_keys):
    """
    Predict tomorrow's price for multiple assets
    
    Args:
        asset_keys (list): List of asset keys
    
    Returns:
        dict: {asset_key: prediction_dict}
    """
    results = {}
    
    for key in asset_keys:
        try:
            predictor = AssetPredictor(key)
            results[key] = predictor.predict_tomorrow()
        except Exception as e:
            results[key] = {'error': str(e)}
    
    return results


def batch_multi_range_forecast(asset_keys):
    """
    Generate multi-range forecasts for multiple assets
    
    Args:
        asset_keys (list): List of asset keys
    
    Returns:
        dict: {asset_key: multi_range_dict}
    """
    results = {}
    
    for key in asset_keys:
        try:
            predictor = AssetPredictor(key)
            results[key] = predictor.get_multi_range_forecast()
            # Add current price for reference
            results[key]['Current'] = predictor.get_latest_price()
        except Exception as e:
            results[key] = {'error': str(e)}
    
    return results


def get_forecast_dataframe(asset_key):
    """
    Generate forecast as a formatted DataFrame for display
    
    Args:
        asset_key (str): Asset identifier
    
    Returns:
        pd.DataFrame: Formatted forecast table
    """
    predictor = AssetPredictor(asset_key)
    forecasts = predictor.get_multi_range_forecast()
    
    df = pd.DataFrame({
        'Timeframe': list(forecasts.keys()),
        'Predicted Price': list(forecasts.values())
    })
    
    return df


# ==================== EXAMPLE USAGE ====================

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
            print(f"{range_name:12s}: ${price:,.2f}")
            
    except Exception as e:
        print(f"Error: {e}")
