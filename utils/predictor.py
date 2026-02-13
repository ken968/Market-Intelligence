"""
Unified Prediction Engine for Multi-Asset Terminal
Handles forecasting for Gold, Bitcoin, and all US Stocks
"""

import numpy as np
import pandas as pd
import pickle
import os
import sys
from utils.config import get_asset_config, FORECAST_RANGES, CONFIDENCE_SCORES

# Handle TensorFlow incompatibility (e.g., Python 3.14)
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    # Fallback for systems where TensorFlow cannot be installed
    TF_AVAILABLE = False
    print("Warning: TensorFlow not found. AI predictions will be disabled.")


def get_confidence_score(asset_key, timeframe):
    """
    Get confidence score for asset and timeframe
    
    Args:
        asset_key (str): Asset identifier (e.g., 'gold', 'btc', 'aapl')
        timeframe (str): Forecast timeframe (e.g., '1 Week', '1 Month')
    
    Returns:
        dict: {'score': float, 'label': str, 'color': str}
    """
    # Determine asset type
    if asset_key == 'gold':
        asset_type = 'gold'
    elif asset_key == 'btc':
        asset_type = 'btc'
    else:
        asset_type = 'stocks'
    
    return CONFIDENCE_SCORES.get(asset_type, {}).get(timeframe, {
        'score': 0.50, 'label': 'Unknown', 'color': 'info'
    })


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
        if not TF_AVAILABLE:
            return False

        model_path = self.config['model_file']
        scaler_path = self.config['scaler_file']
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        try:
            self.model = load_model(model_path)
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
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
        if not TF_AVAILABLE:
            # Return last value in sequence as prediction (no change)
            return sequence[0, -1, 0]

        if not self.is_loaded:
            success = self.load_model()
            if not success:
               return sequence[0, -1, 0]
        
        prediction = self.model.predict(sequence, verbose=0)
        return prediction[0, 0]
    
    def recursive_forecast(self, steps):
        """
        Generate multi-step forecast using recursive prediction with feature drift
        
        Args:
            steps (int): Number of steps to predict
        
        Returns:
            list: Predicted values (in original scale)
        """
        if not TF_AVAILABLE:
             # Return empty list or handle in get_multi_range_forecast
             return []

        if not self.is_loaded:
            success = self.load_model()
            if not success:
                return []
        
        if self.data is None:
            self.load_data()
        
        # Calculate historical averages for features (to use as drift targets)
        # We use the loaded data to find the "normal" state of market indicators
        feature_means = np.mean(self.data, axis=0)
        
        # Normalize data
        scaled_data = self.scaler.transform(self.data)
        scaled_means = self.scaler.transform(feature_means.reshape(1, -1))[0]
        
        # Get initial sequence
        seq_len = self.config['sequence_length']
        current_batch = scaled_data[-seq_len:].reshape(1, seq_len, len(self.config['features']))
        
        predictions = []
        temp_data = current_batch.copy()
        
        for i in range(steps):
            # Predict next step
            pred_scaled = self.predict_next_step(temp_data)
            
            # 1. Get current state info
            prev_frame = temp_data[0, -1, :].copy()
            prev_price_scaled = prev_frame[0]
            start_price_scaled = current_batch[0, -1, 0]
            
            # 2. Adaptive Recursive Damping
            # Short-term (1-5 days) = Model-heavy
            # Long-term (Months) = Stable/Anchor-heavy
            
            ai_delta = pred_scaled - prev_price_scaled
            
            # TIGHTER CLIP: max 0.8% per step in scaled space for stability
            # This prevents the AI from generating high-momentum 'crashes' or 'moons' in isolation
            ai_delta = np.clip(ai_delta, -0.008, 0.008)
            
            # Convergence Factor: Decay trust linearly
            # For 365 steps (1 year), we want trust to be 0.05 at the end (highly anchored)
            # UPDATED: Extended to 365.0 to allow trends to persist for the full year
            trust_factor = max(0.05, 1.0 - (i / 365.0))
            
            # Damping: apply decay to prevent runaway trends
            # Decay determines how fast the AI 'loses confidence' in the trend
            # Relaxed for better long-term trend expression
            decay = 0.99 if self.asset_key != 'btc' else 0.97
            
            ai_movement = (ai_delta * decay * trust_factor)
            
            # STRONGER ANCHOR SPRING (OPTIMIZED):
            # Pull back reduced to 1% (was 5%) to allow more dynamic movement
            # while still preventing runaway hallucinations.
            anchor_pull = (start_price_scaled - prev_price_scaled) * (1.0 - trust_factor) * 0.01
            
            new_price_scaled = prev_price_scaled + ai_movement + anchor_pull
            
            # 3. Build the next frame
            last_frame = prev_frame.copy()
            last_frame[0] = new_price_scaled
            
            # 4. Update Dynamic Features (EMA, Halving, etc.)
            features = self.config['features']
            
            for f_idx in range(1, len(last_frame)):
                f_name = features[f_idx]
                
                if f_name == 'EMA_90':
                    alpha = 2.0 / (90.0 + 1.0)
                    last_frame[f_idx] = (new_price_scaled * alpha) + (prev_frame[f_idx] * (1.0 - alpha))
                
                elif f_name == 'Halving_Cycle':
                    # Decrement days by 1 (adjusted for scaler scale_)
                    last_frame[f_idx] = max(0, prev_frame[f_idx] - self.scaler.scale_[f_idx])
                
                elif f_name in ['Sentiment', 'DXY', 'VIX', 'Yield_10Y']:
                    # Macro indicators drift to their historical means
                    # UPDATED: Slower drift (0.002) allows current market "vision"/news to persist longer
                    drift_rate = 0.002
                    target = scaled_means[f_idx]
                    last_frame[f_idx] = last_frame[f_idx] + (target - last_frame[f_idx]) * drift_rate
            
            # Shift window and add new step
            new_batch = np.append(temp_data[:, 1:, :], [[last_frame]], axis=1)
            temp_data = new_batch
            
            predictions.append(new_price_scaled)
        
        # Inverse transform predictions
        n_features = len(self.config['features'])
        dummy = np.zeros((len(predictions), n_features))
        dummy[:, 0] = predictions
        
        predictions_original = self.scaler.inverse_transform(dummy)[:, 0]
        
        # Final safety check: no price below floor (20% of start price)
        start_price = self.scaler.inverse_transform(current_batch[0, -1, :].reshape(1, -1))[0, 0]
        predictions_original = np.maximum(predictions_original, start_price * 0.2) 
        
        return predictions_original.tolist()
    
    def get_multi_range_forecast(self):
        """
        Generate forecasts for all predefined ranges with confidence scores
        
        Returns:
            dict: {'Current': current_price, range_name: {'price': float, 'confidence': dict}, ...}
        """
        current_price = self.get_latest_price()
        results = {'Current': current_price}
        
        for label, steps in FORECAST_RANGES.items():
            forecast = self.recursive_forecast(steps)
            confidence = get_confidence_score(self.asset_key, label)
            
            if forecast:
                price = forecast[-1]
            else:
                price = current_price

            results[label] = {
                'price': price,
                'confidence': confidence
            }
        
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
        
        if not forecast:
            return {
                'current': current_price,
                'predicted': current_price,
                'change': 0.0,
                'pct_change': 0.0,
                'direction': 'flat',
                'error': 'AI Model Unavailable'
            }
            
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
    
    def predict_week(self):
        """
        Quick prediction for 1 week (7 days) ahead
        
        Returns:
            dict: {'current': float, 'predicted': float, 'change': float, 'pct_change': float}
        """
        current_price = self.get_latest_price()
        forecast = self.recursive_forecast(7)
        
        if not forecast:
            return {
                'current': current_price,
                'predicted': current_price,
                'change': 0.0,
                'pct_change': 0.0,
                'direction': 'flat',
                'error': 'AI Model Unavailable'
            }

        predicted_price = forecast[-1]  # Take the 7th day prediction
        
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


def batch_predict_week(asset_keys):
    """
    Predict 1 week ahead price for multiple assets
    
    Args:
        asset_keys (list): List of asset keys
    
    Returns:
        dict: {asset_key: prediction_dict}
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
