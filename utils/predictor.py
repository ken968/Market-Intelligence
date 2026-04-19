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
    TF_AVAILABLE = False
    print("Warning: TensorFlow not found. AI predictions will be disabled.")

# CEO Layer optional import
try:
    from utils.llm_manager import compute_drift_multiplier
    from utils.macro_processor import build_macro_context
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


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
        # Check if all required features are present
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features in {self.asset_key}: {missing_features}")
            
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
        
        # Feature count safety check (migration support)
        if hasattr(self.scaler, 'n_features_in_'):
            expected_n = self.scaler.n_features_in_
            if sequence.shape[2] != expected_n:
                # Slice to expected number of features
                sequence = sequence[:, :, :expected_n]
        # Run model prediction
        try:
            prediction = self.model.predict(sequence, verbose=0)
            return float(prediction[0, 0])
        except Exception:
            return float(sequence[0, -1, 0])  # Return baseline on failure
    
    def recursive_forecast(self, steps, ceo_bias_vector=None, ceo_drift_multiplier=1.0):
        """
        Generate multi-step forecast using recursive prediction with:
        - Weighted Multi-Scale Anchor (90-day short-term + full-history long-term)
        - CEO Layer drift multiplier injection

        Args:
            steps                : Number of steps to predict
            ceo_bias_vector      : np.ndarray from llm_manager — CEO contextual bias
            ceo_drift_multiplier : float from llm_manager — shifts anchor weight

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

        # -----------------------------------------------------------------------
        # Weighted Multi-Scale Anchor (Level 2 — Manager Layer)
        # Short-term (90 days): captures current market regime/momentum
        # Long-term (full history): provides structural pattern recognition
        # -----------------------------------------------------------------------
        short_window = min(90, len(self.data))
        long_window = len(self.data)

        feature_means_short = np.mean(self.data[-short_window:], axis=0)  # 90-day mean
        feature_means_long = np.mean(self.data, axis=0)                   # Full history mean

        # CEO Layer modulates the balance: higher drift_multiplier = keep current regime longer
        # Clamp multiplier in [0.85, 1.15] to prevent runaway bias
        dm = max(0.85, min(1.15, ceo_drift_multiplier))

        # Short-term weight increases when CEO signals strong current regime (dm > 1.0)
        # and decreases when CEO is neutral/bearish (dm < 1.0)
        w_short = 0.40 * dm   # 40% base weight on 90-day mean, scaled by CEO
        w_long = 1.0 - w_short

        feature_means = w_short * feature_means_short + w_long * feature_means_long
        
        # Normalize data with feature count safety
        data_to_scale = self.data
        means_to_scale = feature_means
        
        if hasattr(self.scaler, 'n_features_in_'):
            expected_n = self.scaler.n_features_in_
            data_to_scale = data_to_scale[:, :expected_n]
            means_to_scale = means_to_scale[:expected_n]

        scaled_data = self.scaler.transform(data_to_scale)
        scaled_means = self.scaler.transform(means_to_scale.reshape(1, -1))[0]
        
        # Get initial sequence
        seq_len = self.config['sequence_length']
        n_scaled_features = scaled_data.shape[1]
        current_batch = scaled_data[-seq_len:].reshape(1, seq_len, n_scaled_features)
        
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
                
                elif f_name in ['Sentiment', 'DXY', 'VIX', 'Yield_10Y', 'Oil_Price']:
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
        n_features = self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else len(self.config['features'])
        dummy = np.zeros((len(predictions), n_features))
        dummy[:, 0] = predictions
        
        predictions_original = self.scaler.inverse_transform(dummy)[:, 0]
        
        # Final safety check: no price below floor (20% of start price)
        start_price = self.scaler.inverse_transform(current_batch[0, -1, :].reshape(1, -1))[0, 0]
        predictions_original = np.maximum(predictions_original, start_price * 0.2) 
        
        return predictions_original.tolist()
    
    def get_multi_range_forecast(self, headlines=None, published_at_list=None):
        """
        Generate forecasts for all predefined ranges with confidence scores.
        Automatically invokes the CEO Layer (Gemini) if available.

        Args:
            headlines       : Optional list of news headlines for CEO Layer
            published_at_list: Optional list of ISO datetimes for staleness filtering

        Returns:
            dict with 'Current', forecast ranges, and 'ceo_context' metadata
        """
        current_price = self.get_latest_price()
        results = {'Current': current_price}

        # -----------------------------------------------------------------------
        # Level 3 — CEO Layer: Get Gemini contextual bias
        # -----------------------------------------------------------------------
        ceo_drift_multiplier = 1.0
        ceo_bias_vector = None
        ceo_context = {'is_fallback': True, 'narrative': 'CEO Layer not active.', 'confidence': 0.0}

        if LLM_AVAILABLE and headlines:
            try:
                from utils.llm_manager import analyze_news_context
                from utils.macro_processor import build_macro_context

                macro_ctx = build_macro_context()
                macro_summary = macro_ctx.get('macro_summary', '')

                analysis = analyze_news_context(
                    headlines=headlines,
                    macro_summary=macro_summary,
                    published_at_list=published_at_list,
                )

                ceo_bias_vector = analysis['bias_vector']
                ceo_drift_multiplier = compute_drift_multiplier(
                    ceo_bias_vector,
                    asset_type='gold' if self.asset_key == 'gold' else
                               'btc'  if self.asset_key == 'btc' else 'stocks'
                )
                ceo_context = {
                    'is_fallback':     analysis['is_fallback'],
                    'narrative':       analysis['narrative'],
                    'dominant_regime': analysis['dominant_regime'],
                    'confidence':      analysis['confidence'],
                    'drift_multiplier': ceo_drift_multiplier,
                    'headlines_used':  analysis['headlines_used'],
                    'macro_summary':   macro_summary,
                }
            except Exception as e:
                print(f"Warning: CEO Layer error for {self.asset_key}: {e}. Using baseline.")

        results['ceo_context'] = ceo_context

        # -----------------------------------------------------------------------
        # Level 1+2 — Run forecasts for all ranges with CEO multiplier injected
        # -----------------------------------------------------------------------
        for label, steps in FORECAST_RANGES.items():
            baseline = self.recursive_forecast(steps, ceo_drift_multiplier=1.0)
            contextual = self.recursive_forecast(steps,
                                                  ceo_bias_vector=ceo_bias_vector,
                                                  ceo_drift_multiplier=ceo_drift_multiplier)
            confidence = get_confidence_score(self.asset_key, label)
            
            # Grounded Monte Carlo (Fan Charts) using Historical AI Error
            fan_p10, fan_p90 = [], []
            if contextual:
                try:
                    # Attempt to load AI's actual historical error from Backtest
                    import json
                    with open(f'reports/backtest_{self.asset_key}.json', 'r') as f:
                        metrics = json.load(f)
                        if isinstance(metrics, dict):
                            rmse = metrics.get('rmse_3layer', metrics.get('rmse', 0))
                            # Convert absolute RMSE to percentage error approximation
                            vol = (rmse / current_price) if current_price > 0 else 0.02
                            # Cap it to realistic extremes (min 0.5%, max 5% daily standard deviation spread)
                            vol = max(0.005, min(0.05, vol))
                        else:
                            vol = 0.015
                except Exception:
                    # Fallback if backtest hasn't been run yet
                    vol = 0.04 if self.asset_key == 'btc' else (0.012 if self.asset_key == 'gold' else 0.015)
                    
                # Random walk drift simulation
                paths = np.zeros((100, steps))
                for i in range(100):
                    noises = np.random.normal(0, vol, steps)
                    # cumulative noise growth scales by sqrt(t) 
                    cum_noise = np.exp(np.cumsum(noises) - 0.5 * vol**2 * np.arange(1, steps + 1))
                    paths[i] = np.array(contextual) * cum_noise
                
                fan_p10 = np.percentile(paths, 10, axis=0).tolist()
                fan_p90 = np.percentile(paths, 90, axis=0).tolist()

            results[label] = {
                'price':            contextual[-1] if contextual else current_price,
                'baseline_price':   baseline[-1]   if baseline   else current_price,
                'confidence':       confidence,
                'series':           contextual,
                'baseline_series':  baseline,
                'fan_p10':          fan_p10,
                'fan_p90':          fan_p90,
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
        
        # Final guard: ensure we return a dict
        res = {
            'current': float(current_price),
            'predicted': float(predicted_price),
            'change': float(change),
            'pct_change': float(pct_change),
            'direction': 'up' if change > 0 else 'down'
        }
        return res
    
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
            fetched_forecasts = predictor.get_multi_range_forecast()
            
            if isinstance(fetched_forecasts, dict):
                results[key] = fetched_forecasts
            else:
                results[key] = {'Current': 0, 'error': 'Invalid data format'}
            
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
