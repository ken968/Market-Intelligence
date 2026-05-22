"""
Unified Prediction Engine for Multi-Asset Terminal
Handles forecasting for Gold, Bitcoin, and all US Stocks
"""

import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, Any, List, Optional, Union

from utils.config import get_asset_config, FORECAST_RANGES
from utils.confidence_engine import get_confidence_score

# Layers
import utils.layers.worker_lstm as worker_layer
import utils.layers.manager_anchor as manager_layer

# Handle TensorFlow incompatibility
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




class AssetPredictor:
    """Universal predictor for all asset types"""
    
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
        
        self.model = None
        self.scaler = None
        self.data = None
        self.is_loaded = False
    
    def load_model(self) -> bool:
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
            self.model, self.scaler = worker_layer.load_lstm_model(model_path, scaler_path)
            if self.model is not None and self.scaler is not None:
                self.is_loaded = True
                return True
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def load_data(self) -> pd.DataFrame:
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
    
    def predict_next_step(self, sequence: np.ndarray) -> float:
        """
        Predict next timestep given a sequence.
        Delegates to worker_layer.
        """
        if not TF_AVAILABLE:
            return float(sequence[0, -1, 0])

        if not self.is_loaded:
            if not self.load_model():
               return float(sequence[0, -1, 0])
        
        return worker_layer.predict_next_step(self.model, self.scaler, sequence)
    
    def recursive_forecast(self, steps: int, ceo_bias_vector: Optional[np.ndarray] = None, ceo_drift_multiplier: float = 1.0) -> List[float]:
        """
        Generate multi-step forecast using recursive prediction.
        Delegates to worker_layer.
        """
        if not TF_AVAILABLE:
             return []

        if not self.is_loaded:
            if not self.load_model():
                return []
        
        if self.data is None:
            self.load_data()

        return worker_layer.recursive_forecast(
            model=self.model,
            scaler=self.scaler,
            data=self.data,
            config=self.config,
            steps=steps,
            asset_key=self.asset_key,
            ceo_drift_multiplier=ceo_drift_multiplier
        )
    
    def get_multi_range_forecast(self, headlines: Optional[List[Dict]] = None, published_at_list: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate forecasts for all predefined ranges with confidence scores.
        Automatically invokes the CEO Layer (Gemini) if available.
        """
        current_price = self.get_latest_price()
        results = {'Current': current_price}

        # Level 3 — CEO Layer
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

        # Level 2 — Manager Layer (Ensemble)
        try:
            ensemble_result = self.ensemble_forecast()
        except Exception as _e:
            ensemble_result = {}
            print(f"[Ensemble] ensemble_forecast failed, using recursive: {_e}")

        # Generate a single 90-day trajectory
        max_steps = 90
        baseline_90 = self.recursive_forecast(max_steps, ceo_drift_multiplier=1.0)
        if not baseline_90:
            baseline_90 = [current_price] * max_steps

        contextual_90 = self.pct_chain_forecast(
            max_steps,
            ceo_drift_multiplier=ceo_drift_multiplier,
            ensemble_7d_pct=ensemble_result.get('pct_change_7d')
        )
        if not contextual_90:
            contextual_90 = list(baseline_90)

        # For 1 Week: override endpoint with Dual-Head Stacker signal (applied on the single 90-day trajectory)
        if (ensemble_result.get('model') == 'dual_head_ensemble'
                and ensemble_result.get('pct_change_7d') is not None):
            try:
                pct_7d    = ensemble_result['pct_change_7d']
                ens_price = current_price * (1.0 + pct_7d)
                # Override the first 7 days with a linear interpolation
                for i in range(7):
                    contextual_90[i] = current_price + (ens_price - current_price) * (i + 1) / 7.0
            except Exception as e:
                print(f"Warning: Failed to apply Stacker override to 90-day trajectory: {e}")

        # Run Monte Carlo simulation ONCE for the 90-day trajectory
        fan_p10_90, fan_p90_90 = [], []
        if contextual_90:
            try:
                from utils.realtime_prices import get_live_dvol, get_live_vix, iv_to_daily_vol

                if self.asset_key == 'btc':
                    iv_annual = get_live_dvol(fallback=60.0)
                    print(f"[Monte Carlo] BTC DVOL (Deribit): {iv_annual:.1f}% annualized")
                else:
                    iv_annual = get_live_vix(fallback=20.0)
                    print(f"[Monte Carlo] VIX (yfinance): {iv_annual:.1f}% annualized")

                vol = iv_to_daily_vol(iv_annual)

                if self.asset_key == 'btc':
                    vol = max(0.010, min(0.12, vol))
                elif self.asset_key == 'gold':
                    vol = max(0.003, min(0.04, vol))
                else:
                    vol = max(0.004, min(0.06, vol))

                print(f"[Monte Carlo] Daily IV vol = {vol:.4f} ({vol*100:.2f}%/day)")

            except Exception as e:
                print(f"[Monte Carlo] IV fetch failed: {e}. Using historical fallback.")
                vol = 0.035 if self.asset_key == 'btc' else (0.010 if self.asset_key == 'gold' else 0.012)

            ceo_conf = ceo_context.get('confidence', 0.0)
            if ceo_conf > 0.0:
                ceo_narrow = 1.0 - (ceo_conf * 0.40)
                vol = vol * max(0.50, ceo_narrow)
                print(f"[Monte Carlo] After CEO confidence ({ceo_conf:.2f}) adjustment: vol={vol:.4f}")

            n_paths = 500
            paths = np.zeros((n_paths, max_steps))
            for i in range(n_paths):
                noises = np.random.normal(0, vol, max_steps)
                cum_noise = np.exp(np.cumsum(noises) - 0.5 * vol**2 * np.arange(1, max_steps + 1))
                paths[i] = np.array(contextual_90) * cum_noise

            fan_p10_90 = np.percentile(paths, 10, axis=0).tolist()
            fan_p90_90 = np.percentile(paths, 90, axis=0).tolist()

        # Slice forecasts for all ranges
        for label, steps in FORECAST_RANGES.items():
            baseline = baseline_90[:steps]
            contextual = contextual_90[:steps]
            fan_p10 = fan_p10_90[:steps] if fan_p10_90 else []
            fan_p90 = fan_p90_90[:steps] if fan_p90_90 else []

            if LLM_AVAILABLE and headlines and not ceo_context.get('is_fallback', True):
                try:
                    from utils.counterfactual_logger import log_forecast
                    df_temp = pd.read_csv(self.config['data_file'])
                    forecast_date = df_temp['Date'].iloc[-1]
                    log_forecast(
                        asset_key=self.asset_key,
                        forecast_date=forecast_date,
                        steps=steps,
                        baseline_prices=baseline,
                        contextual_prices=contextual,
                        llm_scores=analysis.get('scores', {}),
                        llm_narrative=analysis.get('narrative', ''),
                        macro_regime=macro_ctx
                    )
                except Exception as log_err:
                    print(f"[CounterfactualLogger] Error logging forecast: {log_err}")
                    
            confidence = get_confidence_score(
                self.asset_key, label,
                ceo_confidence=ceo_context.get('confidence', 0.0)
            )

            result_entry = {
                'price':            contextual[-1] if contextual else current_price,
                'baseline_price':   baseline[-1]   if baseline   else current_price,
                'confidence':       confidence,
                'series':           contextual,
                'baseline_series':  baseline,
                'fan_p10':          fan_p10,
                'fan_p90':          fan_p90,
            }

            if label == '1 Week' and ensemble_result:
                result_entry['ensemble_meta'] = {
                    'pct_change_7d':  ensemble_result.get('pct_change_7d', 0.0),
                    'direction':      ensemble_result.get('direction', 'flat'),
                    'direction_prob': ensemble_result.get('direction_prob', 0.5),
                    'lstm_signal':    ensemble_result.get('lstm_signal', 0.0),
                    'xgb_signal':     ensemble_result.get('xgb_signal', 0.0),
                    'model':          ensemble_result.get('model', 'fallback'),
                }

            results[label] = result_entry

        if not isinstance(results, dict):
            return {'Current': current_price, 'error': 'Internal forecast error'}
            
        return results
    
    def get_latest_price(self) -> float:
        """Get the most recent actual price"""
        if self.data is None:
            df = self.load_data()
        else:
            df = pd.read_csv(self.config['data_file'])
        
        # First feature is always the price
        return float(df[self.config['features'][0]].iloc[-1])

    def ensemble_forecast(self) -> Dict[str, Any]:
        """
        Generate 7-day % change forecast using Dual-Head Stacker (Ensemble Alpha Engine).
        Delegates to manager_layer.
        """
        current_price = self.get_latest_price()

        stacker_models = manager_layer.load_stacker_models(self.asset_key)
        if not stacker_models:
            legacy = self.recursive_forecast(7)
            if not legacy:
                return {'pct_change_7d': 0.0, 'direction': 'flat',
                        'direction_prob': 0.5, 'predicted_price': current_price,
                        'model': 'fallback', 'lstm_signal': 0.0, 'xgb_signal': 0.0}
            pred_price = legacy[-1]
            pct = (pred_price - current_price) / current_price
            return {
                'pct_change_7d': float(pct),
                'direction': 'up' if pct > 0 else 'down',
                'direction_prob': 0.55,
                'predicted_price': float(pred_price),
                'model': 'fallback_lstm',
                'lstm_signal': float(pct),
                'xgb_signal': 0.0,
            }

        dir_head, mag_head, meta_scaler, stacker_meta = stacker_models

        df_full = pd.read_csv(self.config['data_file'], index_col=0, parse_dates=True).sort_index()
        df_last = df_full.iloc[[-1]]

        lstm_signal = manager_layer.get_lstm_signal(self.asset_key, self.config, df_full)
        xgb_signal = manager_layer.get_xgb_signal(self.asset_key, df_last)

        ctx_features = ['VIX', 'GK_Vol_21d', 'Sentiment', 'Sentiment_Std',
                        'YieldCurve_10Y2Y', 'DXY']
        ctx_values = {}
        for f in ctx_features:
            if f in df_last.columns:
                ctx_values[f] = float(df_last[f].iloc[0])

        return manager_layer.run_dual_head_inference(
            dir_head=dir_head,
            mag_head=mag_head,
            meta_scaler=meta_scaler,
            stacker_meta=stacker_meta,
            lstm_signal=lstm_signal,
            xgb_signal=xgb_signal,
            ctx_values=ctx_values,
            current_price=current_price
        )

    def pct_chain_forecast(self, steps: int, ceo_drift_multiplier: float = 1.0, ensemble_7d_pct: float = None) -> list:
        """
        Build a realistic multi-step price path using the canonical Dual-Head Stacker's
        7-day prediction to scale the non-linear curvature of the LSTM recursive forecast.
        
        Strategy:
          1. Retrieve the 7-day pct change target from the Stacker.
          2. Modulate slightly by CEO drift multiplier.
          3. Generate the non-linear LSTM recursive rollout for the full steps.
          4. Scale the LSTM trajectory percent changes so that the endpoint matches
             the Stacker's projected target price for the given horizon.
          5. Fall back to power-law decay curve if LSTM is flat or fails.
        """
        current_price = self.get_latest_price()

        if ensemble_7d_pct is None:
            try:
                ens = self.ensemble_forecast()
                ensemble_7d_pct = ens.get('pct_change_7d', 0.0)
            except Exception:
                ensemble_7d_pct = 0.0

        # If stacker failed, fallback to recursive absolute-price model
        if ensemble_7d_pct == 0.0:
            return self.recursive_forecast(steps, ceo_drift_multiplier=ceo_drift_multiplier)

        # CEO modulates the expected magnitude slightly
        dm = max(0.85, min(1.15, ceo_drift_multiplier))
        adjusted_7d_pct = ensemble_7d_pct * dm

        # ── Project target change for the full horizon (steps) based on adjusted_7d_pct ──
        momentum_scale = min(12.0, steps / 7.0)
        target_horizon_pct = adjusted_7d_pct * momentum_scale
        target_price = current_price * (1.0 + target_horizon_pct)

        # ── Generate base LSTM recursive forecast for curvature ──
        lstm_path = self.recursive_forecast(steps, ceo_drift_multiplier=ceo_drift_multiplier)
        if not lstm_path or len(lstm_path) < steps:
            # Fallback to curved power-law trajectory if LSTM path unavailable
            prices = []
            for i in range(1, steps + 1):
                t_ratio = i / 7.0
                horizon_pct = adjusted_7d_pct * (t_ratio ** 0.75)
                prices.append(current_price * (1.0 + horizon_pct))
            return prices

        # LSTM predicted full horizon change
        lstm_horizon_pct = (lstm_path[-1] - current_price) / current_price

        # Adjust the path: scale each step's pct change to match the target endpoint
        prices = []
        if abs(lstm_horizon_pct) > 1e-5:
            scale_factor = target_horizon_pct / lstm_horizon_pct
            # Clamp scale factor to avoid extreme artifacts
            if 0.1 <= abs(scale_factor) <= 10.0:
                for price in lstm_path:
                    step_pct = (price - current_price) / current_price
                    adj_step_pct = step_pct * scale_factor
                    prices.append(current_price * (1.0 + adj_step_pct))
                return prices

        # Stable fallback: shift the LSTM path smoothly to the target price
        lstm_target_price = lstm_path[-1]
        for i, price in enumerate(lstm_path):
            t_ratio = (i + 1) / steps
            adj_price = price + (target_price - lstm_target_price) * t_ratio
            prices.append(adj_price)
        return prices

    def predict_tomorrow(self) -> Dict[str, Union[float, str]]:
        """
        Quick prediction for next day only.
        Uses pct_chain_forecast(1) for consistency with multi-range pipeline.
        """
        current_price = self.get_latest_price()
        forecast = self.pct_chain_forecast(1)

        if not forecast:
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
            'current': float(current_price),
            'predicted': float(predicted_price),
            'change': float(change),
            'pct_change': float(pct_change),
            'direction': 'up' if change > 0 else 'down'
        }
    
    def predict_week(self) -> Dict[str, Union[float, str, bool]]:
        """
        Quick prediction for 1 week (7 days) ahead.
        Uses ensemble_forecast() (Dual-Head Stacker) for accuracy,
        falls back to pct_chain_forecast(7) if stacker not available.
        """
        current_price = self.get_latest_price()

        try:
            ens = self.ensemble_forecast()
            if ens.get('model') == 'dual_head_ensemble' and ens.get('pct_change_7d') is not None:
                predicted_price = float(ens['predicted_price'])
                change = predicted_price - current_price
                pct_change = (change / current_price) * 100
                return {
                    'current':        float(current_price),
                    'predicted':      predicted_price,
                    'change':         float(change),
                    'pct_change':     float(pct_change),
                    'direction':      ens.get('direction', 'flat'),
                    'direction_prob': ens.get('direction_prob', 0.5),
                    'lstm_signal':    ens.get('lstm_signal', 0.0),
                    'xgb_signal':     ens.get('xgb_signal', 0.0),
                    'has_ensemble':   True,
                }
        except Exception:
            pass

        forecast = self.pct_chain_forecast(7)
        if not forecast:
            forecast = self.recursive_forecast(7)

        if not forecast:
            return {
                'current':      current_price,
                'predicted':    current_price,
                'change':       0.0,
                'pct_change':   0.0,
                'direction':    'flat',
                'lstm_signal':  0.0,
                'xgb_signal':   0.0,
                'has_ensemble': False,
                'error':        'AI Model Unavailable'
            }

        predicted_price = forecast[-1]
        change = predicted_price - current_price
        pct_change = (change / current_price) * 100

        return {
            'current':        float(current_price),
            'predicted':      float(predicted_price),
            'change':         float(change),
            'pct_change':     float(pct_change),
            'direction':      'up' if change > 0 else 'down',
            'direction_prob': 0.5,
            'lstm_signal':    0.0,
            'xgb_signal':     0.0,
            'has_ensemble':   False,
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
