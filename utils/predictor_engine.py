import numpy as np
import pandas as pd
import os
from typing import Dict, Any, List, Optional, Union

from utils.config import FORECAST_RANGES
from utils.confidence_engine import get_confidence_score

import utils.layers.worker_lstm as worker_layer
import utils.layers.manager_anchor as manager_layer

try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not found. AI predictions will be disabled.")

try:
    from utils.llm_manager import compute_drift_multiplier, analyze_news_context
    from utils.macro_processor import build_macro_context
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

class ForecastEngine:
    """Handles all AI forecasting logic and model orchestration."""
    
    def __init__(self, asset_key: str, config: dict, data_handler):
        self.asset_key = asset_key
        self.config = config
        self.data_handler = data_handler
        
        self.models = {}
        self.scalers = {}
        self.model = None
        self.scaler = None
        self.is_loaded = False

    def load_horizon_model(self, horizon_days: int) -> bool:
        """Load trained model and scalers for a specific horizon (1D, 7D, 14D, 30D, 90D)."""
        if not TF_AVAILABLE:
            return False

        if horizon_days in self.models:
            return True

        import pickle
        model_path = f"models/{self.asset_key}_model_{horizon_days}d.keras"
        scaler_path = f"models/{self.asset_key}_scaler_{horizon_days}d.pkl"
        target_scaler_path = f"models/{self.asset_key}_scaler_{horizon_days}d_target.pkl"

        # Fallback to legacy config files if horizon_days == 7 and horizon files don't exist
        if horizon_days == 7 and (not os.path.exists(model_path) or not os.path.exists(scaler_path)):
            model_path = self.config['model_file']
            scaler_path = self.config['scaler_file']
            target_scaler_path = self.config['scaler_file'].replace('.pkl', '_target.pkl')

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return False

        try:
            model = load_model(model_path)
            with open(scaler_path, 'rb') as fh:
                feat_scaler = pickle.load(fh)
            
            target_scaler = None
            if os.path.exists(target_scaler_path):
                with open(target_scaler_path, 'rb') as fh:
                    target_scaler = pickle.load(fh)

            if model is not None and feat_scaler is not None:
                self.models[horizon_days] = model
                self.scalers[horizon_days] = (feat_scaler, target_scaler)
                # Keep legacy compatible fields updated if it's the 7D horizon
                if horizon_days == 7:
                    self.model = model
                    self.scaler = feat_scaler
                    self.is_loaded = True
                return True
            return False
        except Exception as e:
            print(f"Error loading horizon {horizon_days}d model: {e}")
            return False

    def load_model(self) -> bool:
        """Load trained model and scaler (backward compatibility)"""
        return self.load_horizon_model(7)

    def predict_next_step(self, sequence: np.ndarray) -> float:
        if not TF_AVAILABLE:
            return float(sequence[0, -1, 0])

        if not self.is_loaded:
            if not self.load_model():
               return float(sequence[0, -1, 0])
        
        return worker_layer.predict_next_step(self.model, self.scaler, sequence)

    def predict_horizon(self, horizon_days: int) -> float:
        """
        Run inference for a specific target horizon.
        If the model is not found, fallback using power-law projection from 7D prediction.
        """
        if self.data_handler.data is None:
            self.data_handler.load_data()
        
        # Ensure we have data
        if self.data_handler.data is None or len(self.data_handler.data) == 0:
            return 0.0

        # Try to load and predict using specific horizon model
        if self.load_horizon_model(horizon_days):
            model = self.models[horizon_days]
            feat_scaler, target_scaler = self.scalers[horizon_days]
            
            pred_pct = worker_layer.predict_direct_horizon(
                model=model,
                feature_scaler=feat_scaler,
                target_scaler=target_scaler,
                data=self.data_handler.data,
                config=self.config
            )
            return float(pred_pct)

        # Fallback to power-law scaling of 7D prediction if specific model is missing
        # 1. Get 7D return prediction
        pct_7 = 0.0
        if self.load_horizon_model(7):
            model = self.models[7]
            feat_scaler, target_scaler = self.scalers[7]
            pct_7 = float(worker_layer.predict_direct_horizon(
                model=model,
                feature_scaler=feat_scaler,
                target_scaler=target_scaler,
                data=self.data_handler.data,
                config=self.config
            ))
        else:
            # Last resort fallback: legacy recursive forecast for 7 steps
            legacy_forecast = self.recursive_forecast(7)
            if legacy_forecast:
                current_price = self.data_handler.get_latest_price()
                pct_7 = (legacy_forecast[-1] - current_price) / current_price

        # 2. Power-law projection: pct_H = pct_7 * ((H / 7.0) ** 0.65)
        power_exponent = 0.65
        pct_H = pct_7 * ((horizon_days / 7.0) ** power_exponent)
        return float(pct_H)

    def ensemble_forecast(self) -> Dict[str, Any]:
        current_price = self.data_handler.get_latest_price()

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

        # Try to load historical data from DuckDB, fallback to CSV
        data_path = self.config['data_file']
        table_name = os.path.splitext(os.path.basename(data_path))[0].lower()
        from utils.data_store import MarketDataStore
        store = MarketDataStore()
        df_full = None
        try:
            df_full = store.read_table(table_name, format='pandas')
            if 'Date' in df_full.columns:
                df_full['Date'] = pd.to_datetime(df_full['Date'])
                df_full.set_index('Date', inplace=True)
            df_full = df_full.sort_index()
        except Exception as e:
            print(f"Warning: Could not read table '{table_name}' from DuckDB: {e}. Falling back to CSV.")
            df_full = pd.read_csv(data_path, index_col=0, parse_dates=True).sort_index()

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
        current_price = self.data_handler.get_latest_price()

        if ensemble_7d_pct is None:
            try:
                ens = self.ensemble_forecast()
                ensemble_7d_pct = ens.get('pct_change_7d', 0.0)
            except Exception:
                ensemble_7d_pct = 0.0

        if ensemble_7d_pct == 0.0:
            return self.recursive_forecast(steps, ceo_drift_multiplier=ceo_drift_multiplier)

        dm = max(0.85, min(1.15, ceo_drift_multiplier))
        adjusted_7d_pct = ensemble_7d_pct * dm

        momentum_scale = min(12.0, steps / 7.0)
        target_horizon_pct = adjusted_7d_pct * momentum_scale
        target_price = current_price * (1.0 + target_horizon_pct)

        lstm_path = self.recursive_forecast(steps, ceo_drift_multiplier=ceo_drift_multiplier)
        if not lstm_path or len(lstm_path) < steps:
            prices = []
            for i in range(1, steps + 1):
                t_ratio = i / 7.0
                horizon_pct = adjusted_7d_pct * (t_ratio ** 0.75)
                prices.append(current_price * (1.0 + horizon_pct))
            return prices

        lstm_horizon_pct = (lstm_path[-1] - current_price) / current_price

        prices = []
        if abs(lstm_horizon_pct) > 1e-5:
            scale_factor = target_horizon_pct / lstm_horizon_pct
            if 0.1 <= abs(scale_factor) <= 10.0:
                for price in lstm_path:
                    step_pct = (price - current_price) / current_price
                    adj_step_pct = step_pct * scale_factor
                    prices.append(current_price * (1.0 + adj_step_pct))
                return prices

        lstm_target_price = lstm_path[-1]
        for i, price in enumerate(lstm_path):
            t_ratio = (i + 1) / steps
            adj_price = price + (target_price - lstm_target_price) * t_ratio
            prices.append(adj_price)
        return prices

    def predict_tomorrow(self) -> Dict[str, Union[float, str]]:
        current_price = self.data_handler.get_latest_price()
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
        current_price = self.data_handler.get_latest_price()

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

    def get_multi_range_forecast(self, headlines: Optional[List[Dict]] = None, published_at_list: Optional[List[str]] = None) -> Dict[str, Any]:
        current_price = self.data_handler.get_latest_price()
        results = {'Current': current_price}

        ceo_drift_multiplier = 1.0
        ceo_bias_vector = None
        ceo_context = {'is_fallback': True, 'narrative': 'CEO Layer not active.', 'confidence': 0.0}

        if LLM_AVAILABLE and headlines:
            try:
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

        try:
            ensemble_result = self.ensemble_forecast()
        except Exception as _e:
            ensemble_result = {}
            print(f"[Ensemble] ensemble_forecast failed, using recursive: {_e}")

        # ── Direct Multi-Step Prediction & Interpolation ─────────────────────
        # Target horizons we support
        horizons = [1, 7, 14, 30, 90]
        
        # 1. Fetch direct predictions for baseline (no LLM drift, no Stacker/ensemble adjustments)
        base_pcts = {}
        for h in horizons:
            base_pcts[h] = self.predict_horizon(h)
            
        # 2. Fetch direct predictions for contextual (apply LLM drift and Stacker adjustments)
        context_pcts = {}
        for h in horizons:
            if h == 7 and ensemble_result and ensemble_result.get('pct_change_7d') is not None:
                # Use Stacker override for 7D if available
                pct_7d = ensemble_result['pct_change_7d']
                context_pcts[7] = pct_7d * ceo_drift_multiplier
            else:
                context_pcts[h] = self.predict_horizon(h) * ceo_drift_multiplier

        # ── Volatility-Aware Minimum Scale ───────────────────────────────────
        # When models are trained on a different price regime (e.g., gold at $2k but
        # now at $4.5k), their predicted magnitudes shrink to near-zero while the
        # direction signal remains valid. We apply a volatility floor: predicted
        # magnitude must be at least 25% of recent realized weekly volatility,
        # scaled by sqrt(h/7) for longer horizons. Direction (sign) is preserved.
        try:
            df_recent = pd.read_csv(self.config['data_file'])
            price_col = self.config['features'][0]  # e.g. 'Gold', 'BTC', 'SPY'
            if price_col in df_recent.columns and len(df_recent) >= 20:
                recent_prices = df_recent[price_col].dropna().values[-30:]
                recent_rets = np.diff(recent_prices) / recent_prices[:-1]
                recent_weekly_vol = float(np.std(recent_rets) * np.sqrt(7))  # 7-day vol from daily std
            else:
                recent_weekly_vol = 0.012  # fallback 1.2%/week
        except Exception:
            recent_weekly_vol = 0.012

        # Minimum = 25% of recent weekly vol, scaled for each horizon
        vol_floor_fraction = 0.25
        for h in horizons:
            min_magnitude = recent_weekly_vol * vol_floor_fraction * ((h / 7.0) ** 0.5)
            pct = context_pcts[h]
            if abs(pct) < min_magnitude and recent_weekly_vol > 0.003:
                # Scale up to meet floor, preserving direction
                direction = 1 if pct >= 0 else -1
                # Blend: 40% floor + 60% model direction (so model direction dominates)
                context_pcts[h] = direction * (0.6 * min_magnitude + 0.4 * abs(pct)) if abs(pct) > 1e-6 \
                    else direction * min_magnitude * 0.5

        # Apply same floor to baseline
        for h in horizons:
            min_magnitude = recent_weekly_vol * vol_floor_fraction * ((h / 7.0) ** 0.5)
            pct = base_pcts[h]
            if abs(pct) < min_magnitude and recent_weekly_vol > 0.003:
                direction = 1 if pct >= 0 else -1
                base_pcts[h] = direction * (0.6 * min_magnitude + 0.4 * abs(pct)) if abs(pct) > 1e-6 \
                    else direction * min_magnitude * 0.5

        # 3. Construct price points (x: time steps [0, 1, 7, 14, 30, 90])
        x_points = [0, 1, 7, 14, 30, 90]

        y_base = [current_price] + [current_price * (1.0 + base_pcts[h]) for h in horizons]
        y_context = [current_price] + [current_price * (1.0 + context_pcts[h]) for h in horizons]
        
        # 4. Interpolate over 90-day trajectory
        x_eval = np.arange(1, 91)
        baseline_90 = np.interp(x_eval, x_points, y_base).tolist()
        contextual_90 = np.interp(x_eval, x_points, y_context).tolist()

        fan_p10_90, fan_p90_90 = [], []
        if contextual_90:
            try:
                from utils.realtime_prices import get_live_dvol, get_live_vix, iv_to_daily_vol

                if self.asset_key == 'btc':
                    iv_annual = get_live_dvol(fallback=60.0)
                else:
                    iv_annual = get_live_vix(fallback=20.0)

                vol = iv_to_daily_vol(iv_annual)

                if self.asset_key == 'btc':
                    vol = max(0.010, min(0.12, vol))
                elif self.asset_key == 'gold':
                    vol = max(0.003, min(0.04, vol))
                else:
                    vol = max(0.004, min(0.06, vol))

            except Exception as e:
                vol = 0.035 if self.asset_key == 'btc' else (0.010 if self.asset_key == 'gold' else 0.012)

            ceo_conf = ceo_context.get('confidence', 0.0)
            if ceo_conf > 0.0:
                ceo_narrow = 1.0 - (ceo_conf * 0.40)
                vol = vol * max(0.50, ceo_narrow)

            n_paths = 500
            paths = np.zeros((n_paths, 90))
            for i in range(n_paths):
                noises = np.random.normal(0, vol, 90)
                cum_noise = np.exp(np.cumsum(noises) - 0.5 * vol**2 * np.arange(1, 91))
                paths[i] = np.array(contextual_90) * cum_noise

            fan_p10_90 = np.percentile(paths, 10, axis=0).tolist()
            fan_p90_90 = np.percentile(paths, 90, axis=0).tolist()

        for label, steps in FORECAST_RANGES.items():
            baseline = baseline_90[:steps]
            contextual = contextual_90[:steps]
            fan_p10 = fan_p10_90[:steps] if fan_p10_90 else []
            fan_p90 = fan_p90_90[:steps] if fan_p90_90 else []

            if LLM_AVAILABLE and headlines and not ceo_context.get('is_fallback', True):
                try:
                    from utils.counterfactual_logger import log_forecast
                    data_path = self.config['data_file']
                    table_name = os.path.splitext(os.path.basename(data_path))[0].lower()
                    from utils.data_store import MarketDataStore
                    store = MarketDataStore()
                    df_temp = None
                    try:
                        df_temp = store.read_table(table_name, format='pandas')
                    except Exception as e:
                        df_temp = pd.read_csv(data_path)
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
