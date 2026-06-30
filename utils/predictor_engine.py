import numpy as np
import pandas as pd
import os
from typing import Dict, Any, List, Optional, Union

from utils.config import FORECAST_RANGES
from utils.confidence_engine import get_confidence_score

import utils.layers.worker_lstm as worker_layer
import utils.layers.manager_anchor as manager_layer
import utils.layers.risk_layer as risk_layer

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

        # ── Phase 7: Dual-Model A & B state ──────────────────────────────────
        # Model_A: short-term specialist (horizons 1D, 7D, 14D)
        # Model_B: macro specialist     (horizons 30D, 90D)
        self._phase7 = {
            'model_a':    None,
            'model_b':    None,
            'scaler_a':   None,   # (feature_scaler, target_scaler) tuple
            'scaler_b':   None,
            'horizons_a': [1, 7, 14],
            'horizons_b': [30, 90],
            'loaded':     False,
        }
        # Cache: cleared at start of each forecast cycle to avoid stale preds
        self._phase7_cache: dict = {}
        # Epistemic uncertainty per horizon (MC Dropout std) — used for Kelly sizing
        self._phase7_uncertainty: dict = {}
        # Full uncertainty breakdown: mc_std + cross_window_std per horizon
        self._phase7_uncertainty_detail: dict = {}

        # Quorum: all-window models (loaded once, not cleared per cycle)
        self._quorum_windows: dict = {'a': None, 'b': None}
        self._registry: Optional[dict] = None  # lazy-loaded model_registry.json

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

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 7: DUAL-MODEL INFERENCE
    # ─────────────────────────────────────────────────────────────────────────
    def _load_phase7_models(self) -> bool:
        """
        Load Phase 7 Dual-Model A and B.
        Priority: model_registry.json (best window per group) -> canonical paths.
        Returns True if at least one model loaded.
        """
        if not TF_AVAILABLE:
            return False
        if self._phase7['loaded']:
            return (
                self._phase7['model_a'] is not None
                or self._phase7['model_b'] is not None
            )

        import json
        import joblib

        def _safe_load(model_path: str, scaler_path: str):
            """Load .keras model + joblib scaler bundle. Returns (model, fs, ts) or (None,None,None).
            compile=False: skip custom loss (directional_mse) lookup — not needed for inference.
            """
            try:
                if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                    return None, None, None
                m   = load_model(model_path, compile=False)
                bnd = joblib.load(scaler_path)
                fs  = bnd.get('feature_scaler')
                ts  = bnd.get('target_scaler')
                return m, fs, ts
            except Exception as e:
                print(f"[Phase7] Load error ({model_path}): {e}")
                return None, None, None

        registry_path = os.path.join('models', 'model_registry.json')
        asset = self.asset_key.lower()

        if os.path.exists(registry_path):
            try:
                with open(registry_path, encoding='utf-8') as f:
                    registry = json.load(f)

                for group_key, group_label in [('a', 'Model_A'), ('b', 'Model_B')]:
                    if self._phase7[f'model_{group_key}'] is not None:
                        continue  # already loaded
                    # Find best window entry for this asset + group
                    entries = [
                        v for k, v in registry.items()
                        if not k.startswith('_')
                        and v.get('asset', '').lower() == asset
                        and v.get('model_group') == group_label
                        and v.get('is_best_window') is True
                    ]
                    if entries:
                        e = entries[0]
                        m, fs, ts = _safe_load(e['model_path'], e['scaler_path'])
                        if m is not None:
                            self._phase7[f'model_{group_key}']  = m
                            self._phase7[f'scaler_{group_key}'] = (fs, ts)
                            self._phase7[f'horizons_{group_key}'] = e.get('horizons', [1,7,14] if group_key=='a' else [30,90])
                            print(f"[Phase7] Loaded {group_label} W{e.get('window')} for {asset} from registry.")
            except Exception as e:
                print(f"[Phase7] Registry read error: {e}. Falling back to canonical paths.")

        # Fallback: canonical paths (also used if registry missing)
        for group_key, suffix, default_h in [
            ('a', 'model_a', [1, 7, 14]),
            ('b', 'model_b', [30, 90]),
        ]:
            if self._phase7[f'model_{group_key}'] is None:
                m_path = self.config['model_file'].replace('.keras', f'_{suffix}.keras')
                s_path = self.config['scaler_file'].replace('.pkl', f'_{suffix}.pkl')
                m, fs, ts = _safe_load(m_path, s_path)
                if m is not None:
                    self._phase7[f'model_{group_key}']    = m
                    self._phase7[f'scaler_{group_key}']   = (fs, ts)
                    self._phase7[f'horizons_{group_key}'] = default_h
                    print(f"[Phase7] Loaded {suffix} for {asset} from canonical path.")

        self._phase7['loaded'] = True
        loaded = (
            self._phase7['model_a'] is not None
            or self._phase7['model_b'] is not None
        )
        if not loaded:
            print(f"[Phase7] No Phase 7 models found for {asset}. Using legacy inference.")
        return loaded

    def _get_registry(self) -> dict:
        """Lazy-load and cache model_registry.json."""
        if self._registry is None:
            self._registry = risk_layer._load_registry()
        return self._registry

    def _get_quorum_windows(self, group: str) -> list:
        """
        Load and cache all non-collapsed window models for the given group ('a' or 'b').
        Loaded ONCE per ForecastEngine lifetime (not per forecast cycle).
        """
        if self._quorum_windows[group] is None:
            model_group = 'Model_A' if group == 'a' else 'Model_B'
            self._quorum_windows[group] = risk_layer.load_window_models_for_quorum(
                asset       = self.asset_key,
                model_group = model_group,
                registry    = self._get_registry(),
            )
            n = len(self._quorum_windows[group])
            print(f"[Phase7/Quorum] Loaded {n} windows for {model_group} ({self.asset_key}).")
        return self._quorum_windows[group]

    def _predict_phase7(self, horizon_days: int) -> Optional[float]:
        """
        Run Phase 7 quorum inference for a specific horizon.

        Uses ALL non-collapsed windows (not just best), weighted by IC-EWMA.
        MC Dropout (30 passes per window) runs ONCE per model group per cycle.
        Results cached in self._phase7_cache until cleared by forecast cycle start.

        Uncertainty stored in self._phase7_uncertainty[horizon] (total_uncertainty)
        and self._phase7_uncertainty_detail[horizon] (mc_std + cross_std breakdown).

        Returns None if Phase 7 models unavailable -> falls back to legacy.
        """
        # First check: can we load at least one Phase 7 model?
        if not self._load_phase7_models():
            return None

        # Routing: which model group handles this horizon?
        if horizon_days in self._phase7['horizons_a']:
            group, model_group = 'a', 'Model_A'
        elif horizon_days in self._phase7['horizons_b']:
            group, model_group = 'b', 'Model_B'
        else:
            return None

        horizons = self._phase7[f'horizons_{group}']

        # Check prediction cache
        cache_key = f'phase7_{group}'
        if cache_key not in self._phase7_cache:
            # Load data (prefer DuckDB, fallback to data_handler)
            data = None
            try:
                data_path  = self.config['data_file']
                table_name = os.path.splitext(os.path.basename(data_path))[0].lower()
                from utils.data_store import MarketDataStore
                df = MarketDataStore().read_table(table_name, format='pandas')
                if 'Date' in df.columns:
                    df = df.set_index('Date').sort_index()
                feats = self.config['features']
                avail = [f for f in feats if f in df.columns]
                data  = df[avail].ffill().fillna(0).values
            except Exception:
                if self.data_handler.data is not None:
                    d    = self.data_handler.data
                    data = d.values if hasattr(d, 'values') else d

            if data is None:
                return None

            # Calculate recent market volatility
            recent_weekly_vol = 0.012
            try:
                if data is not None and len(data) >= 20:
                    # Use last 30 days of closing price (assumed feature index 0)
                    recent_prices = data[-30:, 0]
                    if np.min(recent_prices) > 0:
                        recent_rets = np.diff(recent_prices) / recent_prices[:-1]
                        recent_weekly_vol = float(np.std(recent_rets) * np.sqrt(7))
            except Exception:
                pass
            # Fallbacks by asset
            if recent_weekly_vol < 0.001:
                recent_weekly_vol = 0.035 if self.asset_key == 'btc' else (0.010 if self.asset_key == 'gold' else 0.012)

            # Compute IC-EWMA weights for this model group
            ewma_lam = risk_layer.EWMA_LAMBDA_DEFAULT.get(self.asset_key.lower(), 0.92)
            ic_weights, avg_ic = risk_layer.compute_ic_ewma_weights(
                asset       = self.asset_key,
                model_group = model_group,
                horizons    = horizons,
                ewma_lambda = ewma_lam,
                registry    = self._get_registry(),
            )

            # Load all 5 window models for this group (cached across cycles)
            window_models = self._get_quorum_windows(group)

            # Run quorum inference (weighted ensemble of all windows)
            # Market volatility needs to be scaled per horizon: vol * sqrt(h/7)
            # We pass the base weekly vol, and risk_layer could scale it, or we pass a dict.
            # But quorum_inference expects a single float for market_volatility.
            # Wait, quorum_inference iterates over horizons! Let's pass the base weekly vol, 
            # and inside quorum_inference it can scale it.
            # Oh, the replacement will just pass the recent_weekly_vol.
            results = risk_layer.quorum_inference(
                window_models = window_models,
                ic_weights    = ic_weights,
                data          = data,
                config        = self.config,
                horizons      = horizons,
                market_volatility = recent_weekly_vol,
                n_mc_samples  = 30,
                avg_ic        = avg_ic,
            )
            self._phase7_cache[cache_key] = results

            # Store uncertainty breakdown for external use (Kelly sizing)
            for h, r in results.items():
                self._phase7_uncertainty[h]        = r['total_uncertainty']
                self._phase7_uncertainty_detail[h] = {
                    'mc_std':         r['mc_std'],
                    'cross_std':      r['cross_std'],
                    'n_windows':      r['n_windows_used'],
                    'kelly_fraction': r['kelly_fraction'],
                }

        cached = self._phase7_cache.get(cache_key, {})
        entry  = cached.get(horizon_days)
        return entry['mean'] if entry is not None else None

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

        Phase 7 path (priority):
          - Routes to Model_A (1/7/14D) or Model_B (30/90D)
          - Uses MC Dropout for epistemic uncertainty
          - Uncertainty stored in self._phase7_uncertainty[horizon_days]

        Legacy fallback:
          - Loads per-horizon model files (old format)
          - Power-law projection from 7D if specific horizon model is missing
        """
        if self.data_handler.data is None:
            self.data_handler.load_data()

        if self.data_handler.data is None or len(self.data_handler.data) == 0:
            return 0.0

        # ── Phase 7: try dual-model first ────────────────────────────────────
        p7 = self._predict_phase7(horizon_days)
        if p7 is not None:
            return float(p7)

        # ── Legacy: per-horizon model files ──────────────────────────────────
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

    def recursive_forecast(self, steps: int, ceo_drift_multiplier: float = 1.0) -> List[float]:
        """
        Recursive forecast wrapper used by Streamlit UI to draw 30-day projection charts.
        Delegates to worker_lstm layer.
        """
        if not self.load_horizon_model(1):
            current_price = self.data_handler.get_latest_price()
            return [current_price] * steps
            
        model = self.models[1]
        feat_scaler, _ = self.scalers[1]
        
        from utils.layers.worker_lstm import recursive_forecast as worker_recursive
        return worker_recursive(
            model=model,
            scaler=feat_scaler,
            data=self.data_handler.data,
            config=self.config,
            steps=steps,
            asset_key=self.asset_key,
            ceo_drift_multiplier=ceo_drift_multiplier
        )

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
                
                # Save CEO-filtered news for UI display
                filtered_news_list = analysis.get('headlines_list', [])
                if filtered_news_list:
                    filtered_json_path = f'data/ceo_filtered_news_{self.asset_key}.json'
                    try:
                        import json
                        from datetime import datetime
                        news_to_save = []
                        for h in filtered_news_list:
                            news_to_save.append({
                                'title': h,
                                'date': datetime.now().strftime('%Y-%m-%d'),
                                'url': '#',
                                'sentiment': float(analysis.get('confidence', 0.5)),
                                'source': 'CEO Layer Filter'
                            })
                        with open(filtered_json_path, 'w', encoding='utf-8') as f:
                            json.dump(news_to_save, f, indent=2)
                    except Exception as err:
                        print(f"Warning: Could not save CEO filtered news: {err}")

            except Exception as e:
                print(f"Warning: CEO Layer error for {self.asset_key}: {e}. Using baseline.")

        # Clear Phase 7 prediction cache at start of each forecast cycle
        # This ensures fresh MC Dropout inference per call, not stale from prev request
        self._phase7_cache.clear()

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
            
            # Inject Kelly fraction from Phase 7 uncertainty if available
            kelly = self._phase7_uncertainty_detail.get(steps, {}).get('kelly_fraction', 0.0)
            if kelly > 0.0:
                confidence['kelly_fraction'] = float(kelly)
                # Ensure the score reflects the Kelly sizing
                # Kelly ranges from 0 to max_kelly (default 0.25). We can map this to a confidence 0.5 -> 0.95
                mapped_score = 0.50 + (kelly / 0.25) * 0.45
                confidence['score'] = round(float(np.clip(mapped_score, 0.50, 0.95)), 3)
                if confidence['score'] >= 0.75:
                    confidence['label'], confidence['color'] = 'High', 'success'
                elif confidence['score'] >= 0.65:
                    confidence['label'], confidence['color'] = 'Good', 'success'
                elif confidence['score'] >= 0.55:
                    confidence['label'], confidence['color'] = 'Moderate', 'info'
                else:
                    confidence['label'], confidence['color'] = 'Low', 'warning'

            result_entry = {
                'price':            contextual[-1] if contextual else current_price,
                'baseline_price':   baseline[-1]   if baseline   else current_price,
                'confidence':       confidence,
                'series':           contextual,
                'baseline_series':  baseline,
                'fan_p10':          fan_p10,
                'fan_p90':          fan_p90,
                # Phase 7: uncertainty decomposition per horizon
                # mc_std  = epistemic (model uncertainty via MC Dropout)
                # cross_std = cross-window disagreement (regime signal)
                # total    = sqrt(mc^2 + cross^2) -> used for Kelly sizing
                'phase7_uncertainty': {
                    h: {
                        'total': float(self._phase7_uncertainty.get(h, 0.0)),
                        'mc_std': float(
                            self._phase7_uncertainty_detail.get(h, {}).get('mc_std', 0.0)
                        ),
                        'cross_std': float(
                            self._phase7_uncertainty_detail.get(h, {}).get('cross_std', 0.0)
                        ),
                        'n_windows': int(
                            self._phase7_uncertainty_detail.get(h, {}).get('n_windows', 0)
                        ),
                        'kelly_fraction': float(
                            self._phase7_uncertainty_detail.get(h, {}).get('kelly_fraction', 0.0)
                        ),
                    }
                    for h in [1, 7, 14, 30, 90]
                },
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
