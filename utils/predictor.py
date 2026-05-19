"""
Unified Prediction Engine for Multi-Asset Terminal
Handles forecasting for Gold, Bitcoin, and all US Stocks
"""

import numpy as np
import pandas as pd
import pickle
import json
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


def get_confidence_score(asset_key, timeframe, ceo_confidence: float = 0.0):
    """
    Get confidence score for asset and timeframe.

    Priority order:
    1. Load real hit_ratio from backtest JSON (reports/backtest_{asset}.json)
    2. Apply horizon decay (longer forecast = lower confidence)
    3. Apply CEO uplift if Gemini is highly confident (max +0.08 boost)
    4. Fall back to conservative static values if no backtest data exists

    Args:
        asset_key (str)       : Asset identifier (e.g., 'gold', 'btc', 'aapl')
        timeframe (str)       : Forecast timeframe (e.g., '1 Week', '1 Month')
        ceo_confidence (float): Gemini CEO confidence score [0.0 – 1.0]

    Returns:
        dict: {'score': float, 'label': str, 'color': str}
    """
    # Step 1: Load dynamic confidence from backtest JSON
    try:
        from utils.config import get_dynamic_confidence
        base = get_dynamic_confidence(asset_key, timeframe)
    except Exception:
        # Final fallback to static dict
        asset_type = 'gold' if asset_key == 'gold' else ('btc' if asset_key == 'btc' else 'stocks')
        base = CONFIDENCE_SCORES.get(asset_type, {}).get(timeframe, {
            'score': 0.50, 'label': 'Unknown', 'color': 'info'
        }).copy()

    # Step 2: CEO uplift — if Gemini is confident (>= 0.7), add modest boost
    # Cap at +0.08 (reduced from old +0.10) to avoid inflating scores artificially
    if ceo_confidence >= 0.70 and timeframe != '3 Months':
        uplift = (ceo_confidence - 0.50) * 0.16   # max +0.08 at confidence=1.0
        new_score = min(0.92, base['score'] + uplift)
        # Re-label based on new score
        if new_score >= 0.75:
            base = {'score': round(new_score, 3), 'label': 'High',     'color': 'success'}
        elif new_score >= 0.65:
            base = {'score': round(new_score, 3), 'label': 'Good',     'color': 'success'}
        elif new_score >= 0.55:
            base = {'score': round(new_score, 3), 'label': 'Moderate', 'color': 'info'}
        else:
            base['score'] = round(new_score, 3)

    return base



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
        # Ensemble Alpha Engine — called ONCE before the loop (expensive)
        # Produces a 7-day directional signal from Dual-Head Stacker
        # (Direction head: LogisticRegressionCV | Magnitude head: HuberRegressor)
        # Cached here so '1 Week' range can use it without extra I/O
        # -----------------------------------------------------------------------
        try:
            ensemble_result = self.ensemble_forecast()
        except Exception as _e:
            ensemble_result = {}
            print(f"[Ensemble] ensemble_forecast failed, using recursive: {_e}")

        # -----------------------------------------------------------------------
        # Level 1+2 — Run forecasts for all ranges
        # baseline  : old recursive_forecast (kept for comparison column in UI)
        # contextual: new pct_chain_forecast (Scaled Dual-Head Ensemble)
        # -----------------------------------------------------------------------
        for label, steps in FORECAST_RANGES.items():
            # Baseline: legacy absolute-price LSTM (shown as dashed in charts)
            baseline = self.recursive_forecast(steps, ceo_drift_multiplier=1.0)

            # ── Primary contextual path: pct_chain (Dual-Head Ensemble × √t) ──
            contextual = self.pct_chain_forecast(
                steps, 
                ceo_drift_multiplier=ceo_drift_multiplier,
                ensemble_7d_pct=ensemble_result.get('pct_change_7d')
            )

            # ── For 1 Week: override endpoint with Dual-Head Stacker signal ──
            # pct_chain gives the SHAPE of the path; ensemble corrects the target
            if (label == '1 Week'
                    and ensemble_result.get('model') == 'dual_head_ensemble'
                    and ensemble_result.get('pct_change_7d') is not None):
                try:
                    pct_7d    = ensemble_result['pct_change_7d']
                    ens_price = current_price * (1.0 + pct_7d)
                    # Smooth linear interpolation to ensemble target
                    contextual = [
                        current_price + (ens_price - current_price) * (i + 1) / steps
                        for i in range(steps)
                    ]
                except Exception:
                    pass  # keep pct_chain contextual if anything fails

            
            # --- Counterfactual logging (CEO vs Baseline accuracy tracking) ---
            if label == '1 Week' and LLM_AVAILABLE and headlines and not ceo_context.get('is_fallback', True):
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

            # ── Black-Scholes Implied Volatility Monte Carlo ──
            # Cloud width is now driven by real-time market expectations (IV),
            # NOT by historical RMSE. This reflects what institutional money
            # is actually pricing for future volatility via options markets.
            #
            # Sources:
            #   - BTC  : Deribit DVOL (BTC's VIX) via free public API
            #   - Gold/Stocks: CBOE VIX via yfinance (Black-Scholes derived)
            #
            # Modifiers:
            #   - CEO Confidence: still narrows cloud when Gemini is highly certain
            fan_p10, fan_p90 = [], []
            if contextual:
                try:
                    from utils.realtime_prices import get_live_dvol, get_live_vix, iv_to_daily_vol

                    # --- Step 1: Fetch live Implied Volatility ---
                    if self.asset_key == 'btc':
                        iv_annual = get_live_dvol(fallback=60.0)
                        print(f"[Monte Carlo] BTC DVOL (Deribit): {iv_annual:.1f}% annualized")
                    else:
                        # Gold, Stocks: use CBOE VIX as proxy IV
                        iv_annual = get_live_vix(fallback=20.0)
                        print(f"[Monte Carlo] VIX (yfinance): {iv_annual:.1f}% annualized")

                    # --- Step 2: Convert IV to Daily Volatility (Black-Scholes) ---
                    # Standard formula: Daily Vol = IV_annual (%) / 100 / sqrt(365)
                    vol = iv_to_daily_vol(iv_annual)

                    # Safety bounds — prevent degenerate Monte Carlo simulations
                    if self.asset_key == 'btc':
                        vol = max(0.010, min(0.12, vol))   # BTC: 1% - 12% daily
                    elif self.asset_key == 'gold':
                        vol = max(0.003, min(0.04, vol))   # Gold: 0.3% - 4% daily
                    else:
                        vol = max(0.004, min(0.06, vol))   # Stocks: 0.4% - 6% daily

                    print(f"[Monte Carlo] Daily IV vol = {vol:.4f} ({vol*100:.2f}%/day)")

                except Exception as e:
                    # Graceful fallback if both APIs fail
                    print(f"[Monte Carlo] IV fetch failed: {e}. Using conservative historical fallback.")
                    vol = 0.035 if self.asset_key == 'btc' else (0.010 if self.asset_key == 'gold' else 0.012)

                # --- Step 3: CEO Confidence narrows cloud when Gemini is sure ---
                ceo_conf = ceo_context.get('confidence', 0.0)
                if ceo_conf > 0.0:
                    # confidence=1.0 → 40% narrower; confidence=0.5 → 20% narrower
                    ceo_narrow = 1.0 - (ceo_conf * 0.40)
                    vol = vol * max(0.50, ceo_narrow)
                    print(f"[Monte Carlo] After CEO confidence ({ceo_conf:.2f}) adjustment: vol={vol:.4f}")

                # --- Step 4: Run Geometric Brownian Motion Monte Carlo ---
                # Using 500 paths for smoother percentile bands (was 100)
                n_paths = 500
                paths = np.zeros((n_paths, steps))
                for i in range(n_paths):
                    noises = np.random.normal(0, vol, steps)
                    cum_noise = np.exp(np.cumsum(noises) - 0.5 * vol**2 * np.arange(1, steps + 1))
                    paths[i] = np.array(contextual) * cum_noise

                fan_p10 = np.percentile(paths, 10, axis=0).tolist()
                fan_p90 = np.percentile(paths, 90, axis=0).tolist()

            result_entry = {
                'price':            contextual[-1] if contextual else current_price,
                'baseline_price':   baseline[-1]   if baseline   else current_price,
                'confidence':       confidence,
                'series':           contextual,
                'baseline_series':  baseline,
                'fan_p10':          fan_p10,
                'fan_p90':          fan_p90,
            }

            # ── Attach Alpha Engine metadata for 1-Week only ─────────────────
            # Streamlit pages read 'ensemble_meta' key to render the signal panel
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

        # Final type safety guard
        if not isinstance(results, dict):
            return {'Current': current_price, 'error': 'Internal forecast error'}
            
        return results

    
    def get_latest_price(self):
        """Get the most recent actual price"""
        if self.data is None:
            df = self.load_data()
        else:
            df = pd.read_csv(self.config['data_file'])
        
        # First feature is always the price
        return df[self.config['features'][0]].iloc[-1]

    def ensemble_forecast(self) -> dict:
        """
        Generate 7-day % change forecast using Dual-Head Stacker (Ensemble Alpha Engine).

        Architecture:
            LSTM (momentum) + XGBoost (macro) predictions are fed into:
            - Direction Head (LogisticRegressionCV) → probability of UP
            - Magnitude Head (HuberRegressor)       → expected % change size

            Combined output: direction_signal * |magnitude|
            This gives both directional confidence AND magnitude estimate.

        Returns:
            dict with keys:
                pct_change_7d    : float — final % change prediction (signed)
                direction        : 'up' or 'down'
                direction_prob   : float [0.5, 1.0] — confidence in direction
                lstm_signal      : float — LSTM's raw % change prediction
                xgb_signal       : float — XGBoost's raw % change prediction
                predicted_price  : float — implied price after 7 days
                model            : 'ensemble' or 'fallback'

        Falls back to recursive_forecast if stacker files are not found.
        """
        dir_path = f'models/{self.asset_key}_stacker_direction.pkl'
        mag_path = f'models/{self.asset_key}_stacker_magnitude.pkl'
        scl_path = f'models/{self.asset_key}_stacker_meta_scaler.pkl'
        current_price = self.get_latest_price()

        # ── Check if stacker models exist ────────────────────────────────────
        if not all(os.path.exists(p) for p in [dir_path, mag_path, scl_path]):
            # Graceful fallback: use legacy recursive forecast, convert to pct
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

        # ── Load stacker models ───────────────────────────────────────────────
        try:
            with open(dir_path, 'rb') as f: dir_head = pickle.load(f)
            with open(mag_path, 'rb') as f: mag_head = pickle.load(f)
            with open(scl_path, 'rb') as f: meta_scaler = pickle.load(f)
        except Exception as e:
            print(f"[Ensemble] Stacker load error: {e}")
            return {'pct_change_7d': 0.0, 'direction': 'flat',
                    'direction_prob': 0.5, 'predicted_price': current_price,
                    'model': 'fallback', 'lstm_signal': 0.0, 'xgb_signal': 0.0}

        # ── Get base model signals for the CURRENT day ────────────────────────
        # Load last row of data as "test" point
        df_full = pd.read_csv(self.config['data_file'], index_col=0, parse_dates=True)
        df_full = df_full.sort_index()
        df_last = df_full.iloc[[-1]]  # Last row as single-row DataFrame

        # LSTM signal — use last sequence_length rows
        try:
            from tensorflow.keras.models import load_model as keras_load
            feat_scaler_path   = self.config['scaler_file']
            target_scaler_path = self.config['scaler_file'].replace('.pkl', '_target.pkl')

            if all(os.path.exists(p) for p in [self.config['model_file'],
                                                 feat_scaler_path, target_scaler_path]):
                lstm_model = keras_load(self.config['model_file'])
                with open(feat_scaler_path, 'rb') as f:
                    feat_sc = pickle.load(f)
                with open(target_scaler_path, 'rb') as f:
                    tgt_sc = pickle.load(f)

                seq_len  = self.config.get('sequence_length', 60)
                features = [f for f in self.config['features'] if f in df_full.columns]
                window   = df_full[features].ffill().fillna(0).iloc[-seq_len:].values
                if window.shape[0] == seq_len:
                    w_sc    = feat_sc.transform(window)
                    p_sc    = lstm_model.predict(w_sc.reshape(1, seq_len, -1), verbose=0)[0, 0]
                    lstm_signal = float(tgt_sc.inverse_transform([[p_sc]])[0, 0])
                else:
                    lstm_signal = 0.0
            else:
                lstm_signal = 0.0
        except Exception:
            lstm_signal = 0.0

        # XGBoost signal
        xgb_signal = 0.0
        try:
            import xgboost as xgb_lib
            xgb_model_path  = f'models/{self.asset_key}_xgb_macro.json'
            xgb_scaler_path = f'models/{self.asset_key}_xgb_scaler.pkl'
            xgb_feat_path   = f'models/{self.asset_key}_xgb_features.json'
            if all(os.path.exists(p) for p in [xgb_model_path, xgb_scaler_path, xgb_feat_path]):
                xgb_m = xgb_lib.XGBRegressor()
                xgb_m.load_model(xgb_model_path)
                with open(xgb_scaler_path, 'rb') as f:
                    xgb_sc = pickle.load(f)
                with open(xgb_feat_path, 'r') as f:
                    xgb_meta = json.load(f)
                xgb_feats = [ft for ft in xgb_meta['features'] if ft in df_last.columns]
                X_xgb = df_last[xgb_feats].fillna(0).values
                xgb_signal = float(xgb_m.predict(xgb_sc.transform(X_xgb))[0])
        except Exception:
            xgb_signal = 0.0

        # ── Context features ──────────────────────────────────────────────────
        ctx_features = ['VIX', 'GK_Vol_21d', 'Sentiment', 'Sentiment_Std',
                        'YieldCurve_10Y2Y', 'DXY']
        ctx_values = {}
        for f in ctx_features:
            if f in df_last.columns:
                ctx_values[f] = float(df_last[f].iloc[0])

        # ── Build meta-feature vector ─────────────────────────────────────────
        # Must match exact order from training (stored in stacker_meta.json)
        meta_json_path = f'models/{self.asset_key}_stacker_meta.json'
        if os.path.exists(meta_json_path):
            with open(meta_json_path, 'r') as f:
                stacker_meta = json.load(f)
            feature_names = stacker_meta['feature_names']
        else:
            feature_names = ['lstm_pred', 'xgb_pred'] + list(ctx_values.keys())

        row = {}
        row['lstm_pred'] = lstm_signal
        row['xgb_pred']  = xgb_signal
        for f in ctx_features:
            row[f] = ctx_values.get(f, 0.0)

        meta_vec = np.array([[row.get(f, 0.0) for f in feature_names]])

        # ── Dual-Head inference ───────────────────────────────────────────────
        try:
            meta_sc = meta_scaler.transform(meta_vec)
            dir_prob  = float(dir_head.predict_proba(meta_sc)[0, 1])  # P(UP)
            magnitude = float(mag_head.predict(meta_sc)[0])

            # Combined signal: signed by direction confidence, scaled by magnitude
            dir_signal  = (dir_prob - 0.5) * 2.0   # [-1, +1]
            pct_change  = dir_signal * abs(magnitude)

            direction    = 'up' if pct_change > 0 else 'down'
            dir_conf     = max(dir_prob, 1 - dir_prob)  # confidence [0.5, 1.0]
            pred_price   = current_price * (1 + pct_change)

        except Exception as e:
            print(f"[Ensemble] Inference error: {e}")
            pct_change, direction, dir_prob, dir_conf = 0.0, 'flat', 0.5, 0.5
            pred_price = current_price

        return {
            'pct_change_7d':   float(pct_change),
            'direction':       direction,
            'direction_prob':  float(dir_conf),
            'predicted_price': float(pred_price),
            'lstm_signal':     float(lstm_signal),
            'xgb_signal':      float(xgb_signal),
            'model':           'dual_head_ensemble',
        }


    def pct_chain_forecast(self, steps: int, ceo_drift_multiplier: float = 1.0, ensemble_7d_pct: float = None) -> list:
        """
        Build a realistic multi-step price path using sqrt(t) scaling of the 
        canonical Dual-Head Stacker's 7-day prediction.
        
        Strategy:
          1. Retrieve the 7-day pct change target from the Stacker.
          2. Modulate slightly by CEO drift multiplier.
          3. Apply sqrt(t) scaling to project to 1-Day, 2-Weeks, 3-Months, etc.
          4. Interpolate a smooth price path.
        """
        current_price = self.get_latest_price()

        if ensemble_7d_pct is None:
            # If not provided (e.g. called from predict_tomorrow), fetch it
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

        # ── Momentum-scaling: project 7D signal to any horizon ─────────────
        # Instead of conservative √t scaling (which looks like a flat line for long horizons),
        # we scale linearly. This reflects the premise that macro regimes (inflation, etc)
        # persist. If the ensemble predicts a trend, we project that trend confidently.
        momentum_scale = (steps / 7.0) 
        momentum_scale = min(12.0, momentum_scale) # Cap at ~3 months
        horizon_pct  = adjusted_7d_pct * momentum_scale

        # ── Build realistic price path: Warp the LSTM shape to hit our target ───
        target_price = current_price * (1.0 + horizon_pct)
        
        # Get the "bumpy" daily shape from the raw LSTM
        raw_shape = self.recursive_forecast(steps, ceo_drift_multiplier=1.0)
        
        if not raw_shape or len(raw_shape) != steps:
            # Fallback to straight line if raw shape fails
            prices = [
                current_price + (target_price - current_price) * (i + 1) / steps
                for i in range(steps)
            ]
            return prices
            
        # "Tilt" or "Warp" the raw shape so its final point perfectly hits our target_price
        raw_end = raw_shape[-1]
        drift_correction = target_price - raw_end
        
        prices = []
        for i in range(steps):
            # Linearly scale the correction so the start connects to current_price 
            # and the end connects to target_price
            correction_i = drift_correction * ((i + 1) / steps)
            prices.append(raw_shape[i] + correction_i)
            
        return prices

    def predict_tomorrow(self):
        """
        Quick prediction for next day only.
        Uses pct_chain_forecast(1) for consistency with multi-range pipeline.

        Returns:
            dict: {'current': float, 'predicted': float, 'change': float, 'pct_change': float}
        """
        current_price = self.get_latest_price()
        forecast = self.pct_chain_forecast(1)

        if not forecast:
            # Fallback to legacy recursive if pct_chain fails
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
    
    def predict_week(self):
        """
        Quick prediction for 1 week (7 days) ahead.
        Uses ensemble_forecast() (Dual-Head Stacker) for accuracy,
        falls back to pct_chain_forecast(7) if stacker not available.

        Returns:
            dict: {'current': float, 'predicted': float, 'change': float, 'pct_change': float}
        """
        current_price = self.get_latest_price()

        # Try Dual-Head Stacker first (most accurate 7D signal)
        try:
            ens = self.ensemble_forecast()
            if ens.get('model') == 'dual_head_ensemble' and ens.get('pct_change_7d') is not None:
                predicted_price = float(ens['predicted_price'])
                change = predicted_price - current_price
                pct_change = (change / current_price) * 100
                return {
                    'current': float(current_price),
                    'predicted': predicted_price,
                    'change': float(change),
                    'pct_change': float(pct_change),
                    'direction': ens.get('direction', 'flat'),
                    'direction_prob': ens.get('direction_prob', 0.5),
                }
        except Exception:
            pass

        # Fallback: pct_chain 7-step
        forecast = self.pct_chain_forecast(7)
        if not forecast:
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

        predicted_price = forecast[-1]
        change = predicted_price - current_price
        pct_change = (change / current_price) * 100

        return {
            'current': float(current_price),
            'predicted': float(predicted_price),
            'change': float(change),
            'pct_change': float(pct_change),
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
