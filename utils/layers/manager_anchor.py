"""
Manager Layer (Level 2) — Ensemble Alpha Engine & Anchoring
============================================================
Responsible for:
  - Dual-Head Stacker inference (Direction Head + Magnitude Head)
  - LSTM + XGBoost signal extraction for meta-feature vector
  - pct_chain_forecast: converts 7D stacker signal → any horizon price path
  - Correlation enforcement via CorrelationEnforcer

This module contains the pure logic; state management lives in AssetPredictor.
Import from utils.predictor (orchestrator) for end-to-end usage.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd


def load_stacker_models(asset_key: str):
    """
    Load Dual-Head Stacker files for a given asset.

    Returns:
        (dir_head, mag_head, meta_scaler, stacker_meta) or None if not found
    """
    dir_path  = f'models/{asset_key}_stacker_direction.pkl'
    mag_path  = f'models/{asset_key}_stacker_magnitude.pkl'
    scl_path  = f'models/{asset_key}_stacker_meta_scaler.pkl'
    meta_path = f'models/{asset_key}_stacker_meta.json'

    if not all(os.path.exists(p) for p in [dir_path, mag_path, scl_path]):
        return None

    try:
        with open(dir_path,  'rb') as fh: dir_head     = pickle.load(fh)
        with open(mag_path,  'rb') as fh: mag_head     = pickle.load(fh)
        with open(scl_path,  'rb') as fh: meta_scaler  = pickle.load(fh)
        stacker_meta = None
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as fh:
                stacker_meta = json.load(fh)
        return dir_head, mag_head, meta_scaler, stacker_meta
    except Exception as e:
        print(f"[manager_anchor] Stacker load error: {e}")
        return None


def get_lstm_signal(asset_key: str, config: dict, df_full: pd.DataFrame) -> float:
    """
    Extract LSTM 7-day % change prediction from the trained LSTM model.

    Uses the _target scaler (% change target) if available.
    Falls back to 0.0 on any failure.
    """
    try:
        from tensorflow.keras.models import load_model as keras_load
        feat_scaler_path   = config['scaler_file']
        target_scaler_path = config['scaler_file'].replace('.pkl', '_target.pkl')

        if not all(os.path.exists(p) for p in [config['model_file'],
                                                feat_scaler_path,
                                                target_scaler_path]):
            return 0.0

        lstm_model = keras_load(config['model_file'])
        with open(feat_scaler_path, 'rb') as fh:  feat_sc = pickle.load(fh)
        with open(target_scaler_path, 'rb') as fh: tgt_sc  = pickle.load(fh)

        seq_len  = config.get('sequence_length', 60)
        features = [f for f in config['features'] if f in df_full.columns]
        window   = df_full[features].ffill().fillna(0).iloc[-seq_len:].values

        if window.shape[0] != seq_len:
            return 0.0

        w_sc = feat_sc.transform(window)
        p_sc = lstm_model.predict(w_sc.reshape(1, seq_len, -1), verbose=0)[0, 0]
        return float(tgt_sc.inverse_transform([[p_sc]])[0, 0])

    except Exception:
        return 0.0


def get_xgb_signal(asset_key: str, df_last: pd.DataFrame) -> float:
    """
    Extract XGBoost macro model 7-day % change prediction.
    Falls back to 0.0 on any failure.
    """
    try:
        import xgboost as xgb_lib
        import re
        from utils.config import get_asset_config

        xgb_model_path  = f'models/{asset_key}_xgb_macro.json'
        xgb_scaler_path = f'models/{asset_key}_xgb_scaler.pkl'
        xgb_feat_path   = f'models/{asset_key}_xgb_features.json'

        if not all(os.path.exists(p) for p in [xgb_model_path,
                                                xgb_scaler_path,
                                                xgb_feat_path]):
            return 0.0

        xgb_m = xgb_lib.XGBRegressor()
        xgb_m.load_model(xgb_model_path)
        with open(xgb_scaler_path, 'rb') as fh: xgb_sc = pickle.load(fh)
        with open(xgb_feat_path,   'r')  as fh: xgb_meta = json.load(fh)

        # ── Fetch full data for robust lag computation ──
        config = get_asset_config(asset_key)
        if config and os.path.exists(config['data_file']):
            df_full = pd.read_csv(config['data_file'], index_col=0, parse_dates=True).sort_index()
        else:
            df_full = df_last

        # ── Merge COT data for robust inference ──
        cot_file = f"data/cot_{asset_key}.csv"
        if os.path.exists(cot_file):
            try:
                cot_df = pd.read_csv(cot_file, parse_dates=['Date'])
                cot_df.set_index('Date', inplace=True)
                df_full = df_full.join(cot_df, how='left')
                for col in ['Net_Commercial', 'Net_NonCommercial', 'Net_Commercial_Long']:
                    if col in df_full.columns:
                        df_full[col] = df_full[col].ffill()
                        
                # Default fillna for early dates
                df_full['Net_Commercial'] = df_full.get('Net_Commercial', pd.Series(0, index=df_full.index)).fillna(0)
                df_full['Net_NonCommercial'] = df_full.get('Net_NonCommercial', pd.Series(0, index=df_full.index)).fillna(0)
                df_full['Net_Commercial_Long'] = df_full.get('Net_Commercial_Long', pd.Series(0.5, index=df_full.index)).fillna(0.5)
                
                # Synthetic Divergence Feature 1: Spot Sentiment vs Futures Positioning
                if 'Sentiment' in df_full.columns:
                    df_full['Inst_Sentiment_Ratio'] = df_full['Sentiment'] / df_full['Net_Commercial'].replace(0, 1e-5)
                    
                # Synthetic Divergence Feature 2: Retail Fear/Greed vs Smart Money
                if 'Fear_Greed' in df_full.columns:
                    window = 756
                    rolling_min = df_full['Net_Commercial'].rolling(window=window, min_periods=1).min()
                    rolling_max = df_full['Net_Commercial'].rolling(window=window, min_periods=1).max()
                    cot_index = (df_full['Net_Commercial'] - rolling_min) / (rolling_max - rolling_min).replace(0, 1) * 100
                    df_full['Smart_Money_Sentiment_Gap'] = df_full['Fear_Greed'] - cot_index
                    
            except Exception as e:
                print(f"[manager_anchor] Error loading COT: {e}")

        # ── Auto-generate missing lag features on df_full ──
        df_work = df_full.copy()
        required_features = xgb_meta['features']
        for feat in required_features:
            if feat not in df_work.columns:
                match = re.match(r'^(.+)_lag(\d+)$', feat)
                if match:
                    base_col, lag_n = match.group(1), int(match.group(2))
                    if base_col in df_work.columns:
                        trading_days_per_month = 21
                        lag_days = lag_n * trading_days_per_month
                        df_work[feat] = df_work[base_col].shift(lag_days)
                        df_work[feat] = df_work[feat].ffill().fillna(0)
                    else:
                        df_work[feat] = 0.0
                else:
                    df_work[feat] = 0.0

        # Extract last row aligned to required features
        df_last_aligned = df_work[required_features].iloc[[-1]].fillna(0)
        X_xgb = df_last_aligned.values

        # Feature count safety check (pad/truncate to match StandardScaler size)
        n_expected = getattr(xgb_sc, 'n_features_in_', None)
        if n_expected is None and hasattr(xgb_sc, 'mean_'):
            n_expected = len(xgb_sc.mean_)

        if n_expected is not None and X_xgb.shape[1] != n_expected:
            if X_xgb.shape[1] < n_expected:
                pad = np.zeros((X_xgb.shape[0], n_expected - X_xgb.shape[1]))
                X_xgb = np.hstack([X_xgb, pad])
            else:
                X_xgb = X_xgb[:, :n_expected]

        return float(xgb_m.predict(xgb_sc.transform(X_xgb))[0])

    except Exception as e:
        print(f"[manager_anchor] XGBoost predict error: {e}")
        return 0.0


def run_dual_head_inference(
    dir_head,
    mag_head,
    meta_scaler,
    stacker_meta,
    lstm_signal: float,
    xgb_signal: float,
    ctx_values: dict,
    current_price: float,
) -> dict:
    """
    Run Dual-Head Stacker inference given pre-computed signals.

    Returns:
        dict with pct_change_7d, direction, direction_prob, predicted_price, etc.
    """
    CTX_FEATURES = ['VIX', 'GK_Vol_21d', 'Sentiment', 'Sentiment_Std',
                    'YieldCurve_10Y2Y', 'DXY']

    if stacker_meta and 'feature_names' in stacker_meta:
        feature_names = stacker_meta['feature_names']
    else:
        feature_names = ['lstm_pred', 'xgb_pred'] + CTX_FEATURES

    row = {'lstm_pred': lstm_signal, 'xgb_pred': xgb_signal}
    row.update({f: ctx_values.get(f, 0.0) for f in CTX_FEATURES})

    meta_vec = np.array([[row.get(f, 0.0) for f in feature_names]])

    try:
        meta_sc    = meta_scaler.transform(meta_vec)
        dir_prob   = float(dir_head.predict_proba(meta_sc)[0, 1])
        magnitude  = float(mag_head.predict(meta_sc)[0])

        dir_signal  = (dir_prob - 0.5) * 2.0      # [-1, +1]
        pct_change  = dir_signal * abs(magnitude)
        direction   = 'up' if pct_change > 0 else 'down'
        dir_conf    = max(dir_prob, 1 - dir_prob)
        pred_price  = current_price * (1 + pct_change)

    except Exception as e:
        print(f"[manager_anchor] Inference error: {e}")
        pct_change = 0.0
        direction  = 'flat'
        dir_prob   = 0.5
        dir_conf   = 0.5
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


def pct_chain_forecast(
    current_price: float,
    ensemble_7d_pct: float,
    steps: int,
    ceo_drift_multiplier: float = 1.0,
) -> list:
    """
    Convert a 7-day % change signal from the Stacker → curved price path of `steps` length.

    Strategy:
      - Scales 7D signal using a power-law momentum decay curve (t/7)^0.65
      - CEO multiplier modulates magnitude
      - Returns curved path from current → target price

    Args:
        current_price        : latest actual price
        ensemble_7d_pct      : Dual-Head Stacker's 7-day % change prediction
        steps                : number of price points to generate
        ceo_drift_multiplier : [0.85, 1.15] — CEO Layer direction bias

    Returns:
        list of float prices (length = steps)
    """
    dm              = max(0.85, min(1.15, ceo_drift_multiplier))
    adjusted_7d_pct = ensemble_7d_pct * dm

    alpha = 0.65  # power-law decay exponent for return projection
    prices = []
    for i in range(1, steps + 1):
        horizon_pct = adjusted_7d_pct * ((i / 7.0) ** alpha)
        prices.append(current_price * (1.0 + horizon_pct))
    return prices
