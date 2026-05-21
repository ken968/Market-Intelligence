"""
Worker Layer (Level 1) — LSTM Inference Engine
================================================
Responsible for:
  - Loading trained LSTM/Attention-LSTM models
  - Single-step prediction
  - Recursive multi-step forecasting with CEO drift injection
  - Feature updating during recursive rollout

This module is STATELESS — all state lives in AssetPredictor.
Import from utils.predictor (orchestrator) for end-to-end usage.
"""

import numpy as np
import os

try:
    from tensorflow.keras.models import load_model as _tf_load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def load_lstm_model(model_path: str, scaler_path: str):
    """
    Load a trained LSTM model and its MinMaxScaler.

    Args:
        model_path  : Path to .keras model file
        scaler_path : Path to .pkl scaler file

    Returns:
        (model, scaler) tuple, or (None, None) on failure
    """
    import pickle

    if not TF_AVAILABLE:
        return None, None

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    try:
        model  = _tf_load_model(model_path)
        with open(scaler_path, 'rb') as fh:
            scaler = pickle.load(fh)
        return model, scaler
    except Exception as e:
        print(f"[worker_lstm] Error loading model: {e}")
        return None, None


def predict_next_step(model, scaler, sequence: np.ndarray) -> float:
    """
    Run one forward-pass through the LSTM model.

    Args:
        model    : Keras model
        scaler   : MinMaxScaler (for feature count guard)
        sequence : np.ndarray shape (1, seq_len, n_features)

    Returns:
        float: scaled prediction value
    """
    if model is None:
        return float(sequence[0, -1, 0])

    # Feature count safety check (migration: old scalers may have fewer features)
    if hasattr(scaler, 'n_features_in_'):
        expected_n = scaler.n_features_in_
        if sequence.shape[2] != expected_n:
            sequence = sequence[:, :, :expected_n]

    try:
        pred = model.predict(sequence, verbose=0)
        return float(pred[0, 0])
    except Exception:
        return float(sequence[0, -1, 0])


def recursive_forecast(
    model,
    scaler,
    data: np.ndarray,
    config: dict,
    steps: int,
    asset_key: str,
    ceo_drift_multiplier: float = 1.0,
) -> list:
    """
    Generate a multi-step price forecast via recursive LSTM rollout.

    Architecture:
      - Weighted Multi-Scale Anchor (90-day short-term + full-history long-term)
      - Adaptive damping: trust factor decays linearly over 365 steps
      - Dynamic feature updating (EMA_90, Halving_Cycle, macro drift)
      - CEO drift multiplier adjusts anchor balance

    Args:
        model                : Loaded Keras model
        scaler               : MinMaxScaler
        data                 : Raw feature array (n_rows × n_features)
        config               : Asset config dict from utils.config
        steps                : Number of forecast steps
        asset_key            : e.g. 'gold', 'btc'
        ceo_drift_multiplier : float — shifts anchor weight (clamp [0.85, 1.15])

    Returns:
        list of float: predicted prices in original scale
    """
    if not TF_AVAILABLE or model is None:
        return []

    # ── Weighted Multi-Scale Anchor ──────────────────────────────────────────
    short_window        = min(90, len(data))
    feature_means_short = np.mean(data[-short_window:], axis=0)
    feature_means_long  = np.mean(data, axis=0)

    dm      = max(0.85, min(1.15, ceo_drift_multiplier))
    w_short = 0.40 * dm
    w_long  = 1.0 - w_short
    feature_means = w_short * feature_means_short + w_long * feature_means_long

    # ── Scale data ───────────────────────────────────────────────────────────
    data_to_scale  = data
    means_to_scale = feature_means

    if hasattr(scaler, 'n_features_in_'):
        expected_n     = scaler.n_features_in_
        data_to_scale  = data_to_scale[:, :expected_n]
        means_to_scale = means_to_scale[:expected_n]

    scaled_data  = scaler.transform(data_to_scale)
    scaled_means = scaler.transform(means_to_scale.reshape(1, -1))[0]

    seq_len           = config['sequence_length']
    n_scaled_features = scaled_data.shape[1]
    current_batch     = scaled_data[-seq_len:].reshape(1, seq_len, n_scaled_features)

    predictions = []
    temp_data   = current_batch.copy()
    features    = config['features']

    for i in range(steps):
        pred_scaled    = predict_next_step(model, scaler, temp_data)
        prev_frame     = temp_data[0, -1, :].copy()
        prev_price_sc  = prev_frame[0]
        start_price_sc = current_batch[0, -1, 0]

        # Adaptive damping
        ai_delta     = np.clip(pred_scaled - prev_price_sc, -0.008, 0.008)
        trust_factor = max(0.05, 1.0 - (i / 365.0))
        decay        = 0.99 if asset_key != 'btc' else 0.97
        ai_movement  = ai_delta * decay * trust_factor
        anchor_pull  = (start_price_sc - prev_price_sc) * (1.0 - trust_factor) * 0.01
        new_price_sc = prev_price_sc + ai_movement + anchor_pull

        last_frame    = prev_frame.copy()
        last_frame[0] = new_price_sc

        # Dynamic feature updating during rollout
        for f_idx in range(1, len(last_frame)):
            if f_idx >= len(features):
                break
            f_name = features[f_idx]
            if f_name == 'EMA_90':
                alpha = 2.0 / (90.0 + 1.0)
                last_frame[f_idx] = (new_price_sc * alpha) + (prev_frame[f_idx] * (1.0 - alpha))
            elif f_name == 'Halving_Cycle':
                last_frame[f_idx] = max(0, prev_frame[f_idx] - scaler.scale_[f_idx])
            elif f_name in ['Sentiment', 'DXY', 'VIX', 'Yield_10Y', 'Oil_Price']:
                drift_rate = 0.002
                last_frame[f_idx] += (scaled_means[f_idx] - last_frame[f_idx]) * drift_rate

        temp_data = np.append(temp_data[:, 1:, :], [[last_frame]], axis=1)
        predictions.append(new_price_sc)

    # Inverse transform
    n_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else len(features)
    dummy         = np.zeros((len(predictions), n_features))
    dummy[:, 0]   = predictions
    preds_orig    = scaler.inverse_transform(dummy)[:, 0]

    # Price floor: no price below 20% of start price
    start_price = scaler.inverse_transform(current_batch[0, -1, :].reshape(1, -1))[0, 0]
    preds_orig  = np.maximum(preds_orig, start_price * 0.2)

    return preds_orig.tolist()
