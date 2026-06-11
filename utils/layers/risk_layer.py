"""
Risk Layer (Level 3) -- Rolling OOS IC + EWMA Quorum Inference
===============================================================
Implements the Quorum-Based Multi-Window Ensemble (Phase 7).

Architecture:
  Worker Layer  : LSTM windows 1-5 (per model group)
  Manager Layer : Model_A (1/7/14D) and Model_B (30/90D) -- direction + magnitude
  Risk Layer    : IC-EWMA weighting + Kelly shrinkage   -- position sizing

This module handles:
  1. compute_ic_ewma_weights()   -- per-window importance via OOS IC + recency decay
  2. load_window_models_for_quorum() -- load all 5 non-collapsed windows
  3. quorum_inference()          -- weighted ensemble mean + uncertainty decomposition
  4. kelly_shrinkage_factor()    -- position sizing from Kelly Criterion

Uncertainty decomposition (CRITICAL SEPARATION):
  - Epistemic (model) uncertainty : MC Dropout std  -> used for Kelly shrinkage
  - Aleatoric (market) uncertainty: VIX/DVOL        -> used for GBM fan chart width
  These two MUST NOT be summed carelessly (different risk layers, different use).

EWMA Lambda by asset (asset-specific regime speed):
  BTC  : 0.85 (half-life ~4d  -- crypto needs fast regime adaptation)
  Gold : 0.94 (half-life ~11d -- macro-driven, needs stability)
  SPY  : 0.92, QQQ: 0.90, DIA: 0.93
"""

import os
import json
import numpy as np
import joblib
from typing import Optional

try:
    from tensorflow.keras.models import load_model as _tf_load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ---- Default EWMA lambdas (mirrors EWMA_LAMBDA in train_lstm_pct.py) --------
EWMA_LAMBDA_DEFAULT = {
    'gold': 0.94,
    'btc':  0.85,
    'spy':  0.92,
    'qqq':  0.90,
    'dia':  0.93,
}


def _load_registry(registry_path: str = 'models/model_registry.json') -> dict:
    """Load model_registry.json. Returns empty dict on any failure."""
    try:
        with open(registry_path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


# =============================================================================
# 1. EWMA IC WEIGHT COMPUTATION
# =============================================================================

def compute_ic_ewma_weights(
    asset: str,
    model_group: str,
    horizons: list,
    ewma_lambda: Optional[float] = None,
    registry: Optional[dict] = None,
) -> tuple[dict, float]:
    """
    Compute EWMA-weighted window importance for OOS-IC-based ensemble.

    Window ordering: W1 is earliest (trained on least data), W5 is most recent.
    Recency decay gives W5 the highest base weight:
      recency[w] = lambda^(5-w)   (W5=1.0, W4=l, W3=l^2, W2=l^3, W1=l^4)

    IC weighting: only positive IC contributes (negative IC = worse than naive).
      ic_score[w] = mean(IC across all horizons) from pre-computed OOS metrics.

    Combined:
      raw[w] = recency[w] * max(0, ic_score[w])
      If ALL ICs negative -> fall back to recency-only weights.
      Collapsed windows are zeroed out.

    Args:
        asset       : 'gold', 'btc', etc.
        model_group : 'Model_A' or 'Model_B'
        horizons    : e.g. [1, 7, 14] -- used to select IC columns
        ewma_lambda : decay factor (0-1). If None, uses EWMA_LAMBDA_DEFAULT.
        registry    : pre-loaded registry dict (loaded from file if None)

    Returns:
        tuple(
            dict[window_int (1-5)] -> normalized float weight (sum = 1.0),
            float -> weighted average IC score across non-collapsed windows
        )
    """
    if registry is None:
        registry = _load_registry()
    if ewma_lambda is None:
        ewma_lambda = EWMA_LAMBDA_DEFAULT.get(asset.lower(), 0.92)

    asset_lower = asset.lower()
    n_windows   = 5
    ic_scores   = {}

    for w in range(1, n_windows + 1):
        key   = f"{asset_lower}_{model_group.lower()}_w{w}"
        entry = registry.get(key, {})

        if entry.get('collapsed', False):
            ic_scores[w] = -1.0   # collapsed -> exclude
            continue

        metrics = entry.get('metrics', {})
        # Mean IC across horizons that have an OOS IC recorded
        ics = [metrics.get(f'ic_{h}d', None) for h in horizons]
        ics = [v for v in ics if v is not None]
        ic_scores[w] = float(np.mean(ics)) if ics else 0.0

    # Recency weights: W5 most recent, W1 oldest
    recency   = {w: ewma_lambda ** (n_windows - w) for w in range(1, n_windows + 1)}

    # Combined raw weights (IC-gated: negative IC -> 0)
    raw = {w: recency[w] * max(0.0, ic_scores.get(w, 0.0))
           for w in range(1, n_windows + 1)}
    total_raw = sum(raw.values())

    if total_raw < 1e-8:
        # All ICs negative or zero -- fall back to pure recency weights
        raw       = recency.copy()
        total_raw = sum(raw.values())

    if total_raw < 1e-8:
        # Extreme fallback: equal weights
        return {w: 1.0 / n_windows for w in range(1, n_windows + 1)}, 0.0

    normalized = {w: raw[w] / total_raw for w in range(1, n_windows + 1)}
    
    # Calculate weighted average IC
    avg_ic = sum(normalized.get(w, 0.0) * ic_scores.get(w, 0.0) for w in range(1, n_windows + 1))
    
    return normalized, float(avg_ic)


# =============================================================================
# 2. LOAD ALL WINDOW MODELS FOR QUORUM
# =============================================================================

def load_window_models_for_quorum(
    asset: str,
    model_group: str,
    registry: Optional[dict] = None,
) -> list:
    """
    Load all non-collapsed window models for quorum inference.

    Reads model paths from model_registry.json and loads each .keras model
    with compile=False (skips directional_mse custom loss -- not needed for
    inference).

    Args:
        asset       : 'gold', 'btc', etc.
        model_group : 'Model_A' or 'Model_B'
        registry    : pre-loaded registry dict

    Returns:
        list of dicts, one per loaded window:
          {
            'window'        : int (1-5),
            'model'         : keras Model,
            'feature_scaler': MinMaxScaler,
            'target_scaler' : StandardScaler,
            'horizons'      : list[int],
          }
        Empty list if TF unavailable or no models found.
    """
    if not TF_AVAILABLE:
        return []

    if registry is None:
        registry = _load_registry()

    asset_lower = asset.lower()
    loaded      = []

    for w in range(1, 6):
        key   = f"{asset_lower}_{model_group.lower()}_w{w}"
        entry = registry.get(key, {})
        if not entry:
            continue
        if entry.get('collapsed', False):
            continue

        m_path = entry.get('model_path', '')
        s_path = entry.get('scaler_path', '')

        if not os.path.exists(m_path) or not os.path.exists(s_path):
            continue

        try:
            model  = _tf_load_model(m_path, compile=False)
            bundle = joblib.load(s_path)
            loaded.append({
                'window':          w,
                'model':           model,
                'feature_scaler':  bundle.get('feature_scaler'),
                'target_scaler':   bundle.get('target_scaler'),
                'horizons':        entry.get('horizons', []),
            })
        except Exception as exc:
            print(f"[risk_layer] W{w} load error ({m_path}): {exc}")
            continue

    return loaded


# =============================================================================
# 3. QUORUM INFERENCE
# =============================================================================

def quorum_inference(
    window_models: list,
    ic_weights: dict,
    data: np.ndarray,
    config: dict,
    horizons: list,
    market_volatility: float = 0.05,
    n_mc_samples: int = 30,
    avg_ic: float = 0.0,
) -> dict:
    """
    Weighted ensemble inference across all loaded window models.

    For each window w:
      - Run predict_phase7_multi_output (MC Dropout n_mc_samples passes)
      - Produces: mean_pred[h], mc_std[h] (epistemic uncertainty)

    Ensemble per horizon h:
      - ensemble_mean[h]  = weighted mean of mean_pred[w][h]
      - mc_epistemic_std  = weighted mean of mc_std[w][h]     (model uncertainty)
      - cross_window_std  = weighted std of mean_pred[w][h]   (regime disagreement)
      - total_uncertainty = sqrt(mc_epistemic_std^2 + cross_window_std^2)

    The two uncertainty components serve different risk management purposes:
      - mc_epistemic_std  -> Kelly shrinkage (model doesn't know)
      - cross_window_std  -> signals regime change detection
      - total_uncertainty -> overall uncertainty for position sizing

    Args:
        window_models : output of load_window_models_for_quorum()
        ic_weights    : output of compute_ic_ewma_weights()
        data          : (N, n_features) numpy array
        config        : asset config dict
        horizons      : e.g. [1, 7, 14]
        market_volatility: recent market vol (aleatoric) for this horizon
        n_mc_samples  : per-window MC Dropout passes (30 default for speed)
        avg_ic        : average IC score (from compute_ic_ewma_weights)

    Returns:
        dict[horizon_day] -> {
            'mean'              : float,
            'mc_std'            : float,   # epistemic uncertainty
            'cross_std'         : float,   # inter-window disagreement
            'total_uncertainty' : float,
            'kelly_fraction'    : float,   # Recommended position sizing
            'window_preds'      : list[(w, pred)],  # for diagnostics
            'n_windows_used'    : int,
        }
    """
    from utils.layers.worker_lstm import predict_phase7_multi_output

    fallback = {
        h: {
            'mean':               0.0,
            'mc_std':             0.0,
            'cross_std':          0.0,
            'total_uncertainty':  0.0,
            'kelly_fraction':     0.0,
            'window_preds':       [],
            'n_windows_used':     0,
        }
        for h in horizons
    }

    if not window_models:
        return fallback

    # ---- Per-window inference -----------------------------------------------
    window_means  = {}   # w -> {h: mean}
    window_mc_std = {}   # w -> {h: mc_std}

    for wm in window_models:
        w      = wm['window']
        weight = ic_weights.get(w, 0.0)
        if weight < 1e-8:
            continue   # IC weight is zero -> skip (collapsed or all-neg)

        results = predict_phase7_multi_output(
            model          = wm['model'],
            feature_scaler = wm['feature_scaler'],
            target_scaler  = wm['target_scaler'],
            data           = data,
            config         = config,
            horizons       = horizons,
            n_mc_samples   = n_mc_samples,
        )
        window_means[w]  = {h: results[h]['mean'] for h in horizons}
        window_mc_std[w] = {h: results[h]['std']  for h in horizons}

    if not window_means:
        return fallback

    # ---- Weighted ensemble per horizon --------------------------------------
    result = {}
    for h in horizons:
        weights = {w: ic_weights.get(w, 0.0) for w in window_means}
        total_w = sum(weights.values())
        if total_w < 1e-8:
            result[h] = fallback[h]
            continue

        # Normalize to this subset
        w_norm = {w: weights[w] / total_w for w in weights}

        # Weighted mean prediction
        ens_mean = sum(w_norm[w] * window_means[w][h] for w in window_means)

        # Weighted MC epistemic std (model uncertainty)
        ens_mc_std = sum(w_norm[w] * window_mc_std[w][h] for w in window_means)

        # Cross-window std (regime disagreement between windows)
        preds_arr   = np.array([window_means[w][h] for w in sorted(window_means)])
        weights_arr = np.array([w_norm[w] for w in sorted(window_means)])
        cross_std   = float(np.sqrt(
            np.sum(weights_arr * (preds_arr - ens_mean) ** 2)
        ))

        total_unc = float(np.sqrt(ens_mc_std ** 2 + cross_std ** 2))
        
        # Scale market volatility for this horizon (assume input is 7D vol)
        h_vol = market_volatility * np.sqrt(h / 7.0)

        kelly = kelly_shrinkage_factor(
            mean_pred=ens_mean,
            market_volatility=h_vol,
            total_uncertainty=total_unc,
            ic_score=avg_ic,
            cross_window_std=cross_std
        )

        result[h] = {
            'mean':               float(ens_mean),
            'mc_std':             float(ens_mc_std),
            'cross_std':          float(cross_std),
            'total_uncertainty':  total_unc,
            'kelly_fraction':     float(kelly),
            'window_preds':       [(w, window_means[w][h]) for w in sorted(window_means)],
            'n_windows_used':     len(window_means),
        }

    return result


# =============================================================================
# 4. KELLY SHRINKAGE FACTOR
# =============================================================================

def kelly_shrinkage_factor(
    mean_pred: float,
    market_volatility: float,
    total_uncertainty: float,
    ic_score: float,
    cross_window_std: float,
    max_kelly: float = 0.25,
) -> float:
    """
    Compute Kelly-shrunk position size fraction.

    Base formula (Half-Kelly for conservatism):
      f_half = 0.5 * |mean| / (market_variance)
      (Market variance = aleatoric uncertainty).

    Additional shrinkage multipliers:
      ic_penalty     : reward high IC (low IC -> shrink)
      cross_penalty  : punish high inter-window disagreement
      epistemic_pen  : punish high model uncertainty

    Final Kelly fraction clamped to [0, max_kelly].

    Args:
        mean_pred         : ensemble mean predicted return
        market_volatility : std of recent market returns for this horizon
        total_uncertainty : sqrt(mc_std^2 + cross_std^2)
        ic_score          : mean OOS IC across horizons
        cross_window_std  : std of window predictions
        max_kelly         : upper cap (default 0.25)
    """
    if market_volatility < 1e-8 or abs(mean_pred) < 1e-6:
        return 0.0

    # Base: Half-Kelly using market variance (aleatoric risk)
    market_var = market_volatility ** 2
    full_kelly = abs(mean_pred) / market_var
    half_kelly = full_kelly * 0.5

    # IC penalty: ranges from 0.3 (IC=-1) to 1.0 (IC=1)
    ic_penalty = float(np.clip(0.5 + 0.5 * max(0.0, ic_score), 0.3, 1.0))

    # Cross-window coefficient of variation penalty
    if abs(mean_pred) > 1e-8:
        cv = cross_window_std / abs(mean_pred)
        cross_penalty = float(np.clip(1.0 - 0.3 * cv, 0.3, 1.0))
    else:
        cross_penalty = 0.3

    # Epistemic penalty: punish if model is highly uncertain relative to mean prediction
    if abs(mean_pred) > 1e-8:
        epi_cv = total_uncertainty / abs(mean_pred)
        epistemic_penalty = float(np.clip(1.0 - 0.2 * epi_cv, 0.1, 1.0))
    else:
        epistemic_penalty = 0.1

    shrunken = half_kelly * ic_penalty * cross_penalty * epistemic_penalty
    return float(np.clip(shrunken, 0.0, max_kelly))
