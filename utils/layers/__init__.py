"""
utils/layers — 3-Layer AI Architecture
=======================================
Layer 1 (Worker):  LSTM inference  → worker_lstm.py
Layer 2 (Manager): Ensemble anchor → manager_anchor.py
Layer 3 (CEO):     LLM override    → ceo_layer.py

Designed for standalone use OR via utils.predictor (orchestrator).
"""

from utils.layers.worker_lstm    import load_lstm_model, predict_next_step, recursive_forecast
from utils.layers.manager_anchor import (
    load_stacker_models, get_lstm_signal, get_xgb_signal,
    run_dual_head_inference, pct_chain_forecast,
)
from utils.layers.ceo_layer      import (
    get_ceo_override, format_ceo_narrative, get_ceo_confidence_score,
)

__all__ = [
    # Layer 1
    'load_lstm_model', 'predict_next_step', 'recursive_forecast',
    # Layer 2
    'load_stacker_models', 'get_lstm_signal', 'get_xgb_signal',
    'run_dual_head_inference', 'pct_chain_forecast',
    # Layer 3
    'get_ceo_override', 'format_ceo_narrative', 'get_ceo_confidence_score',
]
