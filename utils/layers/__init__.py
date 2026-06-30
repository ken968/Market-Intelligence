"""
utils/layers — 3-Layer AI Architecture
=======================================
Layer 1 (Worker):  LSTM inference  → worker_lstm.py
Layer 2 (Manager): Risk Layer (Phase 7) → risk_layer.py
Layer 3 (CEO):     LLM override    → ceo_layer.py

Designed for standalone use OR via utils.predictor (orchestrator).
"""

from utils.layers.worker_lstm    import load_lstm_model, predict_next_step, recursive_forecast
from utils.layers.ceo_layer      import (
    get_ceo_override, format_ceo_narrative, get_ceo_confidence_score,
)

__all__ = [
    # Layer 1
    'load_lstm_model', 'predict_next_step', 'recursive_forecast',
    # Layer 3
    'get_ceo_override', 'format_ceo_narrative', 'get_ceo_confidence_score',
]
