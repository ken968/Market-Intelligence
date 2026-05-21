"""
CEO Layer (Level 3) — LLM Strategic Override
=============================================
Responsible for:
  - Calling Gemini LLM with macro + news context to generate drift multiplier
  - Computing CEO confidence score for bias injection
  - Providing strategic narrative for UI display

This module wraps utils.llm_manager and utils.macro_processor.
It is the ONLY place that calls the LLM — all other layers are LLM-free.

Import from utils.predictor (orchestrator) for end-to-end usage.
"""

from typing import Optional, Tuple


def get_ceo_override(
    asset_key: str,
    headlines: Optional[list] = None,
    published_at_list: Optional[list] = None,
    macro_context: Optional[dict] = None,
) -> Tuple[float, float, Optional[dict]]:
    """
    Call the CEO Layer (Gemini LLM) to generate:
      - drift_multiplier : float in [0.85, 1.15] — how much to amplify/dampen LSTM
      - ceo_confidence   : float in [0.0, 1.0]   — how confident Gemini is
      - ceo_context      : dict with narrative, rationale, signal keys

    Args:
        asset_key         : e.g. 'gold', 'btc', 'aapl'
        headlines         : list of news headline strings (optional)
        published_at_list : list of ISO datetime strings for staleness check
        macro_context     : pre-built macro context dict (to avoid double-fetch)

    Returns:
        (drift_multiplier, ceo_confidence, ceo_context)
        Falls back to (1.0, 0.0, None) if LLM unavailable or fails.
    """
    try:
        from utils.llm_manager import compute_drift_multiplier
        from utils.macro_processor import build_macro_context

        if macro_context is None:
            macro_context = build_macro_context()

        macro_summary = macro_context.get('macro_summary', '')

        result = compute_drift_multiplier(
            asset_key        = asset_key,
            macro_summary    = macro_summary,
            headlines        = headlines or [],
            published_at_list= published_at_list or [],
        )

        drift_multiplier = float(result.get('drift_multiplier', 1.0))
        ceo_confidence   = float(result.get('confidence', 0.0))
        ceo_context      = result  # Full dict for UI display

        return drift_multiplier, ceo_confidence, ceo_context

    except Exception as e:
        print(f"[ceo_layer] CEO Layer unavailable: {e}")
        return 1.0, 0.0, None


def format_ceo_narrative(ceo_context: Optional[dict], asset_key: str) -> str:
    """
    Extract human-readable narrative from CEO Layer output for UI display.

    Args:
        ceo_context : dict returned by compute_drift_multiplier
        asset_key   : e.g. 'gold'

    Returns:
        str: narrative text, or empty string if unavailable
    """
    if not ceo_context:
        return ""

    narrative = ceo_context.get('narrative', '')
    rationale = ceo_context.get('rationale', '')
    signal    = ceo_context.get('signal', 'neutral')
    drift     = ceo_context.get('drift_multiplier', 1.0)

    parts = []
    if narrative:
        parts.append(narrative)
    if rationale:
        parts.append(f"Rationale: {rationale}")
    if signal and drift:
        direction = "Bullish" if drift > 1.0 else ("Bearish" if drift < 1.0 else "Neutral")
        parts.append(f"CEO Signal: {direction} (drift={drift:.3f})")

    return "\n".join(parts)


def get_ceo_confidence_score(ceo_context: Optional[dict]) -> float:
    """
    Extract CEO confidence score from context dict.

    Returns:
        float in [0.0, 1.0], or 0.0 if unavailable
    """
    if not ceo_context:
        return 0.0
    return float(ceo_context.get('confidence', 0.0))
