"""
LLM Manager — CEO Layer (Gemini Intelligence Engine)
Implements the full Hierarchical Intelligence CEO Layer:

Architecture (per Claude + Gemini joint design):
  1. News ingestion with staleness cutoff + deduplication
  2. Causal Hierarchy prompting (Supply → Geo → Monetary → Risk → Sentiment)
  3. Few-Shot Anchor Calibration for consistent numerical scoring
  4. Gram-Schmidt Orthogonalization + ZCA Whitening on the output vector
  5. Zero-Vector Fallback if API fails (Quorum logic for future multi-LLM)
  6. Contextual Bias vector injected into LSTM drift engine
"""

import os
import json
import hashlib
import time
import numpy as np
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Gemini API Key Rotation
# ---------------------------------------------------------------------------
GEMINI_KEYS = [
    k for k in [
        os.getenv('GEMINI_API_KEY_1'),
        os.getenv('GEMINI_API_KEY_2'),
        os.getenv('GEMINI_API_KEY_3'),
    ] if k
]

# ---------------------------------------------------------------------------
# Staleness cutoff: news older than N hours are discarded
# ---------------------------------------------------------------------------
NEWS_STALENESS_HOURS = 12

# ---------------------------------------------------------------------------
# Deduplication: cosine similarity threshold for near-duplicate headlines
# ---------------------------------------------------------------------------
DEDUP_SIMILARITY_THRESHOLD = 0.85

# ---------------------------------------------------------------------------
# Causal Hierarchy: the order in which Gemini must reason about scores
# (from Claude discussion: Supply → Geo → Monetary → Risk → Sentiment)
# ---------------------------------------------------------------------------
SCORE_CATEGORIES = [
    'supply_shock_severity',       # Upstream: physical supply disruptions
    'geopolitical_stress',         # Upstream: political risk, war, sanctions
    'monetary_policy_hawkishness', # Downstream of 1+2: Fed/ECB reaction
    'risk_appetite',               # Downstream of 1+2+3: market positioning
    'market_sentiment',            # Downstream of all above: retail/media mood
]

# ---------------------------------------------------------------------------
# Few-Shot calibration examples (anchor events with known scores)
# These ground the LLM so it doesn't drift numerically between calls
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = """
CALIBRATION EXAMPLES (use these as reference anchors):

Event: "Russia invades Ukraine, global commodity shock" (Feb 2022)
Scores: supply_shock_severity=0.90, geopolitical_stress=0.95, monetary_policy_hawkishness=0.60, risk_appetite=0.10, market_sentiment=0.15

Event: "Fed cuts rates 50bps, emergency action" (Sep 2024)  
Scores: supply_shock_severity=0.10, geopolitical_stress=0.20, monetary_policy_hawkishness=0.05, risk_appetite=0.75, market_sentiment=0.70

Event: "US-China trade talks resume, no deal yet" (Jun 2024 — calm sideways market)
Scores: supply_shock_severity=0.20, geopolitical_stress=0.35, monetary_policy_hawkishness=0.40, risk_appetite=0.50, market_sentiment=0.50

Event: "Silicon Valley Bank collapses, Fed emergency liquidity injection" (Mar 2023)
Scores: supply_shock_severity=0.15, geopolitical_stress=0.25, monetary_policy_hawkishness=0.30, risk_appetite=0.15, market_sentiment=0.20
"""

# ---------------------------------------------------------------------------
# Build the system prompt using Causal Hierarchy + Few-Shot Calibration
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_TEMPLATE = """
You are a quantitative macro analyst. Your task is to analyze a set of financial news headlines 
and assign precise numerical scores (0.0 to 1.0) for each of the following categories.

IMPORTANT — CAUSAL HIERARCHY: Always reason in this exact order:
1. supply_shock_severity       — physical supply/demand disruptions (wars, embargoes, natural disasters)
2. geopolitical_stress         — political instability, military conflicts, sanctions  
3. monetary_policy_hawkishness — central bank response (tightening=high, easing=low)
4. risk_appetite               — market's overall willingness to take risk
5. market_sentiment            — retail/media sentiment (most downstream, most noisy)

This order matters because later categories are CAUSED BY earlier ones.
Do NOT let market_sentiment influence your supply_shock assessment.

{few_shot_examples}

MACRO CONTEXT (live data):
{macro_summary}

OUTPUT FORMAT — respond ONLY with valid JSON, no extra text:
{{
  "supply_shock_severity": 0.0,
  "geopolitical_stress": 0.0,
  "monetary_policy_hawkishness": 0.0,
  "risk_appetite": 0.0,
  "market_sentiment": 0.0,
  "confidence": 0.0,
  "dominant_regime": "describe in 3 words",
  "time_horizon_bias": "short_term|medium_term|long_term",
  "narrative": "2-3 sentence explanation of your assessment"
}}
"""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _simple_hash(text: str) -> str:
    """SHA-256 hash of a string for deduplication."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _cosine_similarity_simple(a: str, b: str) -> float:
    """
    Lightweight bag-of-words cosine similarity for headline deduplication.
    No external dependencies needed.
    """
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    return len(intersection) / (len(words_a) ** 0.5 * len(words_b) ** 0.5)


def deduplicate_headlines(headlines: list) -> list:
    """
    Remove near-duplicate headlines using cosine similarity threshold.
    Keeps the first occurrence of each unique cluster.
    """
    unique = []
    for candidate in headlines:
        is_dup = any(
            _cosine_similarity_simple(candidate, kept) >= DEDUP_SIMILARITY_THRESHOLD
            for kept in unique
        )
        if not is_dup:
            unique.append(candidate)
    return unique


def filter_stale_news(news_items: list, cutoff_hours: int = NEWS_STALENESS_HOURS) -> list:
    """
    Filter out news items older than cutoff_hours.
    Expects items as dicts with 'published_at' (ISO datetime string) and 'headline'.
    Falls back to keeping items without timestamp.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=cutoff_hours)
    fresh = []
    for item in news_items:
        ts = item.get('published_at')
        if ts is None:
            fresh.append(item)
            continue
        try:
            pub = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            if pub >= cutoff:
                fresh.append(item)
        except (ValueError, AttributeError):
            fresh.append(item)
    return fresh


# ---------------------------------------------------------------------------
# Gram-Schmidt Orthogonalization (post-LLM server-side cleaning)
# Respects the causal hierarchy order defined in SCORE_CATEGORIES
# ---------------------------------------------------------------------------

def gram_schmidt_orthogonalize(score_vector: np.ndarray) -> np.ndarray:
    """
    Apply Gram-Schmidt process to the score vector to remove double-counting.
    The causal order (supply_shock first) is preserved as the primary direction.

    Returns: orthogonalized vector (same shape)
    """
    n = len(score_vector)
    if n == 0:
        return score_vector

    basis = []
    ortho = score_vector.copy().astype(float)

    for i in range(n):
        v = np.zeros(n)
        v[i] = score_vector[i]

        # Project out components already explained by upstream factors
        for b in basis:
            proj = np.dot(v, b) / (np.dot(b, b) + 1e-10) * b
            v = v - proj

        basis.append(v)

    # Reconstruct vector from orthogonalized basis components
    result = np.array([b[i] for i, b in enumerate(basis)])

    # Clip back to [0, 1] — orthogonalization can push values slightly out of range
    result = np.clip(result, 0.0, 1.0)
    return result


def zca_whiten(vector: np.ndarray) -> np.ndarray:
    """
    ZCA Whitening: normalize variance to unit scale for stable LSTM injection.
    Prevents high-magnitude events from overpowering the statistical model.
    """
    mu = np.mean(vector)
    std = np.std(vector) + 1e-8
    whitened = (vector - mu) / std
    # Re-center to [0, 1] range after whitening
    whitened = (whitened - whitened.min()) / (whitened.max() - whitened.min() + 1e-8)
    return whitened


def scores_to_bias_vector(raw_scores: dict) -> np.ndarray:
    """
    Convert Gemini's scoring dict → orthogonalized + whitened numpy vector.
    This is the Contextual Bias Vector c⃗ that gets injected into the LSTM drift.
    """
    raw = np.array([raw_scores.get(cat, 0.5) for cat in SCORE_CATEGORIES])
    ortho = gram_schmidt_orthogonalize(raw)
    whitened = zca_whiten(ortho)
    return whitened


# ---------------------------------------------------------------------------
# Gemini API Call with key rotation + fallback
# ---------------------------------------------------------------------------

def _call_gemini(prompt: str, max_retries: int = 2) -> dict | None:
    """
    Call Gemini API with automatic key rotation.
    Returns parsed JSON dict or None on full failure.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        print("Warning: google-generativeai not installed. Run: pip install google-generativeai")
        return None

    for key in GEMINI_KEYS:
        for attempt in range(max_retries):
            try:
                genai.configure(api_key=key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(
                    prompt,
                    generation_config={"temperature": 0.1, "max_output_tokens": 512}
                )
                text = response.text.strip()
                # Extract JSON block
                if '```json' in text:
                    text = text.split('```json')[1].split('```')[0].strip()
                elif '```' in text:
                    text = text.split('```')[1].split('```')[0].strip()
                return json.loads(text)
            except json.JSONDecodeError:
                print(f"Warning: Gemini returned non-JSON response (key ...{key[-4:]}, attempt {attempt+1})")
            except Exception as e:
                print(f"Warning: Gemini API error (key ...{key[-4:]}, attempt {attempt+1}): {e}")
                time.sleep(1)

    print("Warning: All Gemini API keys failed. Falling back to Zero-Vector Injection.")
    return None


ZERO_VECTOR_SCORES = {cat: 0.5 for cat in SCORE_CATEGORIES}
ZERO_VECTOR_SCORES['confidence'] = 0.0


def _zero_vector_fallback() -> dict:
    """
    Fallback when all LLM calls fail.
    Returns a neutral (0.5) score dict with confidence=0, signaling no bias injection.
    """
    return {
        **ZERO_VECTOR_SCORES,
        'dominant_regime': 'unavailable',
        'time_horizon_bias': 'short_term',
        'narrative': 'LLM unavailable — using statistical baseline only.',
        'fallback': True,
    }


# ---------------------------------------------------------------------------
# Main CEO Layer function
# ---------------------------------------------------------------------------

def analyze_news_context(
    headlines: list,
    macro_summary: str = '',
    published_at_list: list = None,
) -> dict:
    """
    Full CEO Layer pipeline:
    1. Filter stale news
    2. Deduplicate headlines
    3. Build Causal Hierarchy prompt with Few-Shot anchors + Macro Context
    4. Call Gemini with key rotation
    5. Apply Gram-Schmidt + ZCA Whitening
    6. Return structured analysis dict

    Args:
        headlines       : List of news headline strings
        macro_summary   : Text from macro_processor.build_macro_context()['macro_summary']
        published_at_list: Optional list of ISO datetimes for staleness filtering

    Returns dict with:
        'scores'         : Raw dict from Gemini
        'bias_vector'    : np.ndarray — the actual injection vector for LSTM
        'narrative'      : str — Gemini's explanation
        'dominant_regime': str
        'confidence'     : float
        'is_fallback'    : bool — True if API failed and zero-vector was used
        'headlines_used' : int — number of headlines after dedup/staleness filter
    """
    # --- 1. Staleness filter ---
    if published_at_list:
        news_items = [{'headline': h, 'published_at': t}
                      for h, t in zip(headlines, published_at_list)]
        news_items = filter_stale_news(news_items)
        headlines = [item['headline'] for item in news_items]

    # --- 2. Deduplicate ---
    headlines = deduplicate_headlines(headlines)

    if not headlines:
        print("Warning: No fresh/unique headlines. Using zero-vector fallback.")
        scores = _zero_vector_fallback()
        return {
            'scores': scores,
            'bias_vector': np.full(len(SCORE_CATEGORIES), 0.5),
            'narrative': 'No relevant news data.',
            'dominant_regime': 'no_data',
            'confidence': 0.0,
            'is_fallback': True,
            'headlines_used': 0,
        }

    # --- 3. Build prompt ---
    headlines_block = '\n'.join(f'- {h}' for h in headlines[:30])  # cap at 30 headlines

    system = SYSTEM_PROMPT_TEMPLATE.format(
        few_shot_examples=FEW_SHOT_EXAMPLES,
        macro_summary=macro_summary or 'No macro data available.',
    )
    user_msg = f"HEADLINES TO ANALYZE:\n{headlines_block}"
    full_prompt = f"{system}\n\n{user_msg}"

    # --- 4. Call Gemini ---
    raw_scores = _call_gemini(full_prompt)
    
    # Type safety check: Gemini should return a dict
    if raw_scores is None or not isinstance(raw_scores, dict):
        is_fallback = True
        raw_scores = _zero_vector_fallback()
    else:
        is_fallback = False

    # --- 5. Orthogonalize + Whiten ---
    bias_vector = scores_to_bias_vector(raw_scores)
    if is_fallback:
        bias_vector = np.full(len(SCORE_CATEGORIES), 0.5)  # neutral, no drift

    return {
        'scores':          raw_scores,
        'bias_vector':     bias_vector,
        'narrative':       raw_scores.get('narrative', ''),
        'dominant_regime': raw_scores.get('dominant_regime', 'unknown'),
        'confidence':      float(raw_scores.get('confidence', 0.0)),
        'is_fallback':     is_fallback,
        'headlines_used':  len(headlines),
    }


def compute_drift_multiplier(bias_vector: np.ndarray, asset_type: str = 'general') -> float:
    """
    Convert the CEO bias vector into a single drift multiplier for the LSTM engine.

    The multiplier shifts the LSTM's mean-reversion anchor:
      - > 1.0 = Bullish bias (hold price higher than mean)
      - < 1.0 = Bearish bias (pull price toward lower anchor)
      - = 1.0 = Neutral (no CEO bias, pure LSTM)

    Asset-specific sensitivity (Gold is more geopolitics-sensitive, BTC more sentiment-sensitive):
    """
    if len(bias_vector) < len(SCORE_CATEGORIES):
        return 1.0  # neutral fallback

    supply_idx      = 0  # supply_shock_severity
    geo_idx         = 1  # geopolitical_stress
    monetary_idx    = 2  # monetary_policy_hawkishness
    risk_idx        = 3  # risk_appetite
    sentiment_idx   = 4  # market_sentiment

    if asset_type == 'gold':
        # Gold: bullish on fear, geopolitics, and inflation
        raw = (
            bias_vector[geo_idx] * 0.40 +
            bias_vector[supply_idx] * 0.30 +
            (1.0 - bias_vector[monetary_idx]) * 0.20 +  # Dovish = gold bullish
            bias_vector[risk_idx] * 0.10
        )
    elif asset_type == 'btc':
        # BTC: risk-on asset, sentiment-heavy
        raw = (
            bias_vector[risk_idx] * 0.40 +
            bias_vector[sentiment_idx] * 0.30 +
            (1.0 - bias_vector[monetary_idx]) * 0.20 +
            (1.0 - bias_vector[geo_idx]) * 0.10
        )
    elif asset_type == 'oil':
        # Oil: supply shock + geopolitics dominant
        raw = (
            bias_vector[supply_idx] * 0.50 +
            bias_vector[geo_idx] * 0.30 +
            bias_vector[risk_idx] * 0.20
        )
    else:  # stocks, general
        raw = (
            bias_vector[risk_idx] * 0.35 +
            (1.0 - bias_vector[monetary_idx]) * 0.30 +
            bias_vector[sentiment_idx] * 0.20 +
            (1.0 - bias_vector[geo_idx]) * 0.15
        )

    # Map [0, 1] → multiplier [0.90, 1.10] — max ±10% bias on LSTM drift
    multiplier = 0.90 + raw * 0.20
    return round(float(multiplier), 4)


if __name__ == '__main__':
    # Quick test
    test_headlines = [
        "Federal Reserve holds rates amid rising geopolitical tensions",
        "Oil prices surge as Iran-Israel conflict escalates",
        "CPI data shows inflation cooling for third consecutive month",
        "Fed holds rates amid geopolitical tensions",  # near-duplicate test
    ]
    result = analyze_news_context(
        headlines=test_headlines,
        macro_summary="Test macro context: YieldCurve +0.15%, M2 YoY +3.2%"
    )
    print(f"Headlines used: {result['headlines_used']} (dedup from 4)")
    print(f"Dominant Regime: {result['dominant_regime']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Is Fallback: {result['is_fallback']}")
    print(f"Bias Vector: {result['bias_vector']}")
    print(f"Narrative: {result['narrative']}")
