"""
XAI Explainer — Explainable AI for Market Intelligence Terminal

Provides:
  1. Macro Driver Extraction: top 3 macro features with the most extreme
     recent movement vs their historical mean (Z-score ranked).
  2. Asset-Specific Impact Mapping: maps each driver to its known directional
     impact on the specific asset (e.g. rising DXY = bearish for Gold).
  3. Gemini Explainer: structured institutional-style rationale.
  4. Sector-Level Batch Analysis: one Gemini call for all-stock divergence.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta


# ---------------------------------------------------------------------------
# Known directional impact of each macro indicator per asset type
# ---------------------------------------------------------------------------
ASSET_IMPACT_MAP = {
    'gold': {
        'DXY':              'BEARISH',   # Stronger dollar → gold cheaper globally
        'VIX':              'BULLISH',   # Fear → safe-haven demand
        'Yield_10Y':        'BEARISH',   # Higher yields → opportunity cost rises
        'Oil_Price':        'BULLISH',   # Inflation proxy → gold hedge
        'CPI_MoM':          'BULLISH',   # Inflation → gold inflation hedge
        'PPI_MoM':          'BULLISH',   # Upstream inflation → gold hedge
        'PCE_MoM':          'BULLISH',   # Fed inflation gauge → policy signal
        'NFP_Change':       'BEARISH',   # Strong jobs → rate hike risk
        'YieldCurve_10Y2Y': 'MIXED',     # Inversion → recession fear → mixed
        'M2_MoM':           'BULLISH',   # More liquidity → gold rises
        'M2_YoY':           'BULLISH',   # Liquidity trend → gold positive
        'Yield_10Y_Rate':   'BEARISH',   # Higher cost of capital → gold negative
        'Breakeven_5Y5Y':   'BULLISH',   # Inflation expectations → gold positive
    },
    'btc': {
        'DXY':              'BEARISH',   # Strong dollar → risk-off
        'VIX':              'BEARISH',   # Fear → risk assets sold
        'Yield_10Y':        'BEARISH',   # Higher yields → risk-off rotation
        'Oil_Price':        'MIXED',
        'CPI_MoM':          'BEARISH',   # Inflation → rate hike → risk-off
        'PPI_MoM':          'BEARISH',
        'PCE_MoM':          'BEARISH',
        'NFP_Change':       'BEARISH',   # Strong jobs → rate hike → BTC down
        'YieldCurve_10Y2Y': 'BULLISH',   # Normal curve → growth regime → risk-on
        'M2_MoM':           'BULLISH',   # Liquidity → crypto positive
        'M2_YoY':           'BULLISH',
        'Yield_10Y_Rate':   'BEARISH',
        'Breakeven_5Y5Y':   'MIXED',
    },
    'stocks': {
        'DXY':              'MIXED',     # Depends on export exposure
        'VIX':              'BEARISH',   # Fear → sell-off
        'Yield_10Y':        'BEARISH',   # Discount rate rises → equity down
        'Oil_Price':        'MIXED',     # Energy sector up, consumer down
        'CPI_MoM':          'BEARISH',   # Inflation → rate hike risk
        'PPI_MoM':          'BEARISH',   # Margin compression
        'PCE_MoM':          'BEARISH',
        'NFP_Change':       'BULLISH',   # Strong jobs → earnings growth
        'YieldCurve_10Y2Y': 'BULLISH',   # Normal → growth → equities up
        'M2_MoM':           'BULLISH',   # Liquidity → equities bid
        'M2_YoY':           'BULLISH',
        'Yield_10Y_Rate':   'BEARISH',
        'Breakeven_5Y5Y':   'MIXED',
    },
}

MACRO_LABELS = {
    'DXY':               'US Dollar Index (DXY)',
    'VIX':               'Fear Index (VIX)',
    'Yield_10Y':         '10Y Treasury Yield',
    'Oil_Price':         'Crude Oil Price',
    'CPI_MoM':           'CPI Month-over-Month',
    'PPI_MoM':           'PPI Month-over-Month',
    'PCE_MoM':           'PCE Month-over-Month',
    'NFP_Change':        'Non-Farm Payrolls',
    'YieldCurve_10Y2Y':  'Yield Curve (10Y-2Y Spread)',
    'M2_MoM':            'M2 Money Supply MoM',
    'M2_YoY':            'M2 Money Supply YoY',
    'Yield_10Y_Rate':    '10Y Treasury Rate',
    'Breakeven_5Y5Y':    '5Y5Y Breakeven Inflation',
    'M2_Liquidity_Spike': 'M2 Liquidity Spike Flag',
}


# ---------------------------------------------------------------------------
# 1. Macro Driver Extraction (Z-score ranked)
# ---------------------------------------------------------------------------

def get_top_macro_drivers(asset_key: str, lookback_days: int = 14, top_n: int = 3) -> list[dict]:
    """
    Identify the top N macro features with the most extreme Z-score movement
    over the past lookback_days vs the full history.

    Returns list of dicts with keys:
        feature, label, current_value, recent_mean, hist_mean,
        z_score, direction, impact (asset-specific)
    """
    from utils.config import get_asset_config
    config = get_asset_config(asset_key)
    if not config:
        return []

    data_path = config['data_file']
    if not os.path.exists(data_path):
        return []

    df = pd.read_csv(data_path)
    features = [f for f in config['features'] if f in df.columns]

    # Exclude price column and non-numeric/binary flags
    macro_features = [
        f for f in features
        if f not in [config['features'][0], 'M2_Liquidity_Spike', 'MacroEvent_Flag',
                     'Halving_Cycle', 'EMA_90', 'Sentiment']
    ]

    if len(df) < lookback_days + 5:
        return []

    # Determine asset type for impact mapping
    if asset_key == 'gold':
        atype = 'gold'
    elif asset_key == 'btc':
        atype = 'btc'
    else:
        atype = 'stocks'

    impact_map = ASSET_IMPACT_MAP.get(atype, {})

    results = []
    for feat in macro_features:
        series = df[feat].dropna()
        if len(series) < lookback_days + 5:
            continue

        current_val = float(series.iloc[-1])
        recent_mean = float(series.iloc[-lookback_days:].mean())
        hist_mean   = float(series.iloc[:-lookback_days].mean())
        hist_std    = float(series.iloc[:-lookback_days].std())

        if hist_std < 1e-8:
            continue

        z = (recent_mean - hist_mean) / hist_std
        impact = impact_map.get(feat, 'MIXED')

        # Flip impact interpretation if indicator is falling
        effective_impact = impact
        if impact != 'MIXED' and z < 0:
            effective_impact = 'BEARISH' if impact == 'BULLISH' else 'BULLISH'

        results.append({
            'feature':        feat,
            'label':          MACRO_LABELS.get(feat, feat),
            'current_value':  round(current_val, 4),
            'recent_mean':    round(recent_mean, 4),
            'hist_mean':      round(hist_mean, 4),
            'z_score':        round(z, 2),
            'direction':      'Rising' if z > 0 else 'Falling',
            'impact':         effective_impact,   # net impact on this asset
        })

    results.sort(key=lambda x: abs(x['z_score']), reverse=True)
    return results[:top_n]


def build_driver_dataframe(drivers: list[dict]) -> pd.DataFrame:
    """
    Convert driver list into a clean DataFrame for st.dataframe display.
    Columns: Indicator | Current | 14D Avg | vs History | Trend | Asset Impact
    """
    rows = []
    for d in drivers:
        zsign = f"+{d['z_score']}" if d['z_score'] >= 0 else str(d['z_score'])
        rows.append({
            'Indicator':     d['label'],
            'Current':       d['current_value'],
            '14D Avg':       d['recent_mean'],
            'Hist Avg':      d['hist_mean'],
            'Deviation':     f"{zsign}σ",
            'Trend':         d['direction'],
            'Asset Impact':  d['impact'],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Gemini single-asset explainer
# ---------------------------------------------------------------------------

def _call_gemini_text(prompt: str) -> str | None:
    """Minimal Gemini call returning raw text."""
    try:
        import google.generativeai as genai
    except ImportError:
        return None

    from dotenv import load_dotenv
    load_dotenv()
    keys = [k for k in [
        os.getenv('GEMINI_API_KEY_1'),
        os.getenv('GEMINI_API_KEY_2'),
        os.getenv('GEMINI_API_KEY_3'),
    ] if k]

    for key in keys:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            resp = model.generate_content(
                prompt,
                generation_config={'temperature': 0.2, 'max_output_tokens': 500}
            )
            return resp.text.strip()
        except Exception:
            continue
    return None


def explain_forecast(
    asset_key: str,
    asset_name: str,
    direction: str,        # 'up' | 'down' | 'sideways'
    pct_change: float,
    drivers: list[dict],
    macro_summary: str = '',
) -> dict:
    """
    Ask Gemini to produce an institutional-style rationale explaining
    why the listed macro drivers logically support the predicted direction.

    Returns:
        {
          'tailwinds':  [str, ...],
          'headwinds':  [str, ...],
          'summary':    str,
          'available':  bool,
        }
    """
    direction_str = 'BULLISH' if direction == 'up' else ('BEARISH' if direction == 'down' else 'NEUTRAL/SIDEWAYS')

    if not drivers:
        return {
            'tailwinds': ['Quantitative pattern momentum detected by LSTM model.'],
            'headwinds': ['Insufficient macro data for detailed attribution.'],
            'summary': f"Model output: {direction_str} ({pct_change:+.2f}%). No macro driver data available.",
            'available': False,
        }

    driver_lines = '\n'.join(
        f"  - {d['label']}: {d['direction']} by {abs(d['z_score']):.1f} standard deviations "
        f"(current {d['current_value']:.3f}, 14D avg {d['recent_mean']:.3f}, "
        f"historical avg {d['hist_mean']:.3f}) → estimated net impact on {asset_name}: {d['impact']}"
        for d in drivers
    )

    prompt = f"""You are a senior quantitative macro strategist writing a concise forecast attribution note.

ASSET: {asset_name}
MODEL FORECAST: {direction_str} | Projected change: {pct_change:+.2f}%

TOP MACRO DRIVERS (ranked by statistical deviation from historical norm):
{driver_lines}

SUPPLEMENTARY MACRO CONTEXT:
{macro_summary or 'Not available.'}

TASK: Write a professional, plain-language attribution explaining why these macro drivers
logically support the {direction_str} forecast for {asset_name}.

Rules:
- No emoji or decorative symbols.
- Be direct and factual.
- Reference specific indicators by name.
- Do NOT hedge with phrases like "it's important to note" or "it's worth mentioning".

Respond in exactly this format:
TAILWINDS:
- [specific macro factor supporting the forecast direction]
- [second factor]

HEADWINDS:
- [factor working against the forecast, or "None significant" if the direction is very clear]

SUMMARY:
[2 sentences. State the dominant macro theme and its logical effect on {asset_name} pricing.]"""

    raw = _call_gemini_text(prompt)
    if not raw:
        return {
            'tailwinds': ['Statistical momentum in AI model output.'],
            'headwinds': ['Gemini API unreachable. Using rule-based attribution only.'],
            'summary': f"Model forecast: {direction_str} ({pct_change:+.2f}%). Driven by deviations in {', '.join(d['label'] for d in drivers[:2])}.",
            'available': False,
        }

    tailwinds, headwinds, summary_lines = [], [], []
    section = None
    for line in raw.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith('TAILWINDS'):
            section = 'tail'
        elif upper.startswith('HEADWINDS'):
            section = 'head'
        elif upper.startswith('SUMMARY'):
            section = 'sum'
        elif stripped.startswith('-') and section == 'tail':
            tailwinds.append(stripped.lstrip('- ').strip())
        elif stripped.startswith('-') and section == 'head':
            headwinds.append(stripped.lstrip('- ').strip())
        elif section == 'sum' and stripped:
            summary_lines.append(stripped)

    return {
        'tailwinds': tailwinds or ['Statistical momentum detected by LSTM pattern recognition.'],
        'headwinds': headwinds or ['None significant given current macro configuration.'],
        'summary': ' '.join(summary_lines).strip() or f"Forecast direction: {direction_str}.",
        'available': True,
    }


# ---------------------------------------------------------------------------
# 3. Sector-Level Batch Analysis
# ---------------------------------------------------------------------------

def explain_sector_forecast(
    ticker_forecasts: dict,   # {ticker: {'direction': str, 'pct_change': float}}
    macro_summary: str = '',
    top_drivers: list[dict] = None,
) -> str:
    """
    One Gemini call explaining sector-level divergences across all stock forecasts.
    Returns a plain text paragraph (no emoji).
    """
    if not ticker_forecasts:
        return 'No forecast data available for sector analysis.'

    sector_map = {
        'SPY': 'Index', 'QQQ': 'Index', 'DIA': 'Index',
        'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
        'NVDA': 'Technology', 'META': 'Technology', 'TSM': 'Technology',
        'AMZN': 'Consumer', 'TSLA': 'Automotive',
    }

    sector_lines = {}
    for ticker, data in ticker_forecasts.items():
        sector = sector_map.get(ticker.upper(), 'Other')
        line = f"  {ticker.upper()}: {data.get('direction','?').upper()} {data.get('pct_change', 0):+.2f}%"
        sector_lines.setdefault(sector, []).append(line)

    forecast_block = ''
    for sector, lines in sector_lines.items():
        forecast_block += f"{sector}:\n" + '\n'.join(lines) + '\n\n'

    drivers_block = ''
    if top_drivers:
        drivers_block = 'MACRO DRIVERS THIS WEEK:\n' + '\n'.join(
            f"  - {d['label']}: {d['direction']} by {abs(d['z_score']):.1f} standard deviations "
            f"(current {d['current_value']:.3f} vs historical {d['hist_mean']:.3f})"
            for d in top_drivers
        )

    prompt = f"""You are a senior equity strategist writing a concise sector attribution note.

AI MODEL FORECASTS BY SECTOR:
{forecast_block}
{drivers_block}

SUPPLEMENTARY MACRO CONTEXT:
{macro_summary or 'Not available.'}

In 3-4 sentences, explain the macro logic behind the sector divergences:
- What macro factor is the primary driver this week?
- Why are certain sectors outperforming or underperforming others?
- What is the dominant market regime (Risk-On / Risk-Off / Stagflation / Reflation / etc.)?

Rules: No emoji. No bullet points. Plain professional paragraph only. Be direct."""

    result = _call_gemini_text(prompt)
    return result or 'Sector narrative unavailable — Gemini API not reachable.'
