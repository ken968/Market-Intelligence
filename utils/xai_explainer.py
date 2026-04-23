"""
XAI Explainer — Explainable AI for Market Intelligence Terminal

Provides two capabilities:
  1. Macro Driver Extraction: identifies the top 3 macro features with the
     most extreme recent movement vs their historical mean.
  2. Gemini Explainer: sends those drivers + forecast direction to Gemini and
     gets a plain-language rationale (Tailwinds / Headwinds / Summary).
  3. Sector-Level Batch Analysis: one Gemini call that explains why
     Tech/Industrial/Index stocks moved in different directions.

Architecture note: uses the same Gemini key-rotation logic as llm_manager.py
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta


# ---------------------------------------------------------------------------
# Macro feature labels (human-readable names for the UI)
# ---------------------------------------------------------------------------
MACRO_LABELS = {
    'DXY':               'US Dollar Index (DXY)',
    'VIX':               'Fear Index (VIX)',
    'Yield_10Y':         '10Y Treasury Yield',
    'Oil_Price':         'Crude Oil Price',
    'CPI_MoM':           'CPI Month-over-Month',
    'PPI_MoM':           'PPI Month-over-Month',
    'PCE_MoM':           'PCE Month-over-Month',
    'NFP_Change':        'Non-Farm Payrolls',
    'YieldCurve_10Y2Y':  'Yield Curve (10Y-2Y)',
    'M2_MoM':            'M2 Money Supply MoM',
    'M2_YoY':            'M2 Money Supply YoY',
    'Yield_10Y_Rate':    '10Y Treasury Rate',
    'Breakeven_5Y5Y':    '5Y5Y Breakeven Inflation',
    'M2_Liquidity_Spike':'M2 Liquidity Spike Flag',
}

BULLISH_DIRECTION = {
    'gold':   'Bullish (price expected to rise)',
    'btc':    'Bullish (price expected to rise)',
    'stocks': 'Bullish (price expected to rise)',
}


# ---------------------------------------------------------------------------
# 1. Macro Driver Extraction
# ---------------------------------------------------------------------------

def get_top_macro_drivers(asset_key: str, lookback_days: int = 14, top_n: int = 3) -> list[dict]:
    """
    Identify the top N macro features with the most extreme Z-score movement
    over the past `lookback_days` vs the asset's full history.

    Returns list of dicts:
        [{'feature': str, 'label': str, 'recent_mean': float,
          'hist_mean': float, 'z_score': float, 'direction': 'rising'|'falling'}]
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
    # Exclude the price column itself and binary flags
    macro_features = [
        f for f in features
        if f not in [config['features'][0], 'M2_Liquidity_Spike', 'MacroEvent_Flag',
                     'Halving_Cycle', 'EMA_90', 'Sentiment']
    ]

    if len(df) < lookback_days + 1:
        return []

    results = []
    for feat in macro_features:
        series = df[feat].dropna()
        if len(series) < lookback_days + 5:
            continue
        recent = series.iloc[-lookback_days:].mean()
        hist_mean = series.iloc[:-lookback_days].mean()
        hist_std = series.iloc[:-lookback_days].std()
        if hist_std < 1e-8:
            continue
        z = (recent - hist_mean) / hist_std
        results.append({
            'feature': feat,
            'label': MACRO_LABELS.get(feat, feat),
            'recent_mean': round(float(recent), 4),
            'hist_mean': round(float(hist_mean), 4),
            'z_score': round(float(z), 2),
            'direction': 'rising' if z > 0 else 'falling',
        })

    # Sort by absolute Z-score (most extreme first)
    results.sort(key=lambda x: abs(x['z_score']), reverse=True)
    return results[:top_n]


# ---------------------------------------------------------------------------
# 2. Gemini single-asset explainer
# ---------------------------------------------------------------------------

def _call_gemini_text(prompt: str) -> str | None:
    """Minimal Gemini call that returns raw text (not JSON)."""
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
                generation_config={'temperature': 0.2, 'max_output_tokens': 600}
            )
            return resp.text.strip()
        except Exception:
            continue
    return None


def explain_forecast(
    asset_key: str,
    asset_name: str,
    direction: str,          # 'up' | 'down' | 'sideways'
    pct_change: float,
    drivers: list[dict],
    macro_summary: str = '',
) -> dict:
    """
    Ask Gemini to explain WHY the model predicted this direction,
    given the top macro drivers.

    Returns:
        {
          'tailwinds': [str, ...],
          'headwinds': [str, ...],
          'summary':   str,
          'available': bool,
        }
    """
    if not drivers:
        return {'tailwinds': [], 'headwinds': [], 'summary': 'Insufficient macro data for explanation.', 'available': False}

    direction_str = 'UP (Bullish)' if direction == 'up' else ('DOWN (Bearish)' if direction == 'down' else 'SIDEWAYS (Neutral)')
    drivers_text = '\n'.join(
        f"  - {d['label']}: {d['direction'].upper()} by {abs(d['z_score']):.1f} standard deviations "
        f"(recent avg {d['recent_mean']:.3f} vs historical avg {d['hist_mean']:.3f})"
        for d in drivers
    )

    prompt = f"""You are a professional quantitative macro analyst writing a brief forecast rationale.

ASSET: {asset_name}
AI MODEL PREDICTION: {direction_str} ({pct_change:+.2f}% projected change)

TOP MACRO DRIVERS IDENTIFIED (by statistical deviation from historical norm):
{drivers_text}

MACRO CONTEXT:
{macro_summary or 'No additional macro context available.'}

TASK:
Explain in plain, professional language why the combination of these macro drivers logically supports
the AI model's {direction_str} prediction for {asset_name}.

Respond ONLY in this exact format (no extra text):
TAILWINDS:
- [factor supporting the predicted direction]
- [factor supporting the predicted direction]

HEADWINDS:
- [factor working against the predicted direction, or 'None significant' if bullish/bearish is clear]

SUMMARY:
[2 sentences max. Explain the dominant macro narrative and why it drives {asset_name} {direction_str}.]
"""

    raw = _call_gemini_text(prompt)
    if not raw:
        return {
            'tailwinds': ['Model detected statistical momentum.'],
            'headwinds': ['Macro data inconclusive.'],
            'summary': f"AI model predicts {direction_str} based on quantitative pattern recognition.",
            'available': False,
        }

    # Parse response
    tailwinds, headwinds, summary = [], [], ''
    section = None
    for line in raw.splitlines():
        line = line.strip()
        if line.upper().startswith('TAILWINDS'):
            section = 'tail'
        elif line.upper().startswith('HEADWINDS'):
            section = 'head'
        elif line.upper().startswith('SUMMARY'):
            section = 'sum'
        elif line.startswith('-') and section == 'tail':
            tailwinds.append(line.lstrip('- '))
        elif line.startswith('-') and section == 'head':
            headwinds.append(line.lstrip('- '))
        elif section == 'sum' and line:
            summary += line + ' '

    return {
        'tailwinds': tailwinds or ['Statistical momentum detected.'],
        'headwinds': headwinds or ['None significant.'],
        'summary': summary.strip() or f"Forecast direction: {direction_str}.",
        'available': True,
    }


# ---------------------------------------------------------------------------
# 3. Sector-Level Batch Analysis (for All Stocks page)
# ---------------------------------------------------------------------------

def explain_sector_forecast(
    ticker_forecasts: dict,   # {ticker: {'direction': str, 'pct_change': float}}
    macro_summary: str = '',
    top_drivers: list[dict] = None,
) -> str:
    """
    One Gemini call that provides a sector-level narrative explaining
    divergences between Tech, Consumer, Index stocks.
    Returns a plain text explanation string.
    """
    if not ticker_forecasts:
        return 'No forecast data available for sector analysis.'

    # Group by implicit sector
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
        drivers_block = 'KEY MACRO DRIVERS THIS WEEK:\n' + '\n'.join(
            f"  - {d['label']}: {d['direction'].upper()} by {abs(d['z_score']):.1f}σ"
            for d in top_drivers
        )

    prompt = f"""You are a senior equity strategist. Below are AI model forecasts for US stocks, grouped by sector.

{forecast_block}
{drivers_block}

MACRO CONTEXT:
{macro_summary or 'No macro context available.'}

In 3-4 sentences, explain the macro logic behind the forecasts:
- Why are some sectors diverging from others?
- What macro factor is the primary driver this week?
- What is the dominant market regime (Risk-On / Risk-Off / Stagflation / etc.)?

Be direct and professional. No bullet points. Plain paragraph only."""

    result = _call_gemini_text(prompt)
    return result or 'Sector narrative unavailable — Gemini API not reachable.'
