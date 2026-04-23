"""
Dashboard Overview Page
Multi-asset performance tracking and portfolio insights
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import os
from utils.config import ASSETS, STOCK_TICKERS, get_asset_status
from utils.ui_components import (
    inject_custom_css, render_page_header, render_metric_card,
    show_error_message
)
from utils.predictor import batch_predict_tomorrow, batch_predict_week

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="Dashboard | Market Intelligence",
    layout="wide"
)

inject_custom_css()

# ==================== MAIN CONTENT ====================

render_page_header(
    icon="",
    title="Portfolio Dashboard",
    subtitle="Real-time tracking of all assets with AI-powered insights"
)

# ==================== CHECK AVAILABILITY ====================

status = get_asset_status()
available_assets = [key for key, s in status.items() if s['data']]

if not available_assets:
    show_error_message("No data available. Please sync data from Settings page.")
    st.stop()

# ==================== ASSET SELECTOR ====================

st.markdown("### Asset Selection")

col1, col2 = st.columns([3, 1])

with col1:
    # Group selection
    asset_groups = {
        'Core Assets': ['gold', 'btc', 'spy', 'qqq'],
        'Market Indices': ['spy', 'qqq', 'dia'],
        'Magnificent 7': ['aapl', 'msft', 'googl', 'amzn', 'nvda', 'meta', 'tsla'],
        'All Stocks': ['spy', 'qqq', 'dia', 'aapl', 'msft', 'googl', 'amzn', 'nvda', 'meta', 'tsla', 'tsm']
    }
    
    group_choice = st.selectbox("Quick Select", list(asset_groups.keys()))
    default_assets = [a for a in asset_groups[group_choice] if a in available_assets]
    
    selected_assets = st.multiselect(
        "Or customize selection",
        available_assets,
        default=default_assets
    )

with col2:
    timeframe = st.selectbox(
        "Timeframe",
        ["1 Month", "3 Months", "6 Months", "1 Year", "All Time"],
        index=3
    )

if not selected_assets:
    st.warning("Please select at least one asset to analyze.")
    st.stop()

st.markdown("---")

# ==================== MACRO INDICATORS ====================

st.markdown("### Market Prices & Macro Indicators")

# -- Row 1: Tier 2 macro cards (Oil + Yield 10Y + DXY + VIX) --
try:
    macro_df = pd.read_csv('data/macro_indicators.csv')
    latest_macro = macro_df.iloc[-1]
    prev_macro = macro_df.iloc[-2]

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card(
            label="Crude Oil (WTI)",
            value=latest_macro['Oil_Price'],
            delta=latest_macro['Oil_Price'] - prev_macro['Oil_Price']
        )
        st.caption("Tier 2 Macro")
    with m2:
        render_metric_card(
            label="US 10Y Yield",
            value=latest_macro['Yield_10Y'],
            delta=latest_macro['Yield_10Y'] - prev_macro['Yield_10Y']
        )
        st.caption("Tier 2 Macro")
    with m3:
        render_metric_card(
            label="DXY (Dollar Index)",
            value=latest_macro['DXY'],
            delta=latest_macro['DXY'] - prev_macro['DXY']
        )
        st.caption("Tier 2 Macro")
    with m4:
        render_metric_card(
            label="VIX (Fear Index)",
            value=latest_macro['VIX'],
            delta=latest_macro['VIX'] - prev_macro['VIX']
        )
        st.caption("Tier 2 Macro")
except Exception:
    st.warning("Macro indicator data unavailable.")

# -- Row 2: Tier 1 FRED indicators --
try:
    fred_df = pd.read_csv('data/fred_indicators.csv', index_col=0, parse_dates=True)

    def _last2(series):
        s = series[series != 0].dropna()
        v = s.iloc[-1] if len(s) >= 1 else 0
        p = s.iloc[-2] if len(s) >= 2 else 0
        return float(v), float(p)

    # Row 2A: CPI / PPI / PCE / NFP
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        v, p = _last2(fred_df['CPI_MoM'])
        render_metric_card(label="CPI MoM (%)", value=round(v, 3), delta=round(v - p, 3))
        st.caption("Tier 1 — monthly")
    with f2:
        v, p = _last2(fred_df['PPI_MoM'])
        render_metric_card(label="PPI MoM (%)", value=round(v, 3), delta=round(v - p, 3))
        st.caption("Tier 1 — monthly")
    with f3:
        v, p = _last2(fred_df['PCE_MoM'])
        render_metric_card(label="PCE MoM (%)", value=round(v, 3), delta=round(v - p, 3))
        st.caption("Tier 1 — monthly")
    with f4:
        v, p = _last2(fred_df['NFP_Change'])
        render_metric_card(label="NFP Change (K)", value=round(v, 0), delta=round(v - p, 0))
        st.caption("Tier 1 — monthly")

    # Row 2B: Yield Curve (10Y-2Y) + M2 Money Supply
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        if 'YieldCurve_10Y2Y' in fred_df.columns:
            yc = fred_df['YieldCurve_10Y2Y'].dropna()
            v = float(yc.iloc[-1]) if len(yc) >= 1 else 0
            p = float(yc.iloc[-2]) if len(yc) >= 2 else 0
            render_metric_card(label="Yield Curve (10Y-2Y)", value=round(v, 3), delta=round(v - p, 3))
            yc_status = "Normal" if v > 0 else ("Inverted (Recession Signal)" if v < 0 else "Flat")
            st.caption(f"Tier 2 — {yc_status}")
    with g2:
        if 'M2_MoM' in fred_df.columns:
            v, p = _last2(fred_df['M2_MoM'])
            render_metric_card(label="M2 Money Supply MoM (%)", value=round(v, 3), delta=round(v - p, 3))
            st.caption("Tier 2 — monthly")
    with g3:
        if 'Yield_10Y_Rate' in fred_df.columns:
            yr = fred_df['Yield_10Y_Rate'].dropna()
            v = float(yr.iloc[-1]) if len(yr) >= 1 else 0
            p = float(yr.iloc[-2]) if len(yr) >= 2 else 0
            render_metric_card(label="10Y Treasury Rate", value=round(v, 3), delta=round(v - p, 3))
            st.caption("Cost of Capital (daily)")
    with g4:
        if 'Breakeven_5Y5Y' in fred_df.columns:
            be = fred_df['Breakeven_5Y5Y'].dropna()
            v = float(be.iloc[-1]) if len(be) >= 1 else 0
            p = float(be.iloc[-2]) if len(be) >= 2 else 0
            render_metric_card(label="5Y5Y Breakeven", value=round(v, 3), delta=round(v - p, 3))
            be_status = "Anchored" if v < 2.3 else ("Rising" if v < 2.8 else "Elevated")
            st.caption(f"Fed Credibility — {be_status}")

    # Row 2C: M2 YoY (Trend) + M2 Liquidity Spike + Macro Compass
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        if 'M2_YoY' in fred_df.columns:
            v, p = _last2(fred_df['M2_YoY'])
            render_metric_card(label="M2 YoY (%)", value=round(v, 2), delta=round(v - p, 2))
            m2_bias = "Risk-On" if v > 5 else ("Neutral" if v > 2 else "Risk-Off")
            st.caption(f"Liquidity Trend — {m2_bias}")
    with h2:
        if 'M2_MoM' in fred_df.columns:
            v, p = _last2(fred_df['M2_MoM'])
            render_metric_card(label="M2 MoM (%)", value=round(v, 3), delta=round(v - p, 3))
            spike = "Liquidity Spike!" if v > 0.5 else "Normal"
            st.caption(f"Momentum — {spike}")
    with h3:
        # Macro Compass: Recession Risk Score
        try:
            from utils.macro_processor import build_macro_context
            macro_ctx = build_macro_context()
            risk = macro_ctx.get('recession_risk', 0.5)
            regime = macro_ctx.get('yield_regime', 'unknown').upper()
            risk_pct = round(risk * 100, 1)
            risk_color = "LOW" if risk < 0.3 else ("MODERATE" if risk < 0.6 else "HIGH")
            st.metric(label="Recession Risk Score", value=f"{risk_pct}%")
            st.caption(f"Regime: {regime}")
        except Exception:
            st.caption("Recession score unavailable")
    with h4:
        # M2 Liquidity Spike Flag
        if 'M2_Liquidity_Spike' in fred_df.columns:
            spike_flag = int(fred_df['M2_Liquidity_Spike'].iloc[-1])
            if spike_flag:
                st.warning("M2 Liquidity Spike Detected!\nPossible Fed intervention or emergency stimulus.")
            else:
                st.success("M2 Flow: Normal")

except Exception:
    st.info("FRED indicators not yet synced. Run FRED sync from Settings.")


# -- Row 3: Buffett Indicator Gauge --
try:
    gdp_df  = pd.read_csv('data/gdp_series.csv', index_col=0, parse_dates=True)
    wilshire = yf.download('^W5000', period='5d', interval='1d', progress=False)
    if not wilshire.empty:
        # Wilshire 5000 index value closely approximates total US market cap in billions of dollars.
        # FRED GDP is also reported in billions of dollars.
        mkt_cap_billions = float(wilshire['Close'].dropna().iloc[-1])
        gdp_billions = float(gdp_df['GDP'].dropna().iloc[-1])
        buffett_ratio = (mkt_cap_billions / gdp_billions) * 100   # Ratio in %

        gc1, gc2 = st.columns([2, 1])
        with gc1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=buffett_ratio,
                number={'suffix': '%', 'font': {'size': 64}},
                title={'text': "Buffett Indicator<br><sub>Total Mkt Cap / GDP</sub>", 'font': {'size': 32}},
                gauge={
                    'axis': {'range': [0, 250], 'ticksuffix': '%'},
                    'bar': {'color': '#00A8E8'},
                    'steps': [
                        {'range': [0, 100],   'color': 'rgba(0,204,150,0.25)'},
                        {'range': [100, 150], 'color': 'rgba(255,161,90,0.25)'},
                        {'range': [150, 250], 'color': 'rgba(239,85,59,0.25)'},
                    ],
                    'threshold': {
                        'line': {'color': 'white', 'width': 2},
                        'thickness': 0.75,
                        'value': buffett_ratio
                    }
                }
            ))
            fig_gauge.update_layout(
                template='plotly_dark',
                height=450,
                margin=dict(l=30, r=30, t=110, b=10)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with gc2:
            if buffett_ratio < 100:
                level, color = "Undervalued", "success"
            elif buffett_ratio < 150:
                level, color = "Fair Value", "warning"
            else:
                level, color = "Overvalued", "error"

            getattr(st, color)(
                f"**Market is {level}** at {buffett_ratio:.1f}%\n\n"
                "- LOW < 100%: Historically undervalued vs GDP\n"
                "- FAIR 100–150%: Fair / moderately elevated\n"
                "- HIGH > 150%: Overvalued (Buffett historically cautious)"
            )
except ImportError:
    pass
except Exception:
    pass  # Silently skip if yfinance or gdp data unavailable

# -- Row 4: Asset price cards --
st.markdown("#### Asset Prices")
cols_count = min(len(selected_assets), 4)
if cols_count > 0:
    cols = st.columns(cols_count)
    for i, asset_key in enumerate(selected_assets):
        col_idx = i % cols_count
        with cols[col_idx]:
            try:
                config = ASSETS[asset_key]
                df = pd.read_csv(config['data_file'])
                latest = df.iloc[-1]
                prev   = df.iloc[-2]
                price_col   = config['features'][0]
                current     = latest[price_col]
                change      = current - prev[price_col]
                pct_change  = (change / prev[price_col]) * 100
                render_metric_card(label=config['name'], value=current, delta=change)
                if pct_change > 0:
                    st.success(f"+{pct_change:.2f}%")
                else:
                    st.error(f"{pct_change:.2f}%")
            except Exception:
                st.warning(f"{asset_key}: Error")

st.markdown("---")


# ==================== AI PREDICTIONS ====================

st.markdown("### 1 Week AI Predictions")

# Filter assets with models
assets_with_models = [a for a in selected_assets if status[a]['model']]

if not assets_with_models:
    st.warning("No trained models for selected assets. Train models from Settings page.")
else:
    if st.button("Generate Predictions for Selected Assets", use_container_width=True):
        with st.spinner("AI analyzing patterns..."):
            try:
                predictions = batch_predict_week(assets_with_models)
                
                # Create results table
                results = []
                for asset_key in assets_with_models:
                    pred = predictions[asset_key]
                    config = ASSETS[asset_key]
                    
                    if 'error' not in pred:
                        results.append({
                            'Asset': config['name'],
                            'Current': f"${pred['current']:,.2f}",
                            'Predicted': f"${pred['predicted']:,.2f}",
                            'Change': f"${pred['change']:+,.2f}",
                            'Change %': f"{pred['pct_change']:+.2f}%",
                            'Direction': 'UP' if pred['direction'] == 'up' else 'DOWN'
                        })
                
                st.markdown("#### Forecast Results")
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
                
            except Exception as e:
                show_error_message(f"Prediction error: {e}")

st.markdown("---")

# ==================== CEO LAYER — GEMINI NARRATIVE ====================

st.markdown("### CEO Layer — Contextual Market Intelligence")
st.caption("Powered by Gemini AI · Causal Hierarchy: Supply → Geopolitics → Monetary → Risk → Sentiment")

try:
    from utils.llm_manager import analyze_news_context
    from utils.macro_processor import build_macro_context

    col_ceo1, col_ceo2 = st.columns([6, 1])

    with col_ceo1:
        # Show macro context always (hard data, no API needed)
        macro_ctx = build_macro_context()
        macro_summary = macro_ctx.get('macro_summary', '')
        risk_score = macro_ctx.get('recession_risk', 0.5)
        yield_regime = macro_ctx.get('yield_regime', 'unknown').upper()
        m2_bias = macro_ctx.get('m2_bias', 'neutral').upper()
        breakeven_regime = macro_ctx.get('breakeven_regime', 'unknown').upper()

        # Regime labels
        regime_icon = {"INVERSION": "!!!", "FLATTENING": "WARN", "EXPANSION": "OK", "STEEPENING": "OK"}.get(yield_regime, "")
        m2_icon = {"RISK_ON": "OK", "NEUTRAL": "---", "RISK_OFF": "!!!"}.get(m2_bias, "")
        be_icon = {"ANCHORED": "OK", "RISING": "WARN", "ELEVATED": "!!!"}.get(breakeven_regime, "")

        st.markdown(f"""
        **Live Macro Regime:**
        | Signal | Status | Indicator |
        |--------|--------|-----------|
        | Yield Curve | {regime_icon} **{yield_regime}** | T10Y2Y spread |
        | M2 Liquidity | {m2_icon} **{m2_bias}** | YoY growth |
        | Inflation Expectations | {be_icon} **{breakeven_regime}** | 5Y5Y Breakeven |
        | Recession Risk Score | **{risk_score*100:.1f}%** | Composite |
        """)

    with col_ceo2:
        # Counterfactual performance
        try:
            from utils.counterfactual_logger import get_performance_summary
            perf = get_performance_summary()
            if 'error' not in perf:
                st.metric("CEO Hit Ratio", f"{perf['contextual_hit_ratio']*100:.1f}%",
                          delta=f"{perf['ceo_delta']*100:+.1f}% vs baseline")
                st.caption(f"Based on {perf['total_resolved']} resolved forecasts")
                st.caption(perf.get('verdict', ''))
            else:
                st.info("Building accuracy track record...")
        except Exception:
            st.info("Collecting data...")

    # Gemini analysis button — reads from all available news files
    import json
    news_headlines = []
    published_at_list = []

    # Aggregate news from all per-asset news files created by sentiment_fetcher_v2.py
    news_source_files = [
        'data/latest_news_meta.json',
        'data/latest_news_gold.json',
        'data/latest_news_btc.json',
    ]
    seen_titles = set()
    for news_file in news_source_files:
        if os.path.exists(news_file):
            try:
                with open(news_file, 'r', encoding='utf-8') as _f:
                    news_items = json.load(_f)
                for item in news_items:
                    # Support both 'headline' and 'title' key formats
                    headline = item.get('headline') or item.get('title', '')
                    if headline and headline not in seen_titles:
                        seen_titles.add(headline)
                        news_headlines.append(headline)
                        # Support both 'published_at' and 'date' key formats
                        ts = item.get('published_at') or item.get('date')
                        published_at_list.append(ts)
            except Exception:
                pass

    if news_headlines:
        st.caption(f"{len(news_headlines)} headlines loaded from news cache · Ready for CEO analysis")
        if st.button("🔄 Run Gemini CEO Analysis", use_container_width=False, key="ceo_analysis_btn"):
            with st.spinner("Gemini reasoning through Causal Hierarchy..."):
                result = analyze_news_context(
                    headlines=news_headlines,
                    macro_summary=macro_summary,
                    published_at_list=published_at_list,
                )
                if result['is_fallback']:
                    st.warning("Gemini unavailable. System running on Statistical Baseline only.")
                else:
                    st.success(f"✅ Analysis complete · {result['headlines_used']} unique headlines processed")

                    narr_col, score_col = st.columns([3, 2])
                    with narr_col:
                        st.markdown(f"**Dominant Regime:** `{result['dominant_regime'].upper()}`")
                        st.markdown(f"**Confidence:** {result['confidence']*100:.0f}%")
                        st.markdown(f"> *{result['narrative']}*")
                        st.caption("PROBABILISTIC FORECAST — Not a trading signal. Confidence intervals apply.")

                    with score_col:
                        scores = result.get('scores', {})
                        categories = {
                            'supply_shock_severity':        'Supply Shock',
                            'geopolitical_stress':          'Geopolitical',
                            'monetary_policy_hawkishness':  'Monetary',
                            'risk_appetite':                'Risk Appetite',
                            'market_sentiment':             'Sentiment',
                        }
                        for key, label in categories.items():
                            val = scores.get(key, 0.5)
                            bar = "█" * int(val * 10) + "░" * (10 - int(val * 10))
                            st.markdown(f"`{label}` {bar} **{val:.2f}**")
    else:
        with st.expander("Macro Context (CEO Layer waiting for news data)"):
            st.text(macro_summary)
        st.caption("Run Sentiment Sync to enable live Gemini CEO analysis.")

except Exception as _e:
    st.caption(f"CEO Layer: {_e}")

st.markdown("---")

# ==================== PERFORMANCE COMPARISON (MOVED BELOW MACRO) ====================

st.markdown("### Performance Comparison (Normalized)")

days_map = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365,
    "All Time": 99999
}

days = days_map[timeframe]

# Create normalized comparison chart
fig = go.Figure()

for asset_key in selected_assets:
    try:
        config = ASSETS[asset_key]
        df = pd.read_csv(config['data_file'])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Filter timeframe
        if days < 99999:
            df = df[df['Date'] >= df['Date'].max() - pd.Timedelta(days=days)]
        
        # Normalize to 100
        price_col = config['features'][0]
        normalized = (df[price_col] / df[price_col].iloc[0]) * 100
        
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=normalized,
            name=config['name'],
            line=dict(color=config['color'], width=2)
        ))
    
    except Exception as e:
        st.warning(f"Error loading {asset_key}: {e}")

fig.update_layout(
    template="plotly_dark",
    height=500,
    yaxis_title="Normalized Performance (Base 100)",
    xaxis_title="Date",
    hovermode='x unified',
    legend=dict(
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="left",
        x=1.02
    )
)

st.plotly_chart(fig, use_container_width=True)

# ==================== RETURNS SUMMARY ====================

st.markdown("---")
st.markdown("### Period Returns")

returns_data = []

for asset_key in selected_assets:
    try:
        config = ASSETS[asset_key]
        df = pd.read_csv(config['data_file'])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        price_col = config['features'][0]
        current_price = df[price_col].iloc[-1]
        
        # Calculate returns for different periods
        periods = {
            '1D': 1,
            '1W': 5,
            '1M': 21,
            '3M': 63,
            '1Y': 252
        }
        
        row = {'Asset': config['name']}
        
        for period_name, days_back in periods.items():
            if len(df) > days_back:
                past_price = df[price_col].iloc[-days_back]
                return_pct = ((current_price - past_price) / past_price) * 100
                row[period_name] = f"{return_pct:+.2f}%"
            else:
                row[period_name] = "N/A"
        
        returns_data.append(row)
    
    except Exception as e:
        continue

if returns_data:
    df_returns = pd.DataFrame(returns_data)
    st.dataframe(df_returns, use_container_width=True, hide_index=True)

st.markdown("---")

# ==================== CORRELATION MATRIX ====================

st.markdown("### Asset Correlation Matrix")

if len(selected_assets) >= 2:
    # Build correlation matrix
    correlation_data = {}
    common_dates = None
    
    for asset_key in selected_assets:
        try:
            config = ASSETS[asset_key]
            df = pd.read_csv(config['data_file'])
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            
            price_col = config['features'][0]
            
            if common_dates is None:
                common_dates = df.index
            else:
                common_dates = common_dates.intersection(df.index)
            
            correlation_data[config['name']] = df[price_col]
        
        except Exception as e:
            continue
    
    if correlation_data and common_dates is not None and len(common_dates) > 0:
        # Create DataFrame with common dates
        corr_df = pd.DataFrame({k: v.loc[common_dates] for k, v in correlation_data.items()})
        
        # Calculate correlation
        corr_matrix = corr_df.corr()
        
        # Plot heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            template="plotly_dark",
            height=500,
            title="Asset Correlation (Pearson)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Correlation Interpretation:**
        - 🟢 **1.0**: Perfect positive correlation (move together)
        - 🟡 **0.0**: No correlation (independent)
        - 🔴 **-1.0**: Perfect negative correlation (move opposite)
        
        Low or negative correlation between assets = better portfolio diversification.
        """)
else:
    st.info("Select at least 2 assets to show correlation matrix.")

# ==================== FOOTER ====================

st.markdown("---")
st.info("""
**Dashboard Tips:**
- Use "Core Assets" for quick Gold/BTC/SPY comparison
- Compare Mag7 stocks to identify sector leaders
- Check correlation matrix for portfolio diversification
- Set custom timeframes to analyze specific market events
""")
