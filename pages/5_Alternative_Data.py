"""
Alternative Data Page
Display Google Trends, Fear & Greed Index, and Fed Watch data
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import requests

from utils.config import ASSETS, get_all_stock_tickers
from utils.ui_components import inject_custom_css, render_page_header, show_error_message
from scripts.google_trends_fetcher import GoogleTrendsFetcher, batch_fetch_trends
from scripts.macro_sentiment import MacroSentimentFetcher

# Page config
st.set_page_config(
    page_title="Alternative Data | Market Intelligence",
    layout="wide"
)

inject_custom_css()

# Header
render_page_header(
    icon="",
    title="Alternative Data Intelligence",
    subtitle="Google Trends, Fear & Greed Index, and Macro Sentiment for systemic risk insights"
)

# Asset selector
all_assets = ['gold', 'btc'] + [t.lower() for t in get_all_stock_tickers()]
selected_assets = st.multiselect(
    "Select assets to analyze",
    all_assets,
    default=['gold', 'btc', 'msft']
)

if not selected_assets:
    st.warning("Please select at least one asset")
    st.stop()

st.markdown("---")

# Tabs for different data sources
tab1, tab2, tab3 = st.tabs(["Google Trends", "Fear & Greed Index", "Macro Sentiment Tracker"])

# Tab 1: Google Trends
with tab1:
    st.markdown("### Google Trends Analysis")
    st.info("Search volume indicates retail interest. Rising trends often precede price movements.")
    
    if st.button("Fetch Latest Google Trends", use_container_width=True):
        with st.spinner("Fetching Google Trends data..."):
            try:
                trends_data = batch_fetch_trends(selected_assets)
                
                results = []
                for asset, signal in trends_data.items():
                    if 'error' not in signal:
                        results.append({
                            'Asset': asset.upper(),
                            'Current Interest': f"{signal['current_interest']}/100",
                            'Avg Interest': f"{signal['avg_interest']:.1f}/100",
                            'Trend': signal['trend'].title(),
                            'Signal Strength': f"{signal['signal_strength']:.2f}"
                        })
                
                if results:
                    st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
                    
                    st.markdown("#### Trend Charts")
                    fetcher = GoogleTrendsFetcher()
                    cols = st.columns(min(len(selected_assets), 2))
                    
                    for i, asset in enumerate(selected_assets):
                        with cols[i % 2]:
                            data = fetcher.fetch_asset_trends(asset)
                            if not data.empty:
                                fig = go.Figure()
                                keyword = data.columns[0]
                                fig.add_trace(go.Scatter(
                                    x=data.index, y=data[keyword],
                                    name=asset.upper(),
                                    line=dict(color='#00A8E8', width=2)
                                ))
                                fig.update_layout(
                                    title=f"{asset.upper()} Search Interest",
                                    template="plotly_dark", height=300,
                                    yaxis_title="Interest (0-100)", xaxis_title="Date",
                                    margin=dict(l=40, r=40, t=40, b=40)
                                )
                                st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No data available")
            
            except Exception as e:
                show_error_message(f"Error: {e}")

# Tab 2: Fear & Greed Index
with tab2:
    st.markdown("### Crypto Fear & Greed Index")
    st.info("""
    **Source:** [alternative.me](https://alternative.me/crypto/fear-and-greed-index/) — No API key required.  
    Measures overall crypto market sentiment: **0-24** Extreme Fear | **25-49** Fear | **50** Neutral | **51-74** Greed | **75-100** Extreme Greed  
    Useful as a **contrarian indicator** — Extreme Fear often marks buying opportunities; Extreme Greed often marks tops.
    """)

    days_fg = st.slider("Days to display", min_value=7, max_value=90, value=30, step=7)

    if st.button("Fetch Fear & Greed Data", use_container_width=True):
        with st.spinner("Fetching Fear & Greed Index..."):
            try:
                resp = requests.get(
                    f"https://api.alternative.me/fng/?limit={days_fg}&format=json",
                    timeout=10
                )
                resp.raise_for_status()
                data = resp.json().get("data", [])

                if data:
                    records = []
                    for item in data:
                        ts = int(item["timestamp"])
                        date = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
                        records.append({
                            "Date": date,
                            "Value": int(item["value"]),
                            "Classification": item["value_classification"]
                        })
                    df_fg = pd.DataFrame(records).sort_values("Date")

                    # Latest metric
                    latest = records[0]
                    val = latest["Value"]
                    cls = latest["Classification"]
                    color = "#FF4444" if val < 25 else "#FF8C00" if val < 50 else "#00C49A" if val < 75 else "#00FF7F"

                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.metric("Today's Index", f"{val} / 100", delta=cls)

                    # Chart
                    def get_bar_color(v):
                        if v < 25: return "#FF4444"
                        if v < 50: return "#FF8C00"
                        if v < 75: return "#00C49A"
                        return "#00FF7F"

                    bar_colors = [get_bar_color(v) for v in df_fg["Value"]]

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=df_fg["Date"], y=df_fg["Value"],
                        marker_color=bar_colors,
                        text=df_fg["Classification"],
                        textposition="outside",
                        hovertemplate="<b>%{x}</b><br>Value: %{y}<br>%{text}<extra></extra>"
                    ))
                    fig.add_hline(y=25, line_dash="dash", line_color="#FF4444", annotation_text="Extreme Fear")
                    fig.add_hline(y=75, line_dash="dash", line_color="#00FF7F", annotation_text="Extreme Greed")
                    fig.update_layout(
                        title=f"Fear & Greed Index — Last {days_fg} Days",
                        template="plotly_dark", height=400,
                        yaxis_title="Index Value (0-100)", xaxis_title="Date",
                        yaxis=dict(range=[0, 105]),
                        margin=dict(l=40, r=40, t=60, b=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(df_fg[::-1].reset_index(drop=True), use_container_width=True, hide_index=True)

            except Exception as e:
                show_error_message(f"Error fetching Fear & Greed: {e}")

# Tab 3: Macro Sentiment
with tab3:
    st.markdown("### Systemic Macro Sentiment")
    st.info("Continuous analysis of DXY, US10Y Yield, and VIX. Risk-On (Dovish) environment is bullish for risk assets.")
    
    if st.button("Calculate Macro Sentiment", use_container_width=True):
        with st.spinner("Analyzing macro environment..."):
            try:
                fetcher = MacroSentimentFetcher()
                signal = fetcher.get_fed_signal()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Dovish Score", f"{signal['dovish_score']:.0f}/100")
                with col2:
                    st.metric("Stance", signal['stance'].title())
                with col3:
                    st.metric("Next Meeting", signal['probabilities']['next_meeting_date'])
                
                st.markdown("#### Implied Market Regime")
                probs = signal['probabilities']
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Risk-On Probability", f"{probs['prob_cut']:.1%}",
                             delta="Bullish" if probs['prob_cut'] > 0.3 else None)
                with col_b:
                    st.metric("Neutral", f"{probs['prob_hold']:.1%}")
                with col_c:
                    st.metric("Risk-Off Probability", f"{probs['prob_hike']:.1%}",
                             delta="Bearish" if probs['prob_hike'] > 0.3 else None)
                
                st.markdown("#### Impact on Assets")
                impact_data = []
                for asset in selected_assets:
                    impact = signal['signal_for_gold'] if asset in ['gold', 'btc'] else signal['signal_for_stocks']
                    impact_data.append({
                        'Asset': asset.upper(),
                        'Macro Signal': impact.title(),
                        'Confidence': f"{signal['confidence']:.2f}"
                    })
                st.dataframe(pd.DataFrame(impact_data), use_container_width=True, hide_index=True)
                
                fetcher.save_fed_data()
                historical = fetcher.get_historical_data(days=30)
                
                if not historical.empty:
                    st.markdown("#### Historical Dovish Score (30 Days)")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=historical['timestamp'], y=historical['dovish_score'],
                        mode='lines+markers', name='Dovish Score',
                        line=dict(color='#00A8E8', width=2)
                    ))
                    fig.add_hline(y=65, line_dash="dash", line_color="green", annotation_text="Dovish Threshold")
                    fig.add_hline(y=35, line_dash="dash", line_color="red", annotation_text="Hawkish Threshold")
                    fig.update_layout(
                        template="plotly_dark", height=400,
                        yaxis_title="Dovish Score (0-100)", xaxis_title="Date",
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                show_error_message(f"Error: {e}")

# Disclaimer
st.markdown("---")
st.warning("""
**Data Disclaimer**: Alternative data sources provide supplementary insights but should not be the sole basis for trading decisions. 
Google Trends reflects retail search interest which may not align with institutional flows. 
The Fear & Greed Index is a sentiment proxy — use it as a contrarian signal, not a directional indicator.
Macro Sentiment probabilities are algorithmically derived from Treasuries, DXY, and VIX.
""")
