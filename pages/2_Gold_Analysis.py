"""
Gold (XAUUSD) Analysis Page
Complete technical and fundamental analysis for precious metal
"""

import streamlit as st
import pandas as pd
import os
from utils.config import ASSETS
from utils.ui_components import (
    inject_custom_css, render_page_header, render_metric_card,
    render_news_section, create_price_chart, create_forecast_chart,
    show_loading_message, show_error_message, render_prediction_table
)
from utils.predictor import AssetPredictor

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="Gold Analysis | Market Intelligence",
    page_icon="",
    layout="wide"
)

inject_custom_css()

# ==================== MAIN CONTENT ====================

render_page_header(
    icon="",
    title="Gold (XAUUSD) Intelligence",
    subtitle="Comprehensive technical and fundamental analysis for precious metal markets"
)

# ==================== CHECK DATA AVAILABILITY ====================

config = ASSETS['gold']

if not os.path.exists(config['data_file']):
    show_error_message("Gold data not available. Please sync data from Settings page.")
    st.stop()

# Load data
df = pd.read_csv(config['data_file'])
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

latest = df.iloc[-1]
prev = df.iloc[-2]

# ==================== KEY METRICS ====================

st.markdown("###  Current Market Status")

col1, col2, col3, col4 = st.columns(4)

with col1:
    render_metric_card(
        label="Gold Price",
        value=latest['Gold'],
        delta=latest['Gold'] - prev['Gold']
    )

with col2:
    render_metric_card(
        label="DXY Index",
        value=latest['DXY'],
        delta=latest['DXY'] - prev['DXY'],
        format_str="{:.2f}"
    )

with col3:
    render_metric_card(
        label="VIX (Fear Index)",
        value=latest['VIX'],
        delta=latest['VIX'] - prev['VIX'],
        format_str="{:.2f}"
    )

with col4:
    if 'Sentiment' in latest:
        render_metric_card(
            label="Market Sentiment",
            value=latest['Sentiment'],
            delta=latest['Sentiment'] - prev.get('Sentiment', 0),
            format_str="{:.2f}"
        )

st.markdown("---")

# ==================== PRICE CHART ====================

st.markdown("###  Historical Price Performance")

tab1, tab2, tab3 = st.tabs(["1 Year", "5 Years", "All Time"])

with tab1:
    df_1y = df[df['Date'] >= df['Date'].max() - pd.Timedelta(days=365)]
    fig = create_price_chart(df_1y, 'Gold', "Gold Price - Last 12 Months", config['color'])
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    df_5y = df[df['Date'] >= df['Date'].max() - pd.Timedelta(days=1825)]
    fig = create_price_chart(df_5y, 'Gold', "Gold Price - Last 5 Years", config['color'])
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    fig = create_price_chart(df, 'Gold', "Gold Price - Complete History", config['color'])
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ==================== CORRELATION ANALYSIS ====================

st.markdown("###  Macro-Economic Correlations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Gold vs DXY (Inverse Correlation)")
    import plotly.graph_objects as go
    
    fig = go.Figure()
    # Gold Price
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Gold'], name='Gold', yaxis='y1', line=dict(color='#FFD700', width=2)))
    # EMA 90 (Indicator)
    if 'EMA_90' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_90'], name='EMA 90', yaxis='y1', line=dict(color='#FFA500', width=1, dash='dash'), opacity=0.7))
    # DXY Index
    fig.add_trace(go.Scatter(x=df['Date'], y=df['DXY'], name='DXY Index', yaxis='y2', line=dict(color='#4b6bff', width=1.5)))
    
    fig.update_layout(
        template="plotly_dark",
        height=350,
        yaxis=dict(title="Gold Price (USD)"),
        yaxis2=dict(title="DXY Index", overlaying='y', side='right'),
        margin=dict(l=20, r=100, t=40, b=20),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### Gold vs VIX (Risk Indicator)")
    
    fig = go.Figure()
    # Gold Price
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Gold'], name='Gold', yaxis='y1', line=dict(color='#FFD700', width=2)))
    # EMA 90 (Indicator)
    if 'EMA_90' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_90'], name='EMA 90', yaxis='y1', line=dict(color='#FFA500', width=1, dash='dash'), opacity=0.7))
    # VIX Index
    fig.add_trace(go.Scatter(x=df['Date'], y=df['VIX'], name='VIX (Fear)', yaxis='y2', line=dict(color='#FF4D4D', width=1.5)))
    
    fig.update_layout(
        template="plotly_dark",
        height=350,
        yaxis=dict(title="Gold Price (USD)"),
        yaxis2=dict(title="VIX Index", overlaying='y', side='right'),
        margin=dict(l=20, r=100, t=40, b=20),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ==================== AI PREDICTIONS ====================

st.markdown("###  AI-Powered Predictions")

# Add disclaimer
from utils.config import FORECAST_DISCLAIMER
st.info(FORECAST_DISCLAIMER)

if not os.path.exists(config['model_file']):
    st.warning(" Model not trained yet. Please train the Gold model from Settings page.")
else:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button(" Generate Multi-Range Forecast", use_container_width=True):
            with show_loading_message("AI analyzing 60-day patterns..."):
                try:
                    predictor = AssetPredictor('gold')
                    forecasts = predictor.get_multi_range_forecast()
                    
                    # Display table
                    st.markdown("####  Forecast Results")
                    render_prediction_table(forecasts, "Gold")
                    
                    # NEW: Add automated forecast analysis
                    from utils.forecast_analyzer import ForecastAnalyzer
                    
                    analyzer = ForecastAnalyzer()
                    
                    # Extract prices from new format
                    forecast_prices = []
                    for key, value in list(forecasts.items())[1:]:  # Skip 'Current'
                        if isinstance(value, dict):
                            forecast_prices.append(value['price'])
                        else:
                            forecast_prices.append(value)
                    
                    insights = analyzer.analyze_forecast(
                        current_price=forecasts['Current'],
                        forecast_prices=forecast_prices,
                        asset_name='Gold'
                    )
                    
                    st.markdown("###  AI Analysis")
                    st.info(insights['summary'])
                    
                    col_i1, col_i2, col_i3 = st.columns(3)
                    with col_i1:
                        st.metric("Trend", insights['trend'].title())
                    with col_i2:
                        st.metric("Strength", insights['strength'].title())
                    with col_i3:
                        st.metric("Risk", insights['risk_level'].title())
                    
                    st.success(f" **Recommendation**: {insights['recommendation']}")
                    
                    # Highlight tomorrow
                    tomorrow = predictor.predict_tomorrow()
                    
                    st.markdown("---")
                    st.markdown("####  Next Day Prediction")
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Current Price", f"${tomorrow['current']:,.2f}")
                    with col_b:
                        st.metric("Tomorrow", f"${tomorrow['predicted']:,.2f}", f"{tomorrow['change']:+.2f}")
                    with col_c:
                        st.metric("Change %", f"{tomorrow['pct_change']:+.2f}%")
                    
                    if tomorrow['direction'] == 'up':
                        st.success(" **Bullish Signal**: Model predicts upward momentum")
                    else:
                        st.error(" **Bearish Signal**: Model predicts downward pressure")
                    
                    # Show forecast chart
                    st.markdown("####  Forecast Visualization")
                    forecast_30d = predictor.recursive_forecast(30)
                    fig = create_forecast_chart(df.tail(90), forecast_30d, 'Gold', 30)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    show_error_message(f"Prediction failed: {e}")
    
    with col2:
        st.info("""
        **AI Model Details:**
        - Architecture: LSTM 3-Layer
        - Sequence Length: 60 days
        - Features: Gold, DXY, VIX, Yield, Sentiment, EMA_90
        - Training: 10 years historical data
        """)

st.markdown("---")

# ==================== NEWS & SENTIMENT ====================

st.markdown("###  Market News & Sentiment Analysis")

render_news_section('gold', max_items=20)

# ==================== FUNDAMENTAL FACTORS ====================

st.markdown("---")
st.markdown("###  Fundamental Analysis Factors")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ####  Dollar Strength (DXY)
    **Current:** {:.2f}
    
    Strong dollar typically pressures gold prices due to:
    - Higher opportunity cost
    - Reduced international demand
    - USD-denominated pricing
    """.format(latest['DXY']))

with col2:
    st.markdown("""
    ####  Market Fear (VIX)
    **Current:** {:.2f}
    
    Gold as safe-haven asset thrives when:
    - VIX above 20 (uncertainty)
    - Equity market volatility
    - Geopolitical tensions
    """.format(latest['VIX']))

with col3:
    st.markdown("""
    ####  Treasury Yields
    **Current:** {:.2f}%
    
    Bond yields impact gold through:
    - Opportunity cost comparison
    - Real interest rates
    - Inflation expectations
    """.format(latest['Yield_10Y']))

# ==================== DISCLAIMER ====================

st.markdown("---")
st.warning("""
 **Investment Disclaimer**: This analysis is for educational purposes only. 
Gold trading involves significant risk. Always conduct your own research and consult 
financial advisors before making investment decisions.
""")
