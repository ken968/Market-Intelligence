"""
Bitcoin Analysis Page
Comprehensive crypto market analysis with halving cycle tracking
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime
from utils.config import ASSETS
from utils.ui_components import (
    inject_custom_css, render_page_header, render_metric_card,
    render_news_section, create_price_chart, create_forecast_chart,
    show_loading_message, show_error_message, render_prediction_table
)
from utils.predictor import AssetPredictor

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="Bitcoin Analysis | Market Intelligence",
    page_icon="",
    layout="wide"
)

inject_custom_css()

# ==================== MAIN CONTENT ====================

render_page_header(
    icon="",
    title="Bitcoin (BTC) Intelligence",
    subtitle="Digital gold analysis with halving cycle insights and on-chain fundamentals"
)

# ==================== CHECK DATA AVAILABILITY ====================

config = ASSETS['btc']

if not os.path.exists(config['data_file']):
    show_error_message("Bitcoin data not available. Please sync data from Settings page.")
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
        label="Bitcoin Price",
        value=latest['BTC'],
        delta=latest['BTC'] - prev['BTC']
    )

with col2:
    pct_change = ((latest['BTC'] - prev['BTC']) / prev['BTC']) * 100
    render_metric_card(
        label="24H Change",
        value=pct_change,
        format_str="{:+.2f}%"
    )

with col3:
    if 'Halving_Cycle' in latest:
        days_to_halving = int(latest['Halving_Cycle'])
        render_metric_card(
            label="Days to Next Halving",
            value=days_to_halving,
            format_str="{:.0f} days"
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

# ==================== HALVING CYCLE ANALYSIS ====================

st.markdown("###  Bitcoin Halving Cycle")

# Halving dates
halving_dates = {
    'Genesis': '2009-01-03',
    '1st Halving': '2012-11-28',
    '2nd Halving': '2016-07-09',
    '3rd Halving': '2020-05-11',
    '4th Halving': '2024-04-19',
    'Next Halving (Est.)': '2028-04-01'
}

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("#### Halving Timeline & Price History")
    
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['BTC'],
        name='BTC Price',
        line=dict(color='#F7931A', width=2)
    ))
    
    # Add halving event markers
    for event, date_str in halving_dates.items():
        if event == 'Genesis' or event == 'Next Halving (Est.)':
            continue
        
        halving_date = pd.to_datetime(date_str)
        if halving_date >= df['Date'].min() and halving_date <= df['Date'].max():
            # Find closest price
            closest_idx = (df['Date'] - halving_date).abs().idxmin()
            price_at_halving = df.loc[closest_idx, 'BTC']
            
            fig.add_vline(
                x=halving_date.to_pydatetime(),
                line_dash="dash",
                line_color="yellow"
            )
            
            # Manually add annotation to avoid internal mean calculation error
            fig.add_annotation(
                x=halving_date.strftime('%Y-%m-%d'),
                y=1,
                yref="paper",
                text=event,
                showarrow=False,
                font=dict(color="yellow"),
                textangle=-90,
                xanchor="right",
                yanchor="top"
            )
    
    fig.update_layout(
        template="plotly_dark",
        height=400,
        yaxis_title="Price (USD)",
        xaxis_title="Date",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### Halving Events")
    
    for event, date in halving_dates.items():
        if 'Next' in event:
            st.markdown(f"**{event}**")
            st.info(f" {date}")
        else:
            st.markdown(f"**{event}**")
            st.text(f" {date}")
    
    st.markdown("---")
    st.info("""
    **Halving Impact:**
    - Supply reduction: 50% cut in new BTC
    - Historical bull runs follow halvings
    - Typically 12-18 month post-halving rally
    """)

st.markdown("---")

# ==================== PRICE CHARTS ====================

st.markdown("###  Price Performance Analysis")

tab1, tab2, tab3, tab4 = st.tabs(["1 Year", "4 Years (Halving Cycle)", "10 Years", "All Time"])

with tab1:
    df_1y = df[df['Date'] >= df['Date'].max() - pd.Timedelta(days=365)]
    fig = create_price_chart(df_1y, 'BTC', "Bitcoin Price - Last Year", config['color'])
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    df_4y = df[df['Date'] >= df['Date'].max() - pd.Timedelta(days=1460)]
    fig = create_price_chart(df_4y, 'BTC', "Bitcoin Price - Current Halving Cycle", config['color'])
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    df_10y = df[df['Date'] >= df['Date'].max() - pd.Timedelta(days=3650)]
    fig = create_price_chart(df_10y, 'BTC', "Bitcoin Price - Decade View", config['color'])
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    fig = create_price_chart(df, 'BTC', "Bitcoin Price - Complete History (Since 2009)", config['color'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Show growth stats
    first_price = df['BTC'].iloc[0]
    current_price = df['BTC'].iloc[-1]
    total_return = ((current_price - first_price) / first_price) * 100
    
    st.success(f"""
     **Historical Performance:**  
    First recorded price: ${first_price:,.2f} ({df['Date'].iloc[0].date()})  
    Current price: ${current_price:,.2f}  
    **Total Return: {total_return:,.0f}%** 
    """)

st.markdown("---")

# ==================== MACRO CORRELATIONS ====================

st.markdown("###  Bitcoin vs Traditional Markets")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### BTC vs DXY (Dollar Index)")
    import plotly.graph_objects as go
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BTC'], name='BTC', yaxis='y1', line=dict(color='#F7931A')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['DXY'], name='DXY', yaxis='y2', line=dict(color='#4b6bff')))
    
    fig.update_layout(
        template="plotly_dark",
        height=350,
        yaxis=dict(title="BTC Price (USD)"),
        yaxis2=dict(title="DXY Index", overlaying='y', side='right'),
        margin=dict(l=20, r=100, t=40, b=20),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### BTC vs VIX (Risk Appetite)")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BTC'], name='BTC', yaxis='y1', line=dict(color='#F7931A')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['VIX'], name='VIX', yaxis='y2', line=dict(color='#FF4D4D')))
    
    fig.update_layout(
        template="plotly_dark",
        height=350,
        yaxis=dict(title="BTC Price (USD)"),
        yaxis2=dict(title="VIX Index", overlaying='y', side='right'),
        margin=dict(l=20, r=100, t=40, b=20),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ==================== AI PREDICTIONS ====================

st.markdown("###  AI-Powered Predictions")

if not os.path.exists(config['model_file']):
    st.warning(" Bitcoin model not trained yet. Please train from Settings page.")
else:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button(" Generate BTC Forecast", use_container_width=True):
            with show_loading_message("AI analyzing 90-day patterns + halving cycles..."):
                try:
                    predictor = AssetPredictor('btc')
                    forecasts = predictor.get_multi_range_forecast()
                    
                    # Display table
                    st.markdown("####  Multi-Range Forecast")
                    render_prediction_table(forecasts, "Bitcoin")
                    
                    # Tomorrow's prediction
                    tomorrow = predictor.predict_tomorrow()
                    
                    st.markdown("---")
                    st.markdown("####  Next 24H Prediction")
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Current", f"${tomorrow['current']:,.2f}")
                    with col_b:
                        st.metric("Tomorrow", f"${tomorrow['predicted']:,.2f}", f"{tomorrow['change']:+,.2f}")
                    with col_c:
                        st.metric("Change", f"{tomorrow['pct_change']:+.2f}%")
                    
                    if tomorrow['direction'] == 'up':
                        st.success(" **Bullish Momentum**: Model predicts upside")
                    else:
                        st.error(" **Bearish Pressure**: Model predicts downside")
                    
                    # Forecast chart
                    st.markdown("####  30-Day Forecast Visualization")
                    forecast_30d = predictor.recursive_forecast(30)
                    fig = create_forecast_chart(df.tail(120), forecast_30d, 'BTC', 30)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    show_error_message(f"Prediction error: {e}")
    
    with col2:
        st.info("""
        **BTC AI Model:**
        - Architecture: LSTM (128-64-32)
        - Sequence: 90 days (longer for cycles)
        - Features: BTC, DXY, VIX, Yield, Sentiment, Halving Cycle, **EMA_90**
        - Training Data: 2009 - Present
        - Higher dropout (0.3) for volatility
        """)

st.markdown("---")

# ==================== NEWS & SENTIMENT ====================

st.markdown("###  Crypto News & Market Sentiment")

render_news_section('btc', max_items=20)

# ==================== FUNDAMENTAL ANALYSIS ====================

st.markdown("---")
st.markdown("###  Fundamental Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ####  Supply Dynamics
    - **Max Supply**: 21,000,000 BTC
    - **Current Supply**: ~19.7M BTC
    - **Inflation Rate**: ~1.7% (decreasing)
    - **Next Halving**: 2028 (Est.)
    
    Scarcity increases post-halving as new supply is cut 50%.
    """)

with col2:
    st.markdown("""
    ####  Institutional Adoption
    - Spot ETFs approved (2024)
    - Corporate treasury holdings
    - PayPal, Square integration
    - Lightning Network growth
    
    Institutional demand driving long-term bull case.
    """)

with col3:
    st.markdown("""
    ####  Macro Environment
    - DXY: {:.2f} (Dollar strength)
    - VIX: {:.2f} (Market fear)
    - US 10Y: {:.2f}% (Risk-free rate)
    
    BTC performs best in weak dollar + low yield environment.
    """.format(latest['DXY'], latest['VIX'], latest['Yield_10Y']))

# ==================== DISCLAIMER ====================

st.markdown("---")
st.warning("""
 **High Risk Investment**: Bitcoin is extremely volatile. Only invest what you can afford to lose.
This analysis is educational only and not financial advice. Always DYOR (Do Your Own Research).
""")
