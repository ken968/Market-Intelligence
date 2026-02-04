"""
US Stocks Analysis Page
Multi-stock tracking for indices, Mag7, and semiconductors
"""

import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
from utils.config import ASSETS, STOCK_TICKERS, get_all_stock_tickers
from utils.ui_components import (
    inject_custom_css, render_page_header, render_metric_card,
    render_news_section, create_price_chart,
    show_loading_message, show_error_message, render_prediction_table
)
from utils.predictor import AssetPredictor, batch_predict_tomorrow

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="US Stocks Analysis | XAUUSD Terminal",
    page_icon="",
    layout="wide"
)

inject_custom_css()

# ==================== MAIN CONTENT ====================

render_page_header(
    icon="",
    title="US Equities Intelligence",
    subtitle="Comprehensive analysis of market indices, Magnificent 7, and semiconductor leaders"
)

# ==================== STOCK SELECTOR ====================

all_tickers = get_all_stock_tickers()
available_stocks = [t for t in all_tickers if os.path.exists(ASSETS[t.lower()]['data_file'])]

if not available_stocks:
    show_error_message("No stock data available. Please sync data from Settings page.")
    st.stop()

st.markdown("###  Stock Selection")

# Group stocks by category
indices = [t for t in available_stocks if t in ['SPY', 'QQQ', 'DIA']]
mag7 = [t for t in available_stocks if t in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']]
semi = [t for t in available_stocks if t == 'TSM']

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("####  Market Indices")
    selected_indices = st.multiselect("Select indices", indices, default=indices[:1] if indices else [])

with col2:
    st.markdown("####  Magnificent 7")
    selected_mag7 = st.multiselect("Select Mag7", mag7, default=mag7[:2] if len(mag7) >= 2 else mag7)

with col3:
    st.markdown("#### 💻 Semiconductors")
    selected_semi = st.multiselect("Select semis", semi, default=semi)

selected_stocks = selected_indices + selected_mag7 + selected_semi

if not selected_stocks:
    st.warning(" Please select at least one stock to analyze.")
    st.stop()

st.markdown("---")

# ==================== MARKET OVERVIEW ====================

st.markdown("###  Current Market Snapshot")

# Load latest data for all selected stocks
latest_data = {}
for ticker in selected_stocks:
    try:
        config = ASSETS[ticker.lower()]
        df = pd.read_csv(config['data_file'])
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        price_col = config['features'][0]
        latest_data[ticker] = {
            'price': latest[price_col],
            'change': latest[price_col] - prev[price_col],
            'pct_change': ((latest[price_col] - prev[price_col]) / prev[price_col]) * 100
        }
    except Exception as e:
        st.error(f"Error loading {ticker}: {e}")

# Display metrics
cols = st.columns(min(len(selected_stocks), 4))

for i, ticker in enumerate(selected_stocks):
    with cols[i % 4]:
        data = latest_data[ticker]
        render_metric_card(
            label=f"{ticker} - {STOCK_TICKERS[ticker]['name']}",
            value=data['price'],
            delta=data['change']
        )
        
        if data['pct_change'] > 0:
            st.success(f" +{data['pct_change']:.2f}%")
        else:
            st.error(f" {data['pct_change']:.2f}%")

st.markdown("---")

# ==================== COMPARISON CHART ====================

st.markdown("###  Price Performance Comparison")

# Timeframe selector
timeframe = st.selectbox(
    "Select timeframe",
    ["1 Month", "3 Months", "6 Months", "1 Year", "5 Years", "All Time"],
    index=3
)

days_map = {
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365,
    "5 Years": 1825,
    "All Time": 99999
}

days = days_map[timeframe]

# Create comparison chart (normalized to 100 at start)
fig = go.Figure()

for ticker in selected_stocks:
    try:
        config = ASSETS[ticker.lower()]
        df = pd.read_csv(config['data_file'])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Filter by timeframe
        if days < 99999:
            df = df[df['Date'] >= df['Date'].max() - pd.Timedelta(days=days)]
        
        # Normalize to 100
        price_col = config['features'][0]
        normalized = (df[price_col] / df[price_col].iloc[0]) * 100
        
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=normalized,
            name=ticker,
            line=dict(color=config['color'], width=2)
        ))
    
    except Exception as e:
        st.error(f"Error plotting {ticker}: {e}")

fig.update_layout(
    template="plotly_dark",
    height=500,
    yaxis_title="Normalized Price (Base 100)",
    xaxis_title="Date",
    hovermode='x unified',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ==================== INDIVIDUAL STOCK ANALYSIS ====================

st.markdown("### 🔍 Individual Stock Deep Dive")

selected_focus = st.selectbox("Select stock for detailed analysis", selected_stocks)

if selected_focus:
    config = ASSETS[selected_focus.lower()]
    df = pd.read_csv(config['data_file'])
    df['Date'] = pd.to_datetime(df['Date'])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"#### {selected_focus} - {STOCK_TICKERS[selected_focus]['name']}")
        
        fig = create_price_chart(df, selected_focus, f"{selected_focus} Price History", config['color'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Stock Info")
        st.info(f"""
        **Ticker:** {selected_focus}  
        **Company:** {STOCK_TICKERS[selected_focus]['name']}  
        **Sector:** {STOCK_TICKERS[selected_focus]['sector']}  
        **Current Price:** ${latest_data[selected_focus]['price']:,.2f}  
        **24H Change:** {latest_data[selected_focus]['pct_change']:+.2f}%
        """)

st.markdown("---")

# ==================== AI PREDICTIONS ====================

st.markdown("###  AI-Powered Predictions")

# Check which stocks have trained models
stocks_with_models = [t for t in selected_stocks if os.path.exists(ASSETS[t.lower()]['model_file'])]

if not stocks_with_models:
    st.warning(" No trained models for selected stocks. Please train models from Settings page.")
else:
    st.markdown(f"**Models available for:** {', '.join(stocks_with_models)}")
    
    if st.button(" Predict Tomorrow (All Selected)", use_container_width=True):
        with show_loading_message("Generating predictions for all stocks..."):
            try:
                predictions = batch_predict_tomorrow([s.lower() for s in stocks_with_models])
                
                # Create results dataframe
                results = []
                for ticker in stocks_with_models:
                    pred = predictions[ticker.lower()]
                    if 'error' not in pred:
                        results.append({
                            'Stock': ticker,
                            'Company': STOCK_TICKERS[ticker]['name'],
                            'Current': f"${pred['current']:,.2f}",
                            'Tomorrow': f"${pred['predicted']:,.2f}",
                            'Change': f"${pred['change']:+,.2f}",
                            'Change %': f"{pred['pct_change']:+.2f}%",
                            'Signal': ' Bullish' if pred['direction'] == 'up' else ' Bearish'
                        })
                
                st.markdown("####  Tomorrow's Predictions")
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
                
            except Exception as e:
                show_error_message(f"Prediction error: {e}")
    
    # Individual detailed forecast
    st.markdown("---")
    st.markdown("####  Multi-Range Forecast (Individual Stock)")
    
    forecast_stock = st.selectbox("Select stock for detailed forecast", stocks_with_models)
    
    if st.button(f"Generate Forecast for {forecast_stock}"):
        with show_loading_message(f"Analyzing {forecast_stock}..."):
            try:
                predictor = AssetPredictor(forecast_stock.lower())
                forecasts = predictor.get_multi_range_forecast()
                
                render_prediction_table(forecasts, forecast_stock)
                
                # Show chart
                forecast_30d = predictor.recursive_forecast(30)
                df_stock = pd.read_csv(ASSETS[forecast_stock.lower()]['data_file'])
                df_stock['Date'] = pd.to_datetime(df_stock['Date'])
                
                from utils.ui_components import create_forecast_chart
                fig = create_forecast_chart(df_stock.tail(90), forecast_30d, forecast_stock, 30)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                show_error_message(f"Error: {e}")

st.markdown("---")

# ==================== SECTOR ANALYSIS ====================

st.markdown("### 🏭 Sector Performance")

# Group by sector
sector_performance = {}
for ticker in selected_stocks:
    sector = STOCK_TICKERS[ticker]['sector']
    if sector not in sector_performance:
        sector_performance[sector] = []
    sector_performance[sector].append({
        'ticker': ticker,
        'change': latest_data[ticker]['pct_change']
    })

cols = st.columns(len(sector_performance))

for i, (sector, stocks) in enumerate(sector_performance.items()):
    with cols[i]:
        st.markdown(f"#### {sector}")
        
        avg_change = sum(s['change'] for s in stocks) / len(stocks)
        
        for stock in stocks:
            st.text(f"{stock['ticker']}: {stock['change']:+.2f}%")
        
        st.markdown("---")
        if avg_change > 0:
            st.success(f"Avg: +{avg_change:.2f}%")
        else:
            st.error(f"Avg: {avg_change:.2f}%")

# ==================== NEWS (Optional) ====================

st.markdown("---")
st.markdown("### 📰 Latest Market News")

# Show news for selected stock if available
if selected_focus and os.path.exists(ASSETS[selected_focus.lower()]['news_file']):
    render_news_section(selected_focus.lower(), max_items=6)
else:
    st.info(" News sentiment available after running sync from Settings page.")

# ==================== DISCLAIMER ====================

st.markdown("---")
st.warning("""
 **Investment Risk**: Stock market investments carry risk. Past performance does not guarantee future results.
This analysis is for educational purposes only. Conduct your own research and consult a financial advisor.
""")
