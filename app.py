"""
XAUUSD Multi-Asset Terminal - Main Homepage
Central dashboard with overview of all assets
"""

import streamlit as st
import pandas as pd
import os
from utils.config import ASSETS, STOCK_TICKERS, get_asset_status
from utils.ui_components import (
    inject_custom_css, render_page_header, render_metric_card,
    render_status_badge, create_multi_asset_comparison
)
from utils.predictor import batch_predict_tomorrow

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="Market Intelligence Terminal",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()

# ==================== SIDEBAR ====================

with st.sidebar:
    st.markdown("###  Navigation")
    st.info("""
    Use the pages in the sidebar to navigate:
    - **Dashboard**: Overview of all assets
    - **Gold Analysis**: Deep dive into XAUUSD
    - **Bitcoin Analysis**: BTC market insights
    - **Stocks Analysis**: US equities tracking
    - **Settings**: Data sync & model training
    """)
    
    st.markdown("---")
    
    # System Status
    st.markdown("###  System Status")
    status = get_asset_status()
    
    assets_ready = sum(1 for s in status.values() if s['data'] and s['model'])
    total_assets = len(status)
    
    if assets_ready == total_assets:
        render_status_badge('success', f'All Systems Operational ({assets_ready}/{total_assets})')
    elif assets_ready > 0:
        render_status_badge('warning', f'Partial ({assets_ready}/{total_assets} Ready)')
    else:
        render_status_badge('danger', 'No Models Trained')
    
    st.markdown("---")
    st.caption("© 2025 Market Intelligence | AI-Powered Market Terminal")

# ==================== MAIN CONTENT ====================

render_page_header(
    icon="",
    title="Multi-Asset Intelligence Dashboard",
    subtitle="Real-time monitoring and AI predictions for Gold, Bitcoin, and US Equities"
)

# ==================== ASSET STATUS OVERVIEW ====================

st.markdown("### Asset Coverage")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Precious Metals")
    if status['gold']['model']:
        render_status_badge('success', 'Gold (XAUUSD) ')
    else:
        render_status_badge('danger', 'Gold (XAUUSD) ')

with col2:
    st.markdown("#### Cryptocurrency")
    if status['btc']['model']:
        render_status_badge('success', 'Bitcoin (BTC) ')
    else:
        render_status_badge('danger', 'Bitcoin (BTC) ')

with col3:
    st.markdown("#### US Equities")
    stocks_ready = sum(1 for ticker in STOCK_TICKERS.keys() if status[ticker.lower()]['model'])
    total_stocks = len(STOCK_TICKERS)
    
    if stocks_ready == total_stocks:
        render_status_badge('success', f'All Stocks Ready ({stocks_ready}/{total_stocks})')
    elif stocks_ready > 0:
        render_status_badge('warning', f'{stocks_ready}/{total_stocks} Stocks Ready')
    else:
        render_status_badge('danger', 'No Stocks Trained')

st.markdown("---")

# ==================== QUICK PREDICTIONS ====================

st.markdown("###  Tomorrow's AI Predictions")

# Check if any models are available
available_assets = [key for key, s in status.items() if s['model']]

if not available_assets:
    st.warning(" No trained models available. Please go to **Settings** page to sync data and train models.")
else:
    with st.spinner("Generating predictions..."):
        try:
            # Predict for available core assets (Gold, BTC, SPY)
            core_assets = ['gold', 'btc', 'spy']
            assets_to_predict = [a for a in core_assets if a in available_assets]
            
            predictions = batch_predict_tomorrow(assets_to_predict)
            
            # Display predictions
            cols = st.columns(len(assets_to_predict))
            
            for i, asset_key in enumerate(assets_to_predict):
                with cols[i]:
                    config = ASSETS[asset_key]
                    pred = predictions[asset_key]
                    
                    if 'error' not in pred:
                        icon_prefix = f"{config['icon']} " if config['icon'] else ""
                        st.markdown(f"#### {icon_prefix}{config['name']}")
                        render_metric_card(
                            label="Tomorrow's Prediction",
                            value=pred['predicted'],
                            delta=pred['change']
                        )
                        
                        if pred['direction'] == 'up':
                            st.success(f" +{pred['pct_change']:.2f}%")
                        else:
                            st.error(f" {pred['pct_change']:.2f}%")
                    else:
                        st.error(f"Error: {pred['error']}")
        
        except Exception as e:
            st.error(f"Prediction error: {e}")

st.markdown("---")

# ==================== MARKET OVERVIEW ====================

st.markdown("###  Latest Market Data")

# Show latest prices for all available assets
available_data_assets = [key for key, s in status.items() if s['data']]

if available_data_assets:
    # Group by category
    categories = {
        'Precious Metals': ['gold'],
        'Cryptocurrency': ['btc'],
        'Market Indices': ['spy', 'qqq', 'dia'],
        'Magnificent 7': ['aapl', 'msft', 'googl', 'amzn', 'nvda', 'meta', 'tsla'],
        'Semiconductor': ['tsm']
    }
    
    for category, assets in categories.items():
        available = [a for a in assets if a in available_data_assets]
        
        if available:
            st.markdown(f"#### {category}")
            
            cols = st.columns(min(len(available), 4))
            
            for i, asset_key in enumerate(available):
                with cols[i % 4]:
                    try:
                        config = ASSETS[asset_key]
                        df = pd.read_csv(config['data_file'])
                        
                        latest = df.iloc[-1]
                        prev = df.iloc[-2]
                        
                        price_col = config['features'][0]
                        current_price = latest[price_col]
                        prev_price = prev[price_col]
                        change = current_price - prev_price
                        
                        render_metric_card(
                            label=config['name'],
                            value=current_price,
                            delta=change
                        )
                    
                    except Exception as e:
                        st.warning(f"{config['name']}: Data unavailable")
else:
    st.warning(" No market data available. Please sync data from the **Settings** page.")

st.markdown("---")

# ==================== QUICK START GUIDE ====================

st.markdown("###  Quick Start Guide")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ####  Sync Data
    Navigate to **Settings** page and click:
    - "Sync All Assets" to download market data
    - "Sync Sentiment" to analyze news
    """)

with col2:
    st.markdown("""
    ####  Train Models
    After syncing data, train AI models:
    - Train individual assets
    - Or train all stocks at once
    """)

with col3:
    st.markdown("""
    ####  Explore Analysis
    Visit dedicated pages for:
    - Gold technical & fundamental analysis
    - Bitcoin halving cycle insights
    - Stock comparison & predictions
    """)

# ==================== FOOTER ====================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94A3B8; font-size: 0.85rem;">
    <strong>Market Intelligence Terminal</strong> | Powered by Deep Learning LSTM<br>
    AI Intelligence • Real-time Data • Sentiment Analysis<br>
     <em>Educational purpose only. Not financial advice.</em>
</div>
""", unsafe_allow_html=True)
