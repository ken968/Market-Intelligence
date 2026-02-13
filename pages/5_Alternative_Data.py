"""
Alternative Data Page
Display Google Trends, Reddit sentiment, and Fed Watch data
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

from utils.config import ASSETS, get_all_stock_tickers
from utils.ui_components import inject_custom_css, render_page_header, show_error_message
from scripts.google_trends_fetcher import GoogleTrendsFetcher, batch_fetch_trends
from scripts.reddit_sentiment_fetcher import RedditSentimentAnalyzer, batch_analyze_sentiment
from scripts.fed_watch_fetcher import FedWatchFetcher

# Page config
st.set_page_config(
    page_title="Alternative Data | Market Intelligence",
    page_icon="",
    layout="wide"
)

inject_custom_css()

# Header
render_page_header(
    icon="",
    title="Alternative Data Intelligence",
    subtitle="Google Trends, Reddit sentiment, and Fed Watch analysis for market insights"
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
tab1, tab2, tab3 = st.tabs(["Google Trends", "Reddit Sentiment", "Fed Watch"])

# Tab 1: Google Trends
with tab1:
    st.markdown("### Google Trends Analysis")
    st.info("Search volume indicates retail interest. Rising trends often precede price movements.")
    
    if st.button("Fetch Latest Google Trends", use_container_width=True):
        with st.spinner("Fetching Google Trends data..."):
            try:
                trends_data = batch_fetch_trends(selected_assets)
                
                # Display results
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
                    
                    # Show charts for each asset
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
                                    x=data.index,
                                    y=data[keyword],
                                    name=asset.upper(),
                                    line=dict(color='#00A8E8', width=2)
                                ))
                                
                                fig.update_layout(
                                    title=f"{asset.upper()} Search Interest",
                                    template="plotly_dark",
                                    height=300,
                                    yaxis_title="Interest (0-100)",
                                    xaxis_title="Date",
                                    margin=dict(l=40, r=40, t=40, b=40)
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.warning("No data available")
            
            except Exception as e:
                show_error_message(f"Error: {e}")

# Tab 2: Reddit Sentiment
with tab2:
    st.markdown("### Reddit Sentiment Analysis")
    st.info("Sentiment from r/cryptocurrency, r/wallstreetbets, and r/gold. Requires Reddit API credentials for live data.")
    
    # Reddit API configuration
    with st.expander("Reddit API Configuration (Optional)"):
        st.markdown("""
        To enable live Reddit sentiment analysis:
        1. Create a Reddit app at https://www.reddit.com/prefs/apps
        2. Get your client_id and client_secret
        3. Enter them below
        
        **Note:** Without credentials, mock data will be used.
        """)
        
        client_id = st.text_input("Client ID", type="password")
        client_secret = st.text_input("Client Secret", type="password")
    
    if st.button("Analyze Reddit Sentiment", use_container_width=True):
        with st.spinner("Analyzing Reddit sentiment..."):
            try:
                sentiment_data = batch_analyze_sentiment(
                    selected_assets,
                    client_id=client_id if client_id else None,
                    client_secret=client_secret if client_secret else None
                )
                
                # Display results
                results = []
                for asset, signal in sentiment_data.items():
                    if 'error' not in signal:
                        results.append({
                            'Asset': asset.upper(),
                            'Sentiment Score': f"{signal['sentiment_score']:.3f}",
                            'Label': signal['sentiment_label'],
                            'Signal': signal['signal'].title(),
                            'Confidence': f"{signal['confidence']:.2f}",
                            'Posts Analyzed': signal['post_count']
                        })
                
                if results:
                    st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
                    
                    # Sentiment distribution chart
                    st.markdown("#### Sentiment Distribution")
                    
                    fig = go.Figure()
                    
                    assets_list = [r['Asset'] for r in results]
                    scores = [float(r['Sentiment Score']) for r in results]
                    
                    colors = ['#00FF00' if s > 0.1 else '#FF0000' if s < -0.1 else '#FFFF00' for s in scores]
                    
                    fig.add_trace(go.Bar(
                        x=assets_list,
                        y=scores,
                        marker_color=colors,
                        text=[f"{s:.2f}" for s in scores],
                        textposition='outside'
                    ))
                    
                    fig.update_layout(
                        title="Sentiment Scores by Asset",
                        template="plotly_dark",
                        height=400,
                        yaxis_title="Sentiment Score (-1 to 1)",
                        xaxis_title="Asset",
                        margin=dict(l=40, r=40, t=60, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.warning("No data available")
            
            except Exception as e:
                show_error_message(f"Error: {e}")

# Tab 3: Fed Watch
with tab3:
    st.markdown("### Fed Watch Tool")
    st.info("Federal Reserve rate probabilities. Dovish stance (rate cuts) is bullish for Gold/BTC.")
    
    if st.button("Fetch Fed Watch Data", use_container_width=True):
        with st.spinner("Fetching Fed probabilities..."):
            try:
                fetcher = FedWatchFetcher()
                signal = fetcher.get_fed_signal()
                
                # Display current stance
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Dovish Score", f"{signal['dovish_score']:.0f}/100")
                
                with col2:
                    st.metric("Stance", signal['stance'].title())
                
                with col3:
                    st.metric("Next Meeting", signal['probabilities']['next_meeting_date'])
                
                # Probabilities
                st.markdown("#### Rate Probabilities")
                
                probs = signal['probabilities']
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Rate Cut", f"{probs['prob_cut']:.1%}", 
                             delta="Bullish for Gold/BTC" if probs['prob_cut'] > 0.3 else None)
                
                with col_b:
                    st.metric("Hold", f"{probs['prob_hold']:.1%}")
                
                with col_c:
                    st.metric("Rate Hike", f"{probs['prob_hike']:.1%}",
                             delta="Bearish for Gold/BTC" if probs['prob_hike'] > 0.3 else None)
                
                # Impact on assets
                st.markdown("#### Impact on Assets")
                
                impact_data = []
                for asset in selected_assets:
                    if asset in ['gold', 'btc']:
                        impact = signal['signal_for_gold']
                    else:
                        impact = signal['signal_for_stocks']
                    
                    impact_data.append({
                        'Asset': asset.upper(),
                        'Fed Signal': impact.title(),
                        'Confidence': f"{signal['confidence']:.2f}"
                    })
                
                st.dataframe(pd.DataFrame(impact_data), use_container_width=True, hide_index=True)
                
                # Save data
                fetcher.save_fed_data()
                
                # Show historical data if available
                historical = fetcher.get_historical_data(days=30)
                
                if not historical.empty:
                    st.markdown("#### Historical Dovish Score (30 Days)")
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=historical['timestamp'],
                        y=historical['dovish_score'],
                        mode='lines+markers',
                        name='Dovish Score',
                        line=dict(color='#00A8E8', width=2)
                    ))
                    
                    # Add threshold lines
                    fig.add_hline(y=65, line_dash="dash", line_color="green", 
                                 annotation_text="Dovish Threshold")
                    fig.add_hline(y=35, line_dash="dash", line_color="red", 
                                 annotation_text="Hawkish Threshold")
                    
                    fig.update_layout(
                        template="plotly_dark",
                        height=400,
                        yaxis_title="Dovish Score (0-100)",
                        xaxis_title="Date",
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                show_error_message(f"Error: {e}")

# Disclaimer
st.markdown("---")
st.warning("""
**Data Disclaimer**: Alternative data sources provide supplementary insights but should not be the sole basis for trading decisions. 
Google Trends and Reddit sentiment reflect retail interest, which may not align with institutional flows. 
Fed Watch probabilities are market-implied and subject to change.
""")
