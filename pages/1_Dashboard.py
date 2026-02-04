"""
Dashboard Overview Page
Multi-asset performance tracking and portfolio insights
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from utils.config import ASSETS, STOCK_TICKERS, get_asset_status
from utils.ui_components import (
    inject_custom_css, render_page_header, render_metric_card,
    show_error_message
)
from utils.predictor import batch_predict_tomorrow

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="Dashboard | XAUUSD Terminal",
    page_icon="",
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

st.markdown("###  Asset Selection")

col1, col2 = st.columns([3, 1])

with col1:
    # Group selection
    asset_groups = {
        'Core Assets': ['gold', 'btc', 'spy'],
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
    st.warning(" Please select at least one asset to analyze.")
    st.stop()

st.markdown("---")

# ==================== PERFORMANCE COMPARISON ====================

st.markdown("###  Performance Comparison (Normalized)")

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
st.markdown("###  Period Returns")

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

# ==================== CURRENT PRICES ====================

st.markdown("###  Current Prices & Changes")

cols = st.columns(min(len(selected_assets), 4))

for i, asset_key in enumerate(selected_assets):
    with cols[i % 4]:
        try:
            config = ASSETS[asset_key]
            df = pd.read_csv(config['data_file'])
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            price_col = config['features'][0]
            current = latest[price_col]
            change = current - prev[price_col]
            pct_change = (change / prev[price_col]) * 100
            
            render_metric_card(
                label=config['name'],
                value=current,
                delta=change
            )
            
            if pct_change > 0:
                st.success(f" +{pct_change:.2f}%")
            else:
                st.error(f" {pct_change:.2f}%")
        
        except Exception as e:
            st.warning(f"{asset_key}: Error")

st.markdown("---")

# ==================== AI PREDICTIONS ====================

st.markdown("###  Tomorrow's AI Predictions")

# Filter assets with models
assets_with_models = [a for a in selected_assets if status[a]['model']]

if not assets_with_models:
    st.warning(" No trained models for selected assets. Train models from Settings page.")
else:
    if st.button("Generate Predictions for Selected Assets", use_container_width=True):
        with st.spinner("AI analyzing patterns..."):
            try:
                predictions = batch_predict_tomorrow(assets_with_models)
                
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
                            'Direction': '' if pred['direction'] == 'up' else ''
                        })
                
                st.markdown("####  Forecast Results")
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
                
            except Exception as e:
                show_error_message(f"Prediction error: {e}")

st.markdown("---")

# ==================== CORRELATION MATRIX ====================

st.markdown("###  Asset Correlation Matrix")

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
        - **1.0**: Perfect positive correlation (move together)
        - **0.0**: No correlation (independent)
        - **-1.0**: Perfect negative correlation (move opposite)
        
         Diversification tip: Assets with low/negative correlation reduce portfolio risk.
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
