"""
Reusable UI Components for Multi-Asset Terminal
Provides consistent design elements across all pages
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
import os
from utils.config import THEME, get_asset_config

# ==================== GLOBAL CSS ====================

def inject_custom_css():
    """Inject premium terminal CSS theme"""
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@400;600;700&display=swap');

        :root {{
            --bg-deep: {THEME['bg_deep']};
            --bg-surface: {THEME['bg_surface']};
            --accent: {THEME['accent']};
            --accent-muted: {THEME['accent_muted']};
            --border: {THEME['border']};
            --text-pri: {THEME['text_primary']};
            --text-sec: {THEME['text_secondary']};
            --success: {THEME['success']};
            --danger: {THEME['danger']};
            --warning: {THEME['warning']};
        }}

        /* Global Reset */
        .block-container {{
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
            max-width: 95% !important;
        }}
        
        .stApp {{
            background-color: var(--bg-deep);
            color: var(--text-pri);
            font-family: 'Inter', sans-serif;
        }}

        h1, h2, h3, h4 {{
            font-family: 'Outfit', sans-serif !important;
            color: var(--text-pri);
            font-weight: 600 !important;
        }}

        /* Metric Cards */
        .metric-card {{
            background: var(--bg-surface);
            border: 1px solid var(--border);
            padding: 1.25rem;
            border-radius: 4px;
            position: relative;
            overflow: hidden;
        }}
        .metric-card-lbl {{
            font-size: 0.75rem;
            color: var(--text-sec);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }}
        .metric-card-val {{
            font-size: 1.75rem;
            font-weight: 700;
            font-family: 'Outfit', sans-serif;
            color: #FFFFFF;
        }}
        .metric-card-delta {{
            font-size: 0.85rem;
            margin-top: 0.25rem;
        }}
        .up {{ color: var(--success); }}
        .down {{ color: var(--danger); }}

        /* News Cards */
        .news-item {{
            background: var(--bg-surface);
            border-bottom: 1px solid var(--border);
            padding: 1rem 0;
            margin-bottom: 0.5rem;
        }}
        .news-item:last-child {{ border-bottom: none; }}
        .news-title {{
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-pri);
            text-decoration: none !important;
            display: block;
            line-height: 1.4;
        }}
        .news-title:hover {{ color: var(--accent); }}
        .news-meta {{
            font-size: 0.75rem;
            color: var(--text-sec);
            margin-top: 0.4rem;
        }}

        /* Status Badge */
        .status-badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        .status-success {{ background: var(--success); color: #000; }}
        .status-warning {{ background: var(--warning); color: #000; }}
        .status-danger {{ background: var(--danger); color: #fff; }}

        /* Buttons */
        .stButton>button {{
            background: var(--accent) !important;
            color: #000 !important;
            border-radius: 4px !important;
            font-weight: 600 !important;
            border: none !important;
            height: 2.8rem;
            transition: opacity 0.2s;
        }}
        .stButton>button:hover {{
            opacity: 0.9;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: #080C14 !important;
            border-right: 1px solid var(--border);
        }}
        
        /* Charts */
        .stPlotlyChart {{
            background: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.5rem;
        }}

        /* Tables */
        .dataframe {{
            font-family: 'Inter', sans-serif !important;
        }}
        </style>
        """, unsafe_allow_html=True)


# ==================== COMPONENT FUNCTIONS ====================

def render_metric_card(label, value, delta=None, format_str="${:,.2f}"):
    """
    Render a metric card with optional delta
    
    Args:
        label (str): Metric label
        value (float): Current value
        delta (float, optional): Change value
        format_str (str): Format string for value
    """
    formatted_value = format_str.format(value)
    
    if delta is not None:
        delta_formatted = f"{delta:+.2f}"
        delta_class = "up" if delta > 0 else "down"
        delta_symbol = "▲" if delta > 0 else "▼"
        delta_html = f'<div class="metric-card-delta {delta_class}">{delta_symbol} {delta_formatted}</div>'
    else:
        delta_html = ""
    
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-lbl">{label}</div>
            <div class="metric-card-val">{formatted_value}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)


def render_news_section(asset_key, max_items=20):
    """
    Render news section for specific asset
    
    Args:
        asset_key (str): Asset identifier
        max_items (int): Maximum news items to show
    """
    config = get_asset_config(asset_key)
    news_file = config['news_file']
    
    if not os.path.exists(news_file):
        st.info(f"No news available for {config['name']}. Run sentiment sync first.")
        return
    
    with open(news_file, 'r') as f:
        news = json.load(f)
    
    if not news:
        st.info(f"No recent news articles found for {config['name']} in the last 30 days.")
        return
    
    for art in news[:max_items]:
        score = art.get('sentiment', 0)
        s_class = "up" if score > 0.1 else ("down" if score < -0.1 else "text-sec")
        s_label = "POS" if score > 0.1 else ("NEG" if score < -0.1 else "NEU")
        
        st.markdown(f"""
        <div class="news-item">
            <a href="{art['url']}" target="_blank" class="news-title">{art['title']}</a>
            <div class="news-meta">
                {art['date']} • <span class="{s_class}">{s_label} ({score:.2f})</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_status_badge(status, label):
    """
    Render status badge
    
    Args:
        status (str): 'success', 'warning', or 'danger'
        label (str): Badge text
    """
    st.markdown(f'<span class="status-badge status-{status}">{label}</span>', unsafe_allow_html=True)


def create_price_chart(df, price_col, title="Price Chart", color="#FFD700"):
    """
    Create interactive price chart with Plotly, including EMA 90
    
    Args:
        df (pd.DataFrame): Data with Date and price columns
        price_col (str): Name of price column
        title (str): Chart title
        color (str): Line color
    
    Returns:
        go.Figure: Plotly figure
    """
    fig = go.Figure()
    
    # 1. Main Price Line
    fig.add_trace(go.Scatter(
        x=df['Date'] if 'Date' in df.columns else df.index,
        y=df[price_col],
        name=f"{price_col} Price",
        line=dict(color=color, width=2.5),
        fill='tonexty',
        fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.05)'
    ))
    
    # 2. EMA 90 Line (Indicator)
    if 'EMA_90' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'] if 'Date' in df.columns else df.index,
            y=df['EMA_90'],
            name="EMA 90 (Trend)",
            line=dict(color="#FFA500", width=1.5, dash='dash'), # Amber/Orange dash
            opacity=0.8
        ))
    
    fig.update_layout(
        template="plotly_dark",
        title=title,
        height=450, # Slightly taller for indicator
        margin=dict(l=20, r=100, t=40, b=20),
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_multi_asset_comparison(data_dict):
    """
    Create comparison chart for multiple assets
    
    Args:
        data_dict (dict): {asset_name: {'dates': [...], 'prices': [...]}}
    
    Returns:
        go.Figure: Plotly figure
    """
    fig = go.Figure()
    
    for asset_name, data in data_dict.items():
        config = get_asset_config(asset_name)
        color = config['color'] if config else '#FFFFFF'
        
        fig.add_trace(go.Scatter(
            x=data['dates'],
            y=data['prices'],
            name=asset_name.upper(),
            line=dict(color=color, width=2)
        ))
    
    fig.update_layout(
        template="plotly_dark",
        title="Multi-Asset Performance Comparison",
        height=500,
        margin=dict(l=30, r=100, t=40, b=20),
        xaxis_title="Date",
        yaxis_title="Normalized Price",
        hovermode='x unified'
    )
    
    return fig


def create_forecast_chart(historical_df, forecast_values, price_col, forecast_days):
    """
    Create chart showing historical data + forecast
    
    Args:
        historical_df (pd.DataFrame): Historical data
        forecast_values (list): Predicted values
        price_col (str): Price column name
        forecast_days (int): Number of forecast days
    
    Returns:
        go.Figure: Plotly figure
    """
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_df['Date'] if 'Date' in historical_df.columns else historical_df.index,
        y=historical_df[price_col],
        name='Historical',
        line=dict(color='#FFD700', width=2)
    ))
    
    # Forecast data
    last_date = pd.to_datetime(historical_df['Date'].iloc[-1] if 'Date' in historical_df.columns else historical_df.index[-1])
    forecast_dates = pd.date_range(start=last_date, periods=forecast_days+1, freq='D')[1:]
    
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        name='Forecast',
        line=dict(color='#00C076', width=2, dash='dash')
    ))
    
    fig.update_layout(
        template="plotly_dark",
        title="Price Forecast",
        height=400,
        margin=dict(l=20, r=100, t=40, b=20),
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode='x unified'
    )
    
    return fig


def render_prediction_table(predictions_dict, asset_name):
    """
    Render formatted prediction table with confidence scores
    
    Args:
        predictions_dict (dict): {'Current': price, timeframe: {'price': float, 'confidence': dict}}
        asset_name (str): Asset name for column header
    """
    import numpy as np
    
    # Extract current price
    current_price = predictions_dict.get('Current', 0)
    
    # Build table data
    timeframes = []
    prices = []
    changes = []
    change_pcts = []
    confidences = []
    
    for key, value in predictions_dict.items():
        if key == 'Current':
            continue
        
        # Handle both old format (float) and new format (dict with price/confidence)
        if isinstance(value, dict):
            price = value.get('price', 0)
            confidence = value.get('confidence', {})
        else:
            # Fallback for old format
            price = value
            confidence = {'label': 'N/A', 'color': 'info'}
        
        # Safety check for NaN
        if pd.isna(price) or (isinstance(price, float) and np.isnan(price)):
            st.error("""
            ⚠️ **Prediction Error**: Invalid data detected (NaN values).
            
            **Possible causes:**
            - Recent market data not yet available
            - Missing macro indicators (DXY, VIX, Yield)
            
            **Solution**: Try syncing data from the Settings page.
            """)
            return
        
        change = price - current_price
        change_pct = (change / current_price) * 100
        
        timeframes.append(key)
        prices.append(f"${price:,.2f}")
        changes.append(f"${change:+,.2f}")
        change_pcts.append(f"{change_pct:+.2f}%")
        
        # Format confidence badge
        conf_label = confidence.get('label', 'N/A')
        conf_color = confidence.get('color', 'info')
        
        # Map color to emoji
        color_map = {
            'success': '🟢',
            'info': '🔵',
            'warning': '🟡',
            'error': '🔴'
        }
        emoji = color_map.get(conf_color, '⚪')
        confidences.append(f"{emoji} {conf_label}")
    
    df = pd.DataFrame({
        'Timeframe': timeframes,
        f'{asset_name} Price': prices,
        'Change': changes,
        'Change %': change_pcts,
        'Confidence': confidences
    })
    
    st.dataframe(
        df, 
        use_container_width=True, 
        hide_index=True,
        height=(len(df) + 1) * 35 + 3
    )


def show_loading_message(message="Processing..."):
    """Show loading spinner with custom message"""
    return st.spinner(message)


def show_success_message(message):
    """Show success message with icon"""
    st.success(f" {message}")


def show_error_message(message):
    """Show error message with icon"""
    st.error(f"❌ {message}")


def show_warning_message(message):
    """Show warning message with icon"""
    st.warning(f" {message}")


def render_page_header(icon, title, subtitle):
    """
    Render consistent page header
    
    Args:
        icon (str): Emoji icon
        title (str): Page title
        subtitle (str): Page subtitle
    """
    st.markdown(f"""
        <div style="border-bottom: 1px solid {THEME['border']}; padding-bottom: 1rem; margin-bottom: 2rem;">
            <h1 style="margin:0; font-size:2rem;">
                <span style="margin-right:0.5rem;">{icon}</span>{title}
            </h1>
            <div style="font-size:0.9rem; color:{THEME['text_secondary']}; margin-top:0.5rem;">
                {subtitle}
            </div>
        </div>
        """, unsafe_allow_html=True)
