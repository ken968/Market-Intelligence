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


def create_forecast_chart(historical_df, forecast_values, price_col, forecast_days, fan_p10=None, fan_p90=None):
    """
    Create chart showing historical data + forecast with optional Probability Cloud (Fan Chart)
    
    Args:
        historical_df (pd.DataFrame): Historical data
        forecast_values (list): Predicted median values
        price_col (str): Price column name
        forecast_days (int): Number of forecast days
        fan_p10 (list): 10th percentile bound
        fan_p90 (list): 90th percentile bound
    
    Returns:
        go.Figure: Plotly figure
    """
    fig = go.Figure()
    
    # Historical data
    x_hist = historical_df['Date'] if 'Date' in historical_df.columns else historical_df.index
    fig.add_trace(go.Scatter(
        x=x_hist,
        y=historical_df[price_col],
        name='Historical',
        line=dict(color='#FFD700', width=2)
    ))
    
    # Forecast data
    last_date = pd.to_datetime(x_hist.iloc[-1])
    forecast_dates = pd.date_range(start=last_date, periods=forecast_days+1, freq='D')[1:]
    
    # Probability Cloud (Fan Chart)
    if fan_p10 and fan_p90:
        # We append to the last historical point to make continuous
        last_val = historical_df[price_col].iloc[-1]
        x_fan = [last_date] + list(forecast_dates)
        y_p10 = [last_val] + list(fan_p10)
        y_p90 = [last_val] + list(fan_p90)
        
        # Add the 90th percentile line first
        fig.add_trace(go.Scatter(
            x=x_fan,
            y=y_p90,
            name='90% Probable High',
            line=dict(width=0),
            showlegend=False
        ))
        
        # Add the 10th percentile and fill up to the 90th percentile
        fig.add_trace(go.Scatter(
            x=x_fan,
            y=y_p10,
            name='Probability Cloud (10%-90%)',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0, 192, 118, 0.2)', # Transparent green
            showlegend=True
        ))
    
    # Median Forecast line
    y_median = [last_val] + list(forecast_values) if fan_p10 else forecast_values
    x_median = [last_date] + list(forecast_dates) if fan_p10 else forecast_dates
    
    fig.add_trace(go.Scatter(
        x=x_median,
        y=y_median,
        name='Forecast Baseline',
        line=dict(color='#00C076', width=2.5, dash='solid')
    ))
    
    fig.update_layout(
        template="plotly_dark",
        title="Price Forecast (with Probability Cloud)",
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
    
    # Final safety guard: If predictions_dict is not a dict (it's a float), fail gracefully
    if not isinstance(predictions_dict, dict):
        st.error("AI Model returned incomplete data. Please train the model and sync macro data.")
        return

    # Extract current price
    current_price = predictions_dict.get('Current', 0)
    
    # Build table data
    timeframes = []
    prices = []
    changes = []
    change_pcts = []
    confidences = []
    
    # Only process known timeframe keys - skip metadata like 'ceo_context', 'error', etc.
    VALID_TIMEFRAMES = ['1 Day', '1 Week', '2 Weeks', '1 Month', '3 Months']
    
    for key in VALID_TIMEFRAMES:
        if key not in predictions_dict:
            continue
        
        value = predictions_dict[key]
        
        # Handle both old format (float) and new format (dict with price/confidence)
        if isinstance(value, dict):
            price = value.get('price', 0)
            confidence = value.get('confidence', {})
        elif isinstance(value, (int, float)):
            price = value
            confidence = {'label': 'N/A', 'color': 'info'}
        else:
            # Skip non-numeric, non-dict values entirely
            continue
        
        # Safety check for NaN
        if pd.isna(price) or (isinstance(price, float) and np.isnan(price)):
            st.error("**Prediction Error**: Invalid data detected (NaN values). Try syncing data from the Settings page.")
            return
        
        change = price - current_price
        change_pct = (change / current_price) * 100 if current_price != 0 else 0
        
        timeframes.append(key)
        prices.append(f"${price:,.2f}")
        changes.append(f"${change:+,.2f}")
        change_pcts.append(f"{change_pct:+.2f}%")
        
        if isinstance(confidence, dict):
            conf_label = confidence.get('label', 'N/A')
            conf_color = confidence.get('color', 'info')
        else:
            conf_label = 'N/A'
            conf_color = 'info'
        # No more emojis, keep it professional
        confidences.append(f"{conf_label}")

    # Create Display DataFrame
    display_df = pd.DataFrame({
        "Timeframe": timeframes,
        f"Target ({asset_name})": prices,
        "Change ($)": changes,
        "Change (%)": change_pcts,
        "System Confidence": confidences
    })
    
    # Static CSS-styled table approach using markdown for better control
    st.dataframe(display_df, use_container_width=True, hide_index=True)


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


def render_quorum_inference_panel(forecasts: dict, asset_name: str = "Asset") -> None:
    """
    Render the Quorum Inference panel using Phase 7 Risk Layer metrics.
    Displays Kelly Fraction, Epistemic Risk, and Aleatoric Risk.
    """
    week_data = forecasts.get('1 Week', {})
    if not isinstance(week_data, dict):
        return

    p7 = week_data.get('phase7_uncertainty', {})
    # Since Phase 7 uncertainty is nested, get the 7-day metrics
    # In some JSON formats keys are strings, in others integers
    metrics_7d = p7.get(7, p7.get('7', {}))
    
    if not metrics_7d:
        # Fallback if the data isn't there (e.g. old JSON cache)
        return

    kelly = metrics_7d.get('kelly_fraction', 0.0) * 100
    epistemic = metrics_7d.get('epistemic_std', 0.0) * 100
    aleatoric = metrics_7d.get('aleatoric_std', 0.0) * 100
    cross_window = metrics_7d.get('cross_window_std', 0.0) * 100
    
    # Try to grab the 7-day predicted direction and confidence
    conf = week_data.get('confidence', {})
    sys_conf = conf.get('label', 'N/A')
    sys_score = conf.get('score', 0.5) * 100
    
    # Colors based on Kelly
    if kelly >= 20:
        kelly_color = THEME.get('success', '#00FF88')
    elif kelly >= 10:
        kelly_color = THEME.get('info', '#00B8FF')
    else:
        kelly_color = THEME.get('warning', '#FFD700')

    st.markdown("---")
    st.markdown(
        f"<h4 style='margin-bottom:0.3rem;'>Quorum Inference Engine &nbsp;"
        f"<span style='font-size:0.75rem;font-weight:400;color:{THEME.get('text_secondary','#888')};'>"
        f"Phase 7 Risk Architecture</span></h4>",
        unsafe_allow_html=True,
    )

    col_kelly, col_epi, col_alea = st.columns(3)

    with col_kelly:
        st.markdown(
            f"""<div style='background:{THEME.get("bg_surface","#1a1a2e")};border:2px solid {kelly_color};
                border-radius:8px;padding:14px;text-align:center;'>
                <div style='font-size:0.75rem;color:{THEME.get("text_secondary","#888")};margin-bottom:4px;'>
                Kelly Sizing Fraction</div>
                <div style='font-size:0.8rem;color:{THEME.get("text_secondary","#888")};margin-bottom:6px;'>
                Recommended Allocation</div>
                <div style='font-size:1.6rem;font-weight:700;color:{kelly_color};'>
                {kelly:.1f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col_epi:
        st.markdown(
            f"""<div style='background:{THEME.get("bg_surface","#1a1a2e")};border:1px solid {THEME.get("border","#333")};
                border-radius:8px;padding:14px;text-align:center;'>
                <div style='font-size:0.75rem;color:{THEME.get("text_secondary","#888")};margin-bottom:4px;'>
                Epistemic Risk</div>
                <div style='font-size:0.8rem;color:{THEME.get("text_secondary","#888")};margin-bottom:6px;'>
                Model Disagreement (MC)</div>
                <div style='font-size:1.5rem;font-weight:700;color:#FFFFFF;'>
                {epistemic:.2f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col_alea:
        st.markdown(
            f"""<div style='background:{THEME.get("bg_surface","#1a1a2e")};border:1px solid {THEME.get("border","#333")};
                border-radius:8px;padding:14px;text-align:center;'>
                <div style='font-size:0.75rem;color:{THEME.get("text_secondary","#888")};margin-bottom:4px;'>
                Aleatoric Risk</div>
                <div style='font-size:0.8rem;color:{THEME.get("text_secondary","#888")};margin-bottom:6px;'>
                Market Volatility (VIX)</div>
                <div style='font-size:1.5rem;font-weight:700;color:#FFFFFF;'>
                {aleatoric:.2f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.caption(
        f"**Quorum Spread:** {cross_window:.2f}% &nbsp;|&nbsp; "
        f"**System Confidence:** {sys_conf} ({sys_score:.1f}%)"
    )
