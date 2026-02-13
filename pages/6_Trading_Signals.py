"""
Trading Signals Page
Display entry/exit signals based on multi-factor analysis
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from utils.config import ASSETS, get_all_stock_tickers
from utils.ui_components import inject_custom_css, render_page_header, show_error_message
from utils.signal_generator import SignalGenerator, batch_generate_signals

# Page config
st.set_page_config(
    page_title="Trading Signals | Market Intelligence",
    page_icon="",
    layout="wide"
)

inject_custom_css()

# Header
render_page_header(
    icon="",
    title="Trading Signals",
    subtitle="Multi-factor analysis for entry/exit recommendations"
)

# Asset selector
all_assets = ['gold', 'btc'] + [t.lower() for t in get_all_stock_tickers()]
selected_assets = st.multiselect(
    "Select assets for signal analysis",
    all_assets,
    default=['gold', 'btc', 'msft']
)

if not selected_assets:
    st.warning("Please select at least one asset")
    st.stop()

st.markdown("---")

# Generate signals button
if st.button("Generate Trading Signals", use_container_width=True, type="primary"):
    with st.spinner("Analyzing market conditions and generating signals..."):
        try:
            signals = batch_generate_signals(selected_assets)
            
            # Display summary table
            st.markdown("### Signal Summary")
            
            summary_data = []
            for asset, signal in signals.items():
                if 'error' not in signal:
                    summary_data.append({
                        'Asset': asset.upper(),
                        'Signal': signal['signal'],
                        'Confidence': f"{signal['confidence']:.0%}",
                        'Entry': f"${signal['entry_price']:,.2f}",
                        'Target': f"${signal['target_price']:,.2f}",
                        'Stop Loss': f"${signal['stop_loss']:,.2f}",
                        'R/R': f"{signal['risk_reward']:.2f}"
                    })
            
            if summary_data:
                df = pd.DataFrame(summary_data)
                
                # Color code signals
                def highlight_signal(row):
                    if row['Signal'] == 'BUY':
                        return ['background-color: rgba(0, 255, 0, 0.1)'] * len(row)
                    elif row['Signal'] == 'SELL':
                        return ['background-color: rgba(255, 0, 0, 0.1)'] * len(row)
                    else:
                        return ['background-color: rgba(255, 255, 0, 0.05)'] * len(row)
                
                st.dataframe(
                    df.style.apply(highlight_signal, axis=1),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Detailed analysis for each asset
                st.markdown("---")
                st.markdown("### Detailed Signal Analysis")
                
                for asset, signal in signals.items():
                    if 'error' not in signal:
                        with st.expander(f"{asset.upper()} - {signal['signal']} Signal"):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.markdown("#### Signal Details")
                                
                                # Signal metrics
                                metric_cols = st.columns(4)
                                
                                with metric_cols[0]:
                                    signal_color = "🟢" if signal['signal'] == 'BUY' else "🔴" if signal['signal'] == 'SELL' else "🟡"
                                    st.metric("Signal", f"{signal_color} {signal['signal']}")
                                
                                with metric_cols[1]:
                                    st.metric("Confidence", f"{signal['confidence']:.0%}")
                                
                                with metric_cols[2]:
                                    st.metric("Bullish Score", f"{signal['bullish_score']:.2f}")
                                
                                with metric_cols[3]:
                                    st.metric("Bearish Score", f"{signal['bearish_score']:.2f}")
                                
                                # Price targets
                                st.markdown("#### Price Targets")
                                
                                price_cols = st.columns(4)
                                
                                with price_cols[0]:
                                    st.metric("Entry Price", f"${signal['entry_price']:,.2f}")
                                
                                with price_cols[1]:
                                    change = signal['target_price'] - signal['entry_price']
                                    pct = (change / signal['entry_price']) * 100
                                    st.metric("Target Price", f"${signal['target_price']:,.2f}", 
                                             delta=f"{pct:+.1f}%")
                                
                                with price_cols[2]:
                                    st.metric("Stop Loss", f"${signal['stop_loss']:,.2f}")
                                
                                with price_cols[3]:
                                    st.metric("Risk/Reward", f"{signal['risk_reward']:.2f}x")
                                
                                # Reasons
                                st.markdown("#### Analysis Factors")
                                
                                for reason in signal['reasons']:
                                    st.markdown(f"- {reason}")
                            
                            with col2:
                                st.markdown("#### Factor Breakdown")
                                
                                # Factor scores chart
                                factors = signal['factors']
                                factor_names = list(factors.keys())
                                factor_scores = [factors[f]['score'] for f in factor_names]
                                factor_weights = [factors[f]['weight'] for f in factor_names]
                                
                                fig = go.Figure()
                                
                                fig.add_trace(go.Bar(
                                    y=factor_names,
                                    x=factor_scores,
                                    orientation='h',
                                    marker_color='#00A8E8',
                                    text=[f"{s:.2f}" for s in factor_scores],
                                    textposition='outside'
                                ))
                                
                                fig.update_layout(
                                    title="Factor Scores",
                                    template="plotly_dark",
                                    height=300,
                                    xaxis_title="Score",
                                    yaxis_title="Factor",
                                    margin=dict(l=40, r=80, t=60, b=40),
                                    showlegend=False
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Factor weights
                                st.markdown("**Factor Weights:**")
                                for name, data in factors.items():
                                    st.markdown(f"- {name.title()}: {data['weight']:.0%}")
                
                # Signal distribution chart
                st.markdown("---")
                st.markdown("### Signal Distribution")
                
                buy_count = sum(1 for s in signals.values() if s.get('signal') == 'BUY')
                sell_count = sum(1 for s in signals.values() if s.get('signal') == 'SELL')
                hold_count = sum(1 for s in signals.values() if s.get('signal') == 'HOLD')
                
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    # Pie chart
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=['BUY', 'SELL', 'HOLD'],
                        values=[buy_count, sell_count, hold_count],
                        marker_colors=['#00FF00', '#FF0000', '#FFFF00']
                    )])
                    
                    fig_pie.update_layout(
                        title="Signal Distribution",
                        template="plotly_dark",
                        height=400
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col_chart2:
                    # Confidence distribution
                    assets_list = [s['Asset'] for s in summary_data]
                    confidences = [float(s['Confidence'].strip('%'))/100 for s in summary_data]
                    
                    fig_conf = go.Figure(data=[go.Bar(
                        x=assets_list,
                        y=confidences,
                        marker_color='#00A8E8',
                        text=[f"{c:.0%}" for c in confidences],
                        textposition='outside'
                    )])
                    
                    fig_conf.update_layout(
                        title="Signal Confidence by Asset",
                        template="plotly_dark",
                        height=400,
                        yaxis_title="Confidence",
                        xaxis_title="Asset",
                        margin=dict(l=40, r=40, t=80, b=40)
                    )
                    
                    st.plotly_chart(fig_conf, use_container_width=True)
            
            else:
                st.warning("No signals generated")
        
        except Exception as e:
            show_error_message(f"Error generating signals: {e}")
            import traceback
            st.code(traceback.format_exc())

# How it works
with st.expander("How Signal Generation Works"):
    st.markdown("""
    ### Multi-Factor Analysis
    
    Signals are generated by analyzing multiple factors with weighted scoring:
    
    **1. AI Forecast (40% weight)**
    - 1-week price prediction
    - Confidence based on asset type
    - Bullish if forecast > +2%, Bearish if < -2%
    
    **2. Macro Conditions (25-30% weight)**
    - Gold: DXY, VIX, Fed dovish score
    - Bitcoin: DXY, Fed dovish score
    - Stocks: VIX, Yield, Fed stance
    
    **3. Sentiment (20% weight)**
    - Google Trends search volume
    - Reddit sentiment analysis
    - Bullish if rising interest + positive sentiment
    
    **4. Technical (15% weight)**
    - Price vs EMA90
    - Bullish if price > EMA, Bearish if price < EMA
    
    ### Signal Thresholds
    
    - **BUY**: Bullish score > 0.55
    - **SELL**: Bearish score > 0.55
    - **HOLD**: Neither threshold met
    
    ### Risk Management
    
    - Stop loss: 5% from entry
    - Target: Based on 1-week forecast
    - Risk/Reward ratio calculated automatically
    """)

# Disclaimer
st.markdown("---")
st.warning("""
**Trading Disclaimer**: These signals are for informational purposes only and do not constitute financial advice. 
All trading involves risk. Past performance does not guarantee future results. 
Always conduct your own research and consider your risk tolerance before making investment decisions.
""")
