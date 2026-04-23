import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys
from datetime import timedelta
from utils.config import ASSETS, STOCK_TICKERS
from utils.ui_components import (
    inject_custom_css, render_page_header, render_metric_card,
    show_error_message, show_loading_message
)
from utils.predictor import AssetPredictor

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Scenario Simulator | Market Intelligence",
    layout="wide"
)

inject_custom_css()

# ==================== MAIN CONTENT ====================
render_page_header(
    icon="",
    title="Ultimate Scenario Simulator",
    subtitle="Simulate global macro shocks across the 3 Pillars: Stocks, Gold, and Bitcoin"
)

st.info(" **Asset Correlation Engine**: This simulator injects shocks across all three primary assets simultaneously to show how capital might rotate during crises.")

# ==================== CONTROLS ====================
# Find available models for 3 pillars
pillars = ['spy', 'gold', 'btc']
available_pillars = {k: ASSETS[k]['name'] for k in pillars if os.path.exists(ASSETS[k]['model_file'])}

if not available_pillars:
    st.warning(" One or more Pillar models (SPY, Gold, BTC) are not trained yet. Please train them in Settings.")

col_ctrl, col_viz = st.columns([1, 2.5])

with col_ctrl:
    st.markdown("### 1. Historical Stress Tests")
    
    # Preset scenarios
    presets = {
        "None (Custom)": None,
        "1970s Stagflation": {
            "oil": 140.0, "vix": 25.0, "sentiment": -0.4, "cpi": 1.2, "nfp": -50.0, "yc": -0.8
        },
        "2008 Housing Crisis": {
            "oil": 45.0, "vix": 45.0, "sentiment": -0.9, "cpi": -0.2, "nfp": -400.0, "yc": 0.2
        },
        "2020 Covid Black Swan": {
            "oil": 20.0, "vix": 65.0, "sentiment": -1.0, "cpi": 0.1, "nfp": -500.0, "yc": 1.5
        },
        "Dotcom Bubble Burst": {
            "oil": 30.0, "vix": 40.0, "sentiment": -0.8, "cpi": 0.3, "nfp": -150.0, "yc": 0.5
        },
        "Geopolitical War": {
            "oil": 130.0, "vix": 35.0, "sentiment": -0.7, "cpi": 0.8, "nfp": 100.0, "yc": 0.1
        },
        "High Inflation Shock": {
            "oil": 110.0, "vix": 25.0, "sentiment": -0.4, "cpi": 1.5, "nfp": 200.0, "yc": -0.3
        },
        "Deflationary Spiral": {
            "oil": 40.0, "vix": 20.0, "sentiment": -0.5, "cpi": -0.8, "nfp": -250.0, "yc": 0.5
        },
        "Euphoric Bull Run": {
            "oil": 75.0, "vix": 12.0, "sentiment": 0.9, "cpi": 0.2, "nfp": 350.0, "yc": 1.2
        }
    }
    
    selected_preset = st.selectbox("Apply Historical Preset", options=list(presets.keys()))
    preset_data = presets[selected_preset]

    st.markdown("### 2. Manual Shock Vectors")
    
    # Load baselines from Gold (representative for macro)
    try:
        df_base = pd.read_csv(ASSETS['gold']['data_file'])
        latest = df_base.iloc[-1].to_dict()
    except:
        latest = {}

    # Helper for preset values
    def get_val(key, default):
        if preset_data and key in preset_data: return preset_data[key]
        return float(latest.get(default, 0.0))

    sim_oil = st.slider("Crude Oil ($)", 30.0, 150.0, get_val("oil", "Oil_Price"), 1.0)
    sim_vix = st.slider("VIX (Fear Index)", 10.0, 90.0, get_val("vix", "VIX"), 1.0)
    sim_sentiment = st.slider("AI Sentiment", -1.0, 1.0, get_val("sentiment", "Sentiment"), 0.1)
    
    with st.expander("Advanced FRED Shocks"):
        sim_cpi = st.slider("CPI MoM (%)", -1.0, 2.0, get_val("cpi", "CPI_MoM"), 0.05)
        sim_nfp = st.slider("NFP Change (K)", -500, 500, int(get_val("nfp", "NFP_Change")), 10)
        sim_yc = st.slider("Yield Curve", -2.0, 3.0, get_val("yc", "YieldCurve_10Y2Y"), 0.05)

    run_sim = st.button("Run Multi-Asset Simulation", use_container_width=True, type="primary")

with col_viz:
    st.markdown("### 3. Cross-Asset Intelligence")
    
    if run_sim:
        with st.status("Solving multi-path differential patterns...", expanded=True) as status:
            final_results = {}
            # Multi-asset loop
            for asset in pillars:
                st.write(f"Calculating {ASSETS[asset]['name']} response...")
                try:
                    p = AssetPredictor(asset)
                    p.load_data()
                    p.load_model()
                    
                    # Inject Shocks
                    feat_list = ASSETS[asset]['features']
                    dummy = p.data[-1:].copy()
                    
                    # Mapping helper
                    m = {
                        'Oil_Price': sim_oil, 'VIX': sim_vix, 'Sentiment': sim_sentiment,
                        'CPI_MoM': sim_cpi, 'NFP_Change': sim_nfp, 'YieldCurve_10Y2Y': sim_yc
                    }
                    for f_name, f_val in m.items():
                        if f_name in feat_list:
                            dummy[0, feat_list.index(f_name)] = f_val
                    
                    # Scale and Apply
                    scaled_dummy = p.scaler.transform(dummy)
                    p.data[-1] = p.scaler.inverse_transform(scaled_dummy)[0]
                    
                    # Main prediction
                    forecast = p.recursive_forecast(30) # 30 days
                    
                    # Monte Carlo Simulation (Monte Carlo Cloud logic)
                    st.write(f"Running Monte Carlo Stress for {asset}...")
                    paths = []
                    for _ in range(20): # 20 paths for performance
                        noise = np.random.normal(0, 0.01, 30) # 1% daily variance
                        path = [f * (1 + n) for f, n in zip(forecast, noise)]
                        paths.append(path)
                    
                    final_results[asset] = {
                        'forecast': forecast,
                        'paths': paths,
                        'color': ASSETS[asset]['color'],
                        'base_price': p.get_latest_price()
                    }
                except Exception as e:
                    st.error(f"Error for {asset}: {e}")

            status.update(label="Simulation Complete", state="complete", expanded=False)

        # Plotting Multi-Asset Comparison
        fig = go.Figure()
        
        for asset, data in final_results.items():
            days = list(range(31))
            # Normalize to 100 base for comparison
            base = data['base_price']
            norm_forecast = [100.0] + [ (f/base)*100 for f in data['forecast'] ]
            
            # Probability Cloud (Monte Carlo)
            all_paths_norm = []
            for path in data['paths']:
                all_paths_norm.append([100.0] + [ (p_val/base)*100 for p_val in path ])
            
            p10 = np.percentile(all_paths_norm, 10, axis=0)
            p90 = np.percentile(all_paths_norm, 90, axis=0)
            
            # Add Cloud
            fig.add_trace(go.Scatter(
                x=days + days[::-1],
                y=list(p90) + list(p10)[::-1],
                fill='toself',
                fillcolor=f"rgba{tuple(list(int(data['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.1])}",
                line=dict(color='rgba(255,255,255,0)'),
                name=f"{asset.upper()} Range",
                showlegend=False
            ))
            
            # Add Mean Path
            fig.add_trace(go.Scatter(
                x=days, y=norm_forecast,
                name=ASSETS[asset]['name'],
                line=dict(color=data['color'], width=3)
            ))

        fig.update_layout(
            template="plotly_dark",
            height=600,
            title="Relative Impact Comparison (Normalized to 100)",
            yaxis=dict(title="Performance Index (Baseline = 100)"),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Comparison Metrics
        st.markdown("#### Impact Summary")
        m_cols = st.columns(3)
        for i, asset in enumerate(final_results.keys()):
            data = final_results[asset]
            perf = ((data['forecast'][-1] - data['base_price']) / data['base_price']) * 100
            m_cols[i].metric(ASSETS[asset]['name'], f"{perf:+.2f}%", delta=f"{perf:+.1f}%")
            
    else:
        st.info(" Adjust the vectors on the left and click **Run Multi-Asset Simulation** to see how the 3 Pillars react.")
        st.image("https://images.unsplash.com/photo-1611974717482-58-000000000000?auto=format&fit=crop&q=80&w=1200", caption="Intelligence Simulation Environment")
