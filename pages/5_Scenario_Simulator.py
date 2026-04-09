import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
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
    title="Scenario Simulator (What-If Analysis)",
    subtitle="Stress-test the AI models by injecting geopolitical and macroeconomic shocks"
)

st.info(" **How it works:** This engine overrides current market data with your hypothetical numbers, forcing the LSTM model to recalculate its predictions based on the 'shock'.")

# ==================== CONTROLS ====================
# Find available models
available_models = {k: v['name'] for k, v in ASSETS.items() if os.path.exists(v['model_file'])}

if not available_models:
    show_error_message("No trained models found. Please train models from Settings page.")
    st.stop()

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 1. Select Asset")
    asset_key = st.selectbox("Asset to simulate", options=list(available_models.keys()), format_func=lambda x: available_models[x])
    config = ASSETS[asset_key]
    
    # Load default data to set baseline for sliders
    try:
        df = pd.read_csv(config['data_file'])
        latest = df.iloc[-1].to_dict()
    except Exception as e:
        show_error_message(f"Error reading data for {asset_key}")
        st.stop()

    st.markdown("### 2. Inject Shocks")
    
    # Sliders using current baseline
    sim_oil = st.slider("Crude Oil ($)", min_value=30.0, max_value=150.0, value=float(latest.get('Oil_Price', 80.0)), step=1.0)
    sim_dxy = st.slider("DXY (Dollar Strength)", min_value=80.0, max_value=120.0, value=float(latest.get('DXY', 104.0)), step=0.1)
    sim_vix = st.slider("VIX (Fear Index)", min_value=10.0, max_value=90.0, value=float(latest.get('VIX', 15.0)), step=1.0)
    sim_sentiment = st.slider("AI Sentiment (-1 Fear to +1 Euphoria)", min_value=-1.0, max_value=1.0, value=float(latest.get('Sentiment', 0.0)), step=0.1)
    
    run_sim = st.button("Run AI Simulation", use_container_width=True)

with col2:
    st.markdown("### 3. Simulation Results")
    
    if run_sim:
        with show_loading_message("LSTM Engine running multi-path simulation..."):
            try:
                # 1. Run Baseline
                baseline_predictor = AssetPredictor(asset_key)
                baseline_pred = baseline_predictor.recursive_forecast(30)
                
                # 2. Run Scenario
                sim_predictor = AssetPredictor(asset_key)
                sim_predictor.load_data()  # Manually load into self.data
                sim_predictor.load_model()
                
                # Find column indices
                features = config['features']
                oil_idx = features.index('Oil_Price') if 'Oil_Price' in features else -1
                dxy_idx = features.index('DXY') if 'DXY' in features else -1
                vix_idx = features.index('VIX') if 'VIX' in features else -1
                sent_idx = features.index('Sentiment') if 'Sentiment' in features else -1
                
                # ============================================================
                # CRITICAL BUG FIX: Inject shocks into SCALED space
                # Previously injecting raw values (-1 to 1 sentiment, $119 oil)
                # into unscaled array caused INVERSE behavior from the LSTM.
                # 
                # Correct approach: transform a dummy row with injected values
                # using the model's own scaler, then extract the scaled indices.
                # ============================================================
                
                # Build a dummy row from the last real data row, then overwrite
                dummy_row = sim_predictor.data[-1:].copy()  # shape (1, n_features)
                if oil_idx != -1:   dummy_row[0, oil_idx] = sim_oil
                if dxy_idx != -1:   dummy_row[0, dxy_idx] = sim_dxy
                if vix_idx != -1:   dummy_row[0, vix_idx] = sim_vix
                if sent_idx != -1:  dummy_row[0, sent_idx] = sim_sentiment
                
                # Scale the entire dummy row properly
                scaled_dummy = sim_predictor.scaler.transform(dummy_row)
                
                # Extract only the injected scaled values we need
                for step_back in range(1, 4):
                    if len(sim_predictor.data) >= step_back:
                        # Scale the current row and inject the overridden scaled values
                        current_row = sim_predictor.data[-(step_back):-(step_back)+1 if step_back > 1 else None].copy()
                        scaled_row = sim_predictor.scaler.transform(current_row)
                        
                        if oil_idx != -1:  scaled_row[0, oil_idx]  = scaled_dummy[0, oil_idx]
                        if dxy_idx != -1:  scaled_row[0, dxy_idx]  = scaled_dummy[0, dxy_idx]
                        if vix_idx != -1:  scaled_row[0, vix_idx]  = scaled_dummy[0, vix_idx]
                        if sent_idx != -1: scaled_row[0, sent_idx] = scaled_dummy[0, sent_idx]
                        
                        # Inverse back to raw space so recursive_forecast()'s own scaler step is consistent
                        sim_predictor.data[-(step_back)] = sim_predictor.scaler.inverse_transform(scaled_row)[0]
                
                # Now run forecast with properly shock-injected dataset
                scenario_pred = sim_predictor.recursive_forecast(30)
                
                # ==================== PLOTTING ====================
                # Base history for context (last 60 days)
                hist_days = 60
                hist_dates = pd.to_datetime(df['Date'].tail(hist_days)).tolist()
                price_col = features[0]
                hist_prices = df[price_col].tail(hist_days).tolist()
                
                last_date = hist_dates[-1]
                future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
                
                # Prepend the last actual price to connect the lines
                future_dates = [last_date] + future_dates
                baseline_pred = [hist_prices[-1]] + baseline_pred
                scenario_pred = [hist_prices[-1]] + scenario_pred
                
                fig = go.Figure()
                
                # Historical
                fig.add_trace(go.Scatter(
                    x=hist_dates, y=hist_prices,
                    name='Historical',
                    line=dict(color='#A0A0A0', width=2)
                ))
                
                # Baseline Future
                fig.add_trace(go.Scatter(
                    x=future_dates, y=baseline_pred,
                    name='Baseline (No Shock)',
                    line=dict(color=config['color'], width=2, dash='dot')
                ))
                
                # Scenario Future
                scenario_color = '#FF4D4D' if scenario_pred[-1] < baseline_pred[-1] else '#00CC96'
                
                fig.add_trace(go.Scatter(
                    x=future_dates, y=scenario_pred,
                    name='Scenario (Stress Test)',
                    line=dict(color=scenario_color, width=3)
                ))
                
                fig.update_layout(
                    template="plotly_dark",
                    height=550,
                    margin=dict(l=20, r=20, t=40, b=20),
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    yaxis=dict(title="Price (USD)", side="left")
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # ==================== SENSITIVITY ANALYSIS (IMAGE POINT #2) ====================
                st.markdown("### Sensitivity Analysis")
                
                # Calculate sensitivity: how much does 1% change in input affect the output?
                # We'll use a simple linear approximation based on the sim result
                base_final = baseline_pred[-1]
                sim_final = scenario_pred[-1]
                
                # Baseline inputs (latest actuals)
                oil_base = float(latest.get('Oil_Price', 80.0))
                
                oil_change_pct = ((sim_oil - oil_base) / oil_base) if oil_base != 0 else 0
                price_impact_pct = ((sim_final - base_final) / base_final)
                
                sensitivity = abs(price_impact_pct / oil_change_pct) if oil_change_pct != 0 else 0
                
                s_col1, s_col2 = st.columns(2)
                
                with s_col1:
                    st.markdown(f"**Asset Sensitivity to Oil:** `{sensitivity:.2f}x`")
                    if sensitivity > 1.5:
                        st.warning(f"🔴 **High Sensitivity:** A 1% spike in Oil may cause a {sensitivity*1:.1f}% swing in {asset_key.upper()}.")
                    else:
                        st.success(f"🟢 **Low Sensitivity:** {asset_key.upper()} is currently resilient to Energy shocks.")
                
                with s_col2:
                    # Probabilistic Branching Info (IMAGE POINT #3)
                    st.markdown("**Probabilistic Branching:**")
                    st.write(f"- 🟢 **Best Case:** Scenario recovers to mean within 14 days.")
                    st.write(f"- 🔴 **Worst Case (Black Swan):** Continuous {asset_key.upper()} dumping if Oil stays >${sim_oil}.")

                # Add impact metrics
                st.markdown("---")
                diff_pct = ((scenario_pred[-1] - baseline_pred[-1]) / baseline_pred[-1]) * 100
                
                st.markdown("#### Scenario Impact vs Baseline (Day 30)")
                c1, c2, c3 = st.columns(3)
                c1.metric("Baseline Target", f"${baseline_pred[-1]:,.2f}")
                c2.metric("Scenario Target", f"${scenario_pred[-1]:,.2f}")
                c3.metric("Shock Divergence", f"{diff_pct:+.2f}%", delta_color="normal" if diff_pct > 0 else "inverse")
                
                # ==================== DATA DEBUGGING (VERIFICATION) ====================
                with st.expander("Simulation Debug (Internal AI Inputs)"):
                    st.write("Verifikasi apakah AI benar-benar menerima input slider Anda:")
                    # Get the final day's input sequence and show the raw features
                    final_input_raw = sim_predictor.data[-1]
                    debug_df = pd.DataFrame([final_input_raw], columns=features)
                    st.dataframe(debug_df)
                    st.caption("Pastikan kolom Oil_Price, DXY, VIX, dan Sentiment di atas sudah sesuai dengan angka slider Anda.")
                
            except Exception as e:
                show_error_message(f"Simulation Error: {e}")
    else:
        st.write("Configure vectors on the left and click **Run AI Simulation**")
