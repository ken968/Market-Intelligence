"""
Scenario Simulator — Tahap 6 Upgrade
Multi-asset stress testing with VIX Regime injection,
Z-Score Anomaly simulation, and Dynamic Correlation override.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from utils.config import ASSETS
from utils.ui_components import (
    inject_custom_css, render_page_header, show_error_message
)
from utils.predictor import AssetPredictor
from utils.anomaly_detector import detect_asset_anomaly

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Scenario Simulator | Market Intelligence",
    layout="wide"
)

inject_custom_css()

render_page_header(
    icon="",
    title="Scenario Simulator & Stress Tester",
    subtitle="Tahap 6: Inject macro shocks and validate Circuit Breakers across Gold, BTC, and SPY"
)

st.info(
    "**AI Safety Net Validation** — Adjust macro shock vectors below and observe "
    "how the VIX Regime, Anomaly Detector, and Dynamic Correlation Enforcer respond. "
    "This is your primary tool for validating that the system is defensively calibrated."
)

# ==================== PRESETS ====================
PRESETS = {
    "None (Custom)": None,
    "1970s Stagflation": {
        "oil": 140.0, "vix": 25.0, "vix_pct": 0.75, "sentiment": -0.4,
        "cpi": 1.2, "nfp": -50.0, "yc": -0.8, "zscore": -1.5, "corr": 0.4
    },
    "2008 Housing Crisis": {
        "oil": 45.0, "vix": 45.0, "vix_pct": 0.92, "sentiment": -0.9,
        "cpi": -0.2, "nfp": -400.0, "yc": 0.2, "zscore": -3.5, "corr": 0.2
    },
    "2020 Covid Black Swan": {
        "oil": 20.0, "vix": 80.0, "vix_pct": 0.999, "sentiment": -1.0,
        "cpi": 0.1, "nfp": -500.0, "yc": 1.5, "zscore": -5.0, "corr": -0.5
    },
    "Dotcom Bubble Burst": {
        "oil": 30.0, "vix": 40.0, "vix_pct": 0.88, "sentiment": -0.8,
        "cpi": 0.3, "nfp": -150.0, "yc": 0.5, "zscore": -2.8, "corr": 0.35
    },
    "Geopolitical War (Oil Shock)": {
        "oil": 130.0, "vix": 35.0, "vix_pct": 0.82, "sentiment": -0.7,
        "cpi": 0.8, "nfp": 100.0, "yc": 0.1, "zscore": -2.0, "corr": 0.3
    },
    "High Inflation Shock": {
        "oil": 110.0, "vix": 25.0, "vix_pct": 0.72, "sentiment": -0.4,
        "cpi": 1.5, "nfp": 200.0, "yc": -0.3, "zscore": -1.2, "corr": 0.5
    },
    "Euphoric Bull Run": {
        "oil": 75.0, "vix": 12.0, "vix_pct": 0.05, "sentiment": 0.9,
        "cpi": 0.2, "nfp": 350.0, "yc": 1.2, "zscore": 0.3, "corr": 0.92
    },
}

# ==================== CONTROLS ====================
pillars = ['spy', 'gold', 'btc']

col_ctrl, col_viz = st.columns([1, 2.5])

with col_ctrl:
    st.markdown("### 1. Historical Stress Preset")
    selected_preset = st.selectbox("Apply Preset Scenario", list(PRESETS.keys()))
    p = PRESETS[selected_preset]

    def _gv(key, default_val):
        """Get preset value or fall back to default."""
        return p[key] if p and key in p else default_val

    st.markdown("### 2. Macro Shock Vectors")

    # Load live baselines
    try:
        df_base = pd.read_csv(ASSETS['gold']['data_file'])
        latest_gold = df_base.iloc[-1].to_dict()
    except Exception:
        latest_gold = {}

    sim_oil       = st.slider("Crude Oil ($)",       30.0, 150.0, _gv("oil",       float(latest_gold.get("Oil_Price", 75.0))), 1.0)
    sim_vix       = st.slider("VIX (Fear Index)",     10.0, 90.0,  _gv("vix",       float(latest_gold.get("VIX", 18.0))),       0.5)
    sim_sentiment = st.slider("AI Sentiment",         -1.0, 1.0,   _gv("sentiment", 0.0), 0.05)

    st.markdown("### 3. Phase 3 Regime Injectors")
    sim_vix_pct = st.slider(
        "VIX Percentile (0=Calm, 1=Extreme)",
        0.0, 1.0, _gv("vix_pct", 0.5), 0.01,
        help="Forces vix_percentile_252d to this value, overriding historical calculation."
    )
    sim_zscore = st.slider(
        "Return Z-Score (-5 to +5)",
        -5.0, 5.0, _gv("zscore", 0.0), 0.1,
        help="Simulates a multi-sigma price move anomaly (positive=spike, negative=crash)."
    )
    sim_corr = st.slider(
        "Rolling 90d Correlation vs SPY",
        -1.0, 1.0, _gv("corr", 0.6), 0.05,
        help="Simulates coupling/decoupling regime with SPY."
    )

    with st.expander("Advanced FRED Shocks"):
        sim_cpi = st.slider("CPI MoM (%)", -1.0, 2.0, _gv("cpi", 0.2), 0.05)
        sim_nfp = st.slider("NFP Change (K)", -500, 500, int(_gv("nfp", 200)), 10)
        sim_yc  = st.slider("Yield Curve (10Y-2Y)", -2.0, 3.0, _gv("yc", 0.3), 0.05)

    run_sim = st.button("Run Stress Simulation", use_container_width=True, type="primary")

# ==================== CIRCUIT BREAKER STATUS ====================
with col_viz:
    st.markdown("### 4. Live Circuit Breaker Status (Pre-Simulation)")

    # Real-time status from actual data
    cb_cols = st.columns(3)
    for idx, asset_key in enumerate(pillars):
        with cb_cols[idx]:
            try:
                config = ASSETS[asset_key]
                df_asset = pd.read_csv(config['data_file'])
                cb_result = detect_asset_anomaly(asset_key, df_asset)

                vix_pct_live = float(df_asset.iloc[-1].get('vix_percentile_252d', 0.5))
                z_live = float(df_asset.iloc[-1].get('return_zscore_90d', 0.0))
                corr_live_key = [c for c in df_asset.columns if c.startswith('roll_corr_')]
                corr_live = float(df_asset.iloc[-1][corr_live_key[0]]) if corr_live_key else 0.5

                if vix_pct_live >= 0.90:
                    vr, vc = "EXTREME", "error"
                elif vix_pct_live >= 0.70:
                    vr, vc = "ELEVATED", "warning"
                else:
                    vr, vc = "CALM", "success"

                with st.container(border=True):
                    st.markdown(f"**{config['name']}**")
                    getattr(st, vc)(f"VIX Regime: **{vr}** ({vix_pct_live*100:.1f}th pct)")
                    if cb_result['is_anomaly']:
                        st.error(f"ANOMALY: Z={z_live:+.2f} (CB Active)")
                    elif abs(z_live) > 2.0:
                        st.warning(f"CAUTION: Z={z_live:+.2f}")
                    else:
                        st.success(f"Normal: Z={z_live:+.2f}")
                    st.caption(f"Corr vs ref: {corr_live:+.2f}")
            except Exception as e:
                st.warning(f"{asset_key}: {e}")

    st.markdown("---")
    st.markdown("### 5. Simulated Circuit Breaker Response")

    # Simulated CB response from sliders
    sim_cb_cols = st.columns(3)
    for idx, asset_key in enumerate(pillars):
        with sim_cb_cols[idx]:
            config = ASSETS[asset_key]
            with st.container(border=True):
                st.markdown(f"**{config['name']} [SIMULATED]**")

                # VIX Regime from slider
                if sim_vix_pct >= 0.90:
                    sim_vr, sim_vc = "EXTREME", "error"
                elif sim_vix_pct >= 0.70:
                    sim_vr, sim_vc = "ELEVATED", "warning"
                else:
                    sim_vr, sim_vc = "CALM", "success"
                getattr(st, sim_vc)(f"VIX Regime: **{sim_vr}** ({sim_vix_pct*100:.1f}th pct)")
                st.progress(min(max(sim_vix_pct, 0.0), 1.0))

                # Z-Score response
                if abs(sim_zscore) > 3.0:
                    direction = "CRASH" if sim_zscore < 0 else "SPIKE"
                    st.error(f"ANOMALY: Z={sim_zscore:+.2f} ({direction}) — CB Active")
                elif abs(sim_zscore) > 2.0:
                    st.warning(f"CAUTION: Z={sim_zscore:+.2f}")
                else:
                    st.success(f"Normal: Z={sim_zscore:+.2f}")

                # Dynamic Correlation Strength
                if sim_corr >= 0.80:
                    cs, cl = 0.85, "Very Strong (tight enforcement)"
                elif sim_corr >= 0.50:
                    cs, cl = 0.60, "Strong (normal enforcement)"
                elif sim_corr >= 0.30:
                    cs, cl = 0.35, "Weak (light enforcement)"
                else:
                    cs, cl = 0.20, "Decoupled (min enforcement)"

                st.caption(f"Corr={sim_corr:+.2f} | Enforcer strength={cs:.2f}")
                st.caption(f"Mode: {cl}")

    st.markdown("---")

    # ==================== FULL SIMULATION ====================
    if run_sim:
        st.markdown("### 6. Multi-Asset Forecast Under Stress")

        available_pillars = [a for a in pillars if os.path.exists(ASSETS[a]['model_file'])]
        if not available_pillars:
            st.warning("No trained models found. Train models from Settings page first.")
        else:
            with st.status("Running multi-asset stress simulation...", expanded=True) as sim_status:
                final_results = {}

                for asset in available_pillars:
                    st.write(f"Simulating {ASSETS[asset]['name']}...")
                    try:
                        predictor = AssetPredictor(asset)

                        # Inject macro shocks via override
                        result = predictor.predict_week(
                            override_features={
                                'Oil_Price': sim_oil,
                                'VIX': sim_vix,
                                'Sentiment': sim_sentiment,
                                'CPI_MoM': sim_cpi,
                                'NFP_Change': float(sim_nfp),
                                'YieldCurve_10Y2Y': sim_yc,
                                'vix_percentile_252d': sim_vix_pct,
                                'return_zscore_90d': sim_zscore,
                            }
                        )

                        if 'error' not in result:
                            final_results[asset] = result
                            direction_icon = "UP" if result.get('direction') == 'up' else "DOWN"
                            st.write(f"  {ASSETS[asset]['name']}: {result['pct_change']:+.2f}% ({direction_icon}), "
                                     f"Confidence: {result.get('direction_prob', 0.5)*100:.0f}%")
                        else:
                            st.warning(f"  {asset}: {result['error']}")
                    except Exception as e:
                        st.warning(f"  {asset}: {e}")

                sim_status.update(label="Simulation Complete", state="complete", expanded=False)

            if final_results:
                # Results table
                rows = []
                for asset_key, res in final_results.items():
                    direction = res.get('direction', 'flat')
                    rows.append({
                        'Asset':        ASSETS[asset_key]['name'],
                        'Current ($)':  f"${res['current']:,.2f}",
                        'Predicted ($)': f"${res['predicted']:,.2f}",
                        'Change':       f"{res['pct_change']:+.2f}%",
                        'Direction':    "UP" if direction == 'up' else ("DOWN" if direction == 'down' else "FLAT"),
                        'Confidence':   f"{res.get('direction_prob', 0.5)*100:.0f}%",
                        'LSTM Signal':  f"{res.get('lstm_signal', 0)*100:+.2f}%" if res.get('has_ensemble') else "--",
                        'XGB Signal':   f"{res.get('xgb_signal', 0)*100:+.2f}%"  if res.get('has_ensemble') else "--",
                    })

                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                # Monotonicity check visual
                st.markdown("#### Anti-Divergence (Monotonicity) Check")
                from utils.correlation_enforcer import CorrelationEnforcer
                enforcer_inst = CorrelationEnforcer.__new__(CorrelationEnforcer)
                enforcer_inst.reference_ticker = "SPY"
                enforcer_inst.betas = {}

                mock_preds = {}
                for ak, res in final_results.items():
                    mock_preds[ak.upper()] = [res['current'], res['predicted']]

                labels = enforcer_inst.monotonicity_check(mock_preds, roll_corr_90d=sim_corr)
                mc_cols = st.columns(len(labels))
                for idx2, (ticker, lbl) in enumerate(labels.items()):
                    with mc_cols[idx2]:
                        if lbl == "DIVERGENCE_WARNING":
                            st.error(f"{ticker}: {lbl}")
                        elif lbl == "HIGH_UNCERTAINTY":
                            st.warning(f"{ticker}: {lbl}")
                        else:
                            st.success(f"{ticker}: {lbl}")
    else:
        st.info("Configure shock vectors on the left panel and click **Run Stress Simulation**.")
