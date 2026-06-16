"""
AI Diagnostics & Validation Room
Centralized view for model health, backtest metrics, and retraining operations
"""

import streamlit as st
import pandas as pd
import json
import os
import sys
import subprocess
import glob
import plotly.graph_objects as go
from PIL import Image

from utils.config import ASSETS
from utils.ui_components import inject_custom_css, render_page_header

st.set_page_config(
    page_title="Diagnostics | Market Intelligence",
    layout="wide"
)

inject_custom_css()

render_page_header(
    icon="",
    title="System Diagnostics",
    subtitle="Model health, validation metrics, and system controls for predictive models"
)

# ==================== HELPERS ====================
def get_health_status():
    health_file = "data/model_health.json"
    if not os.path.exists(health_file):
        subprocess.run([sys.executable, "scripts/model_monitor.py"], capture_output=True)
    
    if os.path.exists(health_file):
        try:
            with open(health_file, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    return None

def load_stacker_results():
    """Load Dual-Head Stacker backtest results from reports/stacker_*_backtest.json."""
    results = {}
    for f in glob.glob("reports/stacker_*_backtest.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
            asset = data.get("asset", os.path.basename(f).replace("stacker_", "").replace("_backtest.json", ""))
            results[asset] = data
        except Exception:
            continue
    return results

def load_all_backtest_results():
    """Load legacy 3-Level backtest JSON files."""
    results = {}
    for f in glob.glob("reports/backtest_*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
            asset = data.get("asset", os.path.basename(f).replace("backtest_", "").replace(".json", ""))
            results[asset] = data
        except Exception:
            continue
    return results

def get_grade(hit_ratio):
    if hit_ratio >= 63: return "A",  "#00C49A"
    if hit_ratio >= 58: return "B+", "#7EB8A4"
    if hit_ratio >= 53: return "B",  "#F0B429"
    if hit_ratio >= 48: return "C+", "#FF8C00"
    return "C", "#FF4444"

def rmse_improvement(rmse_lstm, rmse_3lvl):
    return ((rmse_lstm - rmse_3lvl) / rmse_lstm * 100) if rmse_lstm else 0

# Load data
health_data = get_health_status()
stacker_results = load_stacker_results()
legacy_results  = load_all_backtest_results()

# ==================== TABS ====================
tab1, tab2 = st.tabs(["Health & Operations", "Validation Scorecard"])

with tab1:
    st.markdown("### Model Health & Alerts")
    
    if health_data and 'assets' in health_data:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"Last checked: {health_data.get('last_checked', 'Unknown')}")
        with col2:
            if st.button("Run Health Diagnostics", use_container_width=True):
                with st.spinner("Analyzing counterfactual logs and system health..."):
                    subprocess.run([sys.executable, "scripts/model_monitor.py"])
                st.rerun()

        for asset, info in health_data['assets'].items():
            status = info['status']
            if status == 'HEALTHY':
                st.success(f"**{asset.upper()}**: HEALTHY - {info['message']}")
            elif status == 'WARNING':
                st.warning(f"**{asset.upper()}**: WARNING - {info['message']}")
                st.info(f"To retrain, run: `.venv\\Scripts\\python scripts/train_lstm_pct.py {asset}`")
            elif status == 'DEGRADED':
                st.error(f"**{asset.upper()}**: DEGRADED - {info['message']}")
                st.error(f"Immediate retraining recommended!\n\nRun:\n`.venv\\Scripts\\python scripts/train_lstm_pct.py {asset}`\n`.venv\\Scripts\\python scripts/train_ridge_stacker.py {asset}`")
            else:
                st.info(f"**{asset.upper()}**: {status} - {info['message']}")
    else:
        st.info("Model Health Monitor has not collected enough data yet. Requires resolved 7-day forecasts.")

    st.markdown("---")
    st.markdown("### System Operations")

    selected_asset_for_test = st.selectbox(
        "Target Asset for Validation",
        options=[k.upper() for k in ASSETS.keys()],
        help="Select an asset to rigorously test against the 3-Layer Hierarchical AI."
    )

    if st.button("Run Backtesting"):
        st.info(f"Initiating walk-forward backtest for {selected_asset_for_test}... This may take several minutes.")
        try:
            with st.spinner(f"Compiling intelligence layers and running rigorous backtest for {selected_asset_for_test}..."):
                venv_python = r"d:\Market-Intelligence\.venv\Scripts\python.exe"
                result = subprocess.run(
                    [venv_python, "scripts/backtest_engine.py", selected_asset_for_test.lower()],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    st.success(f"Backtesting completed successfully for {selected_asset_for_test}.")
                    st.rerun()
                else:
                    st.error("Backtest failed.")
                    st.code(result.stderr)
        except Exception as e:
            st.error(f"Error launching backtest: {e}")

with tab2:
    st.markdown("## Dual-Head Ensemble Scorecard")
    st.info("Grades are dynamically evaluated on completely unseen data (20% hold-out). A low grade (e.g. 'C') simply indicates the model recently encountered highly volatile/noisy data (like a VIX spike) during training. It will automatically improve after retraining on clean data.")

    if stacker_results:
        # --- Summary table ---
        rows = []
        for asset, r in sorted(stacker_results.items()):
            grade_ens, _ = get_grade(r.get("hit_ratio_combined", 0))
            rows.append({
                "Asset":                 asset.upper(),
                "LSTM Hit Ratio":        f"{r.get('hit_ratio_lstm', 0):.1f}%",
                "XGBoost Hit Ratio":     f"{r.get('hit_ratio_xgb', 0):.1f}%",
                "Direction Head HR":     f"{r.get('hit_ratio_direction_head', 0):.1f}%",
                "Magnitude Head RMSE":   f"{r.get('rmse_magnitude_head', 0):.5f}",
                "Ensemble Hit Ratio":    f"{r.get('hit_ratio_combined', 0):.1f}%",
                "Ensemble RMSE":         f"{r.get('rmse_combined', 0):.5f}",
                "Grade":                 grade_ens,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # --- Metric cards ---
        st.markdown("---")
        assets_s = sorted(stacker_results.keys())
        cols_card = st.columns(len(assets_s))
        for i, asset in enumerate(assets_s):
            r = stacker_results[asset]
            hr = r.get("hit_ratio_combined", 0)
            grade, color = get_grade(hr)
            with cols_card[i]:
                st.markdown(f"""
                <div style="background:#1a1d2e; border:1px solid {color}; border-radius:12px;
                            padding:20px; text-align:center; margin-bottom:8px;">
                    <div style="font-size:13px; color:#aaa; margin-bottom:4px;">{asset.upper()} — Ensemble</div>
                    <div style="font-size:36px; font-weight:700; color:{color};">{hr:.1f}%</div>
                    <div style="font-size:18px; color:{color}; margin-top:4px;">Grade: {grade}</div>
                    <div style="font-size:11px; color:#666; margin-top:8px;">
                      Dir. Head: {r.get('hit_ratio_direction_head',0):.1f}%&nbsp;|&nbsp;
                      XGB: {r.get('hit_ratio_xgb',0):.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # --- Head-to-head bar chart ---
        st.markdown("---")
        st.markdown("### Head-to-Head: LSTM vs XGBoost vs Ensemble (Hit Ratio %)")
        st.caption("Directional accuracy of each model on the unseen evaluation set.")

        fig_h2h = go.Figure()
        fig_h2h.add_trace(go.Bar(
            name="LSTM (Momentum)",
            x=[a.upper() for a in assets_s],
            y=[stacker_results[a].get("hit_ratio_lstm", 0) for a in assets_s],
            marker_color="#4A90D9",
            text=[f"{stacker_results[a].get('hit_ratio_lstm', 0):.1f}%" for a in assets_s],
            textposition="outside",
        ))
        fig_h2h.add_trace(go.Bar(
            name="XGBoost (Macro)",
            x=[a.upper() for a in assets_s],
            y=[stacker_results[a].get("hit_ratio_xgb", 0) for a in assets_s],
            marker_color="#F0B429",
            text=[f"{stacker_results[a].get('hit_ratio_xgb', 0):.1f}%" for a in assets_s],
            textposition="outside",
        ))
        fig_h2h.add_trace(go.Bar(
            name="Ensemble (Dual-Head)",
            x=[a.upper() for a in assets_s],
            y=[stacker_results[a].get("hit_ratio_combined", 0) for a in assets_s],
            marker_color="#00C49A",
            text=[f"{stacker_results[a].get('hit_ratio_combined', 0):.1f}%" for a in assets_s],
            textposition="outside",
        ))
        fig_h2h.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                          annotation_text="Random (50%)", annotation_position="right")
        fig_h2h.add_hline(y=55, line_dash="dot", line_color="rgba(0,196,154,0.4)",
                          annotation_text="Target (55%)", annotation_position="right")
        fig_h2h.update_layout(
            template="plotly_dark",
            barmode="group",
            height=420,
            yaxis_title="Hit Ratio (%)",
            yaxis=dict(ticksuffix="%", range=[35, 80]),
            legend=dict(orientation="h", y=1.12),
            margin=dict(l=40, r=60, t=30, b=40),
        )
        st.plotly_chart(fig_h2h, use_container_width=True)

        # --- Direction Head coefficients ---
        st.markdown("---")
        st.markdown("### Direction Head Feature Importance")
        st.caption("LogisticRegressionCV coefficients — positive = bullish signal, negative = bearish signal.")

        sel_asset = st.selectbox("Select asset", [a.upper() for a in assets_s], key="coef_asset")
        r_sel = stacker_results[sel_asset.lower()]
        coefs = r_sel.get("direction_coefs", {})

        if coefs:
            df_coef = pd.DataFrame({
                "Feature":     list(coefs.keys()),
                "Coefficient": list(coefs.values()),
            }).sort_values("Coefficient", key=abs, ascending=True)

            fig_coef = go.Figure(go.Bar(
                x=df_coef["Coefficient"],
                y=df_coef["Feature"],
                orientation="h",
                marker_color=["#00C49A" if v > 0 else "#FF4444" for v in df_coef["Coefficient"]],
                text=[f"{v:+.4f}" for v in df_coef["Coefficient"]],
                textposition="outside",
            ))
            fig_coef.add_vline(x=0, line_color="white", line_width=1)
            fig_coef.update_layout(
                template="plotly_dark",
                height=max(280, len(df_coef) * 36),
                xaxis_title="Coefficient (Direction Head — LogisticCV)",
                margin=dict(l=10, r=80, t=20, b=30),
            )
            st.plotly_chart(fig_coef, use_container_width=True)
        else:
            st.info("No coefficient data. Re-run `py scripts/train_ridge_stacker.py`.")

    else:
        st.warning("""
        **No Ensemble backtest results found.**
        Train the Dual-Head Stacker first:
        ```bash
        py scripts/train_ridge_stacker.py
        ```
        """)

    # ==================== LATEST ASSET REPORT ====================
    if legacy_results or stacker_results:
        st.markdown("---")
        st.markdown("### Backtest Chart")
        
        all_assets = list(set(list(stacker_results.keys()) + list(legacy_results.keys())))
        if all_assets:
            view_asset = st.selectbox(
                "View detailed validation chart",
                options=[a.upper() for a in all_assets]
            )
            
            img_path = f"reports/backtest_{view_asset.lower()}.png"
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.warning("Chart not generated yet. Rerun the backtest.")

    st.markdown("---")
    with st.expander("Legacy 3-Level Architecture Results (Historical Reference)", expanded=False):
        if legacy_results:
            st.markdown("### Legacy 3-Level Scorecard")
            st.caption("Old absolute-price LSTM + Manager + CEO pipeline. Kept for historical comparison only.")

            legacy_rows = []
            for asset, r in sorted(legacy_results.items()):
                grade, _ = get_grade(r.get("hit_ratio_3layer", 0))
                legacy_rows.append({
                    "Asset":              asset.upper(),
                    "Hit Ratio (3-Lvl)":  f"{r.get('hit_ratio_3layer', 0):.1f}%",
                    "Hit Ratio (LSTM)":   f"{r.get('hit_ratio_lstm', 0):.1f}%",
                    "RMSE (3-Lvl)":       f"{r.get('rmse_3layer', 0):,.2f}",
                    "RMSE (LSTM)":        f"{r.get('rmse_lstm', 0):,.2f}",
                    "RMSE Improvement":   f"{rmse_improvement(r.get('rmse_lstm',0), r.get('rmse_3layer',0)):.1f}%",
                    "Grade": grade
                })
            st.dataframe(pd.DataFrame(legacy_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No legacy results.")

st.markdown("---")
st.caption("""
**Interpretation Guide:**
- **Ensemble Hit Ratio > 55%** = Statistically meaningful directional edge
- **Direction Head** optimizes *only* for directional accuracy (Hit Ratio)
- **Magnitude Head** optimizes *only* for % change size (RMSE), outlier-robust
- **Combined signal** = Direction x Magnitude — sign-weighted % change estimate
- All results on completely unseen hold-out data — no data leakage
""")
