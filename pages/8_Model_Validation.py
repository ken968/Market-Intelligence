"""
Model Validation & Backtest Scorecard
Walk-forward backtesting results for 3-Level AI Architecture vs Pure LSTM baseline.
"""

import streamlit as st
import pandas as pd
import json
import os
import sys
import glob
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
from datetime import datetime

from utils.ui_components import inject_custom_css, render_page_header
from utils.config import ASSETS

st.set_page_config(
    page_title="Model Validation | Market Intelligence",
    layout="wide"
)

inject_custom_css()

render_page_header(
    icon="",
    title="Model Validation & Backtest Scorecard",
    subtitle="Walk-forward out-of-sample validation: 3-Level AI Architecture vs Pure LSTM Baseline"
)

# ============================================================
# HELPER
# ============================================================

def load_all_backtest_results():
    """Load all backtest JSON files from reports directory."""
    results = {}
    json_files = glob.glob("reports/backtest_*.json")
    for f in json_files:
        try:
            with open(f) as fp:
                data = json.load(fp)
            asset = data.get("asset", os.path.basename(f).replace("backtest_", "").replace(".json", ""))
            results[asset] = data
        except Exception:
            continue
    return results

def get_grade(hit_ratio):
    """Convert hit ratio to letter grade."""
    if hit_ratio >= 60: return "A", "#00C49A"
    if hit_ratio >= 55: return "B+", "#7EB8A4"
    if hit_ratio >= 50: return "B", "#F0B429"
    if hit_ratio >= 45: return "C+", "#FF8C00"
    return "C", "#FF4444"

def rmse_improvement(rmse_lstm, rmse_3lvl):
    """Return % improvement of 3-level vs LSTM."""
    if rmse_lstm == 0:
        return 0
    return ((rmse_lstm - rmse_3lvl) / rmse_lstm) * 100

# ============================================================
# LOAD DATA
# ============================================================

results = load_all_backtest_results()

if not results:
    st.warning("""
    **No backtest results found.**  
    Run a backtest first from the **Settings** page or via terminal:
    ```bash
    .venv\\Scripts\\python.exe scripts/backtest_engine.py gold
    .venv\\Scripts\\python.exe scripts/backtest_engine.py btc
    .venv\\Scripts\\python.exe scripts/backtest_engine.py spy
    ```
    """)
    st.stop()

# ============================================================
# METHODOLOGY EXPLAINER
# ============================================================

with st.expander("📖 How This Backtest Works (Walk-Forward Methodology)", expanded=False):
    st.markdown("""
    **This is NOT in-sample testing (which would be cheating).**
    
    The engine uses a rigorous **Walk-Forward Out-of-Sample** methodology:
    
    | Step | Detail |
    |---|---|
    | **1. Data Split** | 80% of historical data used for training, 20% held out as unseen test set |
    | **2. Isolated Model** | A fresh model is trained ONLY on the 80% training set — it has never seen the test data |
    | **3. Walk-Forward** | The model predicts one day at a time, then "walks forward" — mimicking real-world deployment |
    | **4. 3-Level System** | The Manager (Anchoring) and CEO (Macro/Sentiment Bias) layers are applied on top of each step |
    | **5. Baseline** | Pure LSTM (no macro layers) serves as the control group |
    
    **Key Metrics:**
    - **Directional Hit Ratio:** % of days where the model correctly predicted UP vs DOWN. Random = 50%. >55% is good.
    - **RMSE:** Root Mean Square Error in price units. Lower = better. The 3-Level system's RMSE should be dramatically lower than raw LSTM.
    
    > ⚠️ Past validation accuracy does not guarantee future performance. Markets change regimes.
    """)

st.markdown("---")

# ============================================================
# SUMMARY SCORECARD TABLE
# ============================================================

st.markdown("### 📊 Model Accuracy Scorecard")

scorecard_rows = []
for asset, r in sorted(results.items()):
    grade, _ = get_grade(r["hit_ratio_3layer"])
    rmse_imp = rmse_improvement(r["rmse_lstm"], r["rmse_3layer"])
    scorecard_rows.append({
        "Asset": asset.upper(),
        "Test Period": f"{r['start_test_date']} → {r['end_test_date']}",
        "Test Samples": r["test_samples"],
        "Hit Ratio (3-Level)": f"{r['hit_ratio_3layer']:.1f}%",
        "Hit Ratio (LSTM Only)": f"{r['hit_ratio_lstm']:.1f}%",
        "RMSE (3-Level)": f"{r['rmse_3layer']:,.2f}",
        "RMSE (LSTM Only)": f"{r['rmse_lstm']:,.2f}",
        "RMSE Improvement": f"{rmse_imp:.1f}%",
        "Grade": grade
    })

df_score = pd.DataFrame(scorecard_rows)
st.dataframe(df_score, use_container_width=True, hide_index=True)

# ============================================================
# METRIC CARDS — HIT RATIO
# ============================================================

st.markdown("---")
st.markdown("### 🎯 Directional Hit Ratio — 3-Level System")
st.caption("How often did the model correctly predict the direction (UP/DOWN) of the next day's price? Baseline random = 50%.")

assets_sorted = sorted(results.keys())
cols = st.columns(len(assets_sorted))

for i, asset in enumerate(assets_sorted):
    r = results[asset]
    hr = r["hit_ratio_3layer"]
    grade, color = get_grade(hr)
    with cols[i]:
        st.markdown(f"""
        <div style="background: #1a1d2e; border: 1px solid {color}; border-radius: 12px;
                    padding: 20px; text-align: center; margin-bottom: 8px;">
            <div style="font-size: 13px; color: #aaa; margin-bottom: 4px;">{asset.upper()}</div>
            <div style="font-size: 36px; font-weight: 700; color: {color};">{hr:.1f}%</div>
            <div style="font-size: 20px; color: {color}; margin-top: 4px;">Grade: {grade}</div>
            <div style="font-size: 11px; color: #666; margin-top: 8px;">{r['test_samples']} samples</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# RMSE IMPROVEMENT BAR CHART
# ============================================================

st.markdown("---")
st.markdown("### 📉 RMSE Improvement: 3-Level vs Raw LSTM")
st.caption("How much did the Manager + CEO layers reduce prediction error vs the raw LSTM worker alone?")

fig_rmse = go.Figure()

asset_labels = [a.upper() for a in assets_sorted]
improvements = [rmse_improvement(results[a]["rmse_lstm"], results[a]["rmse_3layer"]) for a in assets_sorted]
bar_colors = ["#00C49A" if v > 0 else "#FF4444" for v in improvements]

fig_rmse.add_trace(go.Bar(
    x=asset_labels,
    y=improvements,
    marker_color=bar_colors,
    text=[f"{v:.1f}%" for v in improvements],
    textposition="outside",
    hovertemplate="<b>%{x}</b><br>RMSE Improvement: %{y:.1f}%<extra></extra>"
))

fig_rmse.add_hline(y=0, line_color="white", line_width=1)

fig_rmse.update_layout(
    template="plotly_dark",
    height=380,
    yaxis_title="RMSE Improvement (%)",
    xaxis_title="Asset",
    showlegend=False,
    margin=dict(l=40, r=40, t=20, b=40),
    yaxis=dict(ticksuffix="%")
)

st.plotly_chart(fig_rmse, use_container_width=True)

# ============================================================
# INDIVIDUAL ASSET BACKTEST CHARTS
# ============================================================

st.markdown("---")
st.markdown("### 📈 Walk-Forward Backtest Charts")
st.caption("Actual price vs 3-Level System prediction vs Raw LSTM — on completely unseen test data.")

selected_asset = st.selectbox(
    "Select asset to view backtest chart",
    [a.upper() for a in assets_sorted],
    index=0
)

asset_key = selected_asset.lower()
chart_path = f"reports/backtest_{asset_key}.png"

if os.path.exists(chart_path):
    img = Image.open(chart_path)
    st.image(img, use_container_width=True, caption=f"{selected_asset} Walk-Forward Backtest — Actual vs 3-Level vs LSTM")

    # Show metrics for selected asset
    r = results[asset_key]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("3-Level Hit Ratio", f"{r['hit_ratio_3layer']:.1f}%",
                  delta=f"{r['hit_ratio_3layer'] - r['hit_ratio_lstm']:+.1f}% vs LSTM")
    with c2:
        st.metric("3-Level RMSE", f"{r['rmse_3layer']:,.2f}")
    with c3:
        st.metric("LSTM-Only RMSE", f"{r['rmse_lstm']:,.2f}")
    with c4:
        imp = rmse_improvement(r["rmse_lstm"], r["rmse_3layer"])
        st.metric("RMSE Reduction", f"{imp:.1f}%", delta="better" if imp > 0 else "worse")
else:
    st.info(f"No chart found for {selected_asset}. Run: `.venv\\Scripts\\python.exe scripts/backtest_engine.py {asset_key}`")

# ============================================================
# RUN NEW BACKTEST SECTION
# ============================================================

st.markdown("---")
st.markdown("### ⚙️ Run New Backtest")
st.warning("⏱️ Backtest trains a fresh isolated model — Gold/Stocks ~5-7 min, BTC ~10-15 min. Keep this page open.")

available_assets = list(ASSETS.keys())
col_sel, col_btn = st.columns([2, 1])

with col_sel:
    backtest_target = st.selectbox("Select asset to backtest", [a.upper() for a in available_assets])

with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    run_bt = st.button(f"▶ Run Backtest for {backtest_target}", use_container_width=True, type="primary")

if run_bt:
    python_exe = sys.executable
    import subprocess, re

    def strip_ansi(text):
        return re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)

    with st.status(f"Running walk-forward backtest for {backtest_target}...", expanded=True) as status:
        process = subprocess.Popen(
            [python_exe, "scripts/backtest_engine.py", backtest_target.lower()],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        while True:
            line = process.stdout.readline()
            if not line:
                break
            clean = strip_ansi(line.strip())
            if clean:
                st.text(clean)
        process.wait()

        if process.returncode == 0:
            status.update(label=f"✅ Backtest for {backtest_target} complete!", state="complete")
            st.success("Backtest finished. Refresh page to see updated scorecard.")
            st.rerun()
        else:
            status.update(label="❌ Backtest failed", state="error")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("""
**Interpretation Guide:**  
• **Hit Ratio > 55%** = Statistically meaningful directional edge  
• **Hit Ratio 50-55%** = Marginal edge, use with macro confirmation  
• **RMSE Improvement > 80%** = Manager layer successfully correcting raw LSTM overreaction  
• These results are on completely unseen data (20% hold-out) using walk-forward methodology
""")
