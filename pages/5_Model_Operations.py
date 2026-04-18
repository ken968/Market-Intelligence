"""
Model Operations & Validation Room
Centralized view for model health, backtest metrics, and retraining operations
"""

import streamlit as st
import pandas as pd
import json
import os
import subprocess
import glob
from utils.config import ASSETS
from utils.ui_components import inject_custom_css, render_page_header

st.set_page_config(
    page_title="Model Operations | Market Intelligence",
    layout="wide"
)

inject_custom_css()

render_page_header(
    icon="",
    title="Model Operations",
    subtitle="Validation metrics and system controls for predictive models"
)

# ==================== DATA COLLECTOR ====================
def load_backtest_metrics():
    reports_dir = "reports"
    if not os.path.exists(reports_dir):
        return []
        
    metrics = []
    for file in glob.glob(f"{reports_dir}/backtest_*.json"):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                metrics.append(data)
        except Exception:
            pass
    return metrics

metrics_data = load_backtest_metrics()

# ==================== METRICS DISPLAY ====================
st.markdown("### Structural Validation & Health")

if metrics_data:
    # Prepare DataFrame
    df_data = []
    for m in metrics_data:
        asset_name = ASSETS.get(m['asset'].lower(), {}).get('name', getattr(m['asset'], 'upper', lambda: m['asset'])())
        df_data.append({
            "Asset": asset_name,
            "Target Asset": m['asset'].upper(),
            "Date Range (Test)": f"{m.get('start_test_date', '')} to {m.get('end_test_date', '')}",
            "Hit Ratio (3-Layer)": f"{m.get('hit_ratio_3layer', 0):.2f}%",
            "RMSE (3-Layer)": f"{m.get('rmse_3layer', 0):.2f}",
            "RMSE (Raw LSTM)": f"{m.get('rmse_lstm', 0):.2f}"
        })
        
    df = pd.DataFrame(df_data)
    
    st.dataframe(
        df,
        column_config={
            "Asset": st.column_config.TextColumn("Asset", width="medium"),
            "Hit Ratio (3-Layer)": st.column_config.TextColumn("Hit Ratio (3-Layer)"),
        },
        hide_index=True,
        use_container_width=True
    )
else:
    st.info("No prior validation data found. Run a backtest below.")

st.markdown("---")

# ==================== ACTIONS ====================
st.markdown("### System Operations")

selected_asset_for_test = st.selectbox(
    "Target Asset for Validation",
    options=[k.upper() for k in ASSETS.keys()],
    help="Select an asset to rigorously test against the 3-Layer Hierarchical AI."
)

if st.button("Run Backtesting"):
    st.info(f"Initiating walk-forward backtest for {selected_asset_for_test}... This may take several minutes.")
    
    try:
        # Run subprocess silently
        # In a real environment, we'd stream stdout or run this asynchronously
        # For simplicity, we just block until it's done
        with st.spinner(f"Compiling intelligence layers and running rigorous backtest for {selected_asset_for_test}..."):
            result = subprocess.run(
                ["python", "scripts/backtest_engine.py", selected_asset_for_test.lower()],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                st.success(f"Backtesting completed successfully for {selected_asset_for_test}.")
                st.rerun()  # Refresh metrics
            else:
                st.error("Backtest failed.")
                st.code(result.stderr)
    except Exception as e:
        st.error(f"Error launching backtest: {e}")

# ==================== LATEST ASSET REPORT ====================
if metrics_data:
    st.markdown("---")
    st.markdown("### Latest Validation Report")
    
    view_asset = st.selectbox(
        "View detailed validation chart",
        options=[m['asset'].upper() for m in metrics_data]
    )
    
    img_path = f"reports/backtest_{view_asset.lower()}.png"
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)
    else:
        st.warning("Chart not generated yet. Rerun the backtest.")

