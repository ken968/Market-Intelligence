"""
Model Validation & Backtest Scorecard
Dual-Head Ensemble Intelligence vs Legacy 3-Level Architecture.
"""

import streamlit as st
import pandas as pd
import json
import os
import sys
import glob
import plotly.graph_objects as go
from PIL import Image

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
    subtitle="Dual-Head Ensemble (LSTM-pct + XGBoost + Stacker) vs Legacy 3-Level Architecture"
)

# ============================================================
# HELPERS
# ============================================================

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


stacker_results = load_stacker_results()
legacy_results  = load_all_backtest_results()

# ============================================================
# METHODOLOGY
# ============================================================

with st.expander("Architecture Overview (click to expand)", expanded=False):
    st.markdown("""
    ### Dual-Head Ensemble Intelligence
    The current production system combines three models in a meta-learning stack:

    | Layer | Model | Target | Metric |
    |---|---|---|---|
    | **Base 1** | LSTM (pct-change) | 7-day forward % | Momentum/Micro |
    | **Base 2** | XGBoost (macro) | 7-day forward % | Macro/Regime |
    | **Meta Head 1** | LogisticRegressionCV | Direction (UP/DOWN) | Hit Ratio |
    | **Meta Head 2** | HuberRegressor | Magnitude (% size) | RMSE |

    **Walk-Forward Validation:** 80% train / 20% unseen test, stacker trained on 70% of test,
    evaluated on final 30% — triple hold-out. No data leakage.

    > Past validation accuracy does not guarantee future performance.
    """)

st.markdown("---")

# ============================================================
# SECTION 1: DUAL-HEAD ENSEMBLE SCORECARD
# ============================================================

st.markdown("## Dual-Head Ensemble Scorecard")
st.caption("Evaluated on completely unseen data (20% hold-out). Stacker meta-evaluation on final 30% of test period.")

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

# ============================================================
# SECTION 2: LEGACY 3-LEVEL (HISTORICAL REFERENCE)
# ============================================================

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

        assets_l = sorted(legacy_results.keys())
        sel_l = st.selectbox("View backtest chart", [a.upper() for a in assets_l], key="legacy_chart")
        chart_path = f"reports/backtest_{sel_l.lower()}.png"
        if os.path.exists(chart_path):
            st.image(chart_path, use_container_width=True,
                     caption=f"{sel_l} Walk-Forward — 3-Level vs Pure LSTM")
        else:
            st.info(f"No chart for {sel_l}. Run `py scripts/backtest_engine.py {sel_l.lower()}`")
    else:
        st.info("No legacy results. Run `py scripts/backtest_engine.py gold`.")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("""
**Interpretation Guide:**
- **Ensemble Hit Ratio > 55%** = Statistically meaningful directional edge
- **Direction Head** optimizes *only* for directional accuracy (Hit Ratio)
- **Magnitude Head** optimizes *only* for % change size (RMSE), outlier-robust
- **Combined signal** = Direction x Magnitude — sign-weighted % change estimate
- All results on completely unseen hold-out data — no data leakage
""")
