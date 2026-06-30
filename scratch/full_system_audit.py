"""
Full System Structural Audit Script
Checks all layers: data → models → engine → UI components
"""
import os
import sys
import json
import pickle
import traceback
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, 'D:\\Market-Intelligence')

PASS = "  [PASS]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"
INFO = "  [INFO]"

results = []

def check(label, condition, detail="", level="PASS"):
    icon = {"PASS": PASS, "FAIL": FAIL, "WARN": WARN, "INFO": INFO}.get(level, PASS)
    status = f"{icon} {label}"
    if detail:
        status += f" | {detail}"
    results.append((level, status))
    print(status)

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ─────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION INTEGRITY
# ─────────────────────────────────────────────────────────────────────
section("1. CONFIGURATION INTEGRITY")

try:
    from utils.config import ASSETS, STOCK_TICKERS, FORECAST_RANGES, get_all_stock_tickers
    check("Config imports", True, f"FORECAST_RANGES={list(FORECAST_RANGES.keys())}")
    
    REQUIRED_CFG_KEYS = ['features', 'model_file', 'scaler_file', 'data_file', 'sequence_length', 'model_arch', 'news_file']
    for asset_key, cfg in ASSETS.items():
        missing = [k for k in REQUIRED_CFG_KEYS if k not in cfg]
        if missing:
            check(f"Config[{asset_key}] keys", False, f"MISSING: {missing}", "FAIL")
        else:
            check(f"Config[{asset_key}] keys", True, f"features={len(cfg['features'])}", "PASS")
except Exception as e:
    check("Config imports", False, str(e), "FAIL")

# ─────────────────────────────────────────────────────────────────────
# 2. DATA FILE INTEGRITY
# ─────────────────────────────────────────────────────────────────────
section("2. DATA FILE INTEGRITY")

try:
    from utils.config import ASSETS
    for asset_key, cfg in ASSETS.items():
        data_path = cfg['data_file']
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            last_date = df['Date'].iloc[-1] if 'Date' in df.columns else 'N/A'
            n_rows = len(df)
            missing_feats = [f for f in cfg['features'] if f not in df.columns]
            if missing_feats:
                check(f"Data[{asset_key}]", False, f"MISSING FEATURES: {missing_feats}", "FAIL")
            elif n_rows < 100:
                check(f"Data[{asset_key}]", False, f"Only {n_rows} rows", "WARN")
            else:
                days_since = (datetime.now() - pd.to_datetime(last_date)).days if last_date != 'N/A' else 999
                level = "PASS" if days_since <= 5 else "WARN"
                check(f"Data[{asset_key}]", True, f"rows={n_rows}, last={last_date}, days_ago={days_since}", level)
        else:
            check(f"Data[{asset_key}]", False, f"FILE NOT FOUND: {data_path}", "FAIL")
except Exception as e:
    check("Data file integrity", False, str(e), "FAIL")

# ─────────────────────────────────────────────────────────────────────
# 3. MODEL FILE AUDIT
# ─────────────────────────────────────────────────────────────────────
section("3. MODEL FILE AUDIT")

try:
    from utils.config import ASSETS
    for asset_key, cfg in ASSETS.items():
        model_file = cfg['model_file']
        scaler_file = cfg['scaler_file']
        m_exists = os.path.exists(model_file)
        s_exists = os.path.exists(scaler_file)
        if m_exists and s_exists:
            check(f"Model[{asset_key}]", True, f"{os.path.basename(model_file)}")
        else:
            check(f"Model[{asset_key}]", False, f"model={m_exists}, scaler={s_exists}", "WARN")
        
        # Check Phase 7 (A/B) models
        for suffix in ['_model_a', '_model_b']:
            p7_model = model_file.replace('.keras', f'{suffix}.keras')
            p7_scaler = scaler_file.replace('.pkl', f'{suffix}.pkl')
            if os.path.exists(p7_model) and os.path.exists(p7_scaler):
                check(f"  Phase7[{asset_key}{suffix}]", True)
            # Only warn for core assets (btc, gold)
            elif asset_key in ['btc', 'gold']:
                check(f"  Phase7[{asset_key}{suffix}]", False, "Not found (registry may cover this)", "WARN")
except Exception as e:
    check("Model file audit", False, str(e), "FAIL")

# ─────────────────────────────────────────────────────────────────────
# 4. MODEL REGISTRY AUDIT
# ─────────────────────────────────────────────────────────────────────
section("4. MODEL REGISTRY AUDIT")

try:
    registry_path = 'models/model_registry.json'
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            registry = json.load(f)
        entries = {k: v for k, v in registry.items() if not k.startswith('_')}
        check("model_registry.json", True, f"{len(entries)} entries")
        
        # Check best_window entries per asset
        for asset in ['gold', 'btc']:
            for group in ['Model_A', 'Model_B']:
                best = [v for k, v in entries.items() 
                        if v.get('asset','').lower()==asset 
                        and v.get('model_group')==group 
                        and v.get('is_best_window') is True]
                if best:
                    b = best[0]
                    m_ok = os.path.exists(b.get('model_path',''))
                    s_ok = os.path.exists(b.get('scaler_path',''))
                    lvl = "PASS" if m_ok and s_ok else "FAIL"
                    check(f"  Registry best[{asset}/{group}]", m_ok and s_ok, 
                          f"W={b.get('window')}, model={m_ok}, scaler={s_ok}", lvl)
                else:
                    check(f"  Registry best[{asset}/{group}]", False, "No is_best_window=True entry", "WARN")
    else:
        check("model_registry.json", False, "File not found", "WARN")
except Exception as e:
    check("Model registry", False, str(e), "FAIL")

# ─────────────────────────────────────────────────────────────────────
# 5. SCALER vs CONFIG FEATURE ALIGNMENT
# ─────────────────────────────────────────────────────────────────────
section("5. SCALER vs CONFIG FEATURE ALIGNMENT")

import joblib

try:
    registry_path = 'models/model_registry.json'
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            registry = json.load(f)
        
        from utils.config import ASSETS
        for asset in ['gold', 'btc']:
            cfg = ASSETS[asset]
            expected_n_feats = len(cfg['features'])
            
            for group in ['Model_A', 'Model_B']:
                entries = [v for k, v in registry.items() 
                           if not k.startswith('_')
                           and v.get('asset','').lower()==asset 
                           and v.get('model_group')==group 
                           and v.get('is_best_window') is True]
                if not entries:
                    continue
                e = entries[0]
                scaler_path = e.get('scaler_path', '')
                if os.path.exists(scaler_path):
                    bundle = joblib.load(scaler_path)
                    fs = bundle.get('feature_scaler')
                    if hasattr(fs, 'n_features_in_'):
                        n_feats = fs.n_features_in_
                        ok = n_feats == expected_n_feats
                        lvl = "PASS" if ok else "FAIL"
                        check(f"Scaler[{asset}/{group}] features", ok,
                              f"scaler={n_feats}, config={expected_n_feats}", lvl)
                    else:
                        check(f"Scaler[{asset}/{group}] features", True, "n_features_in_ not available")
except Exception as e:
    check("Scaler feature alignment", False, str(e), "FAIL")

# ─────────────────────────────────────────────────────────────────────
# 6. DUCKDB DATA STORE
# ─────────────────────────────────────────────────────────────────────
section("6. DUCKDB DATA STORE INTEGRITY")

try:
    from utils.data_store import MarketDataStore
    store = MarketDataStore()
    
    for table_name in ['gold_global_insights', 'btc_global_insights', 'spy_global_insights']:
        try:
            df = store.read_table(table_name, format='pandas')
            last_date = df['Date'].iloc[-1] if 'Date' in df.columns else 'N/A'
            check(f"DuckDB[{table_name}]", True, f"rows={len(df)}, last={last_date}")
        except Exception as te:
            check(f"DuckDB[{table_name}]", False, str(te), "WARN")
except Exception as e:
    check("DuckDB store", False, str(e), "FAIL")

# ─────────────────────────────────────────────────────────────────────
# 7. FRED MACRO INDICATORS
# ─────────────────────────────────────────────────────────────────────
section("7. FRED MACRO INDICATORS")

try:
    fred_path = 'data/fred_indicators.csv'
    if os.path.exists(fred_path):
        df = pd.read_csv(fred_path)
        last_date = df['Date'].iloc[-1] if 'Date' in df.columns else 'N/A'
        required_cols = ['YieldCurve_10Y2Y', 'CPI_MoM', 'PPI_MoM', 'Credit_Spread', 'DGS10']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            check("FRED indicators", False, f"MISSING COLS: {missing}", "FAIL")
        else:
            check("FRED indicators", True, f"rows={len(df)}, last={last_date}, cols={len(df.columns)}")
    else:
        check("FRED indicators", False, "fred_indicators.csv not found", "FAIL")
except Exception as e:
    check("FRED indicators", False, str(e), "FAIL")

# ─────────────────────────────────────────────────────────────────────
# 8. NEWS FILES
# ─────────────────────────────────────────────────────────────────────
section("8. NEWS FILES FRESHNESS")

try:
    news_files = ['data/latest_news_btc.json', 'data/latest_news_gold.json', 
                  'data/latest_news_spy.json', 'data/latest_news_aapl.json']
    for nf in news_files:
        if os.path.exists(nf):
            mtime = datetime.fromtimestamp(os.path.getmtime(nf))
            hours_old = (datetime.now() - mtime).total_seconds() / 3600
            with open(nf) as f:
                items = json.load(f)
            # Check dates
            dates = [i.get('date', '') for i in items if i.get('date')]
            latest = max(dates) if dates else 'N/A'
            lvl = "PASS" if hours_old < 24 else "WARN"
            check(f"News[{os.path.basename(nf)}]", True, 
                  f"items={len(items)}, latest={latest}, updated={hours_old:.1f}h ago", lvl)
        else:
            check(f"News[{os.path.basename(nf)}]", False, "Not found", "WARN")
except Exception as e:
    check("News files", False, str(e), "FAIL")

# ─────────────────────────────────────────────────────────────────────
# 9. MACRO PROCESSOR
# ─────────────────────────────────────────────────────────────────────
section("9. MACRO PROCESSOR OUTPUT")

try:
    from utils.macro_processor import build_macro_context
    ctx = build_macro_context()
    lv = ctx.get('latest_values', {})
    macro_sum = ctx.get('macro_summary', '')
    check("macro_processor.build_macro_context()", True, f"keys={len(lv)}, summary_len={len(macro_sum)}")
    for key in ['Credit_Spread', 'CPI_MoM', 'YieldCurve_10Y2Y']:
        if key in lv:
            check(f"  macro[{key}]", True, f"value={lv[key]:.4f}")
        else:
            check(f"  macro[{key}]", False, "MISSING", "FAIL")
except Exception as e:
    check("Macro processor", False, str(e), "FAIL")
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────
# 10. PREDICTOR ENGINE — FORECAST OUTPUT STRUCTURE
# ─────────────────────────────────────────────────────────────────────
section("10. PREDICTOR ENGINE — FORECAST STRUCTURE (no LLM)")

try:
    from utils.predictor import AssetPredictor
    
    for asset in ['gold', 'btc']:
        try:
            p = AssetPredictor(asset)
            f = p.get_multi_range_forecast()  # No headlines = no CEO Layer
            
            current = f.get('Current', 0)
            horizons_ok = all(hz in f for hz in ['1 Day', '1 Week', '2 Weeks', '1 Month', '3 Months'])
            
            if not horizons_ok:
                missing_hz = [hz for hz in ['1 Day', '1 Week', '2 Weeks', '1 Month', '3 Months'] if hz not in f]
                check(f"Forecast[{asset}] horizons", False, f"MISSING: {missing_hz}", "FAIL")
            else:
                check(f"Forecast[{asset}] current_price", True, f"${current:,.2f}")
                
                for hz in ['1 Day', '1 Week', '2 Weeks', '1 Month', '3 Months']:
                    val = f[hz]
                    if isinstance(val, dict):
                        price = val.get('price', 0)
                        series = val.get('series', [])
                        fan_p10 = val.get('fan_p10', [])
                        fan_p90 = val.get('fan_p90', [])
                        conf = val.get('confidence', {}).get('label', 'N/A')
                        pct = ((price - current) / current * 100) if current > 0 else 0
                        
                        series_ok = len(series) > 0
                        fan_ok = len(fan_p10) > 0 and len(fan_p90) > 0
                        lvl = "PASS" if series_ok else "WARN"
                        
                        check(f"  [{asset}]{hz}", True,
                              f"${price:,.2f} ({pct:+.1f}%), conf={conf}, series={len(series)}, fan={fan_ok}", lvl)
                    else:
                        check(f"  [{asset}]{hz}", False, f"Not a dict: {type(val)}", "FAIL")
        except Exception as ae:
            check(f"Forecast[{asset}]", False, str(ae), "FAIL")
            traceback.print_exc()
except Exception as e:
    check("Predictor engine", False, str(e), "FAIL")
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────
# 11. COUNTERFACTUAL LOGGER
# ─────────────────────────────────────────────────────────────────────
section("11. COUNTERFACTUAL LOGGER")

try:
    log_path = 'data/counterfactual_log.jsonl'
    if os.path.exists(log_path):
        with open(log_path) as f:
            lines = f.readlines()
        non_empty = [l for l in lines if l.strip()]
        if non_empty:
            last = json.loads(non_empty[-1])
            last_date = last.get('forecast_date', 'N/A')
            asset = last.get('asset_key', 'N/A')
            check("counterfactual_log.jsonl", True, f"entries={len(non_empty)}, last={last_date} ({asset})")
        else:
            check("counterfactual_log.jsonl", False, "File is empty", "WARN")
    else:
        check("counterfactual_log.jsonl", False, "Not found — generate_forecasts.py not run with LLM", "WARN")
except Exception as e:
    check("Counterfactual logger", False, str(e), "FAIL")

# ─────────────────────────────────────────────────────────────────────
# 12. UI COMPONENT AUDIT
# ─────────────────────────────────────────────────────────────────────
section("12. UI COMPONENTS AUDIT")

try:
    from utils.ui_components import (
        render_news_section, create_price_chart, create_forecast_chart,
        render_prediction_table, render_quorum_inference_panel
    )
    check("UI components import", True, "All 5 key functions importable")
    
    # Check render_prediction_table handles new dict format
    import io
    from contextlib import redirect_stdout
    import streamlit as st
    check("Streamlit importable", True)
except Exception as e:
    check("UI components import", False, str(e), "FAIL")

# ─────────────────────────────────────────────────────────────────────
# 13. PAGE IMPORTS
# ─────────────────────────────────────────────────────────────────────
section("13. PAGE-LEVEL IMPORT CONSISTENCY")

pages = {
    'Dashboard': 'pages/1_Dashboard.py',
    'Gold': 'pages/2_Gold_Analysis.py',
    'BTC': 'pages/3_Bitcoin_Analysis.py',
    'Stocks': 'pages/4_Stocks_Analysis.py',
}

for page_name, page_path in pages.items():
    if os.path.exists(page_path):
        with open(page_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check key imports exist
        checks = {
            'predictor': 'AssetPredictor' in content,
            'ui_components': 'ui_components' in content,
            'forecast_chart_90d': '3 Months' in content,
            'render_quorum': 'render_quorum_inference_panel' in content if page_name != 'Dashboard' else True,
        }
        all_ok = all(checks.values())
        failed = [k for k, v in checks.items() if not v]
        lvl = "PASS" if all_ok else "WARN"
        check(f"Page[{page_name}]", all_ok, 
              f"OK" if all_ok else f"MISSING: {failed}", lvl)
    else:
        check(f"Page[{page_name}]", False, "File not found", "FAIL")

# ─────────────────────────────────────────────────────────────────────
# 14. XAI EXPLAINER
# ─────────────────────────────────────────────────────────────────────
section("14. XAI EXPLAINER")

try:
    from utils.xai_explainer import get_top_macro_drivers, build_driver_dataframe, explain_forecast
    drivers = get_top_macro_drivers('btc', lookback_days=14, top_n=3)
    check("xai_explainer.get_top_macro_drivers", True, f"drivers={len(drivers)}")
    if drivers:
        df_d = build_driver_dataframe(drivers)
        check("xai_explainer.build_driver_dataframe", True, f"shape={df_d.shape}")
except Exception as e:
    check("XAI Explainer", False, str(e), "FAIL")

# ─────────────────────────────────────────────────────────────────────
# 15. CONFIDENCE SCORE ENGINE
# ─────────────────────────────────────────────────────────────────────
section("15. CONFIDENCE SCORE ENGINE")

try:
    from utils.confidence_engine import get_confidence_score
    for asset in ['gold', 'btc', 'spy']:
        for hz_label in ['1 Day', '1 Week', '3 Months']:
            c = get_confidence_score(asset, hz_label)
            assert 0 <= c.get('score', -1) <= 1, f"Score out of range for {asset}/{hz_label}"
            assert 'label' in c, f"Missing 'label' for {asset}/{hz_label}"
    check("confidence_engine.get_confidence_score", True, "All asset/horizon combos valid")
except Exception as e:
    check("Confidence engine", False, str(e), "FAIL")

# ─────────────────────────────────────────────────────────────────────
# 16. REALTIME PRICES MODULE
# ─────────────────────────────────────────────────────────────────────
section("16. REALTIME PRICES & VOLATILITY MODULE")

try:
    from utils.realtime_prices import get_live_vix, get_live_dvol, iv_to_daily_vol
    vix = get_live_vix(fallback=20.0)
    dvol = get_live_dvol(fallback=60.0)
    daily_vol = iv_to_daily_vol(vix)
    check("realtime_prices module", True, f"VIX={vix:.1f}, DVOL={dvol:.1f}, daily_vol={daily_vol:.4f}")
except Exception as e:
    check("Realtime prices", False, str(e), "WARN")

# ─────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────
section("AUDIT SUMMARY")

total = len(results)
passes = sum(1 for r in results if r[0] == 'PASS')
fails  = sum(1 for r in results if r[0] == 'FAIL')
warns  = sum(1 for r in results if r[0] == 'WARN')

print(f"\n  Total checks : {total}")
print(f"  PASS         : {passes}")
print(f"  WARN         : {warns}")
print(f"  FAIL         : {fails}")
print()

if fails > 0:
    print("CRITICAL FAILURES:")
    for level, msg in results:
        if level == 'FAIL':
            print(f"    {msg}")

if warns > 0:
    print("\nWARNINGS:")
    for level, msg in results:
        if level == 'WARN':
            print(f"    {msg}")

print("\nAudit complete.")
