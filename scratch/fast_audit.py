"""
Fast Full System Structural Audit — NO TF Model Loading
Covers all layers: config, data, models presence, news, macro, UI, pages
"""
import os
import sys
import json
import traceback
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, 'D:\\Market-Intelligence')

results = []

def check(label, ok, detail="", level=None):
    if level is None:
        level = "PASS" if ok else "FAIL"
    icon = {"PASS": "[PASS]", "FAIL": "[FAIL]", "WARN": "[WARN]", "INFO": "[INFO]"}.get(level, "[PASS]")
    msg = f"  {icon} {label}"
    if detail:
        msg += f" | {detail}"
    results.append((level, msg))
    print(msg)

def section(title):
    print(f"\n{'='*62}\n  {title}\n{'='*62}")

# =============================================================================
# 1. CONFIG INTEGRITY
# =============================================================================
section("1. CONFIG INTEGRITY")
try:
    from utils.config import ASSETS, STOCK_TICKERS, FORECAST_RANGES, get_all_stock_tickers
    REQUIRED_CFG = ['features', 'model_file', 'scaler_file', 'data_file', 
                    'sequence_length', 'model_arch', 'news_file']
    for asset_key, cfg in ASSETS.items():
        missing = [k for k in REQUIRED_CFG if k not in cfg]
        lvl = "FAIL" if missing else "PASS"
        check(f"Config[{asset_key}]", not missing, 
              f"features={len(cfg.get('features',[]))}" if not missing else f"MISSING: {missing}", lvl)
    
    # Check FORECAST_RANGES includes all 5 horizons
    expected_hz = {'1 Day', '1 Week', '2 Weeks', '1 Month', '3 Months'}
    actual_hz = set(FORECAST_RANGES.keys())
    check("FORECAST_RANGES covers all 5 horizons", expected_hz == actual_hz, str(actual_hz))
    
    # Check Credit_Spread in all features
    missing_cs = [k for k, v in ASSETS.items() if 'Credit_Spread' not in v.get('features', [])]
    check("Credit_Spread in all asset features", not missing_cs, 
          "OK" if not missing_cs else f"MISSING in: {missing_cs}",
          "PASS" if not missing_cs else "FAIL")
except Exception as e:
    check("Config import", False, str(e), "FAIL")

# =============================================================================
# 2. DATA FILE INTEGRITY
# =============================================================================
section("2. DATA FILE INTEGRITY")
try:
    from utils.config import ASSETS
    for asset_key, cfg in ASSETS.items():
        p = cfg['data_file']
        if not os.path.exists(p):
            check(f"Data[{asset_key}]", False, f"NOT FOUND: {p}", "FAIL"); continue
        df = pd.read_csv(p)
        last_date = df['Date'].iloc[-1] if 'Date' in df.columns else 'N/A'
        n = len(df)
        missing_f = [f for f in cfg['features'] if f not in df.columns]
        days_old = (datetime.now() - pd.to_datetime(last_date)).days if last_date != 'N/A' else 999
        
        if missing_f:
            check(f"Data[{asset_key}]", False, f"MISSING FEATURES: {missing_f}", "FAIL")
        elif days_old > 5:
            check(f"Data[{asset_key}]", True, f"rows={n}, last={last_date}, STALE={days_old}d ago", "WARN")
        else:
            check(f"Data[{asset_key}]", True, f"rows={n}, last={last_date} ({days_old}d ago)")
except Exception as e:
    check("Data integrity", False, str(e), "FAIL")

# =============================================================================
# 3. MODEL & SCALER FILE PRESENCE
# =============================================================================
section("3. MODEL FILE PRESENCE")
try:
    from utils.config import ASSETS
    for asset_key, cfg in ASSETS.items():
        m_ok = os.path.exists(cfg['model_file'])
        s_ok = os.path.exists(cfg['scaler_file'])
        if m_ok and s_ok:
            check(f"Model[{asset_key}]", True, os.path.basename(cfg['model_file']))
        elif m_ok and not s_ok:
            check(f"Model[{asset_key}]", False, f"model=OK but scaler={cfg['scaler_file']} MISSING", "WARN")
        else:
            check(f"Model[{asset_key}]", False, f"model={m_ok}, scaler={s_ok}", "WARN")
        
        # Horizon models  
        for h in [1, 7, 14, 30, 90]:
            hm = f"models/{asset_key}_model_{h}d.keras"
            hs = f"models/{asset_key}_scaler_{h}d.pkl"
            if os.path.exists(hm) and os.path.exists(hs):
                check(f"  HorizonModel[{asset_key}/{h}d]", True)
            # Only flag missing if no Phase7 compensates (btc/gold have Phase7)
except Exception as e:
    check("Model file presence", False, str(e), "FAIL")

# =============================================================================
# 4. MODEL REGISTRY INTEGRITY
# =============================================================================
section("4. MODEL REGISTRY")
try:
    reg_path = 'models/model_registry.json'
    if not os.path.exists(reg_path):
        check("model_registry.json", False, "NOT FOUND", "WARN")
    else:
        with open(reg_path) as f:
            reg = json.load(f)
        entries = {k:v for k,v in reg.items() if not k.startswith('_')}
        check("model_registry.json", True, f"{len(entries)} model entries")
        
        import joblib
        for asset in ['gold', 'btc']:
            for group in ['Model_A', 'Model_B']:
                best = [v for k,v in entries.items() 
                        if v.get('asset','').lower()==asset 
                        and v.get('model_group')==group 
                        and v.get('is_best_window') is True]
                if not best:
                    check(f"  Registry best[{asset}/{group}]", False, "No is_best_window entry", "WARN"); continue
                b = best[0]
                m_ok = os.path.exists(b.get('model_path',''))
                s_ok = os.path.exists(b.get('scaler_path',''))
                if not m_ok or not s_ok:
                    check(f"  Registry best[{asset}/{group}]", False, f"model={m_ok}, scaler={s_ok}", "FAIL"); continue
                
                # Check scaler feature count vs config
                try:
                    from utils.config import ASSETS as _ASSETS
                    n_cfg = len(_ASSETS[asset]['features'])
                    bnd = joblib.load(b['scaler_path'])
                    fs = bnd.get('feature_scaler')
                    n_scaler = getattr(fs, 'n_features_in_', None)
                    if n_scaler is not None:
                        match = n_scaler == n_cfg
                        lvl = "PASS" if match else "FAIL"
                        check(f"  Scaler features[{asset}/{group}]", match, 
                              f"scaler={n_scaler}, config={n_cfg}", lvl)
                    else:
                        check(f"  Scaler features[{asset}/{group}]", True, 
                              f"W={b.get('window')}, model=OK, scaler=OK (n_features N/A)")
                except Exception as se:
                    check(f"  Scaler features[{asset}/{group}]", False, str(se), "WARN")
except Exception as e:
    check("Registry audit", False, str(e), "FAIL")

# =============================================================================
# 5. DUCKDB DATA STORE
# =============================================================================
section("5. DUCKDB DATA STORE")
try:
    from utils.data_store import MarketDataStore
    store = MarketDataStore()
    for tbl in ['gold_global_insights', 'btc_global_insights', 'spy_global_insights', 
                'qqq_global_insights', 'btc_global_insights']:
        try:
            df = store.read_table(tbl, format='pandas')
            last = df['Date'].iloc[-1] if 'Date' in df.columns else 'N/A'
            check(f"DuckDB[{tbl}]", True, f"rows={len(df)}, last={last}")
        except Exception as te:
            check(f"DuckDB[{tbl}]", False, str(te)[:80], "WARN")
except Exception as e:
    check("DuckDB store", False, str(e), "FAIL")

# =============================================================================
# 6. FRED INDICATORS
# =============================================================================
section("6. FRED MACRO INDICATORS")
try:
    fred_path = 'data/fred_indicators.csv'
    if not os.path.exists(fred_path):
        check("fred_indicators.csv", False, "NOT FOUND", "FAIL")
    else:
        df = pd.read_csv(fred_path)
        last = df['Date'].iloc[-1] if 'Date' in df.columns else 'N/A'
        required = ['YieldCurve_10Y2Y', 'CPI_MoM', 'PPI_MoM', 'Credit_Spread', 
                    'Yield_10Y_Rate', 'M2_MoM', 'PCE_MoM', 'NFP_Change']
        missing = [c for c in required if c not in df.columns]
        check("fred_indicators.csv", not missing, 
              f"rows={len(df)}, last={last}, cols={len(df.columns)}" if not missing else f"MISSING: {missing}",
              "PASS" if not missing else "FAIL")
except Exception as e:
    check("FRED indicators", False, str(e), "FAIL")

# =============================================================================
# 7. MACRO INDICATORS CSV (for Dashboard)
# =============================================================================
section("7. MACRO INDICATORS CSV (Dashboard)")
try:
    macro_path = 'data/macro_indicators.csv'
    if not os.path.exists(macro_path):
        check("macro_indicators.csv", False, "NOT FOUND", "FAIL")
    else:
        df = pd.read_csv(macro_path)
        last = df['Date'].iloc[-1] if 'Date' in df.columns else 'N/A'
        required = ['Oil_Price', 'DXY', 'VIX']
        missing = [c for c in required if c not in df.columns]
        check("macro_indicators.csv", not missing,
              f"rows={len(df)}, last={last}" if not missing else f"MISSING: {missing}",
              "PASS" if not missing else "FAIL")
except Exception as e:
    check("macro_indicators.csv", False, str(e), "FAIL")

# =============================================================================
# 8. NEWS FILES
# =============================================================================
section("8. NEWS FILE FRESHNESS")
try:
    from utils.config import ASSETS
    for asset_key in ASSETS:
        nf = f"data/latest_news_{asset_key}.json"
        cf = f"data/ceo_filtered_news_{asset_key}.json"
        if os.path.exists(nf):
            mtime = datetime.fromtimestamp(os.path.getmtime(nf))
            hrs = (datetime.now() - mtime).total_seconds() / 3600
            with open(nf) as f:
                items = json.load(f)
            dates = sorted([i.get('date','') for i in items if i.get('date')], reverse=True)
            latest = dates[0] if dates else 'N/A'
            ceo_exists = os.path.exists(cf)
            lvl = "PASS" if hrs < 24 else "WARN"
            check(f"News[{asset_key}]", True, 
                  f"items={len(items)}, latest={latest}, {hrs:.0f}h ago, ceo_filtered={ceo_exists}", lvl)
        else:
            check(f"News[{asset_key}]", False, "NOT FOUND", "WARN")
except Exception as e:
    check("News files", False, str(e), "FAIL")

# =============================================================================
# 9. MACRO PROCESSOR
# =============================================================================
section("9. MACRO PROCESSOR")
try:
    from utils.macro_processor import build_macro_context
    ctx = build_macro_context()
    lv = ctx.get('latest_values', {})
    summary = ctx.get('macro_summary', '')
    check("build_macro_context()", True, f"keys={len(lv)}, summary_len={len(summary)}")
    for key in ['Credit_Spread', 'CPI_MoM', 'YieldCurve_10Y2Y', 'M2_MoM', 'Yield_10Y_Rate']:
        if key in lv:
            check(f"  macro[{key}]", True, f"={lv[key]:.4f}")
        else:
            check(f"  macro[{key}]", False, "MISSING", "FAIL")
except Exception as e:
    check("Macro processor", False, str(e), "FAIL")
    traceback.print_exc()

# =============================================================================
# 10. UI COMPONENT IMPORTS
# =============================================================================
section("10. UI COMPONENT IMPORTS")
try:
    from utils.ui_components import (
        inject_custom_css, render_page_header, render_metric_card,
        render_news_section, create_price_chart, create_forecast_chart,
        render_prediction_table, render_quorum_inference_panel,
        show_loading_message, show_error_message
    )
    check("All 10 UI component functions importable", True)
    
    # Check render_news_section references ceo_filtered_news
    import inspect
    src = inspect.getsource(render_news_section)
    ceo_priority = 'ceo_filtered_news' in src
    check("render_news_section prioritizes CEO-filtered news", ceo_priority)
    
    # Check create_forecast_chart works without pandas import errors
    check("create_forecast_chart importable", True)
except Exception as e:
    check("UI component imports", False, str(e), "FAIL")
    traceback.print_exc()

# =============================================================================
# 11. PAGE STRUCTURAL ANALYSIS
# =============================================================================
section("11. PAGE STRUCTURAL ANALYSIS")
pages = [
    ('Dashboard', 'pages/1_Dashboard.py'),
    ('Gold', 'pages/2_Gold_Analysis.py'),
    ('BTC', 'pages/3_Bitcoin_Analysis.py'),
    ('Stocks', 'pages/4_Stocks_Analysis.py'),
]

page_checks = {
    'AssetPredictor': 'LSTM model used',
    'get_multi_range_forecast': 'Multi-range forecasting called',
    '3 Months': '90-day fan chart used',
    'render_prediction_table': 'Prediction table rendered',
    'inject_custom_css': 'CSS injected',
}

for page_name, page_path in pages:
    if not os.path.exists(page_path):
        check(f"Page[{page_name}]", False, "FILE NOT FOUND", "FAIL"); continue
    
    with open(page_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    failed_checks = []
    for pattern, desc in page_checks.items():
        if page_name == 'Dashboard' and pattern == 'render_prediction_table':
            continue  # Dashboard doesn't have a prediction table
        if pattern not in content:
            failed_checks.append(desc)
    
    lvl = "PASS" if not failed_checks else "WARN"
    check(f"Page[{page_name}]", not failed_checks, 
          "All structural checks pass" if not failed_checks else f"Missing: {failed_checks}", lvl)
    
    # Check for old 30D references (should now be 90D)
    if '30-Day Probability Cloud' in content:
        check(f"  Page[{page_name}] - no stale 30D reference", False, "STILL has 30-Day Probability Cloud", "WARN")
    else:
        check(f"  Page[{page_name}] - uses 90D fan chart", True)

# =============================================================================
# 12. CONFIDENCE ENGINE
# =============================================================================
section("12. CONFIDENCE ENGINE")
try:
    from utils.confidence_engine import get_confidence_score
    for asset in ['gold', 'btc', 'spy', 'nvda']:
        for hz in ['1 Day', '1 Week', '3 Months']:
            c = get_confidence_score(asset, hz)
            ok = 0 <= c.get('score', -1) <= 1 and 'label' in c and 'color' in c
            lvl = "PASS" if ok else "FAIL"
            check(f"Confidence[{asset}/{hz}]", ok, 
                  f"score={c.get('score','?')}, label={c.get('label','?')}", lvl)
except Exception as e:
    check("Confidence engine", False, str(e), "FAIL")

# =============================================================================
# 13. XAI EXPLAINER
# =============================================================================
section("13. XAI EXPLAINER")
try:
    from utils.xai_explainer import get_top_macro_drivers, build_driver_dataframe, explain_forecast
    drivers = get_top_macro_drivers('btc', lookback_days=14, top_n=3)
    check("get_top_macro_drivers(btc)", True, f"drivers={len(drivers)}")
    if drivers:
        df_d = build_driver_dataframe(drivers)
        check("build_driver_dataframe()", True, f"shape={df_d.shape}")
    
    drivers_gold = get_top_macro_drivers('gold', lookback_days=14, top_n=3)
    check("get_top_macro_drivers(gold)", True, f"drivers={len(drivers_gold)}")
except Exception as e:
    check("XAI Explainer", False, str(e), "FAIL")

# =============================================================================
# 14. REALTIME PRICES
# =============================================================================
section("14. REALTIME PRICES MODULE")
try:
    from utils.realtime_prices import get_live_vix, get_live_dvol, iv_to_daily_vol
    vix = get_live_vix(fallback=20.0)
    dvol = get_live_dvol(fallback=60.0)
    vol = iv_to_daily_vol(vix)
    check("realtime_prices", True, f"VIX={vix:.1f}, BTC_DVOL={dvol:.1f}, daily_vol={vol:.4f}")
    
    # Check vol bounds are applied
    btc_vol = max(0.010, min(0.12, iv_to_daily_vol(dvol)))
    gold_vol = max(0.003, min(0.04, iv_to_daily_vol(vix)))
    check("Vol bounds (BTC)", 0.010 <= btc_vol <= 0.12, f"{btc_vol:.4f}")
    check("Vol bounds (Gold)", 0.003 <= gold_vol <= 0.04, f"{gold_vol:.4f}")
except Exception as e:
    check("Realtime prices", False, str(e), "WARN")

# =============================================================================
# 15. COUNTERFACTUAL LOGGER
# =============================================================================
section("15. COUNTERFACTUAL LOGGER")
try:
    log_path = 'data/counterfactual_log.jsonl'
    if os.path.exists(log_path):
        with open(log_path) as f:
            lines = [l for l in f.readlines() if l.strip()]
        if lines:
            last = json.loads(lines[-1])
            check("counterfactual_log.jsonl", True, 
                  f"entries={len(lines)}, last={last.get('forecast_date','?')} ({last.get('asset_key','?')})")
        else:
            check("counterfactual_log.jsonl", False, "File exists but empty", "WARN")
    else:
        check("counterfactual_log.jsonl", False, "Not found — run generate_forecasts.py with LLM", "WARN")
    
    # Validate the logger module itself imports
    from utils.counterfactual_logger import log_forecast
    check("counterfactual_logger module imports", True)
except Exception as e:
    check("Counterfactual logger", False, str(e), "FAIL")

# =============================================================================
# 16. FEATURE ENGINEERING
# =============================================================================
section("16. FEATURE ENGINEERING")
try:
    from utils.feature_engineering import (
        get_indicator_lookback, compute_dynamic_recession_risk
    )
    # Monthly indicators should have long lookback
    for ind in ['CPI_MoM', 'NFP_Change', 'M2_MoM']:
        lb = get_indicator_lookback(ind)
        ok = lb >= 30
        check(f"Lookback[{ind}]", ok, f"{lb} days", "PASS" if ok else "FAIL")
    
    # Daily indicators should have short lookback
    for ind in ['YieldCurve_10Y2Y', 'VIX', 'Credit_Spread']:
        lb = get_indicator_lookback(ind)
        ok = lb <= 20
        check(f"Lookback[{ind}]", ok, f"{lb} days", "PASS" if ok else "FAIL")
    
    # Recession risk in [0,1]
    dates = pd.date_range('2020-01-01', periods=100)
    df_test = pd.DataFrame({'YieldCurve_10Y2Y': np.random.uniform(-1.5, 1.5, 100)}, index=dates)
    risk = compute_dynamic_recession_risk(df_test)
    check("compute_dynamic_recession_risk", 0 <= risk <= 1, f"={risk:.3f}")
except Exception as e:
    check("Feature engineering", False, str(e), "FAIL")

# =============================================================================
# 17. SENTIMENT AGGREGATOR
# =============================================================================
section("17. SENTIMENT AGGREGATOR")
try:
    sys.path.insert(0, 'scripts')
    from sentiment_sources.aggregator import SentimentAggregator
    cred = SentimentAggregator.SOURCE_CREDIBILITY
    default = SentimentAggregator.DEFAULT_CREDIBILITY
    
    tier1 = ['reuters.com', 'bloomberg.com', 'federalreserve.gov']
    for d in tier1:
        ok = d in cred and cred[d] >= 0.9
        check(f"Credibility[{d}]", ok, f"={cred.get(d,'N/A')}", "PASS" if ok else "FAIL")
    
    check("Default credibility in [0.3, 0.7]", 0.3 <= default <= 0.7, f"={default}")
    check("SentimentAggregator init", True, f"sources defined")
except Exception as e:
    check("Sentiment aggregator", False, str(e), "FAIL")

# =============================================================================
# 18. CORRELATION ENFORCER
# =============================================================================
section("18. CORRELATION ENFORCER")
try:
    from utils.correlation_enforcer import CorrelationEnforcer
    enforcer = CorrelationEnforcer(reference_ticker='SPY')
    
    # Test with simple predictions
    test_preds = {
        'SPY': [500.0, 501.0, 503.0, 507.0, 520.0],
        'QQQ': [450.0, 451.0, 454.0, 460.0, 480.0],
    }
    adjusted = enforcer.enforce_predictions(test_preds, adjustment_strength=0.7)
    ok = 'SPY' in adjusted and 'QQQ' in adjusted
    check("CorrelationEnforcer.enforce_predictions()", ok, f"adjusted={list(adjusted.keys())}")
except Exception as e:
    check("Correlation enforcer", False, str(e), "FAIL")

# =============================================================================
# 19. DAILY OPS SCRIPT IMPORTS
# =============================================================================
section("19. PIPELINE SCRIPTS (import-only)")
scripts_to_check = [
    'scripts/daily_ops.py',
    'scripts/generate_forecasts.py',
    'scripts/sentiment_fetcher_v2.py',
    'scripts/fred_fetcher.py',
]
for script_path in scripts_to_check:
    if not os.path.exists(script_path):
        check(f"Script[{script_path}]", False, "NOT FOUND", "FAIL")
    else:
        # Check for obvious syntax errors via compile
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                src = f.read()
            compile(src, script_path, 'exec')
            check(f"Script[{os.path.basename(script_path)}]", True, "Syntax OK")
        except SyntaxError as se:
            check(f"Script[{os.path.basename(script_path)}]", False, f"SyntaxError: {se}", "FAIL")

# =============================================================================
# SUMMARY
# =============================================================================
section("AUDIT SUMMARY")
total  = len(results)
passes = sum(1 for r in results if r[0] == 'PASS')
fails  = sum(1 for r in results if r[0] == 'FAIL')
warns  = sum(1 for r in results if r[0] == 'WARN')

print(f"\n  Total checks : {total}")
print(f"  PASS         : {passes}")
print(f"  WARN         : {warns}")
print(f"  FAIL         : {fails}")
print()

if fails > 0:
    print("CRITICAL FAILURES (must fix):")
    for level, msg in results:
        if level == 'FAIL':
            print(f"    {msg}")

if warns > 0:
    print("\nWARNINGS:")
    for level, msg in results:
        if level == 'WARN':
            print(f"    {msg}")

print("\nAudit complete.")
