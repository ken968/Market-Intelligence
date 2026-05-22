# Market Intelligence 2.0

> **Hierarchical Financial Intelligence & Macro Simulation Engine**
> A production-grade 3-layer AI system for Gold (XAUUSD), Bitcoin, and US Equities.
> Combines LSTM sequence learning, XGBoost macro-regime detection, and a Dual-Head Ensemble Stacker
> with Monte Carlo probability clouds and real-time macroeconomic factor injection.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Macroeconomic Framework](#macroeconomic-framework)
4. [Statistical & Quantitative Methods](#statistical--quantitative-methods)
5. [Project Structure](#project-structure)
6. [Model Training Pipeline](#model-training-pipeline)
7. [Daily Workflow](#daily-workflow)
8. [Latest Backtest Performance](#latest-backtest-performance)
9. [Disclaimer](#disclaimer)

---

## Quick Start

### Option A — Double-Click (Windows)
```
Double-click: run_app.bat
```
This automatically activates the virtual environment (`.venv`) and starts the Streamlit server.
The terminal stays open so you can monitor logs. Press any key to close when done.

### Option B — Manual (PowerShell / CMD)
```powershell
# Activate virtual environment
.venv\Scripts\activate

# Start the app
streamlit run app.py
```

### Option C — Fresh Install
```powershell
# 1. Clone
git clone https://github.com/ken968/Market-Intelligence.git
cd Market-Intelligence

# 2. Create virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure API keys (copy the template below to .env)
# GEMINI_API_KEY=your_gemini_key_here
# NEWSAPI_KEY=your_newsapi_key_here
# FINNHUB_API_KEY=your_finnhub_key_here

# 4. Sync data (first run)
python scripts/fred_fetcher.py
python scripts/data_fetcher_v2.py
python scripts/sentiment_fetcher_v2.py all

# 5. Train models
python scripts/train_lstm_pct.py gold
python scripts/train_lstm_pct.py btc
python scripts/train_lstm_pct.py spy
python scripts/train_xgboost_macro.py gold
python scripts/train_xgboost_macro.py btc
python scripts/train_xgboost_macro.py spy
python scripts/train_ridge_stacker.py        # trains Gold + BTC + SPY

# 6. Launch
run_app.bat
```

---

## Architecture Overview

The system uses a strict **3-Layer Causal Hierarchy** to separate model concerns:

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3 — CEO Layer (Gemini LLM Contextual Bias)           │
│  Purpose: Inject macro narrative & regime context           │
│  Input:   News headlines + macro_summary                    │
│  Output:  drift_multiplier ∈ [0.85, 1.15] + bias_vector    │
│  File:    utils/llm_manager.py                              │
├─────────────────────────────────────────────────────────────┤
│  LAYER 2 — Manager Layer (Dual-Head Ensemble Stacker)       │
│  Purpose: Combine LSTM + XGBoost signals, correct endpoint  │
│  Architecture:                                              │
│    Head 1: LogisticRegressionCV → Direction (UP/DOWN)       │
│    Head 2: HuberRegressor       → Magnitude (% change)      │
│    Combined: direction_signal × |magnitude|                 │
│  File:    utils/layers/manager_anchor.py                    │
│           scripts/train_ridge_stacker.py                    │
├─────────────────────────────────────────────────────────────┤
│  LAYER 1 — Worker Layer (LSTM + XGBoost Base Models)        │
│  LSTM:  Sequence-based momentum & micro pattern learning    │
│  XGB:   Macro regime detection via lagged economic features │
│  Files: utils/layers/worker_lstm.py                         │
│         scripts/train_lstm_pct.py                           │
│         scripts/train_xgboost_macro.py                      │
└─────────────────────────────────────────────────────────────┘
```

### Forecast Pipeline (per request)
1. `ensemble_forecast()` — LSTM + XGBoost → Dual-Head Stacker → 7D signal
2. `pct_chain_forecast(N)` — scale LSTM recursive path to stacker's endpoint
3. Monte Carlo fan chart — 500 GBM paths using live IV (Deribit DVOL / CBOE VIX)
4. CEO Layer drift modulation (optional, requires Gemini API key)

---

## Macroeconomic Framework

The system is grounded in standard institutional macro analysis. Every feature maps to a specific economic mechanism:

### Tier 1 — FRED Economic Indicators (Monthly)
| Feature | FRED Series | Economic Role |
|---------|-------------|---------------|
| `CPI_MoM` | CPIAUCSL | Inflation momentum; drives Fed rate expectations |
| `PPI_MoM` | PPIACO | Producer cost pressure; leads CPI by 1-2 months |
| `PCE_MoM` | PCEPI | Fed's preferred inflation gauge (targets 2% YoY) |
| `NFP_Change` | PAYEMS | Labor market strength; risk appetite proxy |
| `M2_MoM` | M2SL | Liquidity cycle; leading indicator for risk assets |
| `M2_YoY` | M2SL | Structural liquidity trend (>5% = Risk-On bias) |
| `Breakeven_5Y5Y` | T5YIFR | Market-implied long-run inflation expectations |
| `Credit_Spread` | BAMLH0A0HYM2EY | ICE BofA High-Yield OAS; systemic stress barometer |

**Credit Spread (BAMLH0A0HYM2EY — ICE HY OAS):**
The ICE BofA High-Yield Option-Adjusted Spread (OAS) measures the yield premium
that investors demand to hold junk bonds over Treasuries. When credit spreads widen
(e.g. >500 bps), it signals tightening financial conditions, recession risk, and
flight-to-quality behavior — a strongly bearish signal for equities and crypto.

### Tier 2 — Daily Market Indicators
| Feature | Source | Role |
|---------|--------|------|
| `VIX` | CBOE | Implied volatility / fear index; narrows Monte Carlo cloud at low values |
| `DXY` | yfinance | Dollar strength; inverse proxy for Gold and risk assets |
| `Yield_10Y` | yfinance | Cost of capital; high yields compress equity multiples (DCF denominator) |
| `YieldCurve_10Y2Y` | FRED | 10Y minus 2Y spread; negative = yield curve inversion (recession signal) |
| `Oil_Price` | yfinance | Commodity cost pressure; geopolitical stress proxy |
| `GK_Vol_21d` | Computed | Garman-Klass 21-day realized volatility (see below) |

### Yield Curve Regimes
The 10Y-2Y Treasury spread maps directly to economic cycle stages:
- **Steepening (>0.5%)**: Early expansion, risk-on, equities outperform
- **Flat (0 to 0.5%)**: Late cycle, uncertainty, increase hedges
- **Inversion (<0)**: Historically precedes recessions by 6–18 months
- **Post-inversion steepening**: Often coincides with actual recession onset

### M2 Liquidity Cycles
M2 money supply YoY growth is a leading indicator for Bitcoin and Gold:
- **M2 YoY > 5%** → Excess liquidity seeking yield → Risk-On (BTC/Gold bullish)
- **M2 YoY 2–5%** → Neutral / moderate expansion
- **M2 YoY < 2%** → Liquidity contraction → Risk-Off

### Buffett Indicator (Dashboard)
`Total Market Cap / GDP` — Warren Buffett's preferred valuation metric:
- **<100%**: Historically undervalued
- **100–150%**: Fair / moderately elevated
- **>150%**: Overvalued (Buffett historically reduces equity exposure)

---

## Statistical & Quantitative Methods

### Garman-Klass Volatility (GK_Vol_21d)
A superior realized volatility estimator that uses OHLC (Open, High, Low, Close)
prices — capturing intraday range that close-to-close estimators miss:

```
GK = (1/2) × ln(H/L)² − (2ln2 − 1) × ln(C/O)²
```

This provides a statistically more efficient estimate of true volatility with
~5× smaller variance than simple close-to-close returns, making it particularly
useful for assets like Bitcoin with high intraday volatility.

### Dual-Head Ensemble (Ridge Stacker — Upgraded)
Traditional Ridge Regression stackers minimize MSE, creating a tension between
hit ratio (direction accuracy) and magnitude accuracy. The Dual-Head approach
separates these objectives:

```
Head 1 (Direction): LogisticRegressionCV
   Input:  [lstm_pred, xgb_pred, VIX, GK_Vol, Sentiment, YieldCurve, DXY]
   Target: y_dir = 1 if price_7d_later > price_today else 0
   Loss:   Cross-entropy (maximizes directional accuracy)
   class_weight='balanced': corrects UP/DOWN imbalance

Head 2 (Magnitude): HuberRegressor(epsilon=1.35)
   Input:  Same meta-features
   Target: y_pct = (price_7d_later - price_today) / price_today
   Loss:   Huber(δ=1.35) — quadratic within 1.35σ, linear beyond
   → Outlier-robust: crash weeks don't dominate the fit

Combined signal:
   dir_signal  = (direction_prob - 0.5) × 2        # ∈ [-1, +1]
   final_pct   = dir_signal × |magnitude_pred|
```

### LSTM Training: Percentage Change Target
The LSTM is **not** trained to predict absolute prices.
Instead, the target is the 7-day forward percentage change:
```
y_t = (Price_{t+7} - Price_t) / Price_t
```
This target is then standardized via `StandardScaler` (zero mean, unit variance).
A separate `target_scaler.pkl` is saved for inverse-transforming predictions
back to raw % change during inference.

**Why % change?**
Predicting % changes is stationary (prices are not), which dramatically improves
LSTM generalization. The model learns directional momentum patterns rather than
memorizing absolute price levels.

### Geometric Brownian Motion Monte Carlo (Fan Chart)
The probability cloud (P10–P90 band) uses **500 simulated GBM paths**:
```
S(t+1) = S(t) × exp(σε - ½σ²)    where ε ~ N(0,1)
```
The daily volatility `σ` is sourced from live market data:
- **BTC**: Deribit DVOL (Bitcoin's own VIX, implied by BTC options market)
- **Gold/Stocks**: CBOE VIX converted via `σ_daily = IV_annual / sqrt(252)`

This means the cloud width dynamically reflects what institutional options
traders are actually pricing as future uncertainty — not historical RMSE.

### Decaying Momentum Paths (Recursive Rollout)
The `recursive_forecast()` uses adaptive trust-factor decay to prevent
trend extrapolation beyond what the LSTM can reliably predict:
```
trust_factor(i) = max(0.05, 1.0 - i/365.0)    # linear decay over 1 year
ai_delta        = clip(pred_t - prev_price, -0.8%, +0.8%)
ai_movement     = ai_delta × decay × trust_factor
anchor_pull     = (start_price - current_price) × (1 - trust_factor) × 0.01
new_price       = current_price + ai_movement + anchor_pull
```
The anchor progressively dominates at longer horizons, while the AI drives
short-term movements. `decay ∈ {0.99 (gold/stocks), 0.97 (BTC)}`.

### Counterfactual Accuracy Tracking
Every forecast is logged with its date, predicted price series, and the actual
baseline. After the forecast horizon elapses, the system automatically scores
directional accuracy. This builds a live track record visible on the Dashboard's
CEO Layer panel.

### Macro Z-Score Anomaly Detection (XAI)
Each macro indicator is normalized to a rolling Z-Score:
```
Z_t = (X_t - μ_{lookback}) / σ_{lookback}
```
Where `lookback` is calibrated per indicator frequency:
- Monthly FRED data → 36-month window
- Daily market data → 252-day window

Z > 2.0 = statistically anomalous (historical 97.7th percentile),
triggering visual alerts in the Macro Anomaly section.

---

## Project Structure

```
Market-Intelligence/
├── app.py                              # Streamlit entry point (sidebar + routing)
├── run_app.bat                         # One-click launcher (Windows)
├── requirements.txt                    # Python dependencies
│
├── pages/
│   ├── 1_Dashboard.py                  # Portfolio overview + AI batch predictions
│   ├── 2_Gold_Analysis.py              # XAUUSD: macro, forecast, Alpha Engine
│   ├── 3_Bitcoin_Analysis.py           # BTC: halving cycle, DVOL, Alpha Engine
│   ├── 4_Stocks_Analysis.py            # Equities: sector view, Alpha Engine
│   ├── 5_Alternative_Data.py           # Fear & Greed + alternative signals
│   ├── 5_Model_Operations.py           # Live trading signals & AI diagnostics
│   ├── 5_Model_Validation.py           # Walk-forward backtest scorecard
│   ├── 5_Scenario_Simulator.py         # Macro stress-testing (shock injection)
│   └── 7_Settings.py                   # Data sync controls
│
├── scripts/                            # Data pipeline & training
│   ├── data_fetcher_v2.py              # OHLCV + macro merge → per-asset CSVs
│   ├── fred_fetcher.py                 # FRED macro series → fred_indicators.csv
│   ├── sentiment_fetcher_v2.py         # News + FearGreed → sentiment scores
│   ├── fed_watch_fetcher.py            # CME FedWatch rate probabilities
│   ├── google_trends_fetcher.py        # Search interest → alternative signal
│   ├── inject_lag_features.py          # Add CPI_lag3, NFP_lag3, etc. to CSVs
│   ├── backtest_engine.py              # Walk-forward 80/20 validation
│   ├── train_lstm_pct.py               # LSTM training (pct-change target)
│   ├── train_xgboost_macro.py          # XGBoost macro model training
│   ├── train_ridge_stacker.py          # Dual-Head Stacker training
│   └── verify_new_forecast.py          # Quick sanity-check script
│
├── utils/                              # Core library
│   ├── config.py                       # ASSETS dict, feature lists, FORECAST_RANGES
│   ├── predictor.py                    # Orchestrator: AssetPredictor class
│   ├── ui_components.py                # Shared Streamlit components
│   ├── macro_processor.py              # Macro context builder (regime detection)
│   ├── llm_manager.py                  # CEO Layer (Gemini API integration)
│   ├── realtime_prices.py              # Live IV fetch (Deribit DVOL / CBOE VIX)
│   ├── xai_explainer.py                # Z-Score anomaly detection
│   ├── counterfactual_logger.py        # Forecast accuracy logging
│   └── layers/                         # 3-Layer inference sub-modules
│       ├── __init__.py
│       ├── worker_lstm.py              # Level 1: LSTM load + recursive forecast
│       ├── manager_anchor.py           # Level 2: Dual-Head stacker inference
│       └── ceo_layer.py                # Level 3: LLM override interface
│
├── models/                             # Trained model artifacts (Git-ignored)
│   ├── {asset}_model.keras             # Trained LSTM model
│   ├── {asset}_scaler.pkl              # Feature MinMaxScaler
│   ├── {asset}_scaler_target.pkl       # Target StandardScaler (for pct output)
│   ├── {asset}_xgb_macro.json          # XGBoost model
│   ├── {asset}_xgb_scaler.pkl          # XGBoost feature scaler
│   ├── {asset}_xgb_features.json       # Saved feature list for inference alignment
│   ├── {asset}_stacker_direction.pkl   # LogisticRegressionCV
│   ├── {asset}_stacker_magnitude.pkl   # HuberRegressor
│   ├── {asset}_stacker_meta_scaler.pkl # Meta-feature StandardScaler
│   └── {asset}_stacker_meta.json       # Stacker coefficients + backtest metrics
│
├── data/                               # Data warehouse (Git-ignored)
│   ├── gold_data.csv                   # Gold OHLCV + macro merged
│   ├── btc_data.csv                    # Bitcoin OHLCV + macro merged
│   ├── {ticker}_data.csv               # Per-stock merged dataset
│   ├── macro_indicators.csv            # Daily macro: VIX, DXY, Oil, Yield
│   ├── fred_indicators.csv             # Monthly FRED: CPI, PPI, PCE, NFP, M2
│   └── gdp_series.csv                  # GDP for Buffett Indicator
│
├── reports/                            # Backtest outputs
│   ├── backtest_{asset}.json           # Walk-forward accuracy metrics
│   └── stacker_{asset}_backtest.json   # Stacker vs. base model comparison
│
└── tests/
    └── test_core.py                    # Pytest suite (25 tests)
```

---

## Model Training Pipeline

### Full Re-training Sequence (after data sync)
```powershell
# Step 1: Sync all data sources
python scripts/fred_fetcher.py
python scripts/data_fetcher_v2.py
python scripts/sentiment_fetcher_v2.py all

# Step 2: Inject lag features into CSVs
python scripts/inject_lag_features.py

# Step 3: Train base LSTM models (pct-change architecture)
python scripts/train_lstm_pct.py gold
python scripts/train_lstm_pct.py btc
python scripts/train_lstm_pct.py spy
python scripts/train_lstm_pct.py qqq
# ... (add other stock tickers as needed)

# Step 4: Train XGBoost macro models
python scripts/train_xgboost_macro.py gold
python scripts/train_xgboost_macro.py btc
python scripts/train_xgboost_macro.py spy

# Step 5: Train Dual-Head Ensemble Stacker (requires Steps 3+4 complete)
python scripts/train_ridge_stacker.py           # trains gold, btc, spy
# or single asset:
python scripts/train_ridge_stacker.py btc

# Step 6: Run backtest validation (optional)
python scripts/backtest_engine.py gold
python scripts/backtest_engine.py btc
python scripts/backtest_engine.py spy
```

### Training Notes
- Models are trained on **100% of historical data** (production training)
- Stacker is evaluated on **20% chronological hold-out** within its own training
- Stacker requires `{asset}_scaler_target.pkl` (created by `train_lstm_pct.py`)
- XGBoost auto-generates lagged features during inference for backward compatibility

---

## Daily Workflow

```powershell
# Run every day before market analysis (takes ~2-3 minutes)
python scripts/fred_fetcher.py              # monthly FRED (fast, checks for new releases)
python scripts/data_fetcher_v2.py           # daily OHLCV + yfinance macro
python scripts/sentiment_fetcher_v2.py all  # news + fear/greed

# Then launch the terminal
run_app.bat
```

---

## Latest Backtest Performance (May 2026)

### Dual-Head Ensemble (Stacker vs. Base Models)
| Asset | LSTM Hit Ratio | XGBoost Hit Ratio | Direction Head | Combined Ensemble |
|-------|---------------|-------------------|---------------|-------------------|
| **Gold** | 36.6% | 62.2% | 43.6% | **43.6%** |
| **BTC** | 51.2% | 47.3% | 52.7% | **52.7%** |
| **SPY** | 47.7% | 61.0% | 61.6% | **61.6%** |

*Notes:*
- Gold LSTM underperforms because Credit_Spread has limited history (FRED free tier starts 2023).
  XGBoost dominates Gold/SPY where macro features have long history.
- BTC LSTM is stronger than XGBoost — momentum and micro patterns dominate BTC price action.
- SPY stacker successfully leverages both XGBoost macro signals and VIX (fear) dynamics.
- To push hit ratios toward 70%+: re-sync FRED, retrain XGBoost with full Credit_Spread history
  (2010–present via FRED paid tier), and add sentiment features to the stacker meta-matrix.

### Walk-Forward Backtest (80/20 Chronological Split)
| Asset | Test Period | Samples | Hit Ratio | RMSE |
|-------|-------------|---------|-----------|------|
| **Gold** | Feb 2024 → May 2026 | 559 | 41.2% | 68.91 |
| **BTC** | Jan 2024 → May 2026 | 833 | 49.0% | 2,266.99 |
| **SPY** | Feb 2024 → May 2026 | 559 | 42.8% | 7.66 |

*The backtest RMSE represents the Manager Layer output (pct_chain_forecast), which is grounded
by the Dual-Head Stacker signal. Run `python scripts/backtest_engine.py {asset}` to regenerate.*

---

## Disclaimer

This platform is a financial intelligence research tool built for educational and
analytical purposes. AI predictions are probabilistic estimates based on historical
patterns and macroeconomic indicators. They are **not** investment advice.

- Past performance does not guarantee future results
- Hit ratios < 60% indicate significant uncertainty; treat all signals directionally only
- Always apply proper risk management (position sizing, stop-losses)
- The CEO Layer (Gemini) narrative is AI-generated and may contain errors or biases
- Credit Spread and yield curve signals have multi-month lead times; short-term predictions
  are inherently noisier than structural macro analysis

*Built with: TensorFlow/Keras · XGBoost · scikit-learn · Streamlit · Plotly · Pandas · yfinance · FRED API*
