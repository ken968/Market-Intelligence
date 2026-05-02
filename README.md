# Market Intelligence 2.0

> **Hierarchical Financial Intelligence & Macro Simulation Engine**
> Advanced 3-Layer AI architecture (Worker, Manager, CEO) for Gold, Bitcoin, and Equities. Features include macro-grounded Monte Carlo simulations and rigorous walk-forward backtesting.

---

## 3-Layer Intelligence Architecture

The platform separates pattern recognition from structural market anchoring and macro-level contextual bias:

1. **Worker Layer (LSTM)**: Handles raw technical pattern recognition and sequence learning.
2. **Manager Layer (Dynamic Anchoring)**: Implements multi-scale anchoring (90-day vs historical) and recursive damping to prevent trend hallucinations and "mean-reversion bias."
3. **CEO Layer (LLM Contextual Bias)**: Modulates the forecast drift based on real-world macro indicators (M2 Money Supply, Yield Curves, Geopolitical Sentiment).

---

## Model Validation & Backtesting

Every model in the system is validated using a strict 80/20 chronological train-test split. The **Model Operations** suite allows for the execution of walk-forward validation to compare raw metrics against the refined 3-layer system.

### Latest Validation Performance
| Asset | Model Type | Hit Ratio | RMSE (Price Error) | Improvement |
|---|---|---|---|---|
| **BTC** | 3-Level Hierarchy | 49.4% | 2,264 | 94% Error Reduction |
| **Gold** | 3-Level Hierarchy | 41.8% | 70.7 | 95% Error Reduction |
| **SPY** | 3-Level Hierarchy | 42.2% | 7.4 | 89% Error Reduction |

*Note: The Root Mean Square Error (RMSE) shows a massive reduction when transitioning from technical-only (LSTM) to the 3-Layer Hierarchy, proving the effectiveness of macro-grounding.*

---

## Monte Carlo Fan Charts (Uncertainty Visualization)

Forecasts are visualized using Institutional-Grade Probabilistic Fan Charts based on Geometric Brownian Motion (GBM). The spread of the fan is grounded in **Live Implied Volatility** (using Deribit DVOL proxies for crypto and CBOE VIX for equities) via a Black-Scholes framework, combined with 500-path Monte Carlo simulations, providing a highly realistic, market-driven view of uncertainty.

---

## Project Structure

```
Market-Intelligence/
├── app.py                          # Main entry point
├── pages/
│   ├── 1_Dashboard.py              # Portfolio overview
│   ├── 2_Gold_Analysis.py          # XAUUSD deep-dive
│   ├── 3_Bitcoin_Analysis.py       # BTC-USD analysis & cycles
│   ├── 4_Stocks_Analysis.py        # Equity & sector tracking
│   ├── 5_Model_Operations.py       # Validation room & Backtesting
│   ├── 6_Scenario_Simulator.py     # Macro stress-testing
│   └── 7_Settings.py               # Data sync controls
├── scripts/
│   ├── backtest_engine.py          # Walk-forward validation engine
│   ├── data_fetcher_v2.py          # Global macro data sync
│   ├── train_ultimate.py           # Model training
│   └── fred_fetcher.py             # FRED economic data provider
├── utils/
│   ├── config.py                   # Centralized configuration
│   ├── predictor.py                # 3-Layer inference engine
│   ├── realtime_prices.py          # Live IV and Black-Scholes volatility fetcher
│   └── llm_manager.py              # CEO Layer (Gemini Integration)
├── models/                         # Trained .keras model files
├── reports/                        # Validation charts and JSON metrics
├── tests/                          # Automated testing and experimental scripts
└── data/                           # Macro data warehouse (CSV)
```

---

## Quick Start

```bash
git clone https://github.com/ken968/Market-Intelligence.git
cd Market-Intelligence

python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows

pip install -r requirements.txt

# Sync data and run validation
python scripts/data_fetcher_v2.py
python scripts/backtest_engine.py btc

# Launch Dashboard
streamlit run app.py
```

---

## Disclaimer

This platform is a financial intelligence research tool. AI predictions are probabilistic and based on historical data. Use forecasts for directional guidance only. Always apply proper risk management systems.

---

Built with Python, TensorFlow, and Streamlit.
