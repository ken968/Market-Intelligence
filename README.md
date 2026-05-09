# Market Intelligence 2.0

> **Hierarchical Financial Intelligence & Macro Simulation Engine**
> Advanced 3-Layer AI architecture (Worker, Manager, CEO) for Gold, Bitcoin, and Equities. Features include macro-grounded Monte Carlo simulations and rigorous walk-forward backtesting.

---

## 3-Layer Intelligence Architecture

The platform separates pattern recognition from structural market anchoring and macro-level contextual bias:

1. **Worker Layer (LSTM)**: Handles raw technical pattern recognition and sequence learning.
2. **Manager Layer (Dynamic Anchoring)**: Implements multi-scale anchoring (90-day vs historical) and recursive damping to prevent trend hallucinations and "mean-reversion bias."
3. **CEO Layer (LLM Contextual Bias)**: Modulates the forecast drift based on real-world macro indicators (M2 Money Supply, Yield Curves, Geopolitical Sentiment) using Gemini API.

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
│   ├── 5_Alternative_Data.py       # Fear & Greed Index + Alternative Signals
│   ├── 5_Model_Operations.py       # Trading Signals & AI Diagnostics
│   ├── 5_Model_Validation.py       # Walk-Forward Backtesting Scorecard
│   ├── 5_Scenario_Simulator.py     # Macro stress-testing
│   └── 7_Settings.py               # Data sync controls
├── scripts/
│   ├── backtest_engine.py          # Walk-forward validation engine
│   ├── data_fetcher_v2.py          # Global macro data & prices sync
│   ├── sentiment_fetcher_v2.py     # News & sentiment sync
│   ├── train_ultimate.py           # Model training for Gold (XAUUSD)
│   ├── train_btc.py                # Model training for Bitcoin
│   └── train_stocks.py             # Batch Model training for US Equities
├── utils/
│   ├── config.py                   # Centralized configuration
│   ├── predictor.py                # 3-Layer inference engine
│   ├── realtime_prices.py          # Live IV and Black-Scholes volatility fetcher
│   ├── xai_explainer.py            # Macro Z-Score Anomaly detection
│   └── llm_manager.py              # CEO Layer (Gemini Integration)
├── models/                         # Trained .keras model files
├── reports/                        # Validation charts and JSON metrics
└── data/                           # Macro data warehouse (CSV) - Excluded from Git
```

---

## Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/ken968/Market-Intelligence.git
cd Market-Intelligence
```

**2. Setup Virtual Environment & Install Dependencies**
```bash
# On Windows PowerShell
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

**3. Setup Environment Variables**
Create a `.env` file in the root directory and add the necessary API keys:
```env
GEMINI_API_KEY=your_gemini_key_here
NEWSAPI_KEY=your_newsapi_key_here
FINNHUB_API_KEY=your_finnhub_key_here
```

---

## Operating Instructions

### 1. Synchronize Data (Daily)
Run these commands to pull the latest closing prices, FRED macro data, and news sentiment. This should be done daily to keep the Dashboard up to date.
*(Alternatively, you can click "Synchronize Data" from the Streamlit Dashboard).*
```bash
# Fetch prices and FRED Macro Data
python scripts/data_fetcher_v2.py

# Fetch Global News Sentiment
python scripts/sentiment_fetcher_v2.py all
```

### 2. Model Training (Weekly/Monthly)
Train the AI "Worker" (LSTM) models using 100% of the historical data. This ensures the models learn the newest market regimes.
```bash
# Train Gold AI Model
python scripts/train_ultimate.py

# Train Bitcoin AI Model
python scripts/train_btc.py

# Train US Stocks (SPY, QQQ, AAPL, etc.)
# You can append a specific ticker (e.g. 'spy') or 'all'
python scripts/train_stocks.py all
```

### 3. Model Validation & Backtesting (Diagnostic)
Run a strict 80/20 chronological train-test split walk-forward backtest. This trains a completely isolated model on 80% of data, and tests its predictive accuracy on the remaining 20% of unseen data.
```bash
# Backtest Gold
python scripts/backtest_engine.py gold

# Backtest Bitcoin
python scripts/backtest_engine.py btc

# Backtest SPY
python scripts/backtest_engine.py spy
```
*Results will automatically populate in the **Model Validation** page of the Dashboard.*

---

## Latest Backtest Performance (May 2026)
| Asset | Test Period | Samples | Hit Ratio | RMSE (3-Level) | RMSE Improvement vs LSTM |
|---|---|---|---|---|---|
| **Gold** | Feb 2024 → May 2026 | 559 | 41.2% | 68.91 | -95.8% |
| **BTC** | Jan 2024 → May 2026 | 833 | 49.0% | 2,266.99 | -94.8% |
| **SPY** | Feb 2024 → May 2026 | 559 | 42.8% | 7.66 | -95.3% |

*Note: The Root Mean Square Error (RMSE) shows a massive 90%+ reduction when transitioning from technical-only (LSTM) to the 3-Layer Hierarchy, proving the Manager Layer successfully prevents hallucinated price spikes.*

---

## Launch the Terminal

```bash
streamlit run app.py
```
*Or simply double-click `run_app.bat` on Windows.*

---

## Disclaimer

This platform is a financial intelligence research tool. AI predictions are probabilistic and based on historical data. Use forecasts for directional guidance only. Always apply proper risk management systems.
