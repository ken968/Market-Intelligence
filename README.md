# Market Intelligence 2.0

> **Macro Stress-Testing & AI Financial Intelligence Engine**  
> Advanced LSTM forecasts for Gold, Bitcoin, and Equities integrated with a multi-factor sentiment aggregator and macro shock simulator.

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)

---

## Overview

Market Intelligence 2.0 is more than a price predictor. It is a decision-support system that analyzes how geopolitics, macroeconomics (DXY, VIX, Oil), and sentiment data interact to move asset prices — and lets you run your own stress-tests against that model.

---

## 🟢 Dashboard Page
The command center for your portfolio. It tracks normalized performance and AI predictions in real-time.

<img src="docs/assets/1_dashboardA.png" width="100%">
<br>
<img src="docs/assets/2_dashboardB.png" width="100%">

---

## 🔴 Scenario Simulator (What-If Analysis)
Inject custom macro shocks (Oil jumps, DXY spikes, Fear index) and observe how the LSTM model recalculates its 30-day forecast under stress conditions. We also calculate sensitivity metrics to see how much an asset reacts to your shock.

<p align="center">
  <img src="docs/assets/3_scenarioA.png" width="50%">
  <br><br>
  <img src="docs/assets/4_scenarioB.png" width="100%">
  <br><br>
  <img src="docs/assets/5_scenarioC.png" width="100%">
</p>

- 🔵 Baseline: AI forecast using current live market data
- 🔴 Stress-Test: AI forecast after your macro shock is injected

---

## 🔵 Sentiment Analysis & Price Correlation
View price action overlayed with weighted sentiment leads. The core aggregation layer filters noise from financial markets:
- **Weighted Sentiment**: Geopolitics (2.5x), On-chain (1.5x), X/Twitter (0.8x).
- **Echo Chamber Fix**: Strips articles with >85% semantic overlap.

<img src="docs/assets/6_sentimenanalisA.png" width="100%">
<br>
<img src="docs/assets/7_sentimenanalisB.png" width="100%">

---

## 🟡 Alternative Data Intelligence
Data sources beyond the standard order book to give an edge on underlying institutional flows.

<img src="docs/assets/8_alternatifA.png" width="100%">
<br>
<img src="docs/assets/9_alternatifB.png" width="100%">

- **Fed Watch**: Market-implied FOMC rate hike/cut probabilities.
- **Google Trends**: Real-time retail search interest tracking.
- **Geopolitical Pulse**: Aggregated risk index from conflict zones and trade data.

---

## 🟢 Project Structure

```
Market-Intelligence/
├── app.py                          # Main entry point
├── pages/
│   ├── 1_Dashboard.py              # Multi-asset overview & correlation
│   ├── 2_Gold_Analysis.py          # XAUUSD deep-dive + sentiment
│   ├── 3_Bitcoin_Analysis.py       # Halving cycle + on-chain metrics
│   ├── 4_Stocks_Analysis.py        # Equity & sector analysis
│   ├── 5_Scenario_Simulator.py     # Macro stress-testing engine
│   ├── 5_Alternative_Data.py       # Fed Watch, Trends, Geopolitics
│   ├── 6_Trading_Signals.py        # Multi-factor entry/exit signals
│   └── 7_Settings.py               # Data sync & model training
├── scripts/
│   ├── aggregator.py               # Weighted sentiment aggregation
│   ├── data_fetcher_v2.py          # Market & macro data sync
│   ├── train_ultimate.py           # Gold LSTM training
│   ├── train_btc.py                # Bitcoin LSTM training
│   └── train_stocks.py             # Equities LSTM training
├── utils/
│   ├── config.py                   # Asset configuration
│   ├── predictor.py                # LSTM inference & recursive forecast
│   ├── signal_generator.py         # Multi-factor signal logic
│   └── ui_components.py            # Shared UI components
├── models/                         # Trained .keras model files
└── data/                           # Normalized CSV data warehouse
```

---

## Quick Start

```bash
git clone https://github.com/ken968/Market-Intelligence.git
cd Market-Intelligence

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

# Sync market data
python scripts/data_fetcher_v2.py

# Train core models
python scripts/train_ultimate.py
python scripts/train_btc.py
python scripts/train_stocks.py SPY

# Launch
streamlit run app.py
```

---

## Model Architecture

| Asset | LSTM Units | Sequence | Epochs | Key Features |
|---|---|---|---|---|
| Gold | 100-50-25 | 60 days | 30 | Price, DXY, VIX, Yield, Oil, Sentiment |
| Bitcoin | 128-64-32 | 90 days | 50 | Price, DXY, VIX, Halving Cycle, On-chain, Sentiment |
| Stocks | 100-50-25 | 60 days | 30 | Price, DXY, VIX, Yield, Sentiment |

All models use recursive forecasting: each prediction step becomes input for the next, allowing multi-day projections with a single trained model.

---

## Covered Assets

**Precious Metals**: GC=F (Gold Futures)  
**Crypto**: BTC-USD (Bitcoin)  
**Indices**: SPY, QQQ, DIA  
**Mag 7**: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA  
**Semiconductors**: TSM

---

## Status Indicators

🟢 System operational / bullish signal  
🔵 AI prediction active / info  
🟡 Warning / neutral signal  
🔴 Error / bearish signal / stress condition

---

## Notes

- Sync data and sentiment frequently, especially during high-volatility periods
- Retrain models weekly to incorporate the latest market data
- Bitcoin model requires full history (2009-present) for cycle-aware predictions
- Stocks forecast is the most complex — sector context matters

---

## Disclaimer

This platform is a market intelligence tool, not financial advice. All trading involves risk. AI predictions are probabilistic. Always apply proper risk management.

---

<div align="center">

Built with Python, TensorFlow, and Streamlit

</div>
