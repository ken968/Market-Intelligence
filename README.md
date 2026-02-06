# Market Intelligence

> **AI-Powered Financial Intelligence Platform**  
> Deep learning predictions for Gold, Bitcoin, and US Equities with real-time sentiment analysis.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)

---

## Features

### Multi-Asset Coverage
- **Gold (XAUUSD)**: Precious metal analysis with macro correlations.
- **Bitcoin**: Full history (2009-present) with halving cycle tracking.
- **US Stocks**: 11 equities including S&P 500, Mag7, and semiconductors.

### AI-Powered Predictions
- **LSTM Deep Learning**: 3-layer neural networks optimized per asset type.
- **Multi-Range Forecasts**: 1 day to 1 year predictions.
- **Recursive Forecasting**: Advanced prediction chaining.

### Comprehensive Analysis
- **Technical**: Price charts, trend analysis, volatility tracking.
- **Fundamental**: Sentiment analysis from 10+ trusted news sources.
- **Macro Correlation**: DXY, VIX, Treasury Yields integration.

### Professional UI
- **Multi-Page App**: Dedicated pages for each asset class.
- **Real-Time Data**: Yahoo Finance + NewsAPI integration.
- **Interactive Charts**: Plotly visualizations with dark theme.

---

## Quick Start

### Prerequisites
```bash
# Python 3.8 or higher
python --version

# Install dependencies
pip install -r requirements.txt
```

### Installation
```bash
# Clone repository
git clone https://github.com/ken968/Market-Intelligence.git
cd Market-Intelligence

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### First Run Setup
```bash
# 1. Sync market data
python scripts/data_fetcher_v2.py

# 2. Analyze sentiment
python scripts/sentiment_fetcher_v2.py all

# 3. Train core models (Gold + BTC + SPY)
python scripts/train_ultimate.py        # Gold
python scripts/train_btc.py             # Bitcoin
python scripts/train_stocks.py SPY      # S&P 500

# 4. (Optional) Train all 11 stocks
python scripts/train_stocks.py ALL      # Takes 25-35 minutes

# 5. Launch app
streamlit run app.py
```

---

## Project Structure

```
Market-Intelligence/
├── app.py                      # Main homepage
├── pages/                      # Multi-page application views
│   ├── 1_Dashboard.py
│   ├── 2_Gold_Analysis.py
│   ├── 3_Bitcoin_Analysis.py
│   ├── 4_Stocks_Analysis.py
│   └── 5_Settings.py
├── scripts/                    # Core logic and training scripts
│   ├── data_fetcher_v2.py
│   ├── sentiment_fetcher_v2.py
│   ├── train_ultimate.py
│   ├── train_btc.py
│   └── train_stocks.py
├── utils/                      # Helper components and config
│   ├── config.py
│   ├── predictor.py
│   └── ui_components.py
├── data/                       # Historical and sentiment data
├── models/                     # Trained LSTM models
└── requirements.txt            # Project dependencies
```

---

## Model Architecture

### Gold Model
- **Architecture**: LSTM (100-50-25 units)
- **Sequence Length**: 60 days
- **Features**: Gold, DXY, VIX, Yield, Sentiment
- **Training Data**: 10 years
- **Epochs**: 30

### Bitcoin Model
- **Architecture**: LSTM (128-64-32 units) + higher dropout
- **Sequence Length**: 90 days
- **Features**: BTC, DXY, VIX, Yield, Sentiment, Halving Cycle
- **Training Data**: 2009-present
- **Epochs**: 50

### Stock Models
- **Architecture**: LSTM (100-50-25 units)
- **Sequence Length**: 60 days
- **Features**: Price, DXY, VIX, Yield, Sentiment
- **Training Data**: 10 years
- **Epochs**: 30

---

## Supported Assets

### Precious Metals
- **GC=F** - Gold Futures (XAUUSD)

### Cryptocurrency
- **BTC-USD** - Bitcoin

### Market Indices
- **SPY** - S&P 500 ETF
- **QQQ** - Nasdaq 100 ETF
- **DIA** - Dow Jones ETF

### Magnificent 7
- **AAPL** - Apple Inc.
- **MSFT** - Microsoft Corp.
- **GOOGL** - Alphabet Inc.
- **AMZN** - Amazon.com Inc.
- **NVDA** - NVIDIA Corp.
- **META** - Meta Platforms
- **TSLA** - Tesla Inc.

### Semiconductors
- **TSM** - Taiwan Semiconductor

---

## Configuration

### API Keys
Edit `utils/config.py` to update NewsAPI key:
```python
NEWS_API_KEY = 'your_api_key_here'
```

### Hyperparameters
Modify training scripts in `scripts/` to adjust model parameters.

---

## UI Navigation

### Homepage
- System status overview
- Quick predictions for core assets
- Latest market data

### Dashboard Page
- Multi-asset performance comparison
- Portfolio correlation matrix
- Batch predictions

### Gold Analysis
- XAUUSD price charts
- DXY/VIX correlation analysis
- Multi-range AI forecasts
- News sentiment

### Bitcoin Analysis
- Full history visualization (2009+)
- Halving cycle timeline
- Macro correlation analysis
- Crypto news feed

### Stocks Analysis
- Multi-stock comparison charts
- Sector performance tracking
- Individual stock deep dives
- Batch predictions for all stocks

---

## Troubleshooting

### Model Training Fails
- Ensure data is synced first using scripts in `scripts/`.
- Check if CSV files exist in `data/`.

### Prediction Errors
- Clear Streamlit cache and restart.
- Ensure models are trained and saved in `models/`.

---

## Future Enhancements
- [ ] Options flow analysis
- [ ] Backtest simulator
- [ ] Alert system (price/sentiment triggers)
- [ ] Portfolio optimizer
- [ ] Additional assets (commodities, forex)

---

## License
MIT License - see [LICENSE](LICENSE) file for details

---

## Contact
**Project Maintainer**: Ken968  
**GitHub**: [https://github.com/ken968/Market-Intelligence](https://github.com/ken968/Market-Intelligence)

---

## Acknowledgments
- Yahoo Finance for market data
- NewsAPI for news aggregation
- TensorFlow/Keras for deep learning
- Streamlit for the dashboard UI
- Plotly for interactive charts

---

<div align="center">

**Disclaimer: seringkan/perbanyak sync data & sentiment dan train model untuk awal-awal. stocks forecast saat ini sangat sulit karna paling complex**

Built with Python, TensorFlow, and Streamlit

</div>
