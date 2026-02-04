# XAUUSD Multi-Asset Terminal 

> **AI-Powered Financial Intelligence Platform**  
> Deep Learning predictions for Gold, Bitcoin, and US Equities with real-time sentiment analysis

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)

---

## 🌟 Features

### Multi-Asset Coverage
- ** Gold (XAUUSD)**: Precious metal analysis with macro correlations
- ** Bitcoin**: Full history (2009-present) with halving cycle tracking
- ** US Stocks**: 11 equities including S&P 500, Mag7, and semiconductors

### AI-Powered Predictions
- **LSTM Deep Learning**: 3-layer neural networks optimized per asset type
- **Multi-Range Forecasts**: 1 day to 1 year predictions
- **Recursive Forecasting**: Advanced prediction chaining

### Comprehensive Analysis
- **Technical**: Price charts, trend analysis, volatility tracking
- **Fundamental**: Sentiment analysis from 10+ trusted news sources
- **Macro Correlation**: DXY, VIX, Treasury Yields integration

### Professional UI
- **Multi-Page App**: Dedicated pages for each asset class
- **Real-Time Data**: Yahoo Finance + NewsAPI integration
- **Interactive Charts**: Plotly visualizations with dark theme

---

##  Quick Start

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
python data_fetcher_v2.py

# 2. Analyze sentiment
python sentiment_fetcher_v2.py all

# 3. Train core models (Gold + BTC + SPY)
python train_ultimate.py        # Gold
python train_btc.py             # Bitcoin
python train_stocks.py SPY      # S&P 500

# 4. (Optional) Train all 11 stocks
python train_stocks.py ALL      # Takes 25-35 minutes

# 5. Launch app
streamlit run app.py
```

---

## 📁 Project Structure

```
Market-Intelligence/
├── app.py                      # Main homepage
├── pages/
│   ├── 1__Dashboard.py       # Multi-asset overview
│   ├── 2__Gold_Analysis.py   # Gold deep dive
│   ├── 3__Bitcoin_Analysis.py # Bitcoin + halving cycles
│   ├── 4__Stocks_Analysis.py # US equities tracking
│   └── 5__Settings.py        # Control panel
├── utils/
│   ├── config.py               # Global configuration
│   ├── predictor.py            # Unified prediction engine
│   └── ui_components.py        # Reusable UI elements
├── data_fetcher_v2.py          # Multi-asset data downloader
├── sentiment_fetcher_v2.py     # News sentiment analyzer
├── train_ultimate.py           # Gold model trainer
├── train_btc.py                # Bitcoin model trainer
├── train_stocks.py             # Stock model trainer
└── requirements.txt            # Python dependencies
```

---

## 🤖 Model Architecture

### Gold Model
- **Architecture**: LSTM (100-50-25 units)
- **Sequence Length**: 60 days
- **Features**: Gold, DXY, VIX, Yield, Sentiment
- **Training Data**: 10 years
- **Epochs**: 30

### Bitcoin Model
- **Architecture**: LSTM (128-64-32 units) + higher dropout
- **Sequence Length**: 90 days (captures longer cycles)
- **Features**: BTC, DXY, VIX, Yield, Sentiment, **Halving Cycle**
- **Training Data**: 2009-present (full history)
- **Epochs**: 50

### Stock Models
- **Architecture**: LSTM (100-50-25 units)
- **Sequence Length**: 60 days
- **Features**: Price, DXY, VIX, Yield, Sentiment
- **Training Data**: 10 years
- **Epochs**: 30

---

##  Supported Assets

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

## 🔧 Configuration

### API Keys
Edit `utils/config.py` to update NewsAPI key:
```python
NEWS_API_KEY = 'your_api_key_here'
```

Get free API key: [https://newsapi.org/](https://newsapi.org/)

### Hyperparameters
Modify training scripts to adjust model parameters:
- `train_ultimate.py` - Gold
- `train_btc.py` - Bitcoin
- `train_stocks.py` - Stocks

---

##  Usage Examples

### Sync Data for Specific Asset
```bash
# Gold only
python data_fetcher_v2.py gold

# Bitcoin only
python data_fetcher_v2.py btc

# Single stock
python data_fetcher_v2.py NVDA

# All stocks
python data_fetcher_v2.py stocks
```

### Train Specific Models
```bash
# Train Gold
python train_ultimate.py

# Train Bitcoin
python train_btc.py

# Train single stock
python train_stocks.py AAPL

# Train all 11 stocks (batch)
python train_stocks.py ALL
```

### Sentiment Analysis
```bash
# All assets
python sentiment_fetcher_v2.py all

# Specific asset
python sentiment_fetcher_v2.py btc
python sentiment_fetcher_v2.py nvda
```

---

## 🎨 UI Navigation

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

### Settings Page
- Data synchronization controls
- Model training interface
- System maintenance tools

---

##  Data Sources

### Market Data
- **Yahoo Finance API** (`yfinance`)
  - Historical OHLC prices
  - Macro indicators (DXY, VIX, US10Y)

### News & Sentiment
- **NewsAPI** 
  - Trusted sources: Bloomberg, Reuters, WSJ, CNBC, etc.
  - NLP analysis via TextBlob

---

##  Disclaimer

**This software is for educational purposes only.**

- ❌ Not financial advice
- ❌ Not investment recommendations
- ❌ Past performance ≠ future results

**High-Risk Assets:**
- Gold and stocks involve significant volatility
- Bitcoin is extremely volatile
- AI predictions are probabilistic, not guaranteed

**Always:**
- Conduct your own research (DYOR)
- Consult licensed financial advisors
- Only invest what you can afford to lose

---

## 🛠️ Troubleshooting

### Model Training Fails
```bash
# Ensure data is synced first
python data_fetcher_v2.py
python sentiment_fetcher_v2.py all

# Check if CSV files exist
ls *_global_insights.csv
```

### Prediction Errors
```bash
# Clear cache and restart
rm -rf __pycache__
streamlit cache clear
```

### Data Download Issues
- Check internet connection
- Verify Yahoo Finance is accessible
- Try syncing individual assets instead of all at once

---

##  Future Enhancements

- [ ] Options flow analysis
- [ ] Backtest simulator
- [ ] Alert system (price/sentiment triggers)
- [ ] Portfolio optimizer
- [ ] Additional assets (commodities, forex)
- [ ] Transformer models (BERT/GPT for sentiment)

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📧 Contact

**Project Maintainer**: Ken968  
**GitHub**: [https://github.com/ken968/Market-Intelligence](https://github.com/ken968/Market-Intelligence)

---

## 🙏 Acknowledgments

- Yahoo Finance for market data API
- NewsAPI for news aggregation
- TensorFlow/Keras for deep learning framework
- Streamlit for rapid UI development
- Plotly for interactive visualizations

---

<div align="center">

**⭐ Star this repo if you find it useful!**

Built with ❤️ using Python, TensorFlow, and Streamlit

</div>
