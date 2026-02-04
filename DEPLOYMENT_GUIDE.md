#  DEPLOYMENT GUIDE - XAUUSD Multi-Asset Terminal

## 📦 Complete File Structure

```
XAUUSD-Analyzer/
├── 📄 Core Application
│   ├── app.py                      # Main homepage (entry point)
│   ├── requirements.txt            # Python dependencies
│   ├── README.md                   # Full documentation
│   └── .gitignore                  # Git ignore rules
│
├── 📂 pages/                       # Streamlit multi-page structure
│   ├── 1__Dashboard.py           # Portfolio overview & comparison
│   ├── 2__Gold_Analysis.py       # Gold deep dive page
│   ├── 3__Bitcoin_Analysis.py     # Bitcoin + halving cycles
│   ├── 4__Stocks_Analysis.py     # US stocks tracking
│   └── 5__Settings.py            # Data sync & training controls
│
├── 📂 utils/                       # Shared utilities
│   ├── __init__.py                 # Package initializer
│   ├── config.py                   # Global configuration
│   ├── predictor.py                # Unified prediction engine
│   └── ui_components.py            # Reusable UI components
│
├── 🔄 Data Pipeline
│   ├── data_fetcher_v2.py          # Multi-asset market data fetcher
│   └── sentiment_fetcher_v2.py     # News sentiment analyzer
│
└── 🤖 AI Training Scripts
    ├── train_ultimate.py           # Gold model trainer (legacy - still works)
    ├── train_btc.py                # Bitcoin model trainer
    └── train_stocks.py             # Stock models trainer (batch support)
```

---

## 🛠️ SETUP INSTRUCTIONS

### Step 1: GitHub Repository Setup

```bash
# Navigate to your local XAUUSD folder
cd /path/to/your/XAUUSD

# Copy all new files from the outputs
# (Replace with actual path where you downloaded files)

# Initialize git (if not already)
git init

# Add remote (your existing GitHub repo)
git remote add origin https://github.com/ken968/XAUUSD-Analyzer.git

# Create new branch for v2
git checkout -b feature/multi-asset-v2

# Add all files
git add .

# Commit
git commit -m "feat: Multi-asset terminal v2 with BTC and US Stocks support"

# Push to GitHub
git push origin feature/multi-asset-v2
```

### Step 2: Install Dependencies

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Step 3: First Time Setup

```bash
# IMPORTANT: Create utils/__init__.py
mkdir -p utils
touch utils/__init__.py   # On Windows: type nul > utils\__init__.py

# 1. Fetch all market data (takes 5-10 minutes)
python data_fetcher_v2.py

# 2. Analyze sentiment for all assets
python sentiment_fetcher_v2.py all

# 3. Train core models
python train_ultimate.py        # Gold (~3 mins)
python train_btc.py             # Bitcoin (~5 mins)
python train_stocks.py SPY      # S&P 500 (~3 mins)

# 4. (Optional) Train all 11 stocks - RECOMMENDED for full experience
python train_stocks.py ALL      # All stocks (~30 mins)
```

### Step 4: Launch Application

```bash
streamlit run app.py
```

Browser will open at: `http://localhost:8501`

---

## 🔑 IMPORTANT: NewsAPI Configuration

### Get Free API Key
1. Visit: https://newsapi.org/
2. Sign up for free account
3. Copy your API key

### Update Configuration
Edit `utils/config.py`:
```python
# Line 130 - Replace with your key
NEWS_API_KEY = 'your_actual_api_key_here'
```

**Note:** Free tier limits:
- 100 requests/day
- 1 month historical data
- Sufficient for daily syncs

---

##  DATA MANAGEMENT

### Daily Update Routine
```bash
# Morning routine (before market open)
python data_fetcher_v2.py        # Update prices
python sentiment_fetcher_v2.py all   # Fresh sentiment
```

### Weekly Model Retraining
```bash
# Recommended: Sunday evening
python train_ultimate.py         # Gold
python train_btc.py              # Bitcoin
python train_stocks.py ALL       # All stocks (optional)
```

### Storage Requirements
- **Data files (.csv)**: ~50-100 MB total
- **Models (.h5)**: ~5-10 MB per model
- **Total disk space**: ~200-300 MB

---

## 🐛 TROUBLESHOOTING

### Issue: "Module not found: utils"
**Solution:**
```bash
# Make sure utils/__init__.py exists
touch utils/__init__.py

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: Training fails with "Data not found"
**Solution:**
```bash
# Sync data first
python data_fetcher_v2.py
python sentiment_fetcher_v2.py all

# Verify files exist
ls *_global_insights.csv
```

### Issue: "scaler.pkl not found" during prediction
**Solution:**
```bash
# Re-train the specific model
python train_ultimate.py    # For Gold
python train_btc.py         # For Bitcoin
python train_stocks.py SPY  # For specific stock
```

### Issue: Streamlit won't start
**Solution:**
```bash
# Clear cache
rm -rf __pycache__ .streamlit/
streamlit cache clear

# Reinstall streamlit
pip install --upgrade streamlit
```

### Issue: Yahoo Finance download fails
**Solution:**
```bash
# Update yfinance
pip install --upgrade yfinance

# Try manual test
python -c "import yfinance as yf; print(yf.download('GC=F', period='1d'))"
```

---

##  FEATURE TESTING CHECKLIST

After setup, test each feature:

###  Homepage
- [ ] System status shows green for available assets
- [ ] Quick predictions display correctly
- [ ] Market data cards show prices

###  Dashboard Page
- [ ] Asset selector works
- [ ] Comparison chart renders
- [ ] Correlation matrix displays (2+ assets)

###  Gold Analysis
- [ ] Price charts load
- [ ] Correlation charts work
- [ ] Predictions generate
- [ ] News section displays

###  Bitcoin Analysis
- [ ] Full history chart shows
- [ ] Halving timeline displays
- [ ] 90-day predictions work
- [ ] Crypto news loads

###  Stocks Analysis
- [ ] Multi-stock comparison works
- [ ] Individual stock deep dive
- [ ] Batch predictions (if models trained)
- [ ] Sector analysis displays

###  Settings Page
- [ ] System status accurate
- [ ] Data sync buttons work
- [ ] Training buttons functional
- [ ] Progress logs display

---

##  PRODUCTION DEPLOYMENT

### Option 1: Streamlit Cloud (Free)
1. Push code to GitHub
2. Visit: https://streamlit.io/cloud
3. Connect GitHub repository
4. Deploy app
5. **Note:** Model files are too large - train locally and exclude from Git

### Option 2: Local Server (Recommended)
```bash
# Run as service (Linux)
streamlit run app.py --server.port 8501 --server.headless true

# Or use PM2 (Node.js process manager)
pm2 start "streamlit run app.py" --name xauusd-terminal
```

### Option 3: Docker (Advanced)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

---

##  PERFORMANCE OPTIMIZATION

### Speed Up Training
- Use GPU: Install `tensorflow-gpu`
- Reduce epochs for faster training
- Train individual models in parallel

### Reduce Memory Usage
- Delete old `.csv` files after training
- Clear Streamlit cache regularly
- Use smaller batch sizes

### Faster Data Loading
- Cache data with `@st.cache_data`
- Load models once with `@st.cache_resource`
- Limit historical data range

---

##  NEXT STEPS & ROADMAP

### Phase 1: Current Features ( Complete)
-  Multi-asset support (Gold, BTC, 11 stocks)
-  Multi-page UI
-  LSTM predictions
-  Sentiment analysis

### Phase 2: Enhancements (Planned)
- [ ] Options flow integration
- [ ] Backtest simulator
- [ ] Alert system (email/Telegram)
- [ ] Portfolio optimizer

### Phase 3: Advanced Features (Future)
- [ ] More assets (commodities, forex)
- [ ] Transformer models (BERT for sentiment)
- [ ] Real-time WebSocket data
- [ ] Mobile app version

---

## 📞 SUPPORT

### Documentation
- Full README: `README.md`
- Inline code comments in all `.py` files
- Docstrings in functions

### Community
- GitHub Issues: Report bugs or request features
- Discussions: Share ideas and improvements

### Contact
- GitHub: @ken968
- Project: https://github.com/ken968/XAUUSD-Analyzer

---

## 📝 VERSION HISTORY

### v2.0.0 (Current)
- Multi-asset support: Gold, Bitcoin, 11 US Stocks
- Multi-page Streamlit app
- Dedicated pages per asset class
- Batch training for all stocks
- Unified prediction engine
- Enhanced UI with correlation matrix

### v1.0.0 (Previous)
- Single-page Gold analysis
- Basic LSTM predictions
- Simple sentiment analysis

---

## ⚖️ LICENSE

MIT License - See LICENSE file for details

---

<div align="center">

**🎉 Setup Complete! Launch the app and explore! 🎉**

```bash
streamlit run app.py
```

</div>
