# XAUUSD Core Terminal - AI Gold Prediction Dashboard

A professional-grade financial dashboard for XAUUSD (Gold) built with **Streamlit**, featuring real-time market data, sentiment analysis, and Deep Learning price forecasting.

![Dashboard Preview](https://via.placeholder.com/800x450.png?text=XAUUSD+Core+Terminal+Preview)

## 🚀 Features

- **Real-time Market Metrics**: Tracks Gold (GC=F), DXY (US Dollar Index), VIX (Volatility Index), and TNX (10Y US Treasury Yield).
- **Sentiment Analysis**: Integrated NLP using `TextBlob` to analyze financial news from premium sources (Bloomberg, Reuters, CNBC, etc.).
- **Deep Learning Forecast**: Multi-range price prediction (1D up to 1Y) powered by an **LSTM** neural network.
- **Premium UI**: High-density financial terminal aesthetic with Glassmorphism and Industrial design.
- **On-the-fly Training**: Re-train the AI model directly from the dashboard using your local GPU (RTX 3050).

## 📁 Project Structure

```text
XAUUSD/
├── app.py                  # Main Dashboard UI (Streamlit)
├── data_fetcher.py         # Market Data Extraction (yfinance)
├── sentiment_fetcher.py    # News Sentiment Analysis (NewsAPI + NLP)
├── train_ultimate.py       # LSTM Model Training Script
├── requirements.txt        # Project Dependencies
├── gold_macro_data.csv     # Raw Market Data
├── gold_global_insights.csv # Combined Market + Sentiment Data
├── latest_news.json        # Cached News for Dashboard
├── scaler.pkl              # Normalization Object
└── gold_ultimate_model.h5  # Trained LSTM Model
```

## 🛠 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/XAUUSD-AI-Dashboard.git
   cd XAUUSD-AI-Dashboard
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Update `API_KEY` in `sentiment_fetcher.py` with your NewsAPI key.

5. **Run the Dashboard**:
   ```bash
   streamlit run app.py
   ```

## 🧠 AI Model Logic

The model is an LSTM (Long Short-Term Memory) network designed for time-series forecasting. It processes the last 60 days of market data (Gold, DXY, VIX, Yield, and Sentiment) to predict the next day's closing price. The dashboard then uses recursive logic for longer timeframe forecasts.

## ⚖️ Disclaimer

This project is for educational purposes only. It is not financial advice. Trading gold involves high risk.

---
Created by [Kenzo Darmawan](mailto:kenzodarmawan968@gmail.com)
