import yfinance as yf
import pandas as pd
import os

def fetch_global_gold_data():
    """Fetches gold and macro-economic correlations from Yahoo Finance."""
    tickers = {
        'Gold': 'GC=F',
        'USD_Index': 'DX-Y.NYB',
        'VIX': '^VIX',
        'US_10Y_Yield': '^TNX'
    }
    
    print("System: Fetching market data...")
    # Period 10y for long-term cycle awareness
    data = yf.download(list(tickers.values()), period="10y", interval="1d")
    
    if data.empty:
        print("Error: No data retrieved from yfinance.")
        return

    df = data['Close'].ffill()
    
    # Map ticker symbols to readable column names
    df.columns = [tickers[col] if col in tickers else col for col in df.columns]
    df = df.rename(columns={
        'GC=F': 'Gold',
        'DX-Y.NYB': 'DXY',
        '^VIX': 'VIX',
        '^TNX': 'Yield_10Y'
    })

    df = df.dropna()
    df.to_csv("gold_macro_data.csv")
    print(f"System: Success. {len(df)} records saved to 'gold_macro_data.csv'.")

if __name__ == "__main__":
    fetch_global_gold_data()