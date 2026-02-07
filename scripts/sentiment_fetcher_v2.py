import requests
import pandas as pd
import json
import os
import sys
from textblob import TextBlob
from datetime import datetime

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
API_KEY = 'cb548b26fc6542c0a6bb871ef3786eba'
TRUSTED_DOMAINS = (
    "bloomberg.com,reuters.com,cnbc.com,wsj.com,finance.yahoo.com,"
    "investing.com,marketwatch.com,economist.com,ft.com,coindesk.com,cointelegraph.com,"
    "businessinsider.com,forbes.com"
)

# Asset-specific search queries
ASSET_QUERIES = {
    'gold': '(XAUUSD OR "Gold Price" OR "Gold Futures" OR "Fed Rate" OR "US Inflation")',
    'btc': '(Bitcoin OR BTC OR "Crypto Market" OR "Bitcoin Price" OR "Cryptocurrency")',
    'stocks': '("Stock Market" OR "S&P 500" OR "Nasdaq" OR "Fed Rate" OR "Wall Street")',
    
    # Individual stocks (optional: can customize per stock)
    'aapl': '(Apple OR AAPL OR iPhone OR "Tim Cook")',
    'nvda': '(Nvidia OR NVDA OR "AI Chips" OR "GPU Market")',
    'tsla': '(Tesla OR TSLA OR "Elon Musk" OR "Electric Vehicle")',
    'msft': '(Microsoft OR MSFT OR "Cloud Computing" OR Azure)',
    'googl': '(Google OR Alphabet OR GOOGL OR "Search Engine")',
    'amzn': '(Amazon OR AMZN OR AWS OR "E-commerce")',
    'meta': '(Meta OR Facebook OR META OR Instagram)',
    'tsm': '(TSMC OR TSM OR "Semiconductor" OR "Chip Manufacturing")',
    'spy': '("S&P 500" OR "SP500" OR "Stock Market Index" OR "S&P Index")',
    'qqq': '(Nasdaq OR QQQ OR "Tech Stocks" OR "Invesco QQQ")',
    'dia': '("Dow Jones" OR DIA OR "Blue Chip Stocks" OR "Industrial Average")'
}

# Blacklist for non-financial noise
BLACKLIST = [
    'wwe', 'wrestling', 'netflix', 'drama', 'movie', 'sport', 'olympic', 
    'medals', 'celebrity', 'gossip', 'entertainment', 'gaming'
]


def get_sentiment(text):
    """Calculates polarity score using TextBlob."""
    if not text: 
        return 0
    return TextBlob(text).sentiment.polarity


def fetch_news_sentiment(asset='gold', max_articles=15):
    """
    Fetch and analyze financial news sentiment for specific asset.
    NOW USING MULTI-SOURCE AGGREGATOR (Yahoo RSS + optional Finnhub/AlphaVantage)
    
    Args:
        asset (str): Asset type ('gold', 'btc', 'stocks', or ticker symbol)
        max_articles (int): Maximum articles to save for display
    
    Returns:
        pd.DataFrame: Daily sentiment scores
    """
    
    # NEW: Use multi-source aggregator instead of NewsAPI
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    
    try:
        from sentiment_sources.aggregator import SentimentAggregator
        
        asset_lower = asset.lower()
        
        print(f"System: Analyzing {asset.upper()} sentiment from multiple sources...")
        
        aggregator = SentimentAggregator()
        sentiment_df = aggregator.fetch_all(asset_lower, days=30)
        
        if sentiment_df.empty:
            print(f"System: No articles found for {asset}.")
            return pd.DataFrame()
        
        # Save news summary for dashboard
        news_file = f'data/latest_news_{asset_lower}.json'
        try:
            summary_data = [{
                'date': str(datetime.now().date()),
                'title': f'Sentiment from {len(aggregator.get_source_names())} sources',
                'description': f'Multi-source sentiment analysis for {asset.upper()}',
                'url': '#',
                'sentiment': float(sentiment_df['Sentiment'].mean())
            }]
            
            with open(news_file, 'w') as f:
                json.dump(summary_data, f, indent=4)
        except Exception as e:
            pass  # Non-critical
        
        print(f"System: Days with data: {len(sentiment_df)}, Non-zero: {(sentiment_df['Sentiment'] != 0).sum()}")
        return sentiment_df
        
    except Exception as e:
        print(f"Error with sentiment aggregator: {e}")
        return pd.DataFrame()


def integrate_sentiment(asset='gold'):
    """
    Integrate sentiment data with macro data.
    
    Args:
        asset (str): 'gold', 'btc', or stock ticker
    """
    
    asset_lower = asset.lower()
    
    # Determine file names based on asset type
    if asset_lower == 'gold':
        macro_file = 'data/gold_macro_data.csv'
        output_file = 'data/gold_global_insights.csv'
    elif asset_lower in ['btc', 'bitcoin']:
        macro_file = 'data/btc_macro_data.csv'
        output_file = 'data/btc_global_insights.csv'
    else:
        # Stock ticker
        ticker = asset.upper()
        macro_file = f'data/{ticker}_macro_data.csv'
        output_file = f'data/{ticker}_global_insights.csv'
    
    # Check if macro data exists
    if not os.path.exists(macro_file):
        print(f"Error: '{macro_file}' missing.")
        print(f"Run: python data_fetcher_v2.py {asset_lower}")
        return False
    
    # Fetch sentiment
    sentiment_df = fetch_news_sentiment(asset)
    
    if sentiment_df.empty:
        print(f"Warning: No sentiment data found for {asset}.")
        print("Creating insights file with zero sentiment...")
        macro_df = pd.read_csv(macro_file)
        macro_df['Sentiment'] = 0
        macro_df.to_csv(output_file, index=False)
        print(f"System: '{output_file}' created with placeholder sentiment.")
        return True
    
    # Merge with macro data
    macro_df = pd.read_csv(macro_file)
    macro_df['Date'] = pd.to_datetime(macro_df['Date']).dt.strftime('%Y-%m-%d')
    
    final_df = pd.merge(macro_df, sentiment_df, on='Date', how='left')
    final_df['Sentiment'] = final_df['Sentiment'].fillna(0)
    
    final_df.to_csv(output_file, index=False)
    print(f"System: '{output_file}' updated successfully.")
    print(f"        Total records: {len(final_df)}")
    print(f"        With sentiment: {final_df['Sentiment'].ne(0).sum()}")
    
    return True


def process_all_assets():
    """Process sentiment for all major assets"""
    assets = ['gold', 'btc', 'SPY', 'NVDA', 'AAPL']  # Core assets
    
    print("\n" + "="*60)
    print("MULTI-ASSET SENTIMENT ANALYSIS")
    print("="*60)
    
    results = {}
    for asset in assets:
        print(f"\n--- Processing {asset.upper()} ---")
        try:
            success = integrate_sentiment(asset)
            results[asset] = "SUCCESS" if success else "FAILED"
        except Exception as e:
            print(f"Error processing {asset}: {e}")
            results[asset] = "FAILED"
    
    print("\n" + "="*60)
    print("SENTIMENT SYNC SUMMARY")
    print("="*60)
    for asset, status in results.items():
        print(f"{asset.upper():6s}: {status}")
    print("="*60)
    
    return all(s == "SUCCESS" for s in results.values())


if __name__ == "__main__":
    from utils.config import STOCK_TICKERS
    
    if len(sys.argv) > 1:
        asset = sys.argv[1]
        
        if asset.lower() == 'all':
            success = process_all_assets()
            if not success:
                sys.exit(1)
        elif asset.lower() == 'stocks':
            print("\n--- Processing ALL STOCKS ---")
            all_success = True
            for ticker in STOCK_TICKERS.keys():
                if not integrate_sentiment(ticker):
                    all_success = False
            if not all_success:
                sys.exit(1)
        else:
            success = integrate_sentiment(asset)
            if not success:
                sys.exit(1)
    else:
        # Default: Gold only (backward compatibility)
        print("Usage: python sentiment_fetcher_v2.py [gold|btc|stocks|AAPL|...|all]")
        print("\nRunning default: Gold sentiment analysis...")
        success = integrate_sentiment('gold')
        if not success:
            sys.exit(1)
