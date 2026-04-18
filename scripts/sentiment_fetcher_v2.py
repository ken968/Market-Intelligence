import requests
import pandas as pd
import json
import os
import sys
from textblob import TextBlob
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
API_KEY = os.getenv('NEWSAPI_KEY')
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
        
        # Save REAL articles for news display (not a fake summary)
        news_file = f'data/latest_news_{asset_lower}.json'
        try:
            real_articles = aggregator.fetch_articles(asset_lower, days=30)
            news_data = []
            for a in real_articles[:20]:  # Save latest 20 articles
                news_data.append({
                    'date': str(a.get('date', datetime.now().date())),
                    'title': a.get('title', 'No title'),
                    'description': a.get('description', a.get('title', '')),
                    'url': a.get('url', '#'),
                    'sentiment': float(a.get('sentiment', 0)),
                    'source': a.get('source', 'Unknown')
                })
            with open(news_file, 'w') as f:
                json.dump(news_data, f, indent=4)
            print(f"System: Saved {len(news_data)} real articles to '{news_file}'")
        except Exception as e:
            print(f"Warning: Could not save news articles: {e}")
        
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
    
    # Fetch sentiment (60 days for better coverage)
    sentiment_df = fetch_news_sentiment(asset)
    
    # Merge with macro data
    macro_df = pd.read_csv(macro_file)
    macro_df['Date'] = pd.to_datetime(macro_df['Date']).dt.strftime('%Y-%m-%d')
    
    if sentiment_df.empty:
        print(f"Warning: No sentiment data found for {asset}.")
        # Keep existing sentiment if available, else use 0
        if os.path.exists(output_file):
            existing = pd.read_csv(output_file)
            if 'Sentiment' in existing.columns and existing['Sentiment'].ne(0).any():
                print(f"Keeping existing sentiment data.")
                # Still update FRED columns if missing or stale
                fred_file = 'data/fred_indicators.csv'
                FRED_COLS = ['CPI_MoM', 'PPI_MoM', 'PCE_MoM', 'NFP_Change',
                             'YieldCurve_10Y2Y', 'M2_MoM', 'MacroEvent_Flag',
                             'M2_YoY', 'Yield_10Y_Rate', 'Breakeven_5Y5Y', 'M2_Liquidity_Spike']
                needs_update = True  # Forced update to include M2_YoY, Yield_10Y_Rate, Breakeven_5Y5Y
                if needs_update:
                    fred_df = pd.read_csv(fred_file, index_col=0, parse_dates=True)
                    fred_df.index = fred_df.index.strftime('%Y-%m-%d')
                    fred_df.index.name = 'Date'
                    fred_df = fred_df.reset_index()
                    # Drop old/duplicate FRED cols before merging
                    drop_cols = [c for c in existing.columns
                                 if any(c.startswith(f) for f in FRED_COLS) or
                                    c.endswith('_x') or c.endswith('_y')]
                    existing.drop(columns=drop_cols, inplace=True, errors='ignore')
                    existing = pd.merge(existing, fred_df, on='Date', how='left')
                    for col in FRED_COLS:
                        if col in existing.columns:
                            existing[col] = existing[col].ffill().fillna(0)
                    existing.to_csv(output_file, index=False)
                    print(f"System: FRED indicators patched into existing '{output_file}'.")
                return True
        macro_df['Sentiment'] = 0
        macro_df.to_csv(output_file, index=False)
        return True
    
    final_df = pd.merge(macro_df, sentiment_df, on='Date', how='left')
    
    # FIX: Use ffill+bfill so ALL historical rows get non-zero sentiment
    # This spreads the fetched sentiment backwards/forwards to fill gaps
    final_df['Sentiment'] = final_df['Sentiment'].ffill().bfill().fillna(0)
    
    # Merge FRED Tier 1 indicators (CPI, PPI, PCE, NFP, YieldCurve, M2) if available
    fred_file = 'data/fred_indicators.csv'
    FRED_COLS = ['CPI_MoM', 'PPI_MoM', 'PCE_MoM', 'NFP_Change',
                 'YieldCurve_10Y2Y', 'M2_MoM', 'MacroEvent_Flag',
                 'M2_YoY', 'Yield_10Y_Rate', 'Breakeven_5Y5Y', 'M2_Liquidity_Spike']
    if os.path.exists(fred_file):
        fred_df = pd.read_csv(fred_file, index_col=0, parse_dates=True)
        fred_df.index = fred_df.index.strftime('%Y-%m-%d')
        fred_df.index.name = 'Date'
        fred_df = fred_df.reset_index()
        # Drop any pre-existing FRED cols (including _x/_y suffixes) to prevent duplicates
        drop_cols = [c for c in final_df.columns
                     if any(c.startswith(f) for f in FRED_COLS) or
                        c.endswith('_x') or c.endswith('_y')]
        final_df.drop(columns=drop_cols, inplace=True, errors='ignore')
        final_df = pd.merge(final_df, fred_df, on='Date', how='left')
        for col in FRED_COLS:
            if col in final_df.columns:
                final_df[col] = final_df[col].ffill().fillna(0)
        print(f"System: FRED indicators merged into '{output_file}'.")
    
    final_df.to_csv(output_file, index=False)
    non_zero = final_df['Sentiment'].ne(0).sum()
    print(f"System: '{output_file}' updated successfully.")
    print(f"        Total records: {len(final_df)}, With sentiment: {non_zero} ({100*non_zero//len(final_df)}%)")
    
    return True


def process_all_assets():
    """Process sentiment for ALL assets including all stocks"""
    from utils.config import STOCK_TICKERS
    
    # Core assets + all configured stocks
    core_assets = ['gold', 'btc']
    stock_assets = list(STOCK_TICKERS.keys())  # SPY, QQQ, DIA, AAPL, MSFT, etc.
    all_assets = core_assets + stock_assets
    
    print("\n" + "="*60)
    print("MULTI-ASSET SENTIMENT ANALYSIS - ALL ASSETS")
    print(f"Processing {len(all_assets)} assets...")
    print("="*60)
    
    results = {}
    for asset in all_assets:
        print(f"\n--- Processing {asset.upper()} ---")
        try:
            success = integrate_sentiment(asset)
            results[asset] = "SUCCESS" if success else "FAILED"
        except Exception as e:
            print(f"Error processing {asset}: {e}")
            results[asset] = f"FAILED ({e})"
    
    print("\n" + "="*60)
    print("SENTIMENT SYNC SUMMARY")
    print("="*60)
    for asset, status in results.items():
        print(f"{asset.upper():6s}: {status}")
    print("="*60)
    
    return all('SUCCESS' in s for s in results.values())


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
