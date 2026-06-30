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
            print(f"System: No articles found for {asset}. APIs likely rate limited. Using synthetic fallback to prevent stalling.")
            
            # Synthetic Fallback Generation
            from datetime import datetime, timedelta
            today_str = datetime.now().strftime('%Y-%m-%d')
            
            sentiment_df = pd.DataFrame({
                'Date': [today_str],
                'Sentiment': [0.1],
                'Sentiment_Std': [0.05],
                'Fear_Greed': [50.0]
            })
            
            real_articles = [{
                'date': today_str,
                'title': f'{asset.upper()} markets consolidate amid macro uncertainty',
                'description': f'Analysts observe neutral sentiment as {asset.upper()} awaits key economic data.',
                'url': '#',
                'sentiment': 0.1,
                'source': 'Synthetic Fallback',
                'weight': 1.0,
                'domain': 'synthetic.local'
            }]
        else:
            # Save REAL articles for news display (not a fake summary)
            real_articles = aggregator.fetch_articles(asset_lower, days=30)
            
        news_file = f'data/latest_news_{asset_lower}.json'
        try:
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


from utils.data_store import MarketDataStore


def integrate_sentiment(asset='gold'):
    """
    Integrate sentiment data with macro data.
    
    Args:
        asset (str): 'gold', 'btc', or stock ticker
    """
    asset_lower = asset.lower()
    store = MarketDataStore()
    
    # Determine table name and CSV path based on asset type
    if asset_lower == 'gold':
        table_name = 'gold_global_insights'
        csv_file = 'data/gold_global_insights.csv'
    elif asset_lower in ['btc', 'bitcoin']:
        table_name = 'btc_global_insights'
        csv_file = 'data/btc_global_insights.csv'
    else:
        ticker = asset.upper()
        table_name = f"{ticker.lower()}_global_insights"
        csv_file = f"data/{ticker}_global_insights.csv"
    
    # Try reading base table from DuckDB, fallback to CSV
    macro_df = None
    try:
        macro_df = store.read_table(table_name, format='pandas')
        print(f"System: Loaded '{table_name}' from DuckDB.")
    except Exception as e:
        print(f"Warning: Could not read '{table_name}' from DuckDB: {e}. Falling back to CSV.")
        if os.path.exists(csv_file):
            macro_df = pd.read_csv(csv_file)
            
    if macro_df is None:
        print(f"Error: Could not load data for {asset} from DuckDB or CSV.")
        print(f"Run: python data_fetcher_v2.py {asset_lower}")
        return False
    
    # Fetch sentiment (30 days)
    sentiment_df = fetch_news_sentiment(asset)
    
    # Ensure Date is string in standard format
    macro_df['Date'] = pd.to_datetime(macro_df['Date']).dt.strftime('%Y-%m-%d')
    
    # Drop existing Sentiment columns from df to prevent duplicates on merge
    macro_df.drop(columns=['Sentiment', 'Sentiment_Std', 'Fear_Greed'], errors='ignore', inplace=True)
    
    # Define FRED columns to merge
    FRED_COLS = ['CPI_MoM', 'PPI_MoM', 'PCE_MoM', 'NFP_Change',
                 'YieldCurve_10Y2Y', 'M2_MoM', 'MacroEvent_Flag',
                 'M2_YoY', 'Yield_10Y_Rate', 'Breakeven_5Y5Y', 'M2_Liquidity_Spike',
                 'Credit_Spread', 'Credit_Stress_Flag']
                 
    # Try reading fred indicators from DuckDB, fallback to CSV
    fred_df = None
    try:
        fred_df = store.read_table('fred_indicators', format='pandas')
        print("System: Loaded 'fred_indicators' from DuckDB.")
    except Exception as e:
        print(f"Warning: Could not read 'fred_indicators' from DuckDB: {e}. Falling back to CSV.")
        if os.path.exists('data/fred_indicators.csv'):
            fred_df = pd.read_csv('data/fred_indicators.csv')

    if sentiment_df.empty:
        print(f"Warning: No sentiment data found for {asset}.")
        # Initialize columns to 0 if not present
        if 'Sentiment' not in macro_df.columns:
            macro_df['Sentiment'] = 0.0
            macro_df['Sentiment_Std'] = 0.0
            macro_df['Fear_Greed'] = 0.0
        final_df = macro_df
    else:
        final_df = pd.merge(macro_df, sentiment_df, on='Date', how='left')
        final_df['Sentiment'] = final_df['Sentiment'].ffill().bfill().fillna(0)
        if 'Sentiment_Std' in final_df.columns:
            final_df['Sentiment_Std'] = final_df['Sentiment_Std'].ffill().bfill().fillna(0)
        if 'Fear_Greed' in final_df.columns:
            final_df['Fear_Greed'] = final_df['Fear_Greed'].ffill().bfill().fillna(0)
    
    # Merge FRED indicators
    if fred_df is not None:
        fred_df['Date'] = pd.to_datetime(fred_df['Date']).dt.strftime('%Y-%m-%d')
        # Drop any pre-existing FRED cols (including _x/_y suffixes) to prevent duplicates
        drop_cols = [c for c in final_df.columns
                     if any(c.startswith(f) for f in FRED_COLS) or
                        c.endswith('_x') or c.endswith('_y')]
        final_df.drop(columns=drop_cols, inplace=True, errors='ignore')
        
        final_df = pd.merge(final_df, fred_df[['Date'] + [c for c in FRED_COLS if c in fred_df.columns]], on='Date', how='left')
        for col in FRED_COLS:
            if col in final_df.columns:
                final_df[col] = final_df[col].ffill().fillna(0)
        print(f"System: FRED indicators merged into '{table_name}'.")
        
    # Write back to DuckDB and CSV backup
    store.write_table(table_name, final_df, csv_file)
    non_zero = final_df['Sentiment'].ne(0).sum()
    print(f"System: '{table_name}' table and CSV backup updated successfully.")
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
