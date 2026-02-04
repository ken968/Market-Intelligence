import requests
import pandas as pd
import json
import os
import sys
from textblob import TextBlob
from datetime import datetime

# Configuration
API_KEY = 'cb548b26fc6542c0a6bb871ef3786eba'
TRUSTED_DOMAINS = (
    "bloomberg.com,reuters.com,cnbc.com,wsj.com,finance.yahoo.com,"
    "investing.com,marketwatch.com,economist.com,ft.com,coindesk.com,cointelegraph.com"
)

# Asset-specific search queries
ASSET_QUERIES = {
    'gold': '(XAUUSD OR "Gold Price" OR "Gold Futures" OR "Fed Rate" OR "US Inflation")',
    'btc': '(Bitcoin OR BTC OR "Crypto Market" OR "Bitcoin Price" OR "Cryptocurrency")',
    'stocks': '("Stock Market" OR "S&P 500" OR "Nasdaq" OR "Fed Rate" OR "Wall Street")',
    
    # Individual stocks (optional: can customize per stock)
    'AAPL': '(Apple OR AAPL OR iPhone OR "Tim Cook")',
    'NVDA': '(Nvidia OR NVDA OR "AI Chips" OR "GPU Market")',
    'TSLA': '(Tesla OR TSLA OR "Elon Musk" OR "Electric Vehicle")',
    'MSFT': '(Microsoft OR MSFT OR "Cloud Computing" OR Azure)',
    'GOOGL': '(Google OR Alphabet OR GOOGL OR "Search Engine")',
    'AMZN': '(Amazon OR AMZN OR AWS OR "E-commerce")',
    'META': '(Meta OR Facebook OR META OR Instagram)',
    'TSM': '(TSMC OR TSM OR "Semiconductor" OR "Chip Manufacturing")',
    'SPY': '("S&P 500" OR SPY OR "Market Index")',
    'QQQ': '(Nasdaq OR QQQ OR "Tech Stocks")',
    'DIA': '("Dow Jones" OR DIA OR "Blue Chip")'
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
    
    Args:
        asset (str): Asset type ('gold', 'btc', 'stocks', or ticker symbol)
        max_articles (int): Maximum articles to save for display
    
    Returns:
        pd.DataFrame: Daily sentiment scores
    """
    
    asset_lower = asset.lower()
    query = ASSET_QUERIES.get(asset_lower, ASSET_QUERIES.get(asset.upper(), ASSET_QUERIES['stocks']))
    
    url = (
        f'https://newsapi.org/v2/everything?'
        f'q={query}&'
        f'domains={TRUSTED_DOMAINS}&'
        f'language=en&'
        f'sortBy=publishedAt&'
        f'pageSize=100&'
        f'apiKey={API_KEY}'
    )
    
    print(f"System: Analyzing {asset.upper()} sentiment from trusted sources...")
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if data.get('status') != 'ok':
            print(f"Error: {data.get('message', 'Unknown error')}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error connecting to NewsAPI: {e}")
        return pd.DataFrame()
    
    articles = data.get('articles', [])
    
    if not articles:
        print(f"System: No articles found for {asset}.")
        return pd.DataFrame()
    
    news_data = []
    display_news = []
    
    for art in articles:
        title = art.get('title', "") or ""
        desc = art.get('description', "") or ""
        full_text = (title + " " + desc).lower()
        
        # Skip blacklisted content
        if any(bad_word in full_text for bad_word in BLACKLIST):
            continue
        
        date = art.get('publishedAt', "")[:10]
        score = get_sentiment(full_text)
        source_name = art.get('source', {}).get('name', "Unknown")
        
        news_data.append({
            'Date': date, 
            'Sentiment': score, 
            'Source': source_name, 
            'Title': title
        })
        
        # Save for dashboard
        if len(display_news) < max_articles:
            display_news.append({
                'date': date,
                'title': title,
                'description': desc,
                'url': art.get('url', '#'),
                'sentiment': score
            })
    
    if not news_data:
        print(f"System: No relevant news found for {asset} after filtering.")
        return pd.DataFrame()
    
    # Save news for dashboard
    news_file = f'latest_news_{asset_lower}.json'
    with open(news_file, 'w') as f:
        json.dump(display_news, f, indent=4)
    print(f"System: {len(display_news)} articles saved to '{news_file}'")
    
    df_news = pd.DataFrame(news_data)
    print(f"System: Processed {len(df_news)} {asset} articles.")
    
    # Calculate daily mean sentiment
    daily_sentiment = df_news.groupby('Date')['Sentiment'].mean().reset_index()
    
    return daily_sentiment


def integrate_sentiment(asset='gold'):
    """
    Integrate sentiment data with macro data.
    
    Args:
        asset (str): 'gold', 'btc', or stock ticker
    """
    
    asset_lower = asset.lower()
    
    # Determine file names based on asset type
    if asset_lower == 'gold':
        macro_file = 'gold_macro_data.csv'
        output_file = 'gold_global_insights.csv'
    elif asset_lower in ['btc', 'bitcoin']:
        macro_file = 'btc_macro_data.csv'
        output_file = 'btc_global_insights.csv'
    else:
        # Stock ticker
        ticker = asset.upper()
        macro_file = f'{ticker}_macro_data.csv'
        output_file = f'{ticker}_global_insights.csv'
    
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
            results[asset] = "" if success else "❌"
        except Exception as e:
            print(f"Error processing {asset}: {e}")
            results[asset] = "❌"
    
    print("\n" + "="*60)
    print("SENTIMENT SYNC SUMMARY")
    print("="*60)
    for asset, status in results.items():
        print(f"{asset.upper():6s}: {status}")
    print("="*60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        asset = sys.argv[1]
        
        if asset.lower() == 'all':
            process_all_assets()
        else:
            integrate_sentiment(asset)
    else:
        # Default: Gold only (backward compatibility)
        print("Usage: python sentiment_fetcher_v2.py [gold|btc|AAPL|...|all]")
        print("\nRunning default: Gold sentiment analysis...")
        integrate_sentiment('gold')
