import requests
import pandas as pd
import json
import os
from textblob import TextBlob
from datetime import datetime

# Configuration
API_KEY = 'cb548b26fc6542c0a6bb871ef3786eba'
TRUSTED_DOMAINS = (
    "bloomberg.com,reuters.com,cnbc.com,wsj.com,finance.yahoo.com,"
    "investing.com,marketwatch.com,economist.com,ft.com"
)

def get_sentiment(text):
    """Calculates polarity score using TextBlob."""
    if not text: return 0
    return TextBlob(text).sentiment.polarity

def fetch_news_sentiment():
    """Fetches and analyzes financial news sentiment from top-tier sources."""
    query = '(XAUUSD OR "Gold Price" OR "Gold Futures" OR "Fed Rate" OR "US Inflation")'
    url = (
        f'https://newsapi.org/v2/everything?'
        f'q={query}&'
        f'domains={TRUSTED_DOMAINS}&'
        f'language=en&'
        f'sortBy=publishedAt&'
        f'apiKey={API_KEY}'
    )
    
    print(f"System: Analyzing sentiment from {TRUSTED_DOMAINS.split(',')[0]} and others...")
    
    try:
        response = requests.get(url)
        data = response.json()
        if data.get('status') != 'ok':
            print(f"Error: {data.get('message')}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error connecting to NewsAPI: {e}")
        return pd.DataFrame()

    articles = data.get('articles', [])
    news_data = []
    display_news = []
    
    # Blacklist for non-financial context
    blacklist = ['wwe', 'wrestling', 'netflix', 'drama', 'movie', 'sport', 'olympic', 'medals']
    
    for art in articles:
        title = art.get('title', "") or ""
        desc = art.get('description', "") or ""
        full_text = (title + " " + desc).lower()
        
        if any(bad_word in full_text for bad_word in blacklist):
            continue
            
        date = art.get('publishedAt', "")[:10]
        score = get_sentiment(full_text)
        source_name = art.get('source', {}).get('name', "Unknown")
        
        news_data.append({'Date': date, 'Sentiment': score, 'Source': source_name, 'Title': title})
        
        # Prepare for Dashboard (top 15)
        if len(display_news) < 15:
            display_news.append({
                'date': date,
                'title': title,
                'description': desc,
                'url': art.get('url', '#'),
                'sentiment': score
            })

    if not news_data:
        print("System: No relevant news found today.")
        return pd.DataFrame()

    # Save news for Dashboard rendering
    with open('latest_news.json', 'w') as f:
        json.dump(display_news, f, indent=4)

    df_news = pd.DataFrame(news_data)
    print(f"System: Processed {len(df_news)} articles.")
    
    # Calculate daily mean sentiment
    daily_sentiment = df_news.groupby('Date')['Sentiment'].mean().reset_index()
    return daily_sentiment

if __name__ == "__main__":
    # Integration with Macro Data
    if not os.path.exists('gold_macro_data.csv'):
        print("Error: 'gold_macro_data.csv' missing. Run data_fetcher.py first.")
        exit()

    macro_df = pd.read_csv('gold_macro_data.csv')
    sentiment_df = fetch_news_sentiment()
    
    if not sentiment_df.empty:
        macro_df['Date'] = pd.to_datetime(macro_df['Date']).dt.strftime('%Y-%m-%d')
        final_df = pd.merge(macro_df, sentiment_df, on='Date', how='left').fillna(0)
        final_df.to_csv('gold_global_insights.csv', index=False)
        print("System: Global Insights database updated successfully.")
    else:
        print("System: No new sentiment data found to update.")