"""
Alpha Vantage Sentiment Fetcher
Free tier: 25 calls/day, market news and sentiment
"""

import requests
from textblob import TextBlob
from datetime import datetime
from typing import List, Dict
from .base_fetcher import BaseSentimentFetcher


class AlphaVantageFetcher(BaseSentimentFetcher):
    """Fetch news and sentiment from Alpha Vantage API"""
    
    def __init__(self, api_key: str):
        super().__init__('Alpha Vantage')
        self.api_key = api_key
        self.base_url = 'https://www.alphavantage.co/query'
    
    def fetch_news(self, asset: str, days: int = 30) -> List[Dict]:
        """
        Fetch news sentiment from Alpha Vantage
        
        Args:
            asset: Asset ticker (e.g., 'AAPL', 'BTC')
            days: Number of days to look back
        
        Returns:
            List of articles with sentiment scores
        """
        # Map assets to tickers
        ticker_map = {
            'btc': 'CRYPTO:BTC',
            'gold': 'FOREX:XAU'
        }
        
        ticker = ticker_map.get(asset.lower(), asset.upper())
        
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': ticker,
            'limit': 50,  # Max articles
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=15)
            
            if response.status_code != 200:
                print(f"  {self.source_name}: API error {response.status_code}")
                return []
            
            data = response.json()
            
            if 'feed' not in data:
                print(f"  {self.source_name}: No data for {asset}")
                return []
            
            results = []
            for article in data['feed'][:30]:
                try:
                    # Alpha Vantage provides sentiment score, but we'll use TextBlob for consistency
                    text = f"{article.get('title', '')} {article.get('summary', '')}"
                    sentiment = TextBlob(text).sentiment.polarity
                    
                    # Parse date
                    pub_date = article.get('time_published', '')[:10]
                    if pub_date:
                        # Format: YYYYMMDD -> YYYY-MM-DD
                        pub_date = f"{pub_date[:4]}-{pub_date[4:6]}-{pub_date[6:8]}"
                    else:
                        pub_date = datetime.now().strftime('%Y-%m-%d')
                    
                    results.append({
                        'title': article.get('title', ''),
                        'url': article.get('url', ''),
                        'date': pub_date,
                        'sentiment': sentiment,
                        'source': self.source_name
                    })
                except Exception as e:
                    continue
            
            print(f"  {self.source_name}: Fetched {len(results)} articles for {asset}")
            return results
            
        except Exception as e:
            print(f"  {self.source_name}: Error fetching {asset} - {e}")
            return []
