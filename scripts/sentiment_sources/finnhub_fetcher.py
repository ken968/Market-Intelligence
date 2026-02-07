"""
Finnhub Sentiment Fetcher
Free tier: 60 calls/minute, company news
"""

import requests
from datetime import datetime, timedelta
from textblob import TextBlob
from typing import List, Dict
from .base_fetcher import BaseSentimentFetcher


class FinnhubFetcher(BaseSentimentFetcher):
    """Fetch news and sentiment from Finnhub API"""
    
    def __init__(self, api_key: str):
        super().__init__('Finnhub')
        self.api_key = api_key
        self.base_url = 'https://finnhub.io/api/v1'
    
    def fetch_news(self, asset: str, days: int = 30) -> List[Dict]:
        """
        Fetch company news from Finnhub
        
        Args:
            asset: Asset ticker (e.g., 'AAPL', 'BTC')
            days: Number of days to look back
        
        Returns:
            List of articles with sentiment scores
        """
        # Map crypto to supported symbols
        ticker_map = {
            'btc': 'BINANCE:BTCUSDT',  # Crypto trading pair
            'gold': 'OANDA:XAU_USD'    # Gold CFD
        }
        
        symbol = ticker_map.get(asset.lower(), asset.upper())
        
        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        params = {
            'symbol': symbol,
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d'),
            'token': self.api_key
        }
        
        try:
            response = requests.get(
                f'{self.base_url}/company-news',
                params=params,
                timeout=10
            )
            
            if response.status_code != 200:
                print(f"  {self.source_name}: API error {response.status_code}")
                return []
            
            articles = response.json()
            
            if not articles:
                print(f"  {self.source_name}: No articles found for {asset}")
                return []
            
            results = []
            for article in articles[:30]:  # Limit to 30
                try:
                    # Analyze sentiment
                    text = f"{article.get('headline', '')} {article.get('summary', '')}"
                    sentiment = TextBlob(text).sentiment.polarity
                    
                    pub_date = datetime.fromtimestamp(article.get('datetime', 0))
                    
                    results.append({
                        'title': article.get('headline', ''),
                        'url': article.get('url', ''),
                        'date': pub_date.strftime('%Y-%m-%d'),
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
