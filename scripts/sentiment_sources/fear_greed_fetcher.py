"""
Fear & Greed Index Fetcher
Fetches the Crypto Fear & Greed Index from Alternative.me (free, no API key).
Normalizes the 0-100 score to -1.0 to 1.0 to match sentiment polarity.
"""

import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict

try:
    from sentiment_sources.base_fetcher import BaseSentimentFetcher
except ImportError:
    from base_fetcher import BaseSentimentFetcher


class FearGreedFetcher(BaseSentimentFetcher):
    """Fetches the Crypto Fear & Greed Index."""
    
    def __init__(self):
        super().__init__("Fear & Greed Index")
        self.endpoint = "https://api.alternative.me/fng/"
    
    def fetch_news(self, asset: str, days: int = 30) -> List[Dict]:
        """
        Fetch historical Fear & Greed Index.
        Note: The F&G Index is mostly relevant for Crypto (BTC), so if asset 
        is completely un-crypto (like SPY), we could technically return empty, 
        but as a general proxy for market sentiment, it's very useful.
        For now, we fetch it if asset is btc/bitcoin/crypto, and maybe others.
        """
        asset_lower = asset.lower()
        if asset_lower not in ['btc', 'bitcoin', 'crypto'] and asset_lower not in ['gold', 'spy', 'qqq']:
            # We can still fetch it, but let's just allow it for major assets if needed, 
            # though it's crypto-specific. Let's fetch it for BTC prominently.
            pass
        
        url = f"{self.endpoint}?limit={days}"
        articles = []
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'data' not in data:
                print("Warning: F&G API didn't return data array.")
                return []
                
            for item in data['data']:
                try:
                    timestamp = int(item['timestamp'])
                    date_str = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d')
                    
                    value = float(item['value'])
                    classification = item['value_classification']
                    
                    # Normalize 0-100 to -1.0 to 1.0
                    normalized_sentiment = (value - 50.0) / 50.0
                    
                    articles.append({
                        'title': f"Fear & Greed: {value} ({classification})",
                        'url': 'https://alternative.me/crypto/fear-and-greed-index/',
                        'date': date_str,
                        'sentiment': normalized_sentiment,
                        'source': self.source_name
                    })
                except Exception as e:
                    print(f"Error parsing F&G item: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error fetching Fear & Greed Index: {e}")
            
        print(f"System: Fetched {len(articles)} days of Fear & Greed Index.")
        return articles


if __name__ == "__main__":
    fetcher = FearGreedFetcher()
    data = fetcher.fetch_news('btc', 5)
    for d in data:
        print(d)
