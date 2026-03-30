"""
Geopolitical & Regulatory News Fetcher
Filters Tiingo for geopolitical and high-impact regulatory keywords.
Scores text using FinBERT.
"""

import os
import requests
from typing import List, Dict
from datetime import datetime, timedelta
import urllib.parse

try:
    from sentiment_sources.base_fetcher import BaseSentimentFetcher
except ImportError:
    from base_fetcher import BaseSentimentFetcher

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from utils.finbert_analyzer import get_finbert_sentiment
except ImportError:
    def get_finbert_sentiment(text):
        return 0.0

from dotenv import load_dotenv
load_dotenv()


class GeopoliticalFetcher(BaseSentimentFetcher):
    """Fetches geopolitical and crypto regulatory news."""
    
    def __init__(self):
        super().__init__("Geopolitical & Regulatory")
        self.tiingo_api_key = os.getenv("TIINGO_API_KEY")
        self.keywords = [
            "war", "sanctions", "embargo", "tariff", 
            "Middle East", "Russia", "China tradeoff",
            "SEC lawsuit", "MiCA", "crypto ban", "crypto regulation"
        ]

    def fetch_news(self, asset: str, days: int = 30) -> List[Dict]:
        """
        Fetch geopolitical and regulatory news.
        """
        articles = []
        if not self.tiingo_api_key:
            return articles
            
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        start_str = start_date.strftime('%Y-%m-%d')
        
        # Tiingo allows searching across all news
        query = " OR ".join(f'"{kw}"' for kw in self.keywords)
        query_encoded = urllib.parse.quote(query)
        
        url = f"https://api.tiingo.com/tiingo/news?query={query_encoded}&startDate={start_str}&limit=30"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Token {self.tiingo_api_key}'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for item in data:
                    title = item.get('title', '')
                    description = item.get('description', '')
                    
                    full_text = f"{title}. {description}"
                    sentiment = get_finbert_sentiment(full_text)
                    
                    date_str = item.get('publishedDate', '').split('T')[0]
                    if not date_str:
                        date_str = datetime.utcnow().strftime('%Y-%m-%d')
                        
                    articles.append({
                        'title': title,
                        'url': item.get('url', ''),
                        'date': date_str,
                        'sentiment': sentiment,
                        'source': 'Tiingo Geopolitics'
                    })
        except Exception as e:
            print(f"Error fetching Tiingo geopolitical news: {e}")
            
        print(f"System: Fetched and scored {len(articles)} geopolitical/regulatory events.")
        return articles

if __name__ == "__main__":
    fetcher = GeopoliticalFetcher()
    data = fetcher.fetch_news('btc', 7)
    for d in data[:5]:
        print(f"{d['date']}: {d['sentiment']} -> {d['title']}")
