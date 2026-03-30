"""
Macro Economic News Fetcher
Fetches economic calendar events via FMP API and macro headlines via Tiingo News API.
Scores text using FinBERT.
"""

import os
import requests
from typing import List, Dict
from datetime import datetime, timedelta

try:
    from sentiment_sources.base_fetcher import BaseSentimentFetcher
except ImportError:
    from base_fetcher import BaseSentimentFetcher

# Use global FinBERT instance
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from utils.finbert_analyzer import get_finbert_sentiment
except ImportError:
    def get_finbert_sentiment(text):
        from textblob import TextBlob
        return TextBlob(text).sentiment.polarity

from dotenv import load_dotenv
load_dotenv()


class MacroNewsFetcher(BaseSentimentFetcher):
    """Fetches macro-economic news and events."""
    
    def __init__(self):
        super().__init__("Macro News")
        self.fmp_api_key = os.getenv("FMP_API_KEY")
        self.tiingo_api_key = os.getenv("TIINGO_API_KEY")

    def fetch_news(self, asset: str, days: int = 30) -> List[Dict]:
        """
        Fetch macro news. Always relevant to all assets (Gold, Stocks, BTC).
        """
        articles = []
        
        if self.tiingo_api_key:
            self._fetch_tiingo_news(articles, days, asset)
            
        if self.fmp_api_key:
            self._fetch_fmp_calendar(articles, days)
            
        return articles
        
    def _fetch_tiingo_news(self, articles: List[Dict], days: int, asset: str):
        """Fetch general economic news using Tiingo."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        start_str = start_date.strftime('%Y-%m-%d')
        
        # Tags for macro
        tags = "economy,inflation,federal reserve,fomc,cpi,interest rates"
        url = f"https://api.tiingo.com/tiingo/news?tags={tags}&startDate={start_str}&limit=50"
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
                        'source': 'Tiingo Macro'
                    })
        except Exception as e:
            print(f"Error fetching Tiingo macro news: {e}")

    def _fetch_fmp_calendar(self, articles: List[Dict], days: int):
        """Fetch high-impact economic events (CPI, FOMC, etc)."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        url = f"https://financialmodelingprep.com/api/v3/economic_calendar?from={start_str}&to={end_str}&apikey={self.fmp_api_key}"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                events = response.json()
                for event in events:
                    # Filter for High impact events in US
                    if event.get('country') == 'US' and event.get('impact') == 'High':
                        actual = event.get('actual')
                        estimate = event.get('estimate')
                        event_name = event.get('event', '')
                        
                        if actual is not None and estimate is not None:
                            # Heuristic sentiment:
                            # if Actual > Estimate for CPI -> Negative for market
                            # NFP higher -> Positive for economy, maybe negative for rate cuts
                            text_for_bert = f"The US reported {event_name} actual {actual} vs estimate {estimate}."
                            sentiment = get_finbert_sentiment(text_for_bert)
                            
                            date_str = event.get('date', '').split(' ')[0]
                            
                            articles.append({
                                'title': f"Economic Event: {event_name}",
                                'url': 'https://financialmodelingprep.com',
                                'date': date_str,
                                'sentiment': sentiment,
                                'source': 'FMP Calendar'
                            })
        except Exception as e:
            print(f"Error fetching FMP Economic Calendar: {e}")

if __name__ == "__main__":
    fetcher = MacroNewsFetcher()
    data = fetcher.fetch_news('spy', 7)
    for d in data[:5]:
        print(f"{d['date']}: {d['sentiment']} -> {d['title']}")
