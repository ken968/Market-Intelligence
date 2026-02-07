"""
Yahoo Finance RSS Sentiment Fetcher
Unlimited free access, no API key required
"""

import feedparser
from textblob import TextBlob
from datetime import datetime, timedelta
from typing import List, Dict
from .base_fetcher import BaseSentimentFetcher


class YahooRSSFetcher(BaseSentimentFetcher):
    """Fetch news and sentiment from Yahoo Finance RSS feeds"""
    
    def __init__(self):
        super().__init__('Yahoo RSS')
    
    def fetch_news(self, asset: str, days: int = 30) -> List[Dict]:
        """
        Fetch news from Yahoo Finance RSS
        
        Args:
            asset: Asset ticker (e.g., 'AAPL', 'BTC')
            days: Number of days to look back
        
        Returns:
            List of articles with sentiment scores
        """
        # Convert asset to ticker format
        ticker_map = {
            'btc': 'BTC-USD',
            'gold': 'GC=F',  # Gold futures
        }
        
        ticker = ticker_map.get(asset.lower(), asset.upper())
        
        # Yahoo Finance RSS URL
        url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
        
        try:
            feed = feedparser.parse(url)
            
            if not feed.entries:
                print(f"  {self.source_name}: No articles found for {asset}")
                return []
            
            cutoff_date = datetime.now() - timedelta(days=days)
            results = []
            
            for entry in feed.entries[:30]:  # Limit to 30 most recent
                try:
                    # Parse date
                    pub_date = datetime(*entry.published_parsed[:6])
                    
                    if pub_date < cutoff_date:
                        continue
                    
                    # Analyze sentiment using TextBlob
                    text = f"{entry.title} {entry.get('summary', '')}"
                    sentiment = TextBlob(text).sentiment.polarity
                    
                    results.append({
                        'title': entry.title,
                        'url': entry.link,
                        'date': pub_date.strftime('%Y-%m-%d'),
                        'sentiment': sentiment,
                        'source': self.source_name
                    })
                    
                except Exception as e:
                    # Skip problematic entries
                    continue
            
            print(f"  {self.source_name}: Fetched {len(results)} articles for {asset}")
            return results
            
        except Exception as e:
            print(f"  {self.source_name}: Error fetching {asset} - {e}")
            return []
