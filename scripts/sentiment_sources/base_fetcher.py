"""
Base class for sentiment fetchers
Defines common interface for all sentiment sources
"""

from abc import ABC, abstractmethod
from typing import List, Dict
from datetime import datetime


class BaseSentimentFetcher(ABC):
    """Abstract base class for sentiment data sources"""
    
    def __init__(self, source_name: str):
        """
        Initialize base fetcher
        
        Args:
            source_name: Name of the sentiment source (e.g., 'Yahoo RSS')
        """
        self.source_name = source_name
    
    @abstractmethod
    def fetch_news(self, asset: str, days: int = 30) -> List[Dict]:
        """
        Fetch news articles for given asset
        
        Args:
            asset: Asset ticker or keyword (e.g., 'AAPL', 'Bitcoin')
            days: Number of days to look back
        
        Returns:
            List of dicts with keys: {title, url, date, sentiment, source}
        """
        pass
    
    def _normalize_asset_query(self, asset: str) -> str:
        """
        Convert asset ticker to search-friendly query
        
        Args:
            asset: Ticker symbol (e.g., 'AAPL', 'BTC')
        
        Returns:
            Search query string
        """
        asset_queries = {
            'gold': 'Gold',
            'btc': 'Bitcoin',
            'spy': 'S&P 500',
            'qqq': 'Nasdaq',
            'dia': 'Dow Jones',
            'aapl': 'Apple',
            'msft': 'Microsoft',
            'googl': 'Google Alphabet',
            'amzn': 'Amazon',
            'nvda': 'NVIDIA',
            'meta': 'Meta Facebook',
            'tsla': 'Tesla',
            'tsm': 'TSMC Semiconductor'
        }
        
        return asset_queries.get(asset.lower(), asset.upper())
