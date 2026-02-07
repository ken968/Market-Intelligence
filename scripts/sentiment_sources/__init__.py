"""
Sentiment Sources Package
Multi-source sentiment fetching to replace NewsAPI
"""

from .base_fetcher import BaseSentimentFetcher
from .yahoo_rss import YahooRSSFetcher
# from .finnhub_fetcher import FinnhubFetcher  # Optional
# from .alpha_vantage_fetcher import AlphaVantageFetcher  # Optional
from .aggregator import SentimentAggregator

__all__ = [
    'BaseSentimentFetcher',
    'YahooRSSFetcher',
    'SentimentAggregator'
]
