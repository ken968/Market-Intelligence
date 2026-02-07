"""
Multi-Source Sentiment Aggregator
Combines data from multiple free sentiment sources
"""

import pandas as pd
from typing import List, Dict
from datetime import datetime, timedelta
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentiment_sources.yahoo_rss import YahooRSSFetcher


class SentimentAggregator:
    """Aggregate sentiment from multiple sources"""
    
    def __init__(self):
        """Initialize all available sentiment sources"""
        self.sources = []
        
        # Yahoo RSS - Always available (no key needed)
        self.sources.append(YahooRSSFetcher())
        
        # Finnhub - Use provided API key
        try:
            from sentiment_sources.finnhub_fetcher import FinnhubFetcher
            finn_key = 'd63j2u9r01ql6dj0a470d63j2u9r01ql6dj0a47g'
            self.sources.append(FinnhubFetcher(finn_key))
        except Exception as e:
            print(f"Finnhub not available: {e}")
        
        # Alpha Vantage - Use provided API key
        try:
            from sentiment_sources.alpha_vantage_fetcher import AlphaVantageFetcher
            av_key = '1FC9SC9YNDAT7DCF'
            self.sources.append(AlphaVantageFetcher(av_key))
        except Exception as e:
            print(f"Alpha Vantage not available: {e}")
        
        print(f"Sentiment Aggregator initialized with {len(self.sources)} source(s)")
    
    def fetch_all(self, asset: str, days: int = 30) -> pd.DataFrame:
        """
        Fetch sentiment from all sources and aggregate
        
        Args:
            asset: Asset ticker
            days: Number of days to fetch
        
        Returns:
            DataFrame with columns: Date, Sentiment
        """
        all_articles = []
        
        print(f"\nFetching sentiment for {asset.upper()}...")
        
        # Fetch from all sources
        for source in self.sources:
            try:
                articles = source.fetch_news(asset, days)
                all_articles.extend(articles)
            except Exception as e:
                print(f"  {source.source_name}: Failed - {e}")
                continue
        
        if not all_articles:
            print(f"  No sentiment data found from any source")
            # Return empty DataFrame with expected structure
            return pd.DataFrame(columns=['Date', 'Sentiment'])
        
        # Convert to DataFrame
        df = pd.DataFrame(all_articles)
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        # Aggregate sentiment by date (average across all articles)
        daily_sentiment = df.groupby('date').agg({
            'sentiment': 'mean',  # Average sentiment per day
            'title': 'count'      # Number of articles
        }).reset_index()
        
        daily_sentiment.columns = ['Date', 'Sentiment', 'ArticleCount']
        
        # Convert date back to string for consistency
        daily_sentiment['Date'] = daily_sentiment['Date'].astype(str)
        
        print(f"  Aggregated: {len(daily_sentiment)} days with sentiment data")
        print(f"  Non-zero sentiment: {(daily_sentiment['Sentiment'] != 0).sum()} days")
        print(f"  Mean sentiment: {daily_sentiment['Sentiment'].mean():+.4f}")
        
        return daily_sentiment[['Date', 'Sentiment']]
    
    def get_source_names(self) -> List[str]:
        """Get list of active source names"""
        return [source.source_name for source in self.sources]


# ==================== TESTING ====================

if __name__ == "__main__":
    print("="*60)
    print("SENTIMENT AGGREGATOR - TESTING")
    print("="*60)
    
    aggregator = SentimentAggregator()
    print(f"\nActive sources: {', '.join(aggregator.get_source_names())}")
    
    # Test with multiple assets
    test_assets = ['AAPL', 'BTC', 'MSFT']
    
    for asset in test_assets:
        print(f"\n{'='*60}")
        sentiment_data = aggregator.fetch_all(asset, days=7)
        
        if not sentiment_data.empty:
            print(f"\nSample data for {asset}:")
            print(sentiment_data.head())
        else:
            print(f"No data retrieved for {asset}")
