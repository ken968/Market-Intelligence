"""
Multi-Source Sentiment Aggregator
Combines data from multiple free sentiment sources
"""

import pandas as pd
from typing import List, Dict
from datetime import datetime, timedelta
import sys
import os
import re

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentiment_sources.yahoo_rss import YahooRSSFetcher
from sentiment_sources.fear_greed_fetcher import FearGreedFetcher
from sentiment_sources.twitter_fetcher import TwitterSentimentFetcher
from sentiment_sources.onchain_fetcher import OnChainFetcher
from sentiment_sources.macro_news_fetcher import MacroNewsFetcher
from sentiment_sources.geopolitical_fetcher import GeopoliticalFetcher


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
            
        # Add New Advanced Sources
        self.sources.append(FearGreedFetcher())
        self.sources.append(TwitterSentimentFetcher())
        self.sources.append(OnChainFetcher())
        self.sources.append(MacroNewsFetcher())
        self.sources.append(GeopoliticalFetcher())
        
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
                
                # Apply structural weighting to solve "Sama Rata" Anomaly
                weight = 1.0
                if isinstance(source, (GeopoliticalFetcher, MacroNewsFetcher)):
                    weight = 2.5  # Critical Macro/Geo news gets 2.5x weight
                elif isinstance(source, OnChainFetcher):
                    weight = 1.5  # Fundamental data gets 1.5x weight
                elif isinstance(source, TwitterSentimentFetcher):
                    weight = 0.8  # Social noise reduced slightly
                
                for a in articles:
                    a['weight'] = weight
                    
                all_articles.extend(articles)
            except Exception as e:
                print(f"  {source.source_name}: Failed - {e}")
                continue
        
        if not all_articles:
            print(f"  No sentiment data found from any source")
            # Return empty DataFrame with expected structure
            return pd.DataFrame(columns=['Date', 'Sentiment'])
            
        # ==========================================
        # FLAW #1 FIX: Title De-duplication (Echo Chamber)
        # ==========================================
        print(f"  Pre-deduplication article count: {len(all_articles)}")
        unique_articles = []
        seen_titles = set()
        
        # Sort by weight descending so we keep the highest weighted source if duplicates exist
        all_articles.sort(key=lambda x: x.get('weight', 1.0), reverse=True)
        
        for article in all_articles:
            raw_title = article.get('title', '')
            # Clean title: lowercase, remove special characters and extra spaces
            clean_title = re.sub(r'[^a-z0-9]', '', raw_title.lower())
            
            # Simple hash/set matching for near-exact duplicates
            if clean_title not in seen_titles:
                seen_titles.add(clean_title)
                unique_articles.append(article)
                
        print(f"  Post-deduplication article count: {len(unique_articles)}")
        all_articles = unique_articles
        # ==========================================
        
        # Convert to DataFrame
        df = pd.DataFrame(all_articles)
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        # Calculate weighted sentiment
        df['weight'] = df.get('weight', 1.0)
        df['weighted_sentiment'] = df['sentiment'] * df['weight']
        
        # Aggregate sentiment by date (weighted average across all articles)
        daily_sum = df.groupby('date')[['weighted_sentiment', 'weight']].sum().reset_index()
        daily_sum['Sentiment'] = (daily_sum['weighted_sentiment'] / daily_sum['weight']).clip(-1.0, 1.0)
        
        # Also get article count
        article_counts = df.groupby('date').size().reset_index(name='ArticleCount')
        
        # Merge
        daily_sentiment = pd.merge(daily_sum[['date', 'Sentiment']], article_counts, on='date')
        
        # Sort chronologically for decay processing
        daily_sentiment = daily_sentiment.sort_values('date')
        
        # ==========================================
        # FLAW #3 FIX: Sentiment Decay (Memory)
        # ==========================================
        # Apply Exponential Moving Average to give sentiment a "memory tail"
        # span=3 roughly gives 50% decay over 3 days (simulates lingering fear/greed)
        daily_sentiment['Sentiment'] = daily_sentiment['Sentiment'].ewm(span=3, adjust=False).mean()
        # ==========================================
        
        daily_sentiment.rename(columns={'date': 'Date'}, inplace=True)
        
        # Convert date back to string for consistency
        daily_sentiment['Date'] = daily_sentiment['Date'].astype(str)
        
        print(f"  Aggregated: {len(daily_sentiment)} days with weighted sentiment data")
        print(f"  Non-zero sentiment: {(daily_sentiment['Sentiment'] != 0).sum()} days")
        print(f"  Mean sentiment: {daily_sentiment['Sentiment'].mean():+.4f}")
        
        return daily_sentiment[['Date', 'Sentiment']]
    
    def get_source_names(self) -> List[str]:
        """Get list of active source names"""
        return [source.source_name for source in self.sources]
    
    def fetch_articles(self, asset: str, days: int = 30) -> List[Dict]:
        """
        Fetch raw articles from all sources for news display.
        
        Returns:
            List of article dicts with: title, url, date, sentiment, source
        """
        all_articles = []
        
        for source in self.sources:
            try:
                articles = source.fetch_news(asset, days)
                all_articles.extend(articles)
            except Exception as e:
                continue
        
        # Sort by date descending (newest first)
        all_articles.sort(key=lambda x: x.get('date', ''), reverse=True)
        return all_articles


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
