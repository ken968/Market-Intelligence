"""
Reddit Sentiment Analyzer
Analyzes sentiment from relevant subreddits for market assets
"""

import praw
from textblob import TextBlob
import pandas as pd
import os
from datetime import datetime, timedelta

class RedditSentimentAnalyzer:
    """Analyze sentiment from Reddit for market intelligence"""
    
    def __init__(self, client_id=None, client_secret=None, user_agent=None):
        """
        Initialize Reddit API client
        
        Args:
            client_id (str): Reddit app client ID
            client_secret (str): Reddit app client secret
            user_agent (str): User agent string
        
        Note: If credentials not provided, will use read-only mode with limitations
        """
        self.data_dir = 'data/alternative'
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize Reddit client
        if client_id and client_secret:
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent or 'MarketIntelligence/1.0'
            )
            self.authenticated = True
        else:
            # Read-only mode (limited functionality)
            self.reddit = None
            self.authenticated = False
            print("Warning: Reddit API not authenticated. Using mock data mode.")
    
    def analyze_subreddit(self, subreddit_name, limit=100, keyword_filter=None):
        """
        Analyze sentiment from subreddit posts
        
        Args:
            subreddit_name (str): Subreddit name without r/
            limit (int): Number of recent posts to analyze
            keyword_filter (str): Optional keyword to filter posts
        
        Returns:
            dict: {
                'sentiment_score': float (-1 to 1),
                'post_count': int,
                'bullish_count': int,
                'bearish_count': int,
                'neutral_count': int
            }
        """
        if not self.authenticated:
            return self._mock_sentiment()
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts = subreddit.hot(limit=limit)
            
            sentiments = []
            bullish = 0
            bearish = 0
            neutral = 0
            
            for post in posts:
                # Filter by keyword if provided
                if keyword_filter:
                    if keyword_filter.lower() not in post.title.lower() and \
                       keyword_filter.lower() not in post.selftext.lower():
                        continue
                
                # Analyze sentiment
                text = post.title + ' ' + post.selftext
                sentiment = TextBlob(text).sentiment.polarity
                sentiments.append(sentiment)
                
                # Categorize
                if sentiment > 0.1:
                    bullish += 1
                elif sentiment < -0.1:
                    bearish += 1
                else:
                    neutral += 1
            
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            
            return {
                'sentiment_score': avg_sentiment,
                'post_count': len(sentiments),
                'bullish_count': bullish,
                'bearish_count': bearish,
                'neutral_count': neutral
            }
        
        except Exception as e:
            print(f"Error analyzing r/{subreddit_name}: {e}")
            return self._mock_sentiment()
    
    def _mock_sentiment(self):
        """Return mock sentiment data when API unavailable"""
        return {
            'sentiment_score': 0.0,
            'post_count': 0,
            'bullish_count': 0,
            'bearish_count': 0,
            'neutral_count': 0,
            'note': 'Mock data - Reddit API not configured'
        }
    
    def get_asset_sentiment(self, asset_key):
        """
        Get sentiment for specific asset
        
        Args:
            asset_key (str): 'gold', 'btc', 'msft', etc.
        
        Returns:
            dict: Sentiment analysis results
        """
        subreddit_map = {
            'btc': ('cryptocurrency', 'Bitcoin'),
            'gold': ('Gold', 'gold'),
            'spy': ('wallstreetbets', 'SPY'),
            'qqq': ('wallstreetbets', 'QQQ'),
            'aapl': ('wallstreetbets', 'AAPL'),
            'msft': ('wallstreetbets', 'MSFT'),
            'googl': ('wallstreetbets', 'GOOGL'),
            'amzn': ('wallstreetbets', 'AMZN'),
            'nvda': ('wallstreetbets', 'NVDA'),
            'meta': ('wallstreetbets', 'META'),
            'tsla': ('wallstreetbets', 'TSLA'),
            'tsm': ('wallstreetbets', 'TSM')
        }
        
        subreddit, keyword = subreddit_map.get(asset_key.lower(), ('investing', asset_key))
        return self.analyze_subreddit(subreddit, limit=100, keyword_filter=keyword)
    
    def save_sentiment_data(self, asset_key):
        """
        Fetch and save sentiment data to CSV
        
        Args:
            asset_key (str): Asset identifier
        
        Returns:
            str: Path to saved file
        """
        sentiment = self.get_asset_sentiment(asset_key)
        
        # Create dataframe with timestamp
        df = pd.DataFrame([{
            'timestamp': datetime.now(),
            'asset': asset_key,
            **sentiment
        }])
        
        filepath = os.path.join(self.data_dir, f'reddit_sentiment_{asset_key}.csv')
        
        # Append to existing file or create new
        if os.path.exists(filepath):
            existing = pd.read_csv(filepath)
            df = pd.concat([existing, df], ignore_index=True)
        
        df.to_csv(filepath, index=False)
        print(f"Saved Reddit sentiment for {asset_key} to {filepath}")
        
        return filepath
    
    def get_sentiment_signal(self, asset_key):
        """
        Analyze sentiment and generate trading signal
        
        Args:
            asset_key (str): Asset identifier
        
        Returns:
            dict: {
                'sentiment_score': float,
                'sentiment_label': str,
                'signal': 'bullish' | 'bearish' | 'neutral',
                'confidence': float (0-1)
            }
        """
        sentiment = self.get_asset_sentiment(asset_key)
        score = sentiment['sentiment_score']
        
        # Determine label and signal
        if score > 0.3:
            label = 'Very Bullish'
            signal = 'bullish'
            confidence = min(score, 1.0)
        elif score > 0.1:
            label = 'Bullish'
            signal = 'bullish'
            confidence = score * 0.8
        elif score < -0.3:
            label = 'Very Bearish'
            signal = 'bearish'
            confidence = min(abs(score), 1.0)
        elif score < -0.1:
            label = 'Bearish'
            signal = 'bearish'
            confidence = abs(score) * 0.8
        else:
            label = 'Neutral'
            signal = 'neutral'
            confidence = 0.5
        
        return {
            'sentiment_score': score,
            'sentiment_label': label,
            'signal': signal,
            'confidence': confidence,
            'post_count': sentiment['post_count']
        }


def batch_analyze_sentiment(asset_keys, client_id=None, client_secret=None):
    """
    Analyze sentiment for multiple assets
    
    Args:
        asset_keys (list): List of asset identifiers
        client_id (str): Reddit API client ID
        client_secret (str): Reddit API client secret
    
    Returns:
        dict: {asset_key: sentiment_signal}
    """
    analyzer = RedditSentimentAnalyzer(client_id, client_secret)
    results = {}
    
    for key in asset_keys:
        try:
            signal = analyzer.get_sentiment_signal(key)
            results[key] = signal
            analyzer.save_sentiment_data(key)
        except Exception as e:
            print(f"Error processing {key}: {e}")
            results[key] = {'error': str(e)}
    
    return results


if __name__ == '__main__':
    # Test the analyzer
    print("Testing Reddit Sentiment Analyzer...")
    print("Note: Without API credentials, using mock data mode\n")
    
    test_assets = ['gold', 'btc', 'msft']
    
    # To use real data, provide credentials:
    # analyzer = RedditSentimentAnalyzer(
    #     client_id='YOUR_CLIENT_ID',
    #     client_secret='YOUR_CLIENT_SECRET'
    # )
    
    analyzer = RedditSentimentAnalyzer()
    
    for asset in test_assets:
        print(f"--- {asset.upper()} ---")
        signal = analyzer.get_sentiment_signal(asset)
        print(f"Sentiment Score: {signal['sentiment_score']:.3f}")
        print(f"Label: {signal['sentiment_label']}")
        print(f"Signal: {signal['signal']}")
        print(f"Confidence: {signal['confidence']:.2f}")
        print(f"Posts Analyzed: {signal['post_count']}\n")
    
    print("Done!")
