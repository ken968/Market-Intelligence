"""
Reddit Sentiment Analyzer (DISABLED LOCALLY - Waiting for API Approval)
Analyzes sentiment from relevant subreddits for market assets
"""

import praw
from textblob import TextBlob
import pandas as pd
import os
from datetime import datetime, timedelta

class RedditSentimentAnalyzer:
    """DISABLED LOCALLY - Mock analyzer to prevent script failures"""
    def __init__(self, *args, **kwargs):
        self.authenticated = False
    
    def get_sentiment_signal(self, asset_key):
        return {
            'sentiment_score': 0.0,
            'sentiment_label': 'Neutral (Disabled)',
            'signal': 'neutral',
            'confidence': 0.5,
            'post_count': 0
        }
    
    def save_sentiment_data(self, asset_key):
        return None

def batch_analyze_sentiment(asset_keys, *args, **kwargs):
    return {key: {
        'sentiment_score': 0.0,
        'sentiment_label': 'Neutral (Disabled)',
        'signal': 'neutral',
        'confidence': 0.5,
        'post_count': 0
    } for key in asset_keys}

# Original code is preserved on GitHub.
# This local version is simplified to avoid errors while Reddit API is pending.
