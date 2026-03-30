"""
Twitter/X Sentiment Fetcher with xAI (Grok) Classification
Fetches recent tweets for an asset using X API v2.
Classifies sentiment using xAI Grok API.
"""

import os
import sys
from typing import List, Dict
from datetime import datetime

try:
    from sentiment_sources.base_fetcher import BaseSentimentFetcher
except ImportError:
    from base_fetcher import BaseSentimentFetcher

import tweepy
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class TwitterSentimentFetcher(BaseSentimentFetcher):
    """Fetches tweets and scores sentiment using xAI Grok API."""
    
    def __init__(self):
        super().__init__("X/Twitter Sentiment")
        self.bearer_token = os.getenv("X_BEARER_TOKEN")
        self.xai_api_key = os.getenv("XAI_API_KEY")
        
        if not self.bearer_token:
            print("Warning: X_BEARER_TOKEN missing. Twitter fetcher disabled.")
        if not self.xai_api_key:
            print("Warning: XAI_API_KEY missing. Sentiment classification will fallback to TextBlob.")
            
        # Optional fallback to TextBlob if xAI is missing
        if not self.xai_api_key:
            from textblob import TextBlob
            self._textblob_fallback = True
        else:
            self._textblob_fallback = False
            self.xai_client = OpenAI(
                api_key=self.xai_api_key,
                base_url="https://api.x.ai/v1"
            )

    def fetch_news(self, asset: str, days: int = 7) -> List[Dict]:
        """
        Fetch recent tweets and score sentiment.
        Since standard X API v2 search_recent_tweets only goes back 7 days, 
        days > 7 will be capped at 7.
        """
        if not self.bearer_token:
            return []
            
        articles = []
        try:
            client = tweepy.Client(bearer_token=self.bearer_token)
            
            # Build search query based on asset
            query_base = self._normalize_asset_query(asset)
            # Add conditions: text contains asset, english lang, no retweets
            query = f"{query_base} -is:retweet lang:en"
            
            print(f"System: Fetching X/Twitter data for '{query}'...")
            
            # Fetch up to 20 recent relevant tweets
            response = client.search_recent_tweets(
                query=query, 
                max_results=20, 
                tweet_fields=['created_at', 'text', 'id']
            )
            
            if not response.data:
                print(f"Warning: No tweets found for {asset}.")
                return []
                
            tweets = response.data
            
            for tweet in tweets:
                # Classify sentiment
                sentiment = self._classify_sentiment(tweet.text)
                
                created_at = tweet.created_at
                if created_at is None:
                    # fallback if created_at not provided
                    created_at = datetime.utcnow()
                    
                date_str = created_at.strftime('%Y-%m-%d')
                
                # To assign to daily sentiment, we aggregate these later in aggregator.py
                articles.append({
                    'title': tweet.text.replace('\n', ' ')[:100] + '...', # First 100 chars as title
                    'url': f"https://x.com/i/web/status/{tweet.id}",
                    'date': date_str,
                    'sentiment': sentiment,
                    'source': self.source_name
                })
                
        except Exception as e:
            print(f"Error fetching X/Twitter sentiment: {e}")
            
        print(f"System: Fetched and scored {len(articles)} tweets.")
        return articles

    def _classify_sentiment(self, text: str) -> float:
        """
        Score sentiment from -1.0 (extreme fear/bearish) to 1.0 (extreme greed/bullish).
        Uses xAI API if available.
        """
        if self._textblob_fallback:
            from textblob import TextBlob
            return TextBlob(text).sentiment.polarity
            
        try:
            # Call Grok
            completion = self.xai_client.chat.completions.create(
                model="grok-beta",  # Adjust model name if needed
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a highly capable financial sentiment classifier. Respond ONLY with a number between -1.0 and 1.0. -1.0 means extremely bearish or fearful, 1.0 means extremely bullish or greedy, and 0.0 means neutral. Do not explain, just return the float number."
                    },
                    {
                        "role": "user", 
                        "content": f"Classify the financial sentiment of this tweet: '{text}'"
                    }
                ],
                max_tokens=5,
                temperature=0.0
            )
            
            score_str = completion.choices[0].message.content.strip()
            score = float(score_str)
            # Ensure within bounds
            return max(-1.0, min(1.0, score))
            
        except Exception as e:
            # Silently fallback to TextBlob if API fails for one tweet
            from textblob import TextBlob
            return TextBlob(text).sentiment.polarity

if __name__ == "__main__":
    fetcher = TwitterSentimentFetcher()
    data = fetcher.fetch_news('btc')
    for d in data:
        print(f"{d['date']}: {d['sentiment']} -> {d['title']}")
