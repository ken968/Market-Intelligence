"""
Google Trends Data Fetcher
Tracks search volume for assets to gauge retail interest
"""

from pytrends.request import TrendReq
import pandas as pd
import os
from datetime import datetime, timedelta

class GoogleTrendsFetcher:
    """Fetch and analyze Google Trends data for market assets"""
    
    def __init__(self):
        """Initialize Google Trends API client"""
        self.pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25))
        self.data_dir = 'data/alternative'
        os.makedirs(self.data_dir, exist_ok=True)
    
    def fetch_trends(self, keywords, timeframe='today 3-m'):
        """
        Fetch Google Trends data for keywords
        
        Args:
            keywords (list): Search terms e.g. ['Bitcoin', 'Gold price']
            timeframe (str): 'today 3-m', 'today 12-m', 'now 7-d', etc.
        
        Returns:
            pd.DataFrame: Normalized search volume (0-100)
        """
        try:
            self.pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo='', gprop='')
            data = self.pytrends.interest_over_time()
            
            if not data.empty:
                # Remove 'isPartial' column if exists
                if 'isPartial' in data.columns:
                    data = data.drop(columns=['isPartial'])
            
            return data
        
        except Exception as e:
            print(f"Error fetching trends for {keywords}: {e}")
            return pd.DataFrame()
    
    def get_current_interest(self, keyword):
        """
        Get latest search interest score
        
        Args:
            keyword (str): Search term
        
        Returns:
            int: Current interest score (0-100)
        """
        data = self.fetch_trends([keyword], timeframe='now 7-d')
        
        if not data.empty and keyword in data.columns:
            return int(data[keyword].iloc[-1])
        
        return 0
    
    def fetch_asset_trends(self, asset_key):
        """
        Fetch trends for specific asset with appropriate keywords
        
        Args:
            asset_key (str): 'gold', 'btc', 'msft', etc.
        
        Returns:
            pd.DataFrame: Trends data
        """
        keyword_map = {
            'gold': 'Gold price',
            'btc': 'Bitcoin',
            'spy': 'S&P 500',
            'qqq': 'Nasdaq',
            'aapl': 'Apple stock',
            'msft': 'Microsoft stock',
            'googl': 'Google stock',
            'amzn': 'Amazon stock',
            'nvda': 'Nvidia stock',
            'meta': 'Meta stock',
            'tsla': 'Tesla stock',
            'tsm': 'TSMC stock'
        }
        
        keyword = keyword_map.get(asset_key.lower(), asset_key)
        return self.fetch_trends([keyword], timeframe='today 3-m')
    
    def save_trends_data(self, asset_key):
        """
        Fetch and save trends data to CSV
        
        Args:
            asset_key (str): Asset identifier
        
        Returns:
            str: Path to saved file
        """
        data = self.fetch_asset_trends(asset_key)
        
        if not data.empty:
            filepath = os.path.join(self.data_dir, f'google_trends_{asset_key}.csv')
            data.to_csv(filepath)
            print(f"Saved Google Trends data for {asset_key} to {filepath}")
            return filepath
        
        return None
    
    def get_trend_signal(self, asset_key):
        """
        Analyze trend data and generate signal
        
        Args:
            asset_key (str): Asset identifier
        
        Returns:
            dict: {
                'current_interest': int,
                'avg_interest': float,
                'trend': 'rising' | 'falling' | 'stable',
                'signal_strength': float (0-1)
            }
        """
        data = self.fetch_asset_trends(asset_key)
        
        if data.empty:
            return {
                'current_interest': 0,
                'avg_interest': 0,
                'trend': 'unknown',
                'signal_strength': 0
            }
        
        keyword = data.columns[0]
        current = int(data[keyword].iloc[-1])
        avg = float(data[keyword].mean())
        
        # Calculate trend
        recent_avg = float(data[keyword].tail(7).mean())
        older_avg = float(data[keyword].head(7).mean())
        
        if recent_avg > older_avg * 1.1:
            trend = 'rising'
            signal_strength = min((recent_avg - older_avg) / older_avg, 1.0)
        elif recent_avg < older_avg * 0.9:
            trend = 'falling'
            signal_strength = min((older_avg - recent_avg) / older_avg, 1.0)
        else:
            trend = 'stable'
            signal_strength = 0.5
        
        return {
            'current_interest': current,
            'avg_interest': avg,
            'trend': trend,
            'signal_strength': signal_strength
        }


def batch_fetch_trends(asset_keys):
    """
    Fetch trends for multiple assets
    
    Args:
        asset_keys (list): List of asset identifiers
    
    Returns:
        dict: {asset_key: trend_signal}
    """
    fetcher = GoogleTrendsFetcher()
    results = {}
    
    for key in asset_keys:
        try:
            signal = fetcher.get_trend_signal(key)
            results[key] = signal
            fetcher.save_trends_data(key)
        except Exception as e:
            print(f"Error processing {key}: {e}")
            results[key] = {'error': str(e)}
    
    return results


if __name__ == '__main__':
    # Test the fetcher
    print("Testing Google Trends Fetcher...")
    
    test_assets = ['gold', 'btc', 'msft']
    
    fetcher = GoogleTrendsFetcher()
    
    for asset in test_assets:
        print(f"\n--- {asset.upper()} ---")
        signal = fetcher.get_trend_signal(asset)
        print(f"Current Interest: {signal['current_interest']}/100")
        print(f"Average Interest: {signal['avg_interest']:.1f}/100")
        print(f"Trend: {signal['trend']}")
        print(f"Signal Strength: {signal['signal_strength']:.2f}")
        
        fetcher.save_trends_data(asset)
    
    print("\nDone!")
