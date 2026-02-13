"""
Fed Watch Data Fetcher
Scrapes Fed rate probabilities from CME FedWatch Tool
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from datetime import datetime
import json

class FedWatchFetcher:
    """Fetch Fed rate cut/hike probabilities"""
    
    def __init__(self):
        """Initialize Fed Watch fetcher"""
        self.data_dir = 'data/alternative'
        os.makedirs(self.data_dir, exist_ok=True)
        self.filepath = os.path.join(self.data_dir, 'fed_watch_history.csv')
    
    def fetch_probabilities(self):
        """
        Fetch Fed rate probabilities from CME FedWatch
        
        Note: This is a simplified version. Real implementation would
        scrape from CME Group website or use alternative data source.
        
        Returns:
            dict: {
                'next_meeting_date': str,
                'prob_cut': float (0-1),
                'prob_hold': float (0-1),
                'prob_hike': float (0-1),
                'current_rate': float
            }
        """
        # Mock data for now - in production, would scrape from:
        # https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html
        
        # For real implementation, use requests + BeautifulSoup
        # or alternative API like FRED (Federal Reserve Economic Data)
        
        return self._get_mock_probabilities()
    
    def _get_mock_probabilities(self):
        """
        Return mock probabilities based on current market conditions
        
        In production, replace this with actual scraping logic
        """
        # Mock data reflecting current dovish sentiment
        return {
            'next_meeting_date': '2026-03-18',
            'prob_cut': 0.25,
            'prob_hold': 0.70,
            'prob_hike': 0.05,
            'current_rate': 4.50,
            'note': 'Mock data - implement real scraper for production'
        }
    
    def get_dovish_score(self):
        """
        Calculate dovish score (0-100)
        
        Higher score = more dovish (good for Gold/BTC)
        Lower score = more hawkish (bad for Gold/BTC)
        
        Returns:
            float: Dovish score (0-100)
        """
        probs = self.fetch_probabilities()
        
        # Formula: (prob_cut * 100) + (prob_hold * 50) + (prob_hike * 0)
        dovish_score = (probs['prob_cut'] * 100) + (probs['prob_hold'] * 50)
        
        return dovish_score
    
    def get_fed_signal(self):
        """
        Generate trading signal based on Fed probabilities
        
        Returns:
            dict: {
                'dovish_score': float (0-100),
                'stance': 'dovish' | 'neutral' | 'hawkish',
                'signal_for_gold': 'bullish' | 'neutral' | 'bearish',
                'signal_for_stocks': 'bullish' | 'neutral' | 'bearish',
                'confidence': float (0-1)
            }
        """
        probs = self.fetch_probabilities()
        dovish_score = self.get_dovish_score()
        
        # Determine stance
        if dovish_score > 65:
            stance = 'dovish'
            gold_signal = 'bullish'
            stock_signal = 'bullish'
            confidence = (dovish_score - 65) / 35
        elif dovish_score < 35:
            stance = 'hawkish'
            gold_signal = 'bearish'
            stock_signal = 'bearish'
            confidence = (35 - dovish_score) / 35
        else:
            stance = 'neutral'
            gold_signal = 'neutral'
            stock_signal = 'neutral'
            confidence = 0.5
        
        return {
            'dovish_score': dovish_score,
            'stance': stance,
            'signal_for_gold': gold_signal,
            'signal_for_stocks': stock_signal,
            'confidence': confidence,
            'probabilities': probs
        }
    
    def save_fed_data(self):
        """
        Save current Fed probabilities to CSV
        
        Returns:
            str: Path to saved file
        """
        signal = self.get_fed_signal()
        
        # Create dataframe with timestamp
        df = pd.DataFrame([{
            'timestamp': datetime.now(),
            'dovish_score': signal['dovish_score'],
            'stance': signal['stance'],
            'prob_cut': signal['probabilities']['prob_cut'],
            'prob_hold': signal['probabilities']['prob_hold'],
            'prob_hike': signal['probabilities']['prob_hike'],
            'current_rate': signal['probabilities']['current_rate']
        }])
        
        # Append to existing file or create new
        if os.path.exists(self.filepath):
            existing = pd.read_csv(self.filepath)
            df = pd.concat([existing, df], ignore_index=True)
        
        df.to_csv(self.filepath, index=False)
        print(f"Saved Fed Watch data to {self.filepath}")
        
        return self.filepath
    
    def get_historical_data(self, days=30):
        """
        Get historical Fed Watch data
        
        Args:
            days (int): Number of days to retrieve
        
        Returns:
            pd.DataFrame: Historical data
        """
        if not os.path.exists(self.filepath):
            return pd.DataFrame()
        
        df = pd.read_csv(self.filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by date range
        cutoff = datetime.now() - pd.Timedelta(days=days)
        df = df[df['timestamp'] >= cutoff]
        
        return df


if __name__ == '__main__':
    # Test the fetcher
    print("Testing Fed Watch Fetcher...")
    print("Note: Using mock data - implement real scraper for production\n")
    
    fetcher = FedWatchFetcher()
    
    # Get current signal
    signal = fetcher.get_fed_signal()
    
    print("--- Fed Watch Signal ---")
    print(f"Dovish Score: {signal['dovish_score']:.1f}/100")
    print(f"Stance: {signal['stance'].title()}")
    print(f"Signal for Gold: {signal['signal_for_gold'].title()}")
    print(f"Signal for Stocks: {signal['signal_for_stocks'].title()}")
    print(f"Confidence: {signal['confidence']:.2f}")
    print(f"\nProbabilities:")
    print(f"  Rate Cut: {signal['probabilities']['prob_cut']:.1%}")
    print(f"  Hold: {signal['probabilities']['prob_hold']:.1%}")
    print(f"  Rate Hike: {signal['probabilities']['prob_hike']:.1%}")
    
    # Save data
    fetcher.save_fed_data()
    
    print("\nDone!")
