"""
Macro Sentiment Tracker
Analyzes DXY, US10Y Yield, and VIX to determine systemic Risk-On/Risk-Off (Dovish/Hawkish) sentiment.
Replaces the deprecated mock Fed Watch fetcher.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

class MacroSentimentFetcher:
    """Calculate systemic risk sentiment from macro indicators"""
    
    def __init__(self):
        self.data_dir = 'data/alternative'
        os.makedirs(self.data_dir, exist_ok=True)
        self.filepath = os.path.join(self.data_dir, 'macro_sentiment_history.csv')
        self.macro_file = 'data/macro_indicators.csv'
    
    def fetch_probabilities(self):
        """
        Calculate macro probabilities (Risk-On, Neutral, Risk-Off)
        """
        if not os.path.exists(self.macro_file):
            return self._fallback_data()
            
        try:
            df = pd.read_csv(self.macro_file, index_col=0, parse_dates=True)
            if len(df) < 90:
                return self._fallback_data()
                
            # Latest data
            latest = df.iloc[-1]
            
            # Historical means (90-day)
            dxy_90d = df['DXY'].tail(90).mean()
            yield_90d = df['Yield_10Y'].tail(90).mean()
            vix_90d = df['VIX'].tail(90).mean()
            
            # Current vs 90d
            dxy_diff = (latest['DXY'] - dxy_90d) / dxy_90d
            yield_diff = (latest['Yield_10Y'] - yield_90d) / yield_90d
            vix_diff = (latest['VIX'] - vix_90d) / vix_90d
            
            # Score each component (negative diff = dovish/risk-on)
            dxy_score = max(0, min(100, 50 - (dxy_diff * 1000)))  # if DXY is 5% below mean, score is 50 - (-50) = 100
            yield_score = max(0, min(100, 50 - (yield_diff * 500)))
            vix_score = max(0, min(100, 50 - (vix_diff * 200)))
            
            dovish_score = (dxy_score * 0.4) + (yield_score * 0.4) + (vix_score * 0.2)
            
            # Convert to probabilities
            if dovish_score > 65:
                prob_cut = 0.70
                prob_hold = 0.20
                prob_hike = 0.10
            elif dovish_score < 35:
                prob_cut = 0.10
                prob_hold = 0.20
                prob_hike = 0.70
            else:
                prob_cut = 0.30
                prob_hold = 0.40
                prob_hike = 0.30
                
            return {
                'next_meeting_date': 'Macro Engine (Continuous)',
                'prob_cut': prob_cut, # Risk-On
                'prob_hold': prob_hold, # Neutral
                'prob_hike': prob_hike, # Risk-Off
                'current_rate': latest['Yield_10Y'],
                'dovish_score': dovish_score
            }
            
        except Exception as e:
            print(f"Error calculating macro sentiment: {e}")
            return self._fallback_data()

    def _fallback_data(self):
        return {
            'next_meeting_date': 'Macro Engine (Continuous)',
            'prob_cut': 0.33,
            'prob_hold': 0.34,
            'prob_hike': 0.33,
            'current_rate': 4.50,
            'dovish_score': 50.0
        }
    
    def get_dovish_score(self):
        probs = self.fetch_probabilities()
        return probs['dovish_score']
    
    def get_fed_signal(self):
        probs = self.fetch_probabilities()
        dovish_score = probs['dovish_score']
        
        if dovish_score > 65:
            stance = 'Risk-On (Dovish)'
            gold_signal = 'bullish'
            stock_signal = 'bullish'
            confidence = min(1.0, (dovish_score - 50) / 50)
        elif dovish_score < 35:
            stance = 'Risk-Off (Hawkish)'
            gold_signal = 'bearish'
            stock_signal = 'bearish'
            confidence = min(1.0, (50 - dovish_score) / 50)
        else:
            stance = 'Neutral'
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
        signal = self.get_fed_signal()
        
        df = pd.DataFrame([{
            'timestamp': datetime.now(),
            'dovish_score': signal['dovish_score'],
            'stance': signal['stance'],
            'prob_cut': signal['probabilities']['prob_cut'],
            'prob_hold': signal['probabilities']['prob_hold'],
            'prob_hike': signal['probabilities']['prob_hike'],
            'current_rate': signal['probabilities']['current_rate']
        }])
        
        if os.path.exists(self.filepath):
            try:
                existing = pd.read_csv(self.filepath)
                existing['temp_date'] = pd.to_datetime(existing['timestamp']).dt.date
                today = datetime.now().date()
                
                existing = existing[existing['temp_date'] != today]
                existing = existing.drop(columns=['temp_date'])
                
                df = pd.concat([existing, df], ignore_index=True)
            except Exception as e:
                print(f"Error processing existing history: {e}")
        
        df.to_csv(self.filepath, index=False)
        return self.filepath
    
    def get_historical_data(self, days=30):
        if not os.path.exists(self.filepath):
            return pd.DataFrame()
        
        df = pd.read_csv(self.filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        cutoff = datetime.now() - pd.Timedelta(days=days)
        df = df[df['timestamp'] >= cutoff]
        
        return df

if __name__ == '__main__':
    fetcher = MacroSentimentFetcher()
    signal = fetcher.get_fed_signal()
    print(f"Dovish Score: {signal['dovish_score']:.1f}/100")
    print(f"Stance: {signal['stance']}")
