"""
Trading Signal Generator
Multi-factor analysis for entry/exit signals
"""

import pandas as pd
import numpy as np
from utils.predictor import AssetPredictor
from utils.config import get_asset_config, CONFIDENCE_SCORES
from scripts.google_trends_fetcher import GoogleTrendsFetcher
from scripts.reddit_sentiment_fetcher import RedditSentimentAnalyzer
from scripts.fed_watch_fetcher import FedWatchFetcher

class SignalGenerator:
    """Generate trading signals based on multi-factor analysis"""
    
    def __init__(self, asset_key):
        """
        Initialize signal generator
        
        Args:
            asset_key (str): Asset identifier ('gold', 'btc', 'msft', etc.)
        """
        self.asset_key = asset_key
        self.config = get_asset_config(asset_key)
        self.predictor = AssetPredictor(asset_key)
        
        # Initialize alternative data fetchers
        self.trends_fetcher = GoogleTrendsFetcher()
        self.sentiment_analyzer = RedditSentimentAnalyzer()
        self.fed_fetcher = FedWatchFetcher()
    
    def generate_signal(self):
        """
        Generate comprehensive trading signal
        
        Returns:
            dict: {
                'signal': 'BUY' | 'SELL' | 'HOLD',
                'confidence': float (0-1),
                'reasons': list of str,
                'factors': dict of factor scores,
                'entry_price': float,
                'target_price': float,
                'stop_loss': float
            }
        """
        # Load latest data
        df = pd.read_csv(self.config['data_file'])
        latest = df.iloc[-1]
        current_price = latest[self.config['features'][0]]
        
        # Get forecast
        forecast_1w = self.predictor.predict_week()
        
        # Analyze all factors
        factors = self._analyze_factors(latest, forecast_1w)
        
        # Calculate signal
        signal_data = self._calculate_signal(factors, current_price, forecast_1w)
        
        return signal_data
    
    def _analyze_factors(self, latest, forecast):
        """
        Analyze all factors for signal generation
        
        Args:
            latest (pd.Series): Latest data row
            forecast (dict): 1-week forecast
        
        Returns:
            dict: Factor analysis results
        """
        factors = {
            'forecast': self._analyze_forecast(forecast),
            'macro': self._analyze_macro(latest),
            'sentiment': self._analyze_sentiment(),
            'technical': self._analyze_technical(latest)
        }
        
        return factors
    
    def _analyze_forecast(self, forecast):
        """Analyze AI forecast"""
        pct_change = forecast['pct_change']
        
        # Get confidence for 1-week forecast
        asset_type = 'gold' if self.asset_key == 'gold' else \
                     'btc' if self.asset_key == 'btc' else 'stocks'
        confidence_data = CONFIDENCE_SCORES.get(asset_type, {}).get('1 Week', {})
        base_confidence = confidence_data.get('score', 0.5)
        
        return {
            'bullish': pct_change > 2,
            'bearish': pct_change < -2,
            'score': abs(pct_change) / 10,  # Normalize to 0-1
            'weight': 0.4,  # 40% weight
            'confidence': base_confidence,
            'detail': f"Forecast: {pct_change:+.1f}%"
        }
    
    def _analyze_macro(self, latest):
        """Analyze macro conditions"""
        if self.asset_key == 'gold':
            return self._analyze_gold_macro(latest)
        elif self.asset_key == 'btc':
            return self._analyze_btc_macro(latest)
        else:
            return self._analyze_stock_macro(latest)
    
    def _analyze_gold_macro(self, latest):
        """Gold-specific macro analysis"""
        dxy = latest.get('DXY', 105)
        vix = latest.get('VIX', 15)
        
        # Fed Watch signal
        fed_signal = self.fed_fetcher.get_fed_signal()
        fed_bullish = fed_signal['signal_for_gold'] == 'bullish'
        
        # Combine factors
        bullish = (dxy < 105 and vix > 15) or fed_bullish
        bearish = (dxy > 108 and vix < 12) or fed_signal['signal_for_gold'] == 'bearish'
        
        return {
            'bullish': bullish,
            'bearish': bearish,
            'score': 0.7 if bullish else (0.3 if bearish else 0.5),
            'weight': 0.3,
            'confidence': 0.75,
            'detail': f"DXY: {dxy:.1f}, VIX: {vix:.1f}, Fed: {fed_signal['stance']}"
        }
    
    def _analyze_btc_macro(self, latest):
        """Bitcoin-specific macro analysis"""
        dxy = latest.get('DXY', 105)
        
        # Fed Watch signal (dovish = bullish for BTC)
        fed_signal = self.fed_fetcher.get_fed_signal()
        fed_bullish = fed_signal['dovish_score'] > 60
        
        bullish = dxy < 104 or fed_bullish
        bearish = dxy > 107 or fed_signal['dovish_score'] < 40
        
        return {
            'bullish': bullish,
            'bearish': bearish,
            'score': 0.7 if bullish else (0.3 if bearish else 0.5),
            'weight': 0.25,
            'confidence': 0.65,
            'detail': f"DXY: {dxy:.1f}, Fed Dovish Score: {fed_signal['dovish_score']:.0f}"
        }
    
    def _analyze_stock_macro(self, latest):
        """Stock-specific macro analysis"""
        vix = latest.get('VIX', 15)
        yield_10y = latest.get('Yield_10Y', 4.0)
        
        # Fed Watch signal
        fed_signal = self.fed_fetcher.get_fed_signal()
        
        bullish = vix < 18 and fed_signal['signal_for_stocks'] == 'bullish'
        bearish = vix > 25 or fed_signal['signal_for_stocks'] == 'bearish'
        
        return {
            'bullish': bullish,
            'bearish': bearish,
            'score': 0.7 if bullish else (0.3 if bearish else 0.5),
            'weight': 0.25,
            'confidence': 0.70,
            'detail': f"VIX: {vix:.1f}, Yield: {yield_10y:.2f}%, Fed: {fed_signal['stance']}"
        }
    
    def _analyze_sentiment(self):
        """Analyze sentiment indicators"""
        # Google Trends
        trends_signal = self.trends_fetcher.get_trend_signal(self.asset_key)
        
        # Reddit Sentiment
        reddit_signal = self.sentiment_analyzer.get_sentiment_signal(self.asset_key)
        
        # Combine signals
        trends_bullish = trends_signal['trend'] == 'rising' and trends_signal['current_interest'] > 60
        reddit_bullish = reddit_signal['signal'] == 'bullish'
        
        trends_bearish = trends_signal['trend'] == 'falling' and trends_signal['current_interest'] < 40
        reddit_bearish = reddit_signal['signal'] == 'bearish'
        
        bullish = trends_bullish or reddit_bullish
        bearish = trends_bearish or reddit_bearish
        
        return {
            'bullish': bullish,
            'bearish': bearish,
            'score': 0.7 if bullish else (0.3 if bearish else 0.5),
            'weight': 0.2,
            'confidence': 0.60,
            'detail': f"Trends: {trends_signal['trend']}, Reddit: {reddit_signal['sentiment_label']}"
        }
    
    def _analyze_technical(self, latest):
        """Analyze technical indicators"""
        price = latest[self.config['features'][0]]
        ema_90 = latest.get('EMA_90', price)
        
        bullish = price > ema_90 * 1.01  # Price 1% above EMA
        bearish = price < ema_90 * 0.99  # Price 1% below EMA
        
        return {
            'bullish': bullish,
            'bearish': bearish,
            'score': 0.7 if bullish else (0.3 if bearish else 0.5),
            'weight': 0.15,
            'confidence': 0.65,
            'detail': f"Price vs EMA90: {((price/ema_90 - 1) * 100):+.1f}%"
        }
    
    def _calculate_signal(self, factors, current_price, forecast):
        """Calculate final signal from all factors"""
        bullish_score = 0
        bearish_score = 0
        reasons = []
        
        # Aggregate scores
        for factor_name, factor in factors.items():
            if factor['bullish']:
                bullish_score += factor['weight']
                reasons.append(f"{factor_name.title()}: {factor['detail']}")
            if factor['bearish']:
                bearish_score += factor['weight']
                reasons.append(f"{factor_name.title()}: {factor['detail']}")
        
        # Determine signal
        if bullish_score > 0.55:
            signal = 'BUY'
            confidence = min(bullish_score, 1.0)
            target_price = forecast['predicted']
            stop_loss = current_price * 0.95  # 5% stop loss
        elif bearish_score > 0.55:
            signal = 'SELL'
            confidence = min(bearish_score, 1.0)
            target_price = forecast['predicted']
            stop_loss = current_price * 1.05  # 5% stop loss (for shorts)
        else:
            signal = 'HOLD'
            confidence = 1 - abs(bullish_score - bearish_score)
            target_price = current_price
            stop_loss = current_price
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reasons': reasons,
            'factors': {k: {'score': v['score'], 'weight': v['weight']} for k, v in factors.items()},
            'bullish_score': bullish_score,
            'bearish_score': bearish_score,
            'entry_price': current_price,
            'target_price': target_price,
            'stop_loss': stop_loss,
            'risk_reward': abs(target_price - current_price) / abs(stop_loss - current_price) if stop_loss != current_price else 0
        }


def batch_generate_signals(asset_keys):
    """
    Generate signals for multiple assets
    
    Args:
        asset_keys (list): List of asset identifiers
    
    Returns:
        dict: {asset_key: signal_data}
    """
    results = {}
    
    for key in asset_keys:
        try:
            generator = SignalGenerator(key)
            signal = generator.generate_signal()
            results[key] = signal
        except Exception as e:
            print(f"Error generating signal for {key}: {e}")
            results[key] = {'error': str(e)}
    
    return results


if __name__ == '__main__':
    # Test signal generator
    print("Testing Signal Generator...")
    
    test_assets = ['gold', 'btc']
    
    for asset in test_assets:
        print(f"\n{'='*50}")
        print(f"{asset.upper()} SIGNAL")
        print('='*50)
        
        try:
            generator = SignalGenerator(asset)
            signal = generator.generate_signal()
            
            print(f"\nSignal: {signal['signal']}")
            print(f"Confidence: {signal['confidence']:.1%}")
            print(f"\nEntry Price: ${signal['entry_price']:,.2f}")
            print(f"Target Price: ${signal['target_price']:,.2f}")
            print(f"Stop Loss: ${signal['stop_loss']:,.2f}")
            print(f"Risk/Reward: {signal['risk_reward']:.2f}")
            
            print(f"\nFactor Scores:")
            print(f"  Bullish: {signal['bullish_score']:.2f}")
            print(f"  Bearish: {signal['bearish_score']:.2f}")
            
            print(f"\nReasons:")
            for reason in signal['reasons']:
                print(f"  - {reason}")
        
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nDone!")
