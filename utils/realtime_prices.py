"""
Real-Time Price Service
Fetch latest market prices using yfinance (free, 15-min delay acceptable)
"""

import yfinance as yf
from datetime import datetime, time
import pytz
from typing import Dict
import pandas as pd


class RealtimePriceService:
    """Service for fetching latest market prices"""
    
    def __init__(self, cache_minutes: int = 15):
        """
        Initialize price service
        
        Args:
            cache_minutes: Cache duration in minutes to avoid rate limits
        """
        self.cache_minutes = cache_minutes
        self.cache = {}
        self.cache_timestamps = {}
    
    def get_latest_price(self, ticker: str) -> Dict:
        """
        Get latest price for ticker
        
        Args:
            ticker: Stock ticker (e.g., 'AAPL', 'BTC-USD')
        
        Returns:
            dict: {
                'price': float,
                'change': float,
                'change_pct': float,
                'timestamp': datetime,
                'market_status': 'open' | 'closed' | 'pre' | 'post',
                'volume': int
            }
        """
        
        # Check cache
        cache_key = ticker.upper()
        if cache_key in self.cache:
            cache_age = (datetime.now() - self.cache_timestamps[cache_key]).seconds / 60
            if cache_age < self.cache_minutes:
                return self.cache[cache_key]
        
        try:
            # Fetch from yfinance
            stock = yf.Ticker(ticker)
            
            # Get latest data
            hist = stock.history(period='2d')  # Get 2 days for change calculation
            
            if hist.empty:
                return self._error_response(ticker, "No data available")
            
            # Latest price
            latest_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2] if len(hist) >= 2 else latest_price
            
            change = latest_price - prev_price
            change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
            
            # Volume
            volume = int(hist['Volume'].iloc[-1])
            
            # Market status
            market_status = self._get_market_status(ticker)
            
            result = {
                'price': float(latest_price),
                'change': float(change),
                'change_pct': float(change_pct),
                'timestamp': datetime.now(),
                'market_status': market_status,
                'volume': volume,
                'ticker': ticker.upper()
            }
            
            # Update cache
            self.cache[cache_key] = result
            self.cache_timestamps[cache_key] = datetime.now()
            
            return result
            
        except Exception as e:
            return self._error_response(ticker, str(e))
    
    def _get_market_status(self, ticker: str) -> str:
        """
        Determine if market is open, closed, pre-market, or after-hours
        
        Args:
            ticker: Stock ticker
        
        Returns:
            'open' | 'closed' | 'pre' | 'post'
        """
        
        try:
            # US markets (NYSE/NASDAQ) - Eastern Time
            now_et = datetime.now(pytz.timezone('US/Eastern'))
            current_time = now_et.time()
            
            # Market hours: 9:30 AM - 4:00 PM ET
            market_open = time(9, 30)
            market_close = time(16, 0)
            pre_market_start = time(4, 0)
            post_market_end = time(20, 0)
            
            # Check if weekday
            if now_et.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return 'closed'
            
            # Check time
            if market_open <= current_time < market_close:
                return 'open'
            elif pre_market_start <= current_time < market_open:
                return 'pre'
            elif market_close <= current_time < post_market_end:
                return 'post'
            else:
                return 'closed'
                
        except:
            return 'unknown'
    
    def _error_response(self, ticker: str, error_msg: str) -> Dict:
        """Return error response dict"""
        return {
            'price': 0,
            'change': 0,
            'change_pct': 0,
            'timestamp': datetime.now(),
            'market_status': 'error',
            'volume': 0,
            'ticker': ticker.upper(),
            'error': error_msg
        }
    
    def get_multiple_prices(self, tickers: list) -> Dict[str, Dict]:
        """
        Get prices for multiple tickers
        
        Args:
            tickers: List of ticker symbols
        
        Returns:
            dict: {ticker: price_data}
        """
        results = {}
        
        for ticker in tickers:
            results[ticker] = self.get_latest_price(ticker)
        
        return results


# ==================== TESTING ====================

if __name__ == "__main__":
    print("="*60)
    print("REAL-TIME PRICE SERVICE - TESTING")
    print("="*60)
    
    service = RealtimePriceService()
    
    # Test individual ticker
    print("\n--- Test 1: Single Ticker (AAPL) ---")
    apple_data = service.get_latest_price('AAPL')
    
    if 'error' not in apple_data:
        print(f"Ticker: {apple_data['ticker']}")
        print(f"Price: ${apple_data['price']:.2f}")
        print(f"Change: ${apple_data['change']:.2f} ({apple_data['change_pct']:+.2f}%)")
        print(f"Market Status: {apple_data['market_status']}")
        print(f"Volume: {apple_data['volume']:,}")
        print(f"Timestamp: {apple_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"Error: {apple_data['error']}")
    
    # Test multiple tickers
    print("\n--- Test 2: Multiple Tickers ---")
    tickers = ['SPY', 'QQQ', 'AAPL', 'BTC-USD']
    all_prices = service.get_multiple_prices(tickers)
    
    print(f"\n{'Ticker':<10} {'Price':<12} {'Change %':<10} {'Status':<10}")
    print("-" * 45)
    for ticker, data in all_prices.items():
        if 'error' not in data:
            print(f"{data['ticker']:<10} ${data['price']:<11,.2f} {data['change_pct']:<+9.2f}% {data['market_status']:<10}")
        else:
            print(f"{ticker:<10} ERROR: {data['error']}")
