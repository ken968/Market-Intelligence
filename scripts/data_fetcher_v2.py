import yfinance as yf
import pandas as pd
import os
from datetime import datetime

class MultiAssetFetcher:
    """
    Unified data fetcher for Gold, Bitcoin, and US Stocks.
    Supports different date ranges per asset type.
    """
    
    def __init__(self):
        self.macro_tickers = {
            'USD_Index': 'DX-Y.NYB',
            'VIX': '^VIX',
            'US_10Y_Yield': '^TNX'
        }
        
        # Bitcoin: Full history from 2009
        self.btc_config = {
            'ticker': 'BTC-USD',
            'start_date': '2009-01-01',  # Genesis block era
            'filename': 'data/btc_macro_data.csv'
        }
        
        # Gold: 10 years (existing)
        self.gold_config = {
            'ticker': 'GC=F',
            'period': '10y',
            'filename': 'data/gold_macro_data.csv'
        }
        
        # US Stocks: 10 years
        self.stock_tickers = {
            # Indices
            'SPY': 'SPY',      # S&P 500
            'QQQ': 'QQQ',      # Nasdaq 100
            'DIA': 'DIA',      # Dow Jones
            
            # Magnificent 7
            'AAPL': 'AAPL',
            'MSFT': 'MSFT',
            'GOOGL': 'GOOGL',
            'AMZN': 'AMZN',
            'NVDA': 'NVDA',
            'META': 'META',
            'TSLA': 'TSLA',
            
            # TSMC
            'TSM': 'TSM'
        }
        
        self.stock_config = {
            'period': '10y',
            'filename_template': 'data/{ticker}_macro_data.csv'  # Per-stock files
        }
    
    def fetch_macro_indicators(self):
        """Fetch shared macro indicators (DXY, VIX, Yield)"""
        print("System: Fetching macro indicators (DXY, VIX, US10Y)...")
        
        try:
            data = yf.download(
                list(self.macro_tickers.values()), 
                period="10y", 
                interval="1d",
                progress=False
            )
            
            if data.empty:
                print("Error: No macro data retrieved.")
                return None
            
            df = data['Close'].ffill()
            
            # Map columns explicitly by ticker to avoid swapping
            column_mapping = {
                self.macro_tickers['USD_Index']: 'DXY',
                self.macro_tickers['VIX']: 'VIX',
                self.macro_tickers['US_10Y_Yield']: 'Yield_10Y'
            }
            
            # Rename based on actual ticker names in columns
            df.columns = [column_mapping.get(col, col) for col in df.columns]
            
            # Ensure they are in the expected order
            df = df[['DXY', 'VIX', 'Yield_10Y']]
            df = df.dropna()
            
            df.to_csv('data/macro_indicators.csv')
            print(f"System: {len(df)} macro records saved.")
            return df
            
        except Exception as e:
            print(f"Error fetching macro data: {e}")
            return None
    
    def _calculate_ema(self, data, window=90):
        """Calculate Exponential Moving Average"""
        return data.ewm(span=window, adjust=False).mean()

    def fetch_gold_data(self):
        """Fetch Gold data (existing logic)"""
        print("\n=== GOLD DATA ===")
        print("System: Fetching Gold futures (GC=F) - 10 years...")
        
        try:
            data = yf.download(
                self.gold_config['ticker'], 
                period=self.gold_config['period'], 
                interval="1d",
                progress=False
            )
            
            if data.empty:
                print("Error: No Gold data retrieved.")
                return False
            
            # Robust way to handle both single and MultiIndex from yfinance
            if 'Close' in data.columns and isinstance(data.columns, pd.MultiIndex):
                df = data['Close'].iloc[:, 0].to_frame(name='Gold')
            elif 'Close' in data.columns:
                df = data[['Close']].rename(columns={'Close': 'Gold'})
            else:
                # Fallback if structure is unexpected
                df = data.iloc[:, 0].to_frame(name='Gold')
                
            df = df.ffill().dropna()
            
            # Add Technical Indicators
            df['EMA_90'] = self._calculate_ema(df['Gold'], 90)
            
            # Merge with macro indicators
            if os.path.exists('data/macro_indicators.csv'):
                macro = pd.read_csv('data/macro_indicators.csv', index_col=0, parse_dates=True)
                df = df.join(macro, how='inner')
            
            df.to_csv(self.gold_config['filename'])
            print(f"System: Success. {len(df)} Gold records saved to '{self.gold_config['filename']}'.")
            return True
            
        except Exception as e:
            print(f"Error fetching Gold data: {e}")
            return False
    
    def fetch_bitcoin_data(self):
        """Fetch Bitcoin data from 2009 (full history)"""
        print("\n=== BITCOIN DATA ===")
        print("System: Fetching Bitcoin (BTC-USD) from 2009...")
        
        try:
            # Use start date instead of period for full history
            data = yf.download(
                self.btc_config['ticker'], 
                start=self.btc_config['start_date'],
                end=datetime.now().strftime('%Y-%m-%d'),
                interval="1d",
                progress=False
            )
            
            if data.empty:
                print("Error: No Bitcoin data retrieved.")
                return False
            
            # Robust way to handle both single and MultiIndex from yfinance
            if 'Close' in data.columns and isinstance(data.columns, pd.MultiIndex):
                df = data['Close'].iloc[:, 0].to_frame(name='BTC')
            elif 'Close' in data.columns:
                df = data[['Close']].rename(columns={'Close': 'BTC'})
            else:
                df = data.iloc[:, 0].to_frame(name='BTC')
                
            df = df.ffill().dropna()
            
            # Add BTC-specific features & Indicators
            df['Halving_Cycle'] = self._calculate_halving_cycle(df.index)
            df['EMA_90'] = self._calculate_ema(df['BTC'], 90)
            
            # Merge with macro indicators (only where dates overlap)
            if os.path.exists('data/macro_indicators.csv'):
                macro = pd.read_csv('data/macro_indicators.csv', index_col=0, parse_dates=True)
                df = df.join(macro, how='left')
                # Fill NaN for early dates where macro data doesn't exist
                df['DXY'].fillna(method='bfill', inplace=True)
                df['VIX'].fillna(method='bfill', inplace=True)
                df['Yield_10Y'].fillna(method='bfill', inplace=True)
            
            df.to_csv(self.btc_config['filename'])
            print(f"System: Success. {len(df)} BTC records saved (from {df.index[0].date()} to {df.index[-1].date()}).")
            return True
            
        except Exception as e:
            print(f"Error fetching Bitcoin data: {e}")
            return False
    
    def fetch_stock_data(self, ticker=None):
        """
        Fetch US Stock data.
        If ticker=None, fetches all configured stocks.
        """
        print("\n=== US STOCKS DATA ===")
        
        tickers_to_fetch = [ticker] if ticker else list(self.stock_tickers.values())
        
        success_count = 0
        for tick in tickers_to_fetch:
            try:
                print(f"System: Fetching {tick}...")
                
                data = yf.download(
                    tick, 
                    period=self.stock_config['period'], 
                    interval="1d",
                    progress=False
                )
                
                if data.empty:
                    print(f"Warning: No data for {tick}")
                    continue
                
                # Robust way to handle both single and MultiIndex from yfinance
                if 'Close' in data.columns and isinstance(data.columns, pd.MultiIndex):
                    df = data['Close'].iloc[:, 0].to_frame(name=tick)
                elif 'Close' in data.columns:
                    df = data[['Close']].rename(columns={'Close': tick})
                else:
                    df = data.iloc[:, 0].to_frame(name=tick)
                    
                df = df.ffill().dropna()
                
                # Add Technical Indicators
                df['EMA_90'] = self._calculate_ema(df[tick], 90)
                
                # Merge with macro indicators
                if os.path.exists('data/macro_indicators.csv'):
                    macro = pd.read_csv('data/macro_indicators.csv', index_col=0, parse_dates=True)
                    df = df.join(macro, how='inner')
                
                filename = self.stock_config['filename_template'].format(ticker=tick)
                df.to_csv(filename)
                print(f"  -> {len(df)} records saved to '{filename}'")
                success_count += 1
                
            except Exception as e:
                print(f"Error fetching {tick}: {e}")
                continue
        
        print(f"System: {success_count}/{len(tickers_to_fetch)} stocks fetched successfully.")
        return success_count > 0
    
    def _calculate_halving_cycle(self, dates):
        """
        Calculate Bitcoin halving cycle feature.
        Halving events: 2012-11-28, 2016-07-09, 2020-05-11, 2024-04-19
        Next expected: ~2028-04
        """
        halving_dates = pd.to_datetime([
            '2012-11-28', 
            '2016-07-09', 
            '2020-05-11', 
            '2024-04-19',
            '2028-04-01'  # Estimated
        ])
        
        cycle = []
        for date in dates:
            # Find closest halving (past or future)
            days_to_halving = min(abs((date - h).days) for h in halving_dates)
            cycle.append(days_to_halving)
        
        return cycle
    
    def fetch_all(self):
        """Fetch all assets (Gold + BTC + Stocks)"""
        print("="*50)
        print("MULTI-ASSET DATA SYNC INITIATED")
        print("="*50)
        
        # Step 1: Macro indicators (shared)
        macro = self.fetch_macro_indicators()
        if macro is None:
            print("Critical Error: Cannot proceed without macro data.")
            return False
        
        # Step 2: Individual assets
        results = {
            'Gold': self.fetch_gold_data(),
            'Bitcoin': self.fetch_bitcoin_data(),
            'Stocks': self.fetch_stock_data()
        }
        
        print("\n" + "="*50)
        print("SYNC SUMMARY")
        print("="*50)
        for asset, status in results.items():
            status_text = 'Success' if status else 'Failed'
            print(f"{asset}: {status_text}")
        print("="*50)
        
        return all(results.values())


def fetch_global_gold_data():
    """
    Legacy function for backward compatibility.
    Calls the new MultiAssetFetcher for Gold only.
    """
    fetcher = MultiAssetFetcher()
    fetcher.fetch_macro_indicators()
    return fetcher.fetch_gold_data()


if __name__ == "__main__":
    import sys
    
    fetcher = MultiAssetFetcher()
    
    # CLI support
    if len(sys.argv) > 1:
        asset = sys.argv[1].lower()
        
        if asset == 'gold':
            fetcher.fetch_macro_indicators()
            success = fetcher.fetch_gold_data()
        elif asset == 'btc' or asset == 'bitcoin':
            fetcher.fetch_macro_indicators()
            success = fetcher.fetch_bitcoin_data()
        elif asset == 'stocks':
            fetcher.fetch_macro_indicators()
            success = fetcher.fetch_stock_data()
        elif asset.upper() in fetcher.stock_tickers:
            fetcher.fetch_macro_indicators()
            success = fetcher.fetch_stock_data(asset.upper())
        else:
            print(f"Unknown asset: {asset}")
            print("Usage: python data_fetcher_v2.py [gold|btc|stocks|AAPL|NVDA|...]")
            sys.exit(1)
            
        if not success:
            sys.exit(1)
    else:
        # Default: Fetch all if no args
        success = fetcher.fetch_all()
        if not success:
            sys.exit(1)
