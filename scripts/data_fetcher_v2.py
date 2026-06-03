import yfinance as yf
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from utils.data_store import MarketDataStore


class MultiAssetFetcher:
    """
    Unified data fetcher for Gold, Bitcoin, and US Stocks.
    Supports different date ranges per asset type.
    """
    
    def __init__(self):
        self.store = MarketDataStore()
        self.macro_tickers = {
            'USD_Index': 'DX-Y.NYB',
            'VIX': '^VIX',
            'US_10Y_Yield': '^TNX',
            'Crude_Oil': 'CL=F'
        }
        
        # Bitcoin: Full history from 2009
        self.btc_config = {
            'ticker': 'BTC-USD',
            'start_date': '2009-01-01',  # Genesis block era
            'filename': 'data/btc_global_insights.csv'
        }
        
        # Gold: 10 years — use start date (not period) to avoid yfinance cache returning stale data
        self.gold_config = {
            'ticker': 'GC=F',
            'start_date': '2015-01-01',
            'filename': 'data/gold_global_insights.csv'
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
            'start_date': '2015-01-01',  # Explicit start date — avoids yfinance period cache returning stale data
            'filename_template': 'data/{ticker}_global_insights.csv'  # Unified naming
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
                self.macro_tickers['US_10Y_Yield']: 'Yield_10Y',
                self.macro_tickers['Crude_Oil']: 'Oil_Price'
            }
            
            # Rename based on actual ticker names in columns
            df.columns = [column_mapping.get(col, col) for col in df.columns]
            
            # Ensure they are in the expected order
            df = df[['DXY', 'VIX', 'Yield_10Y', 'Oil_Price']]
            df = df.dropna()
            
            self.store.write_table('macro_indicators', df, 'data/macro_indicators.csv')
            print(f"System: {len(df)} macro records saved.")
            return df
            
        except Exception as e:
            print(f"Error fetching macro data: {e}")
            return None
    
    def _calculate_ema(self, data, window=90):
        """Calculate Exponential Moving Average"""
        return data.ewm(span=window, adjust=False).mean()

    def _robust_fill_nas(self, df, columns):
        """Multi-tier NaN filling: ffill -> bfill -> fillna(median/0)."""
        for col in columns:
            if col in df.columns:
                # 1. Forward fill (carry last known value)
                df[col] = df[col].ffill()
                # 2. Backward fill (for early dates where no prior data exists)
                df[col] = df[col].bfill()
                
                # 3. If still NaN (meaning the column was completely empty)
                if df[col].isna().any():
                    median_val = df[col].median()
                    fill_val = 0 if pd.isna(median_val) else median_val
                    df[col] = df[col].fillna(fill_val)
                    print(f"    Warning: Column '{col}' had missing values. Filled with {fill_val}.")
            else:
                # Column entirely missing from merged data
                df[col] = 0.0
                print(f"    Warning: Column '{col}' completely missing! Filled with 0.0.")
        return df

    def _extract_ohlcv(self, data: 'pd.DataFrame', price_col: str) -> 'pd.DataFrame':
        """
        Extract full OHLCV from yfinance download result.
        Handles both flat columns and MultiIndex (multi-ticker download).
        Always returns DataFrame with columns: [price_col, Open, High, Low, Volume]
        """
        cols_needed = ['Open', 'High', 'Low', 'Close', 'Volume']

        if isinstance(data.columns, pd.MultiIndex):
            # MultiIndex: (PriceType, Ticker) — flatten to single level
            extracted = {}
            for price_type in cols_needed:
                if price_type in data.columns.get_level_values(0):
                    extracted[price_type] = data[price_type].iloc[:, 0]
            df = pd.DataFrame(extracted)
        else:
            available = [c for c in cols_needed if c in data.columns]
            df = data[available].copy()

        # Rename Close to the asset-specific name (e.g., 'Gold', 'BTC', 'SPY')
        if 'Close' in df.columns:
            df = df.rename(columns={'Close': price_col})

        return df.ffill().dropna(subset=[price_col])

    def fetch_gold_data(self):
        """Fetch Gold data (existing logic)"""
        print("\n=== GOLD DATA ===")
        print("System: Fetching Gold futures (GC=F) - 10 years...")
        
        try:
            data = yf.download(
                self.gold_config['ticker'],
                start=self.gold_config['start_date'],
                # No 'end' parameter — forces yfinance to return absolute latest data
                interval="1d",
                progress=False
            )
            
            if data.empty:
                print("Error: No Gold data retrieved.")
                return False
            
            # Extract full OHLCV (Open, High, Low, Close, Volume)
            df = self._extract_ohlcv(data, 'Gold')

            # Add Technical Indicators
            df['EMA_90'] = self._calculate_ema(df['Gold'], 90)

            # Garman-Klass Volatility (requires OHLCV)
            try:
                from utils.feature_engineering import compute_garman_klass_vol
                df['GK_Vol_21d'] = compute_garman_klass_vol(
                    df, open_col='Open', high_col='High',
                    low_col='Low', close_col='Gold', window=21
                )
            except Exception as gk_err:
                print(f"Warning: GK Vol skipped: {gk_err}")
                df['GK_Vol_21d'] = 0.0
            
            # Merge with macro indicators
            macro = None
            try:
                macro = self.store.read_table('macro_indicators', format='pandas')
                if 'Date' in macro.columns:
                    macro = macro.set_index('Date')
                    macro.index = pd.to_datetime(macro.index)
            except Exception:
                if os.path.exists('data/macro_indicators.csv'):
                    macro = pd.read_csv('data/macro_indicators.csv', index_col=0, parse_dates=True)
            
            if macro is not None:
                df = df.join(macro, how='left')
                df = self._robust_fill_nas(df, ['DXY', 'VIX', 'Yield_10Y', 'Oil_Price'])
            
            # Merge FRED indicators (CPI, PPI, PCE, NFP)
            fred = None
            try:
                fred = self.store.read_table('fred_indicators', format='pandas')
                if 'Date' in fred.columns:
                    fred = fred.set_index('Date')
                    fred.index = pd.to_datetime(fred.index)
            except Exception:
                if os.path.exists('data/fred_indicators.csv'):
                    fred = pd.read_csv('data/fred_indicators.csv', index_col=0, parse_dates=True)
            
            if fred is not None:
                df = df.join(fred, how='left')
                fred_cols = ['CPI_MoM', 'PPI_MoM', 'PCE_MoM', 'NFP_Change', 'MacroEvent_Flag', 'M2_MoM', 'M2_YoY', 'YieldCurve_10Y2Y', 'Yield_10Y_Rate', 'Breakeven_5Y5Y', 'M2_Liquidity_Spike', 'Credit_Spread', 'Credit_Stress_Flag']
                df = self._robust_fill_nas(df, fred_cols)
            
            # Lagged macro features & Sentiment_Std
            try:
                from utils.feature_engineering import add_lagged_macro_features
                df = add_lagged_macro_features(df, lags_months=[3, 6])
            except Exception as lag_err:
                print(f"Warning: Lagged features skipped: {lag_err}")

            # Preserve or Initialize Sentiment Column
            df = self._preserve_sentiment(df, self.gold_config['filename'])
            if 'Sentiment' in df.columns:
                df['Sentiment_Std'] = df['Sentiment'].rolling(5, min_periods=1).std().fillna(0)
            
            # ── PHASE 3: Dynamic Regime Features ──────────────────────────────
            try:
                from utils.feature_utils import add_dynamic_regime_features
                df = add_dynamic_regime_features(df, price_col='Gold', asset_key='gold')
            except Exception as feat_err:
                print(f"Warning: Dynamic regime features skipped for Gold: {feat_err}")
            # ──────────────────────────────────────────────────────────────────

            df_to_save = df.reset_index()
            self.store.write_table('gold_global_insights', df_to_save, self.gold_config['filename'])
            try:
                from utils.counterfactual_logger import auto_resolve_all_outcomes
                auto_resolve_all_outcomes('gold', df, 'Gold')
            except Exception as e:
                print(f"Warning: Failed to auto-resolve Gold outcomes: {e}")
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
            # UPDATED: Removed 'end' parameter to ensure we get the absolute latest data (including today)
            data = yf.download(
                self.btc_config['ticker'], 
                start=self.btc_config['start_date'],
                # end=datetime.now().strftime('%Y-%m-%d'),  <-- REMOVED to fix off-by-one error
                interval="1d",
                progress=False
            )
            
            if data.empty:
                print("Error: No Bitcoin data retrieved.")
                return False
            
            # Extract full OHLCV (Open, High, Low, Close, Volume)
            df = self._extract_ohlcv(data, 'BTC')

            # Add BTC-specific features & Indicators
            df['Halving_Cycle'] = self._calculate_halving_cycle(df.index)
            df['EMA_90'] = self._calculate_ema(df['BTC'], 90)

            # Garman-Klass Volatility — especially relevant for BTC (high vol asset)
            try:
                from utils.feature_engineering import compute_garman_klass_vol
                df['GK_Vol_21d'] = compute_garman_klass_vol(
                    df, open_col='Open', high_col='High',
                    low_col='Low', close_col='BTC', window=21
                )
            except Exception as gk_err:
                print(f"Warning: GK Vol skipped: {gk_err}")
                df['GK_Vol_21d'] = 0.0
            
            # Merge with macro indicators (only where dates overlap)
            macro = None
            try:
                macro = self.store.read_table('macro_indicators', format='pandas')
                if 'Date' in macro.columns:
                    macro = macro.set_index('Date')
                    macro.index = pd.to_datetime(macro.index)
            except Exception:
                if os.path.exists('data/macro_indicators.csv'):
                    macro = pd.read_csv('data/macro_indicators.csv', index_col=0, parse_dates=True)
            
            if macro is not None:
                df = df.join(macro, how='left')
                df = self._robust_fill_nas(df, ['DXY', 'VIX', 'Yield_10Y', 'Oil_Price'])
            
            # Merge FRED indicators (CPI, PPI, PCE, NFP)
            fred = None
            try:
                fred = self.store.read_table('fred_indicators', format='pandas')
                if 'Date' in fred.columns:
                    fred = fred.set_index('Date')
                    fred.index = pd.to_datetime(fred.index)
            except Exception:
                if os.path.exists('data/fred_indicators.csv'):
                    fred = pd.read_csv('data/fred_indicators.csv', index_col=0, parse_dates=True)
            
            if fred is not None:
                df = df.join(fred, how='left')
                fred_cols = ['CPI_MoM', 'PPI_MoM', 'PCE_MoM', 'NFP_Change', 'MacroEvent_Flag', 'M2_MoM', 'M2_YoY', 'YieldCurve_10Y2Y', 'Yield_10Y_Rate', 'Breakeven_5Y5Y', 'M2_Liquidity_Spike']
                df = self._robust_fill_nas(df, fred_cols)
            
            # Lagged macro features & Sentiment_Std
            try:
                from utils.feature_engineering import add_lagged_macro_features
                df = add_lagged_macro_features(df, lags_months=[3, 6])
            except Exception as lag_err:
                print(f"Warning: Lagged features skipped: {lag_err}")

            # Preserve or Initialize Sentiment Column
            df = self._preserve_sentiment(df, self.btc_config['filename'])
            if 'Sentiment' in df.columns:
                df['Sentiment_Std'] = df['Sentiment'].rolling(5, min_periods=1).std().fillna(0)
            
            # ── PHASE 3: Dynamic Regime Features ──────────────────────────────
            try:
                from utils.feature_utils import add_dynamic_regime_features
                df = add_dynamic_regime_features(df, price_col='BTC', asset_key='btc')
            except Exception as feat_err:
                print(f"Warning: Dynamic regime features skipped for BTC: {feat_err}")
            # ──────────────────────────────────────────────────────────────────

            df_to_save = df.reset_index()
            self.store.write_table('btc_global_insights', df_to_save, self.btc_config['filename'])
            try:
                from utils.counterfactual_logger import auto_resolve_all_outcomes
                auto_resolve_all_outcomes('btc', df, 'BTC')
            except Exception as e:
                print(f"Warning: Failed to auto-resolve BTC outcomes: {e}")
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
                    start=self.stock_config['start_date'],
                    # No 'end' parameter — forces yfinance to return absolute latest data
                    interval="1d",
                    progress=False
                )
                
                if data.empty:
                    print(f"Warning: No data for {tick}")
                    continue
                
                # Extract full OHLCV
                df = self._extract_ohlcv(data, tick)

                # Add Technical Indicators
                df['EMA_90'] = self._calculate_ema(df[tick], 90)

                # Garman-Klass Volatility
                try:
                    from utils.feature_engineering import compute_garman_klass_vol
                    df['GK_Vol_21d'] = compute_garman_klass_vol(
                        df, open_col='Open', high_col='High',
                        low_col='Low', close_col=tick, window=21
                    )
                except Exception as gk_err:
                    df['GK_Vol_21d'] = 0.0
                
                # Merge with macro indicators
                macro = None
                try:
                    macro = self.store.read_table('macro_indicators', format='pandas')
                    if 'Date' in macro.columns:
                        macro = macro.set_index('Date')
                        macro.index = pd.to_datetime(macro.index)
                except Exception:
                    if os.path.exists('data/macro_indicators.csv'):
                        macro = pd.read_csv('data/macro_indicators.csv', index_col=0, parse_dates=True)
                
                if macro is not None:
                    df = df.join(macro, how='left')
                    df = self._robust_fill_nas(df, ['DXY', 'VIX', 'Yield_10Y', 'Oil_Price'])
                
                # Merge FRED indicators (CPI, PPI, PCE, NFP)
                fred = None
                try:
                    fred = self.store.read_table('fred_indicators', format='pandas')
                    if 'Date' in fred.columns:
                        fred = fred.set_index('Date')
                        fred.index = pd.to_datetime(fred.index)
                except Exception:
                    if os.path.exists('data/fred_indicators.csv'):
                        fred = pd.read_csv('data/fred_indicators.csv', index_col=0, parse_dates=True)
                
                if fred is not None:
                    df = df.join(fred, how='left')
                    fred_cols = ['CPI_MoM', 'PPI_MoM', 'PCE_MoM', 'NFP_Change', 'MacroEvent_Flag', 'M2_MoM', 'M2_YoY', 'YieldCurve_10Y2Y', 'Yield_10Y_Rate', 'Breakeven_5Y5Y', 'M2_Liquidity_Spike']
                    df = self._robust_fill_nas(df, fred_cols)

                # Lagged macro features & Sentiment_Std
                try:
                    from utils.feature_engineering import add_lagged_macro_features
                    df = add_lagged_macro_features(df, lags_months=[3, 6])
                except Exception as lag_err:
                    pass
                if 'Sentiment' in df.columns:
                    df['Sentiment_Std'] = df['Sentiment'].rolling(5, min_periods=1).std().fillna(0)
                
                filename = self.stock_config['filename_template'].format(ticker=tick)
                
                # Preserve or Initialize Sentiment Column
                df = self._preserve_sentiment(df, filename)

                # ── PHASE 3: Dynamic Regime Features ──────────────────────────
                try:
                    from utils.feature_utils import add_dynamic_regime_features
                    df = add_dynamic_regime_features(df, price_col=tick, asset_key=tick)
                except Exception as feat_err:
                    print(f"  Warning: Dynamic regime features skipped for {tick}: {feat_err}")
                # ──────────────────────────────────────────────────────────────

                df_to_save = df.reset_index()
                table_name = f"{tick.lower()}_global_insights"
                self.store.write_table(table_name, df_to_save, filename)
                try:
                    from utils.counterfactual_logger import auto_resolve_all_outcomes
                    auto_resolve_all_outcomes(tick.lower(), df, tick)
                except Exception as e:
                    print(f"Warning: Failed to auto-resolve {tick} outcomes: {e}")
                print(f"  -> {len(df)} records saved to '{filename}'")
                success_count += 1
                
            except Exception as e:
                print(f"Error fetching {tick}: {e}")
                continue
        
        print(f"System: {success_count}/{len(tickers_to_fetch)} stocks fetched successfully.")
        return success_count > 0
    
    def _calculate_ema(self, series, period):
        return series.ewm(span=period, adjust=False).mean()
    
    def _preserve_sentiment(self, new_df, filename):
        """
        Retains 'Sentiment' column from existing file if available,
        otherwise initializes with 0.
        """
        table_name = os.path.splitext(os.path.basename(filename))[0].lower()
        old_df = None
        try:
            old_df = self.store.read_table(table_name, format='pandas')
        except Exception:
            if os.path.exists(filename):
                try:
                    old_df = pd.read_csv(filename)
                except Exception:
                    pass
        
        if old_df is not None and 'Sentiment' in old_df.columns:
            try:
                # Merge based on Date (assuming index is date in new_df)
                old_df['Date'] = pd.to_datetime(old_df['Date'])
                old_sentiment = old_df[['Date', 'Sentiment']].set_index('Date')
                
                # Align dates and join
                new_df = new_df.join(old_sentiment, how='left')
                # Fill gaps in sentiment (new dates) with 0 or ffill
                new_df['Sentiment'] = new_df['Sentiment'].ffill().fillna(0)
                return new_df
            except Exception as e:
                print(f"Warning: Could not preserve sentiment: {e}")
        
        # Default: add empty sentiment column if missing
        if 'Sentiment' not in new_df.columns:
            new_df['Sentiment'] = 0.0
            
        return new_df

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
