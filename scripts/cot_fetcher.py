"""
COT (Commitments of Traders) Fetcher
Downloads weekly COT data from the CFTC, extracting smart money positions for Gold and Bitcoin.
"""

import requests
from io import BytesIO
import zipfile
import pandas as pd
import os
import sys
from datetime import datetime

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_store import MarketDataStore

class COTFetcher:
    def __init__(self):
        # We fetch the current year's COT report. 
        # In a real production system, this could loop over multiple years.
        current_year = datetime.now().year
        self.url = f"https://www.cftc.gov/files/dea/history/deacot{current_year}.zip"
        self.store = MarketDataStore()
        
        # Target markets by CFTC Contract Market Code
        self.MARKETS = {
            'gold': '088691',
            'btc': '133097',
            'spy': '13874A'
        }
        
    def fetch_and_process(self):
        print(f"Downloading COT Data from {self.url}...")
        
        try:
            r = requests.get(self.url, timeout=30)
            r.raise_for_status()
        except Exception as e:
            print(f"Failed to download COT data: {e}")
            return False
            
        try:
            with zipfile.ZipFile(BytesIO(r.content)) as z:
                filename = z.namelist()[0]
                # Read CSV and ensure market code is read as string
                df = pd.read_csv(z.open(filename), low_memory=False, dtype={'CFTC Contract Market Code': str})
        except Exception as e:
            print(f"Failed to parse COT zip: {e}")
            return False
            
        print(f"Successfully loaded {len(df)} records from CFTC.")
        
        # Required columns mapping
        cols_to_keep = {
            'As of Date in Form YYYY-MM-DD': 'Date',
            'CFTC Contract Market Code': 'Market_Code',
            'Noncommercial Positions-Long (All)': 'NonComm_Long',
            'Noncommercial Positions-Short (All)': 'NonComm_Short',
            'Commercial Positions-Long (All)': 'Comm_Long',
            'Commercial Positions-Short (All)': 'Comm_Short'
        }
        
        missing = [c for c in cols_to_keep.keys() if c not in df.columns]
        if missing:
            print(f"Error: Missing columns in COT data: {missing}")
            return False
            
        df = df[list(cols_to_keep.keys())].rename(columns=cols_to_keep)
        
        # Process each asset
        all_success = True
        for asset, market_code in self.MARKETS.items():
            asset_df = df[df['Market_Code'] == market_code].copy()
            if asset_df.empty:
                print(f"No COT data found for {asset} (Code: {market_code})")
                continue
                
            # Convert to datetime and apply 3-day shift (Tuesday 'As of Date' -> Friday 'Release Date')
            # This is critical to prevent Look-Ahead Bias / Data Leakage in ML backtesting.
            asset_df['Date'] = pd.to_datetime(asset_df['Date']) + pd.to_timedelta('3 days')
            asset_df = asset_df.sort_values('Date').reset_index(drop=True)
            
            # Calculate Net Positions
            # Commercial (Smart Money / Hedgers)
            asset_df['Net_Commercial'] = asset_df['Comm_Long'] - asset_df['Comm_Short']
            # Non-Commercial (Speculators)
            asset_df['Net_NonCommercial'] = asset_df['NonComm_Long'] - asset_df['NonComm_Short']
            
            # Calculate ratios or normalized features for ML
            asset_df['Net_Commercial_Long'] = asset_df['Comm_Long'] / (asset_df['Comm_Long'] + asset_df['Comm_Short'] + 1e-5)
            
            # Keep final columns
            final_df = asset_df[['Date', 'Net_Commercial', 'Net_NonCommercial', 'Net_Commercial_Long']]
            final_df['Date'] = final_df['Date'].dt.strftime('%Y-%m-%d')
            
            # Save to DuckDB
            table_name = f'cot_{asset}'
            csv_path = f'data/cot_{asset}.csv'
            
            try:
                self.store.write_table(table_name, final_df, csv_path)
                print(f"Saved COT data for {asset.upper()} (Records: {len(final_df)})")
            except Exception as e:
                print(f"Failed to save COT data for {asset}: {e}")
                all_success = False
                
        return all_success

if __name__ == "__main__":
    fetcher = COTFetcher()
    success = fetcher.fetch_and_process()
    if not success:
        sys.exit(1)
