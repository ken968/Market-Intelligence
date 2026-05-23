import os
import sys
import requests
import zipfile
import pandas as pd
from io import BytesIO
import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_store import MarketDataStore

MARKETS = {
    'gold': '088691',
    'btc': '133741',
    'spy': '13874A'
}

def backfill_cot_history(years):
    print("Backfilling COT History...")
    all_dfs = {asset: [] for asset in MARKETS.keys()}
    
    for year in years:
        url = f"https://www.cftc.gov/files/dea/history/deacot{year}.zip"
        print(f"  Fetching COT for {year}...")
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                print(f"    Failed to fetch {year} (status: {r.status_code}). Trying fut_fin_txt_{year}.zip for Financials...")
                continue
                
            with zipfile.ZipFile(BytesIO(r.content)) as z:
                filename = z.namelist()[0]
                df = pd.read_csv(z.open(filename), low_memory=False, dtype={'CFTC Contract Market Code': str})
                
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
                print(f"    Missing columns in {year}: {missing}")
                continue
                
            df = df[list(cols_to_keep.keys())].rename(columns=cols_to_keep)
            
            for asset, market_code in MARKETS.items():
                asset_df = df[df['Market_Code'] == market_code].copy()
                if not asset_df.empty:
                    # Convert to datetime and apply 3-day shift (Tuesday 'As of Date' -> Friday 'Release Date')
                    asset_df['Date'] = pd.to_datetime(asset_df['Date']) + pd.to_timedelta('3 days')
                    asset_df = asset_df.sort_values('Date').reset_index(drop=True)
                    
                    # Net positions
                    asset_df['Net_Commercial'] = asset_df['Comm_Long'] - asset_df['Comm_Short']
                    asset_df['Net_NonCommercial'] = asset_df['NonComm_Long'] - asset_df['NonComm_Short']
                    
                    # Commercial Long Ratio
                    total_comm = asset_df['Comm_Long'] + asset_df['Comm_Short']
                    asset_df['Net_Commercial_Long'] = asset_df['Comm_Long'] / total_comm.replace(0, 1)
                    
                    asset_df = asset_df[['Date', 'Net_Commercial', 'Net_NonCommercial', 'Net_Commercial_Long']]
                    asset_df['Date'] = asset_df['Date'].dt.strftime('%Y-%m-%d')
                    all_dfs[asset].append(asset_df)
                else:
                    print(f"    No records found for {asset} ({market_code}) in {year}.")
                    
        except Exception as e:
            print(f"    Error processing year {year}: {e}")

    # Combine and save
    store = MarketDataStore()
    for asset, dfs in all_dfs.items():
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            combined = combined.drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)
            
            csv_path = f"data/cot_{asset}.csv"
            if os.path.exists(csv_path):
                existing = pd.read_csv(csv_path)
                combined = pd.concat([combined, existing]).drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
            
            combined.to_csv(csv_path, index=False)
            try:
                store.write_table(f'cot_{asset}', combined, format='pandas')
                print(f"  Saved combined COT for {asset} (Total Records: {len(combined)})")
            except Exception as e:
                print(f"  Warning: DuckDB write failed for cot_{asset}: {e}. CSV saved.")
        else:
            print(f"  No historical COT data collected for {asset}.")

def backfill_fear_greed():
    print("Backfilling Fear & Greed for Crypto...")
    try:
        url = "https://api.alternative.me/fng/?limit=0&format=json&date_format=kr"
        r = requests.get(url, timeout=30)
        data = r.json()
        fg_data = data.get('data', [])
        
        records = []
        for d in fg_data:
            records.append({
                'Date': d['timestamp'],
                'Fear_Greed': float(d['value'])
            })
            
        fg_df = pd.DataFrame(records)
        fg_df['Date'] = pd.to_datetime(fg_df['Date']).dt.strftime('%Y-%m-%d')
        fg_df = fg_df.sort_values('Date').drop_duplicates(subset=['Date']).reset_index(drop=True)
        
        store = MarketDataStore()
        try:
            btc_df = store.read_table('btc_global_insights', format='pandas')
        except Exception:
            btc_df = pd.read_csv('data/btc_global_insights.csv', index_col=0, parse_dates=True)
            btc_df.index.name = 'Date'
            btc_df = btc_df.reset_index()
            
        btc_df['Date'] = pd.to_datetime(btc_df['Date'])
        fg_df['Date'] = pd.to_datetime(fg_df['Date'])
        
        if 'Fear_Greed' in btc_df.columns:
            btc_df = btc_df.drop(columns=['Fear_Greed'])
            
        btc_df = pd.merge(btc_df, fg_df, on='Date', how='left')
        
        # Forward fill and fillna with 50 (neutral)
        btc_df['Fear_Greed'] = btc_df['Fear_Greed'].ffill().fillna(50.0)
        
        btc_df['Date'] = btc_df['Date'].dt.strftime('%Y-%m-%d')
        btc_df.set_index('Date', inplace=True)
        btc_df.to_csv('data/btc_global_insights.csv')
        
        try:
            btc_df_reset = btc_df.reset_index()
            store.write_table('btc_global_insights', btc_df_reset, format='pandas')
        except Exception as e:
            pass
            
        print(f"  Merged {len(fg_df)} Fear & Greed records into BTC dataset. Current range: {fg_df['Date'].min()} to {fg_df['Date'].max()}")
        
    except Exception as e:
        print(f"Failed to backfill Fear & Greed: {e}")

if __name__ == "__main__":
    backfill_cot_history(years=[2020, 2021, 2022, 2023, 2024, 2025, 2026])
    backfill_fear_greed()
    print("Backfill complete.")
