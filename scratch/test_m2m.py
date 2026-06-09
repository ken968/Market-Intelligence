import pandas as pd
import numpy as np
import os
from utils.config import ASSETS

def calculate_m2m_table(year=2026):
    # Target assets and macros
    targets = {
        'BTC': ('btc', 'Close'),
        'SPY': ('spy', 'Close'),
        'Gold': ('gold', 'Gold'),
    }
    
    # Try to load macro indicators for DXY and Yield
    macro_df = None
    if os.path.exists('data/macro_indicators.csv'):
        macro_df = pd.read_csv('data/macro_indicators.csv', parse_dates=['Date'])
        macro_df.set_index('Date', inplace=True)
        macro_df.sort_index(inplace=True)
    
    # Prepare monthly data container
    monthly_data = pd.DataFrame()
    
    # Process assets
    for name, (asset_key, price_col) in targets.items():
        if asset_key in ASSETS:
            file_path = ASSETS[asset_key]['data_file']
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, parse_dates=['Date'])
                df.set_index('Date', inplace=True)
                df.sort_index(inplace=True)
                
                # Resample to monthly last day
                monthly = df[price_col].resample('ME').last()
                monthly_data[name] = monthly
                
    # Process macros
    if macro_df is not None:
        if 'DXY' in macro_df.columns:
            monthly_data['DXY'] = macro_df['DXY'].resample('ME').last()
        if 'Yield_10Y' in macro_df.columns:
            monthly_data['10Y Yield'] = macro_df['Yield_10Y'].resample('ME').last()
            
    # Filter for target year and previous year December (for base calculation)
    # Actually just calculate pct_change() then filter by year
    m2m_returns = monthly_data.pct_change() * 100
    m2m_returns = m2m_returns[m2m_returns.index.year == year]
    
    # YTD Cumulative
    # Get last day of previous year
    ytd_returns = {}
    last_year_end = f"{year-1}-12-31"
    
    # A bit more robust calculation for YTD
    for col in monthly_data.columns:
        series = monthly_data[col].dropna()
        try:
            # Find the closest date before or at end of previous year
            base_prices = series[series.index <= pd.to_datetime(last_year_end)]
            if not base_prices.empty:
                base_price = base_prices.iloc[-1]
                # Get current price in target year
                current_prices = series[series.index.year == year]
                if not current_prices.empty:
                    current_price = current_prices.iloc[-1]
                    ytd_returns[col] = ((current_price - base_price) / base_price) * 100
        except Exception:
            pass
            
    # Formatting
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    formatted_data = []
    
    for i in range(1, 13):
        # Find if we have data for this month
        month_data = m2m_returns[m2m_returns.index.month == i]
        row = {'Month': months[i-1]}
        if not month_data.empty:
            for col in m2m_returns.columns:
                val = month_data[col].iloc[-1]
                if pd.isna(val):
                    row[col] = ""
                else:
                    if val < 0:
                        row[col] = f"({abs(val):.1f}%)"
                    else:
                        row[col] = f"{val:.1f}%"
        else:
            # Empty row if no data
            for col in m2m_returns.columns:
                row[col] = ""
                
        # Only append if we have data up to this month or it's the future
        # Let's just append all 12 months for visual layout
        formatted_data.append(row)
        
    # Append YTD row
    ytd_row = {'Month': 'YTD Cumulative'}
    for col in m2m_returns.columns:
        if col in ytd_returns:
            val = ytd_returns[col]
            if val < 0:
                ytd_row[col] = f"({abs(val):.1f}%)"
            else:
                ytd_row[col] = f"{val:.1f}%"
        else:
            ytd_row[col] = ""
            
    formatted_data.append(ytd_row)
    
    return pd.DataFrame(formatted_data)

if __name__ == "__main__":
    df = calculate_m2m_table(2026)
    print(df)
