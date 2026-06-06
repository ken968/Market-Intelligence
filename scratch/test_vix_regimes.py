import pandas as pd
import numpy as np

# Load data
try:
    # Assuming SPY_macro_data.csv has Close (SPY) and VIX
    df = pd.read_csv('data/SPY_macro_data.csv')
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
    # We need SPY return and VIX close. Let's assume standard names or try to find them
    spy_col = [c for c in df.columns if 'SPY' in c or 'Close' in c][0]
    vix_col = [c for c in df.columns if 'VIX' in c][0]
    
    df['ret_spy'] = df[spy_col].pct_change()
    
    # Calculate 252d rolling percentile for VIX
    def get_percentile_rank(window):
        if len(window) < 30: return np.nan
        current_value = window.iloc[-1]
        sorted_window = np.sort(window.values)
        return np.searchsorted(sorted_window, current_value) / len(window)
        
    df['vix_pctile'] = df[vix_col].rolling(252, min_periods=30).apply(get_percentile_rank, raw=False)
    df = df.dropna(subset=['ret_spy', 'vix_pctile'])
    
    # Analyze regimes
    bins = [0, 0.90, 0.95, 0.99, 1.01]
    labels = ['Normal (<90%)', 'Warning (90-95%)', 'Siaga (95-99%)', 'Krisis (>=99%)']
    df['regime'] = pd.cut(df['vix_pctile'], bins=bins, labels=labels, right=False)
    
    summary = df.groupby('regime')['ret_spy'].agg(['count', 'mean', 'std', 'min']).reset_index()
    summary['mean'] = (summary['mean'] * 100).round(3).astype(str) + '%'
    summary['std'] = (summary['std'] * 100).round(3).astype(str) + '%'
    summary['min'] = (summary['min'] * 100).round(3).astype(str) + '%'
    
    print(summary.to_string(index=False))
except Exception as e:
    print(f"Error: {e}")
