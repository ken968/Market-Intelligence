import pandas as pd
import numpy as np

# 1. Check SPY-QQQ correlation
spy = pd.read_csv('data/SPY_global_insights.csv')
qqq = pd.read_csv('data/QQQ_global_insights.csv')
merged = pd.merge(spy[['Date','SPY']], qqq[['Date','QQQ']], on='Date')
correlation = merged['SPY'].corr(merged['QQQ'])
print(f"\n{'='*60}")
print("CORRELATION ANALYSIS: SPY vs QQQ")
print(f"{'='*60}")
print(f"Historical Correlation: {correlation:.4f}")
print(f"Total data points: {len(merged)}")
print(f"Date range: {merged['Date'].iloc[0]} to {merged['Date'].iloc[-1]}")

# 2. Check sentiment data quality
print(f"\n{'='*60}")
print("SENTIMENT DATA ANALYSIS")
print(f"{'='*60}")
for ticker in ['SPY', 'QQQ', 'AAPL', 'BTC']:
    try:
        df = pd.read_csv(f'data/{ticker}_global_insights.csv' if ticker != 'BTC' else 'data/btc_global_insights.csv')
        non_zero = (df['Sentiment'] != 0).sum()
        total = len(df)
        pct = (non_zero / total) * 100
        mean_val = df['Sentiment'].mean()
        print(f"{ticker:6s}: {non_zero:4d}/{total:4d} non-zero ({pct:5.1f}%), mean={mean_val:+.4f}")
    except Exception as e:
        print(f"{ticker:6s}: ERROR - {e}")

# 3. Check latest forecast predictions (from screenshot data)
print(f"\n{'='*60}")
print("FORECAST DIVERGENCE ANALYSIS")
print(f"{'='*60}")
forecast_data = {
    'SPY': {'current': 690.62, '3_months': 742.18},
    'QQQ': {'current': 597.03, '3_months': 289.23},
    'DIA': {'current': 494.75, '3_months': 512.48},
}

for ticker, prices in forecast_data.items():
    change = ((prices['3_months'] - prices['current']) / prices['current']) * 100
    print(f"{ticker}: ${prices['current']:.2f} → ${prices['3_months']:.2f} ({change:+.1f}%)")

spy_change = ((forecast_data['SPY']['3_months'] - forecast_data['SPY']['current']) / forecast_data['SPY']['current']) * 100
qqq_change = ((forecast_data['QQQ']['3_months'] - forecast_data['QQQ']['current']) / forecast_data['QQQ']['current']) * 100
divergence = abs(spy_change - qqq_change)

print(f"\nSPY 3-month change: {spy_change:+.1f}%")
print(f"QQQ 3-month change: {qqq_change:+.1f}%")
print(f"Divergence: {divergence:.1f} percentage points")
print(f"Expected divergence (if corr={correlation:.2f}): < 5 percentage points")
print(f"\n⚠️  CORRELATION PARADOX: {'CONFIRMED' if divergence > 30 else 'NOT CONFIRMED'}")
