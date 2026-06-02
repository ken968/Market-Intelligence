import pandas as pd, numpy as np, pickle, os
from tensorflow.keras.models import load_model
from utils.config import ASSETS

df = pd.read_csv('data/gold_global_insights.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

current = df['Gold'].iloc[-1]
print(f'Current price: {current:.2f}')

for h in [1, 7, 14, 30, 90]:
    past_price = df['Gold'].iloc[-h-1]
    actual_realized = (current - past_price) / past_price
    print(f'Actual {h:3d}d return (realized): {actual_realized*100:+.3f}%')

print()
split_idx = int(len(df) * 0.80)
cutoff_date = df['Date'].iloc[split_idx]
gold_at_cutoff = df['Gold'].iloc[split_idx]
print(f'Training/test split date: {cutoff_date.date()}')
print(f'Gold at split: {gold_at_cutoff:.2f}')
print(f'Gold now: {current:.2f}')
print(f'Change since split: {(current/gold_at_cutoff-1)*100:+.2f}%')
print()

# Check what percent of training data has gold > 3500 (new regime)
print('Distribution of Gold price in training data:')
print(df['Gold'].describe())
print()
print('Rows where Gold > 3500:', (df['Gold'] > 3500).sum())
print('Rows where Gold > 4000:', (df['Gold'] > 4000).sum())
print('Total rows:', len(df))
