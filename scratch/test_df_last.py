import json
import pickle
import pandas as pd
import os

asset = 'gold'
data_path = f'data/{asset}_global_insights.csv'
xgb_feat_path = f'models/{asset}_xgb_features.json'
xgb_scaler_path = f'models/{asset}_xgb_scaler.pkl'

df_full = pd.read_csv(data_path, index_col=0, parse_dates=True)
df_full = df_full.sort_index()
df_last = df_full.iloc[[-1]]

with open(xgb_feat_path, 'r') as f:
    xgb_meta = json.load(f)
    
xgb_feats = [ft for ft in xgb_meta['features'] if ft in df_last.columns]
print("All features in meta:", xgb_meta['features'])
print("Features in df_last:", xgb_feats)
missing = [ft for ft in xgb_meta['features'] if ft not in df_last.columns]
print("Missing:", missing)
