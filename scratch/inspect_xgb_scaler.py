import json
import pickle
import pandas as pd
import os

for asset in ['gold', 'btc', 'spy']:
    print(f"=== Asset: {asset} ===")
    data_path = f'data/{asset}_global_insights.csv'
    xgb_feat_path = f'models/{asset}_xgb_features.json'
    xgb_scaler_path = f'models/{asset}_xgb_scaler.pkl'
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print("CSV Columns:", len(df.columns))
    else:
        print("CSV not found:", data_path)
        
    if os.path.exists(xgb_feat_path):
        with open(xgb_feat_path, 'r') as f:
            feat_meta = json.load(f)
        print("XGB features meta:", len(feat_meta['features']), feat_meta['features'])
    else:
        print("Meta not found:", xgb_feat_path)
        
    if os.path.exists(xgb_scaler_path):
        with open(xgb_scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        if hasattr(scaler, 'n_features_in_'):
            print("Scaler features count:", scaler.n_features_in_)
            if hasattr(scaler, 'feature_names_in_'):
                print("Scaler feature names:", scaler.feature_names_in_)
        else:
            print("Scaler has no n_features_in_")
    else:
        print("Scaler not found:", xgb_scaler_path)
