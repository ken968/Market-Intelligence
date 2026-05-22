import pickle
import numpy as np

def inspect_scaler(path):
    print(f"--- {path} ---")
    with open(path, 'rb') as f:
        scaler = pickle.load(f)
    print("Type:", type(scaler))
    for attr in ['mean_', 'scale_', 'var_', 'n_samples_seen_']:
        if hasattr(scaler, attr):
            print(f"  {attr}: {getattr(scaler, attr)}")
            
inspect_scaler("models/scaler_target.pkl")
inspect_scaler("models/btc_scaler_target.pkl")
