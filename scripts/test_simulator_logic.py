import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load config-like info
features = ['Gold', 'DXY', 'VIX', 'Yield_10Y', 'Oil_Price', 'Sentiment', 'EMA_90']
model_path = 'd:/Market-Intelligence/models/gold_ultimate_model.keras'
scaler_path = 'd:/Market-Intelligence/models/scaler.pkl'
data_path = 'd:/Market-Intelligence/data/gold_global_insights.csv'

def test_simulation():
    if not os.path.exists(model_path):
        print("Model not found")
        return

    # Load resources
    model = load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    df = pd.read_csv(data_path)
    raw_data = df[features].values
    
    # 1. Baseline Sequence
    baseline_raw = raw_data[-60:].copy()
    baseline_scaled = scaler.transform(baseline_raw)
    baseline_input = baseline_scaled.reshape(1, 60, len(features))
    baseline_pred = model.predict(baseline_input, verbose=0)[0, 0]
    
    # 2. "Bad News" Shock (-1.0 Sentiment, $120 Oil)
    # Using the "Fixed" logic from the simulator
    shock_raw = raw_data[-60:].copy()
    
    # Target values
    sim_oil = 120.0
    sim_sentiment = -1.0
    
    # Indices
    oil_idx = features.index('Oil_Price')
    sent_idx = features.index('Sentiment')
    
    # Scaling the shock
    dummy_row = raw_data[-1:].copy()
    dummy_row[0, oil_idx] = sim_oil
    dummy_row[0, sent_idx] = sim_sentiment
    scaled_dummy = scaler.transform(dummy_row)
    
    # Apply to last 3 days
    for i in range(1, 4):
        row = shock_raw[-i:-i+1 if i > 1 else None].copy()
        s_row = scaler.transform(row)
        s_row[0, oil_idx] = scaled_dummy[0, oil_idx]
        s_row[0, sent_idx] = scaled_dummy[0, sent_idx]
        shock_raw[-i] = scaler.inverse_transform(s_row)[0]
        
    shock_scaled = scaler.transform(shock_raw)
    shock_input = shock_scaled.reshape(1, 60, len(features))
    shock_pred = model.predict(shock_input, verbose=0)[0, 0]
    
    # Print results
    print(f"Baseline Scaled Pred: {baseline_pred:.6f}")
    print(f"Shock Scaled Pred:    {shock_pred:.6f}")
    print(f"Delta (Scaled):       {shock_pred - baseline_pred:.6f}")
    
    # Inverse prices
    dummy_b = np.zeros((1, len(features)))
    dummy_b[0, 0] = baseline_pred
    price_b = scaler.inverse_transform(dummy_b)[0, 0]
    
    dummy_s = np.zeros((1, len(features)))
    dummy_s[0, 0] = shock_pred
    price_s = scaler.inverse_transform(dummy_s)[0, 0]
    
    print(f"Baseline Price: ${price_b:,.2f}")
    print(f"Shock Price:    ${price_s:,.2f}")
    print(f"Direction:      {'UP (Bullish)' if price_s > price_b else 'DOWN (Bearish)'}")

    # Check Sentiment Scaling
    print(f"\nScaled Sentiment (-1.0 raw) = {scaled_dummy[0, sent_idx]:.4f}")
    
if __name__ == "__main__":
    test_simulation()
