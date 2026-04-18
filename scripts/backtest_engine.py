import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

from utils.config import ASSETS
from utils.predictor import AssetPredictor

def calculate_halving_cycle(dates):
    halving_dates = pd.to_datetime([
        '2012-11-28', '2016-07-09', '2020-05-11', '2024-04-19', '2028-04-01'
    ])
    cycle = []
    for date in dates:
        days_to_halving = min(abs((date - h).days) for h in halving_dates)
        cycle.append(days_to_halving)
    return cycle

def run_backtest(asset_key):
    asset_key = asset_key.lower()
    if asset_key not in ASSETS:
        print(f"Error: Unknown asset {asset_key}")
        return False
        
    config = ASSETS[asset_key]
    data_file = config['data_file']
    
    if not os.path.exists(data_file):
        print(f"Error: Data file {data_file} not found.")
        return False
        
    print(f"{"="*60}")
    print(f" BACKTEST ENGINE: CLAUDE'S 3-LEVEL ARCHITECTURE - {asset_key.upper()}")
    print(f"{"="*60}")
    
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Robust feature handling: Only use features that actually exist in the CSV
    raw_features = config['features']
    available_features = [f for f in raw_features if f in df.columns]
    missing_features = [f for f in raw_features if f not in df.columns]
    
    if missing_features:
        print(f"Warning: Missing features for {asset_key}: {missing_features}")
    
    features = available_features
    if 'Sentiment' not in df.columns and 'Sentiment' in raw_features:
        df['Sentiment'] = 0
        features.append('Sentiment')
        
    data = df[features].values
    dates = df['Date'].values
    prediction_days = config.get('sequence_length', 60 if asset_key != 'btc' else 90)
    
    # 1. Normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # 2. X, Y Setup
    X_all, Y_all, valid_dates = [], [], []
    for x in range(prediction_days, len(scaled_data)):
        X_all.append(scaled_data[x-prediction_days:x, :])
        Y_all.append(scaled_data[x, 0])
        valid_dates.append(dates[x])
        
    X_all, Y_all = np.array(X_all), np.array(Y_all)
    
    # 3. 80/20 Split
    split_idx = int(len(X_all) * 0.8)
    x_train, y_train = X_all[:split_idx], Y_all[:split_idx]
    x_test, y_test   = X_all[split_idx:], Y_all[split_idx:]
    test_dates       = valid_dates[split_idx:]
    
    print(f"Train samples (80%): {len(x_train)}")
    print(f"Test samples (20%): {len(x_test)} -> {pd.to_datetime(test_dates[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(test_dates[-1]).strftime('%Y-%m-%d')}")
    
    # 4. Train 80% Model
    model = Sequential([
        Input(shape=(x_train.shape[1], x_train.shape[2])),
        LSTM(units=128 if asset_key == 'btc' else 100, return_sequences=True, kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        LSTM(units=64 if asset_key == 'btc' else 50, return_sequences=True if asset_key == 'btc' else False, kernel_regularizer=l2(0.001)),
        Dropout(0.3)
    ])
    if asset_key == 'btc':
        model.add(LSTM(units=32, return_sequences=False, kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.3))
        
    model.add(Dense(units=16 if asset_key == 'btc' else 25, kernel_regularizer=l2(0.001)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)
    
    print("\nTraining Isolated 80% Model...")
    model.fit(
        x_train, y_train, epochs=100, batch_size=32, validation_split=0.1,
        callbacks=[early_stop, reduce_lr], verbose=0  # Silent for cleaner output
    )
    
    # 5. Evaluate USING The 3-Level Architecture
    print("\nTesting Claude's 3-Level Architecture on 20% Unseen Data...")
    
    # We instantiate AssetPredictor but inject our 80% model
    predictor = AssetPredictor(asset_key)
    predictor.model = model
    predictor.scaler = scaler
    predictor.is_loaded = True
    predictor.config['features'] = features  # FIX: Prevent reshape mismatch for assets lacking some macro features
    
    # We will simulate "walking forward" day by day in the 20% test set
    predictions_3level = []
    baseline_lstm_only = model.predict(x_test, verbose=0)
    
    # Inverse transform pure LSTM
    pred_padding = np.zeros((len(baseline_lstm_only), len(features)))
    pred_padding[:, 0] = baseline_lstm_only[:, 0]
    lstm_predictions = scaler.inverse_transform(pred_padding)[:, 0]
    
    actual_padding = np.zeros((len(y_test), len(features)))
    actual_padding[:, 0] = y_test
    actuals = scaler.inverse_transform(actual_padding)[:, 0]
    
    print("Walking forward and applying Manager (Anchoring) and CEO (Sentiment/Macro) Layers...")
    for i in range(len(x_test)):
        # To simulate the exact point in time, we set the predictor's data up to this point
        # The predictor expects raw (unscaled) data for its history anchor
        current_idx = prediction_days + split_idx + i
        predictor.data = data[:current_idx]
        
        # Simulate CEO Bias Multiplier based on Sentiment at that time
        hist_sentiment = predictor.data[-1][features.index('Sentiment')] if 'Sentiment' in features else 0
        hist_m2        = predictor.data[-1][features.index('M2_YoY')] if 'M2_YoY' in features else 0
        
        # Super simple macro emulation for CEO bias (since we can't call Gemini 800 times)
        # If sentiment is high and M2 is expanding, push multiplier up
        ceo_multiplier = 1.0 + (hist_sentiment * 0.05) + (hist_m2 * 0.01)
        ceo_multiplier = max(0.85, min(1.15, ceo_multiplier)) # Bound it exactly as in script
        
        # Predict 1 step ahead using the Recursive Forecast (which applies Damping & Anchor Spring)
        step_forecast = predictor.recursive_forecast(steps=1, ceo_drift_multiplier=ceo_multiplier)
        predictions_3level.append(step_forecast[0])

    predictions_3level = np.array(predictions_3level)
    
    # Calculate directional hit ratio
    def get_hit_ratio(preds, acts):
        hits = 0
        total_moves = len(acts) - 1
        for i in range(1, len(acts)):
            actual_dir = acts[i] > acts[i-1]
            pred_dir = preds[i] > acts[i-1]
            if actual_dir == pred_dir:
                hits += 1
        return (hits / total_moves) * 100 if total_moves > 0 else 0

    hit_ratio_lstm = get_hit_ratio(lstm_predictions, actuals)
    hit_ratio_3lvl = get_hit_ratio(predictions_3level, actuals)
    
    rmse_lstm = np.sqrt(np.mean((lstm_predictions - actuals)**2))
    rmse_3lvl = np.sqrt(np.mean((predictions_3level - actuals)**2))
    
    print("\n" + "="*60)
    print(" [DONE] FINAL VALIDATION RESULTS")
    print("="*60)
    print(f"[Worker Only]  Hit Ratio: {hit_ratio_lstm:.2f}% | RMSE: {rmse_lstm:.2f}")
    print(f"[Full 3-Level] Hit Ratio: {hit_ratio_3lvl:.2f}% | RMSE: {rmse_3lvl:.2f}")
    print("="*60)
    
    # 6. Save Plot
    os.makedirs('reports', exist_ok=True)
    plt.style.use('dark_background')
    plt.figure(figsize=(14, 7))
    plt.plot(pd.to_datetime(test_dates), actuals, color='#4169E1', label='Actual Price', linewidth=2.5) # Royal Blue
    plt.plot(pd.to_datetime(test_dates), lstm_predictions, color='#EF553B', label='Worker (Pure LSTM)', alpha=0.5, linestyle='--')
    plt.plot(pd.to_datetime(test_dates), predictions_3level, color='#00C076', label='3-Level System (Claude)', linewidth=3) # Success Green
    
    plt.title(f"{asset_key.upper()} Walk-Forward Backtest: LSTM vs 3-Level Architecture", color='white', pad=20)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    # Add text overlay
    plt.text(pd.to_datetime(test_dates[int(len(test_dates)*0.05)]), max(actuals)*0.95, 
            f"3-Level Hit Ratio: {hit_ratio_3lvl:.1f}%\nLSTM Hit Ratio: {hit_ratio_lstm:.1f}%", 
            color='white', bbox=dict(facecolor='black', alpha=0.7))
            
    plot_path = f'reports/backtest_{asset_key}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='#0E1117')
    plt.close()
    
    # 7. Save Metrics
    metrics = {
        "asset": asset_key,
        "hit_ratio_3layer": hit_ratio_3lvl,
        "hit_ratio_lstm": hit_ratio_lstm,
        "rmse_3layer": rmse_3lvl,
        "rmse_lstm": rmse_lstm,
        "test_samples": len(x_test),
        "start_test_date": pd.to_datetime(test_dates[0]).strftime('%Y-%m-%d'),
        "end_test_date": pd.to_datetime(test_dates[-1]).strftime('%Y-%m-%d')
    }
    
    with open(f'reports/backtest_{asset_key}.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Saved visualization to '{plot_path}'")
    return metrics

if __name__ == '__main__':
    if len(sys.argv) > 1:
        run_backtest(sys.argv[1])
    else:
        print("Usage: python backtest_engine.py [asset_key]")
