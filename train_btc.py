import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def train_btc_model():
    """
    Train Bitcoin prediction model with longer sequence window.
    Uses 90-day lookback to capture halving cycle patterns.
    """
    
    print("="*60)
    print("BITCOIN AI MODEL TRAINING - DEEP LEARNING LSTM")
    print("="*60)
    
    # 1. Load Data
    if not os.path.exists('btc_global_insights.csv'):
        print("Error: 'btc_global_insights.csv' not found.")
        print("Run: python sentiment_fetcher_v2.py btc")
        return False
    
    df = pd.read_csv('btc_global_insights.csv')
    
    # Features: BTC price + macro + sentiment + halving cycle
    features = ['BTC', 'DXY', 'VIX', 'Yield_10Y', 'Sentiment', 'Halving_Cycle']
    
    # Handle missing columns (if sentiment not run yet)
    if 'Sentiment' not in df.columns:
        print("Warning: Sentiment data missing. Using 0 as placeholder.")
        df['Sentiment'] = 0
    
    if 'Halving_Cycle' not in df.columns:
        print("Warning: Halving_Cycle missing. Recalculating...")
        df['Halving_Cycle'] = calculate_halving_cycle(pd.to_datetime(df['Date']))
    
    data = df[features].values
    
    print(f"Dataset: {len(df)} records from {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
    print(f"Features: {features}")
    
    # 2. Normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Save scaler
    with open('btc_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("System: Scaler saved as 'btc_scaler.pkl'")
    
    # 3. Data Preparation
    # BTC uses 90-day window (vs 60 for Gold) to capture longer cycles
    prediction_days = 90
    x_train, y_train = [], []
    
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x, :])
        y_train.append(scaled_data[x, 0])  # Predict BTC price (index 0)
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    print(f"Training samples: {len(x_train)}")
    print(f"Sequence shape: {x_train.shape}")
    
    # 4. Model Architecture
    # Slightly deeper than Gold model due to higher volatility
    model = Sequential([
        LSTM(units=128, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
        Dropout(0.3),  # Higher dropout for BTC volatility
        LSTM(units=64, return_sequences=True),
        Dropout(0.3),
        LSTM(units=32, return_sequences=False),
        Dropout(0.2),
        Dense(units=16),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    
    # 5. Training
    print("\nStarting training...")
    print("Note: BTC has higher volatility - training may take longer")
    print("-" * 60)
    
    history = model.fit(
        x_train, y_train, 
        epochs=50,  # More epochs for BTC
        batch_size=32, 
        verbose=2,
        callbacks=[early_stop]
    )
    
    # 6. Save Model
    model.save('btc_ultimate_model.h5')
    print("\n" + "="*60)
    print(" Bitcoin model saved as 'btc_ultimate_model.h5'")
    print(f" Final Loss: {history.history['loss'][-1]:.6f}")
    print("="*60)
    
    return True


def calculate_halving_cycle(dates):
    """Calculate days to nearest Bitcoin halving event"""
    halving_dates = pd.to_datetime([
        '2012-11-28', 
        '2016-07-09', 
        '2020-05-11', 
        '2024-04-19',
        '2028-04-01'
    ])
    
    cycle = []
    for date in dates:
        days_to_halving = min(abs((date - h).days) for h in halving_dates)
        cycle.append(days_to_halving)
    
    return cycle


if __name__ == "__main__":
    success = train_btc_model()
    if success:
        print("\n Bitcoin AI is ready for predictions!")
    else:
        print("\n❌ Training failed. Check error messages above.")
