import pandas as pd
import numpy as np
import pickle
import os
import sys
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# List of supported stocks
SUPPORTED_STOCKS = [
    'SPY', 'QQQ', 'DIA',  # Indices
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',  # Mag7
    'TSM'  # TSMC
]

def train_stock_model(ticker):
    """
    Train prediction model for a specific US stock.
    Using Keras 3 format.
    """
    
    ticker = ticker.upper()
    
    print("="*60)
    print(f"{ticker} AI MODEL TRAINING - DEEP LEARNING LSTM")
    print("="*60)
    
    # 1. Load Data
    data_file = f'data/{ticker}_global_insights.csv'
    if not os.path.exists(data_file):
        print(f"Error: '{data_file}' not found.")
        print(f"Run: python scripts/sentiment_fetcher_v2.py {ticker.lower()}")
        return False
    
    df = pd.read_csv(data_file)
    
    # Features: Stock price + macro + sentiment + EMA 90
    features = [ticker, 'DXY', 'VIX', 'Yield_10Y', 'Sentiment', 'EMA_90']
    
    # Handle missing sentiment
    if 'Sentiment' not in df.columns:
        print("Warning: Sentiment data missing. Using 0 as placeholder.")
        df['Sentiment'] = 0
    
    data = df[features].values
    
    print(f"Dataset: {len(df)} records from {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
    print(f"Features: {features}")
    
    # 2. Normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Save scaler
    scaler_file = f'models/{ticker}_scaler.pkl'
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"System: Scaler saved as '{scaler_file}'")
    
    # 3. Data Preparation
    prediction_days = 60  # Same as Gold
    x_train, y_train = [], []
    
    if len(scaled_data) <= prediction_days:
        print("Not enough data to train.")
        return False

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x, :])
        y_train.append(scaled_data[x, 0])  # Predict stock price
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    print(f"Training samples: {len(x_train)}")
    print(f"Sequence shape: {x_train.shape}")
    
    # 4. Model Architecture
    model = Sequential([
        Input(shape=(x_train.shape[1], x_train.shape[2])),
        LSTM(units=100, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    # Early stopping
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    
    # 5. Training
    print("\nStarting training...")
    print("-" * 60)
    
    history = model.fit(
        x_train, y_train, 
        epochs=30, 
        batch_size=32, 
        verbose=2,
        callbacks=[early_stop]
    )
    
    # 6. Save Model
    # Save as .keras
    model_file = f'models/{ticker}_ultimate_model.keras'
    model.save(model_file)
    
    print("\n" + "="*60)
    print(f" {ticker} model saved as '{model_file}'")
    print(f" Final Loss: {history.history['loss'][-1]:.6f}")
    print("="*60)
    
    return True


def train_all_stocks():
    """Train models for all supported stocks"""
    print("\n" + "="*60)
    print("BATCH TRAINING: ALL US STOCKS")
    print("="*60)
    
    results = {}
    
    for i, ticker in enumerate(SUPPORTED_STOCKS, 1):
        print(f"\n[{i}/{len(SUPPORTED_STOCKS)}] Training {ticker}...")
        try:
            success = train_stock_model(ticker)
            results[ticker] = " Success" if success else " Failed"
        except Exception as e:
            print(f"Error training {ticker}: {e}")
            results[ticker] = f" Error: {str(e)[:50]}"
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for ticker, status in results.items():
        print(f"{ticker:6s}: {status}")
    print("="*60)
    
    success_count = sum(1 for s in results.values() if "Success" in s)
    print(f"\nCompleted: {success_count}/{len(SUPPORTED_STOCKS)} models trained successfully")
    
    return success_count == len(SUPPORTED_STOCKS)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1].upper()
        
        if arg == 'ALL':
            train_all_stocks()
        elif arg in SUPPORTED_STOCKS:
            success = train_stock_model(arg)
            if success:
                print(f"\n {arg} AI is ready for predictions!")
            else:
                print(f"\n[!] Training failed for {arg}")
        else:
            print(f"Unknown stock: {arg}")
    else:
        print("Usage: python train_stocks.py [TICKER|ALL]")
