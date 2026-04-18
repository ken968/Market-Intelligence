import pandas as pd
import numpy as np
import pickle
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN info messages
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# 1. Load Data
if not os.path.exists('data/gold_global_insights.csv'):
    print("Error: data/gold_global_insights.csv not found")
    exit(1)

df = pd.read_csv('data/gold_global_insights.csv')
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.config import ASSETS
    features = ASSETS['gold']['features']
except ImportError:
    features = ['Gold', 'DXY', 'VIX', 'Yield_10Y', 'Oil_Price',
                'CPI_MoM', 'PPI_MoM', 'PCE_MoM', 'NFP_Change',
                'YieldCurve_10Y2Y', 'M2_MoM', 'M2_YoY', 'Yield_10Y_Rate', 
                'Breakeven_5Y5Y', 'M2_Liquidity_Spike', 'MacroEvent_Flag',
                'Sentiment', 'EMA_90']
missing = [f for f in features if f not in df.columns]
if missing:
    print(f"Missing features: {missing}")
    # fill missing with 0 for robustness
    for f in missing:
        df[f] = 0

data = df[features].values

# 2. Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Save scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 3. Data Preparation
prediction_days = 60
x_train, y_train = [], []

if len(scaled_data) <= prediction_days:
    print("Not enough data to train")
    exit(1)

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, :])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# 4. Model Architecture
model = Sequential([
    Input(shape=(x_train.shape[1], x_train.shape[2])),
    LSTM(units=100, return_sequences=True, kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    LSTM(units=50, return_sequences=False, kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(units=25, kernel_regularizer=l2(0.001)),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping - monitor validation loss with high patience
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=15, 
    restore_best_weights=True,
    verbose=1
)

# Reduce learning rate when stuck
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=7, 
    min_lr=1e-6,
    verbose=1
)

# 5. Training
print("\nStarting training...")
history = model.fit(
    x_train, y_train, 
    epochs=75, 
    batch_size=32, 
    validation_split=0.1,
    verbose=1,
    callbacks=[early_stop, reduce_lr]
)

# 6. Save Model
model.save('models/gold_ultimate_model.keras')
print("\nModel saved as 'models/gold_ultimate_model.keras'")