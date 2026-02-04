import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 1. Load Data
if not os.path.exists('data/gold_global_insights.csv'):
    print("Error: 'data/gold_global_insights.csv' not found. Fetch data first.")
    exit()

df = pd.read_csv('data/gold_global_insights.csv')
features = ['Gold', 'DXY', 'VIX', 'Yield_10Y', 'Sentiment']
data = df[features].values

# 2. Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Save Scaler for consistent inference in app.py
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("System: Scaler saved as 'models/scaler.pkl'")

# 3. Data Preparation
prediction_days = 60
x_train, y_train = [], []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, :])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# 4. Model Architecture
model = Sequential([
    LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 5. Training
print(f"System: Starting training on {len(x_train)} samples...")
# verbose=2 prevents ANSI progress bars that mess up Streamlit's UI
model.fit(x_train, y_train, epochs=30, batch_size=32, verbose=2)

# 6. Save Model
model.save('models/gold_ultimate_model.h5')
print("System: Model saved as 'models/gold_ultimate_model.h5'")