import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Load data yang sudah didownload oleh data_fetcher.py
df = pd.read_csv('gold_data.csv')

# Perbaikan: Pastikan kolom 'Close' adalah angka (karena yfinance kadang kasih header extra)
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna(subset=['Close'])

data = df['Close'].values.reshape(-1, 1)

# 2. Normalisasi Data (Penting agar model tidak bingung)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 3. Membuat struktur data: melihat 60 hari ke belakang untuk tebak hari ke-61
prediction_days = 60
x_train, y_train = [], []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# 4. Build Model LSTM (Otak AI kamu)
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    Dropout(0.2), # Biar gak "menghafal" tapi "belajar"
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 5. Training! (Pantau Task Manager, RAM kamu harusnya aman sekarang)
print("Memulai training...")
model.fit(x_train, y_train, epochs=25, batch_size=32)
print("Training selesai!")

# Simpan model agar tidak hilang
model.save('gold_model.h5')
