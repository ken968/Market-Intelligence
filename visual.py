import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# 1. Load Data dan Model
df = pd.read_csv('gold_data.csv')

# Perbaikan: Pastikan kolom 'Close' adalah angka
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna(subset=['Close'])

model = load_model('gold_model.h5')
data = df['Close'].values.reshape(-1, 1)

# 2. Persiapkan Data untuk Testing (sama dengan saat training)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

prediction_days = 60
x_test = []

# Ambil data terakhir untuk dites
for x in range(prediction_days, len(scaled_data)):
    x_test.append(scaled_data[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# 3. Prediksi
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices) # Kembalikan ke angka Dollar asli

# 4. Visualisasi dengan Matplotlib
plt.figure(figsize=(12,6))
plt.plot(data[prediction_days:], color="black", label="Harga Asli XAUUSD")
plt.plot(predicted_prices, color="green", label="Prediksi AI (LSTM)")
plt.title("Monitoring & Prediksi Harga Emas")
plt.xlabel("Waktu (Hari)")
plt.ylabel("Harga (USD)")
plt.legend()
plt.grid(True)
plt.show()