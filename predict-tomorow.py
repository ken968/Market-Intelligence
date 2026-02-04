import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# 1. Load Data & Model
# Kita gunakan file hasil Step 2 yang sudah lengkap 5 kolom
df = pd.read_csv('gold_global_insights.csv')
model = load_model('gold_ultimate_model.h5')

# 2. Persiapan Fitur
# Urutan fitur HARUS sama dengan saat training di train_ultimate.py
features = ['Gold', 'DXY', 'VIX', 'Yield_10Y', 'Sentiment']
data = df[features].values

# 3. Scaling (Normalisasi)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 4. Mengambil 60 Hari Terakhir
# Pastikan window tetap 60 hari agar otak AI tidak bingung (jangan diubah ke 5)
prediction_days = 60
if len(scaled_data) >= prediction_days:
    # Reshape menjadi (1, 60, 5) -> 1 batch, 60 hari, 5 kolom fitur
    last_60_days = scaled_data[-prediction_days:].reshape(1, prediction_days, 5)
    
    # 5. Eksekusi Prediksi
    prediction_scaled = model.predict(last_60_days)
    
    # 6. Inverse Transform (Mengembalikan ke angka USD)
    # Scaler mengharapkan 5 kolom, jadi kita buat array 'dummy'
    dummy = np.zeros((1, 5))
    # Masukkan hasil prediksi (angka tunggal) ke posisi kolom 'Gold' (index 0)
    dummy[0, 0] = prediction_scaled.item() 
    
    prediction_final = scaler.inverse_transform(dummy)[0][0]
    
    # 7. Output Hasil
    print("\n" + "="*40)
    print("      HASIL PREDIKSI AI GLOBAL INSIGHTS")
    print("="*40)
    print(f"Harga Penutupan Terakhir : ${df['Gold'].iloc[-1]:,.2f}")
    print(f"Prediksi Harga Besok     : ${prediction_final:,.2f}")
    
    # Hitung Selisih
    diff = prediction_final - df['Gold'].iloc[-1]
    arah = "NAIK 📈" if diff > 0 else "TURUN 📉"
    print(f"Estimasi Pergerakan      : {arah} (${abs(diff):,.2f})")
    print("="*40)
    print("Analisis: Berdasarkan tren 60 hari, DXY, VIX, Yield, & Sentimen.")
else:
    print(f"Error: Data tidak cukup. Butuh minimal {prediction_days} hari.")
