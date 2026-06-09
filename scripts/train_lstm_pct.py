"""
XAUUSD Multi-Asset Terminal
Phase 7: The Hybrid High-Alpha Path (Independent Models A & B)
"""
import os
import gc
import json
import numpy as np
import pandas as pd
import warnings
import joblib

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

import tensorflow as tf
tf.random.set_seed(42)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input as KInput, LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2 as keras_l2

from utils.config import ASSETS, STOCK_TICKERS

CORE_ASSETS = ['gold', 'btc', 'spy', 'qqq', 'dia']

def directional_mse(y_true, y_pred):
    """
    Directional Penalty Loss Function.
    Penalizes predictions with wrong direction by a factor of 2.
    """
    mse = tf.square(y_true - y_pred)
    penalty = tf.where(tf.sign(y_true) * tf.sign(y_pred) < 0, 2.0, 1.0)
    return tf.reduce_mean(mse * penalty)

class LSTMTrainer:
    def __init__(self, asset_key: str):
        self.asset_key = asset_key.lower()
        if self.asset_key not in ASSETS:
            raise ValueError(f"Unknown asset {self.asset_key}")
        self.config = ASSETS[self.asset_key]
        self.data_file = self.config['data_file']
        self.seq_len = self.config.get('sequence_length', 90)

    def _get_price_col(self) -> str:
        for candidate in self.config['features']:
            if candidate in ['Gold', 'BTC'] or candidate in STOCK_TICKERS:
                return candidate
        return self.config['features'][0]

    def build_model(self, seq_len: int, n_features: int, out_dim: int) -> tf.keras.Model:
        arch      = self.config.get('model_arch', {'units': [100, 50], 'dropout': 0.3, 'attention': False})
        units     = arch.get('units', [100, 50])
        dropout   = arch.get('dropout', 0.3)
        use_attn  = arch.get('attention', False)

        inp = KInput(shape=(seq_len, n_features))
        x = inp
        for i, u in enumerate(units):
            is_last = (i == len(units) - 1)
            return_seq = (not is_last) or use_attn
            x = LSTM(u, return_sequences=return_seq, kernel_regularizer=keras_l2(0.001))(x)
            x = Dropout(dropout)(x)
            if use_attn and i == 0:
                attn_out = MultiHeadAttention(num_heads=4, key_dim=max(1, u // 4))(x, x)
                attn_out = LayerNormalization()(attn_out + x)
                x = attn_out if not is_last else Flatten()(attn_out)
            elif is_last and use_attn:
                x = Flatten()(x)

        x = Dense(max(16, units[-1] // 2), kernel_regularizer=keras_l2(0.001), activation='relu')(x)
        out = Dense(out_dim)(x)

        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                      loss=directional_mse, metrics=['mae'])
        return model

    def train_walk_forward(self, df_base: pd.DataFrame, horizons: list, model_name: str) -> dict:
        print(f"\n  --- Training {model_name} (Horizons: {horizons}) Walk-Forward CV ---")
        price_col = self._get_price_col()
        features = self.config['features']
        
        df = df_base.copy()
        target_cols = []
        for h in horizons:
            col = f'_pct_target_{h}'
            df[col] = (df[price_col].shift(-h) - df[price_col]) / df[price_col]
            target_cols.append(col)
            
        df = df.dropna(subset=target_cols)
        if len(df) < self.seq_len + 100:
            print("Not enough data")
            return {}

        raw_features = df[features].ffill().fillna(0).values
        raw_targets = df[target_cols].values
        
        X_raw, y_raw = [], []
        for t in range(self.seq_len, len(raw_features)):
            X_raw.append(raw_features[t - self.seq_len: t, :])
            y_raw.append(raw_targets[t])
        X_raw = np.array(X_raw)
        y_raw = np.array(y_raw)
        
        # 5 Windows Walk-Forward
        n_windows = 5
        window_size = len(X_all) // (n_windows + 1)
        
        best_model = None
        best_val_loss = float('inf')
        
        for w in range(n_windows):
            train_end = (w + 1) * window_size
            test_end = (w + 2) * window_size if w < n_windows - 1 else len(X_all)
            
            # Gap buffer = max(horizons)
            gap = max(horizons)
            split_idx = train_end - gap
            if split_idx <= 0: continue
            
            X_train_raw, y_train_raw = X_raw[:split_idx], y_raw[:split_idx]
            X_val_raw, y_val_raw = X_raw[train_end:test_end], y_raw[train_end:test_end]
            
            # PREVENT DATA LEAKAGE: Fit scaler strictly on training window
            feature_scaler = MinMaxScaler(feature_range=(0, 1))
            # Reshape for scaling (N*seq_len, n_features)
            n_features = X_train_raw.shape[2]
            X_train_flat = X_train_raw.reshape(-1, n_features)
            feature_scaler.fit(X_train_flat)
            
            X_train = feature_scaler.transform(X_train_flat).reshape(X_train_raw.shape)
            if len(X_val_raw) > 0:
                X_val = feature_scaler.transform(X_val_raw.reshape(-1, n_features)).reshape(X_val_raw.shape)
            else:
                X_val = np.empty_like(X_val_raw)
            
            target_scaler = StandardScaler()
            y_train = target_scaler.fit_transform(y_train_raw)
            if len(y_val_raw) > 0:
                y_val = target_scaler.transform(y_val_raw)
            else:
                y_val = np.empty_like(y_val_raw)
            
            model = self.build_model(self.seq_len, X_train.shape[2], len(horizons))
            cb = [EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
                  ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)]
                  
            model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val),
                      callbacks=cb, verbose=0)
                      
            val_loss = model.evaluate(X_val, y_val, verbose=0)
            # Evaluate returns a list [loss, mae]. We take index 0
            if isinstance(val_loss, list): val_loss = val_loss[0]
            print(f"    Window {w+1}/5 - Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                
        # Save Model and Scalers
        base_model_file = self.config['model_file']
        out_model_file = base_model_file.replace('.keras', f'_{model_name.lower()}.keras')
        best_model.save(out_model_file)
        
        base_scaler_file = self.config['scaler_file']
        out_scaler_file = base_scaler_file.replace('.pkl', f'_{model_name.lower()}.pkl')
        joblib.dump({'feature_scaler': feature_scaler, 'target_scaler': target_scaler}, out_scaler_file)
        
        # Eval HR on the final validation set
        if len(X_val_raw) > 0:
            preds = best_model.predict(X_val, verbose=0)
            unscaled_preds = target_scaler.inverse_transform(preds)
            unscaled_y = y_val_raw
        else:
            # If no validation set left, use train set for final HR logging
            preds = best_model.predict(X_train, verbose=0)
            unscaled_preds = target_scaler.inverse_transform(preds)
            unscaled_y = y_train_raw
        
        metrics = {}
        for i, h in enumerate(horizons):
            hr = float((np.sign(unscaled_preds[:, i]) == np.sign(unscaled_y[:, i])).mean()) * 100.0
            metrics[f'hr_{h}'] = hr
            print(f"    [Final] Horizon {h}D Hit Ratio: {hr:.1f}%")
            
        return metrics

def run_training():
    for asset in CORE_ASSETS:
        print(f"\n{'='*50}\nTraining Core Asset: {asset.upper()}\n{'='*50}")
        try:
            trainer = LSTMTrainer(asset)
            if not os.path.exists(trainer.data_file):
                print(f"Data not found for {asset}. Skipping.")
                continue
                
            df = pd.read_csv(trainer.data_file, index_col=0, parse_dates=True).sort_index()
            
            # Model A: Short Term
            m_a = trainer.train_walk_forward(df, [1, 7, 14], 'Model_A')
            # Model B: Macro Term
            m_b = trainer.train_walk_forward(df, [30, 90], 'Model_B')
            
        except Exception as e:
            print(f"Error training {asset}: {e}")
            
if __name__ == '__main__':
    run_training()
