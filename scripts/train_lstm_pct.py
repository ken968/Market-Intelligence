"""
Unified LSTM Training Engine — Pct Change Target
==================================================
Menggantikan train_ultimate.py, train_btc.py, dan train_stocks.py.

PERUBAHAN FUNDAMENTAL dari versi sebelumnya:
    SEBELUM: y_train = scaled_price[t]         (harga absolut ternormalisasi)
    SEKARANG: y_train = pct_change_7d[t]        (% change 7 hari ke depan)

Mengapa penting:
    1. Satuan output seragam dengan XGBoost → Ridge Stacker bisa bekerja
    2. % change adalah nilai stationary (tidak drift seperti harga absolut)
    3. LSTM belajar ARAH pergerakan, bukan level harga → hit ratio meningkat
    4. Direct prediction (bukan recursive) → menghindari error compounding

Target definition:
    y[t] = (price[t + HORIZON] - price[t]) / price[t]
    Ini adalah % change dari hari t ke t+7.

Dua scaler yang disimpan:
    - feature_scaler   : normalisasi input features (MinMaxScaler, semua kolom)
    - target_scaler    : normalisasi target % change (StandardScaler, 1 nilai)
    Memisahkan keduanya agar inverse transform bisa dilakukan secara benar.

Usage:
    python scripts/train_lstm_pct.py gold
    python scripts/train_lstm_pct.py btc
    python scripts/train_lstm_pct.py spy
    python scripts/train_lstm_pct.py all     <- train semua sekaligus
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'   # Force fully deterministic TF CPU ops
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Reproducibility: fix all random seeds ──────────────────────────────────
import random
random.seed(42)
import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

from utils.config import ASSETS, STOCK_TICKERS

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
HORIZON_DAYS = 7      # Predict 7-day forward % change
ALL_ASSETS = ['gold', 'btc'] + [t.lower() for t in STOCK_TICKERS.keys()]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: directional hit ratio
# ─────────────────────────────────────────────────────────────────────────────
def hit_ratio(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """% of predictions with correct sign (direction)."""
    return float((np.sign(y_pred) == np.sign(y_true)).mean()) * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE (per asset type)
# ─────────────────────────────────────────────────────────────────────────────
class LSTMTrainer:
    def __init__(self, asset_key: str):
        self.asset_key = asset_key.lower()
        if self.asset_key not in ASSETS:
            raise ValueError(f"Unknown asset '{self.asset_key}'. Available: {list(ASSETS.keys())}")
        self.config = ASSETS[self.asset_key]
        self.data_file = self.config['data_file']
        self.seq_len = self.config.get('sequence_length', 60)

    def _get_price_col(self) -> str:
        for candidate in self.config['features']:
            if candidate in ['Gold', 'BTC'] or candidate in STOCK_TICKERS:
                return candidate
        return self.config['features'][0]

    def build_model(self, seq_len: int, n_features: int) -> tf.keras.Model:
        """
        Build LSTM model from config['model_arch']
        """
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            Input as KInput, LSTM, Dense, Dropout,
            MultiHeadAttention, LayerNormalization, Flatten
        )
        from tensorflow.keras.regularizers import l2 as keras_l2

        arch      = self.config.get('model_arch', {'units': [100, 50], 'dropout': 0.3, 'attention': False})
        units     = arch.get('units',    [100, 50])
        dropout   = arch.get('dropout',  0.3)
        use_attn  = arch.get('attention', False)

        print(f"  Model arch: units={units}, dropout={dropout}, attention={use_attn}")

        inp = KInput(shape=(seq_len, n_features))
        x   = inp

        for i, u in enumerate(units):
            is_last    = (i == len(units) - 1)
            return_seq = (not is_last) or use_attn
            x = LSTM(u, return_sequences=return_seq, kernel_regularizer=keras_l2(0.001))(x)
            x = Dropout(dropout)(x)

            if use_attn and i == 0:
                attn_out = MultiHeadAttention(num_heads=4, key_dim=max(1, u // 4))(x, x)
                attn_out = LayerNormalization()(attn_out + x)
                x = attn_out if not is_last else Flatten()(attn_out)
            elif is_last and use_attn:
                x = Flatten()(x)

        x   = Dense(max(16, units[-1] // 2), kernel_regularizer=keras_l2(0.001))(x)
        out = Dense(1)(x)

        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model

    def train_horizon(self, df_base: pd.DataFrame, horizon_days: int) -> dict:
        print(f"\n  --- Horizon: {horizon_days} Days ---")
        price_col = self._get_price_col()
        features = self.config['features']

        df = df_base.copy()
        # Compute target: horizon_days forward % change
        df['_pct_target'] = (
            df[price_col].shift(-horizon_days) - df[price_col]
        ) / df[price_col]

        df = df.dropna(subset=['_pct_target'])

        raw_features = df[features].ffill().fillna(0).values   # (N, n_features)
        raw_target   = df['_pct_target'].values                  # (N,)

        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = feature_scaler.fit_transform(raw_features)

        target_scaler = StandardScaler()
        scaled_target = target_scaler.fit_transform(raw_target.reshape(-1, 1)).flatten()

        X_all, y_all = [], []
        for t in range(self.seq_len, len(scaled_features)):
            X_all.append(scaled_features[t - self.seq_len: t, :])
            y_all.append(scaled_target[t])

        X_all = np.array(X_all)
        y_all = np.array(y_all)

        split_idx  = int(len(X_all) * 0.80)
        X_train, X_test = X_all[:split_idx], X_all[split_idx:]
        y_train, y_test = y_all[:split_idx], y_all[split_idx:]

        print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

        model = self.build_model(self.seq_len, X_train.shape[2])

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=8, min_lr=1e-6, verbose=0),
        ]

        # Recency-weighted sample weights: recent samples get up to 3x more weight
        # This forces the model to focus on the new price regime ($3500-$4500+)
        n_train = len(X_train)
        linear_weights = np.linspace(1.0, 3.0, n_train)
        sample_weights = linear_weights / linear_weights.mean()  # normalize so total weight stays same

        model.fit(
            X_train, y_train,
            epochs=150,
            batch_size=32,
            validation_split=0.1,
            callbacks=callbacks,
            sample_weight=sample_weights,
            verbose=0,
        )

        y_pred_scaled = model.predict(X_test, verbose=0).flatten()

        y_pred_pct = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true_pct = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        hr     = hit_ratio(y_pred_pct, y_true_pct)
        rmse   = np.sqrt(mean_squared_error(y_true_pct, y_pred_pct))

        print(f"  Test Hit Ratio : {hr:.1f}%")
        print(f"  Test RMSE      : {rmse:.6f}  (in % change units)")

        os.makedirs('models', exist_ok=True)

        # ── Save files for this specific horizon ─────────────────────────────────
        model_path = f"models/{self.asset_key}_model_{horizon_days}d.keras"
        model.save(model_path)

        feat_scaler_path = f"models/{self.asset_key}_scaler_{horizon_days}d.pkl"
        with open(feat_scaler_path, 'wb') as f:
            pickle.dump(feature_scaler, f)

        target_scaler_path = f"models/{self.asset_key}_scaler_{horizon_days}d_target.pkl"
        with open(target_scaler_path, 'wb') as f:
            pickle.dump(target_scaler, f)

        meta = {
            'asset': self.asset_key,
            'price_col': price_col,
            'features': features,
            'horizon_days': horizon_days,
            'sequence_length': self.seq_len,
            'target_type': f'pct_change_{horizon_days}d',
            'hit_ratio_test': hr,
            'rmse_test': rmse,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'target_scaler_path': target_scaler_path,
        }
        meta_path = f"models/{self.asset_key}_scaler_{horizon_days}d_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=4)

        # ── Save legacy files if this is the 7-day model ─────────────────────────
        if horizon_days == 7:
            legacy_model_path = self.config['model_file']
            model.save(legacy_model_path)
            
            legacy_feat_scaler_path = self.config['scaler_file']
            with open(legacy_feat_scaler_path, 'wb') as f:
                pickle.dump(feature_scaler, f)
                
            legacy_target_scaler_path = self.config['scaler_file'].replace('.pkl', '_target.pkl')
            with open(legacy_target_scaler_path, 'wb') as f:
                pickle.dump(target_scaler, f)
                
            legacy_meta_path = self.config['scaler_file'].replace('.pkl', '_meta.json')
            with open(legacy_meta_path, 'w') as f:
                json.dump(meta, f, indent=4)

        return meta

    def train(self) -> dict:
        print(f"\n{'='*60}")
        print(f" LSTM Multi-Horizon Training — {self.asset_key.upper()}")
        print(f"{'='*60}")

        # ── Load data using MarketDataStore with CSV fallback ────────────────────
        from utils.data_store import MarketDataStore
        store = MarketDataStore()
        data_path = self.config['data_file']
        table_name = os.path.splitext(os.path.basename(data_path))[0].lower()
        
        df = None
        try:
            df = store.read_table(table_name, format='pandas')
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            df = df.sort_index()
            print(f"  Loaded data from DuckDB table '{table_name}'")
        except Exception as e:
            print(f"  Warning: Could not read table '{table_name}' from DuckDB: {e}. Falling back to CSV.")
            if not os.path.exists(data_path):
                print(f"Error: Data file not found: {data_path}")
                return {}
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            df = df.sort_index()

        for feat in self.config['features']:
            if feat not in df.columns:
                print(f"  Warning: '{feat}' missing, filling with 0")
                df[feat] = 0

        features = self.config['features']
        print(f"  Dataset: {len(df)} samples | {len(features)} features")
        
        horizons = [1, 7, 14, 30, 90]
        results = {}
        for h in horizons:
            try:
                meta = self.train_horizon(df, h)
                results[h] = meta
            except Exception as e:
                print(f"  [ERROR] Horizon {h}d failed: {e}")
                
        return results.get(7, {})



# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/train_lstm_pct.py [asset | all | stocks]")
        print("       asset: gold, btc, spy, qqq, aapl, msft, ...")
        print("       all  : train all assets sequentially")
        print("       stocks: train all stock assets sequentially")
        sys.exit(1)

    arg = sys.argv[1].lower()

    if arg == 'all':
        print(f"Training {len(ALL_ASSETS)} assets: {ALL_ASSETS}")
        results = {}
        for a in ALL_ASSETS:
            try:
                trainer = LSTMTrainer(a)
                r = trainer.train()
                results[a] = r.get('hit_ratio_test', 0)
            except Exception as e:
                print(f"Error on {a}: {e}")
                results[a] = None

        print(f"\n{'='*60}")
        print(" SUMMARY")
        print(f"{'='*60}")
        for a, hr_val in results.items():
            status = f"{hr_val:.1f}%" if hr_val is not None else "FAILED"
            print(f"  {a:<12} Hit Ratio: {status}")

    elif arg == 'stocks':
        stock_assets = [t.lower() for t in STOCK_TICKERS.keys()]
        print(f"Training {len(stock_assets)} stock assets: {stock_assets}")
        results = {}
        for a in stock_assets:
            try:
                trainer = LSTMTrainer(a)
                r = trainer.train()
                results[a] = r.get('hit_ratio_test', 0)
            except Exception as e:
                print(f"Error on {a}: {e}")
                results[a] = None

        print(f"\n{'='*60}")
        print(" STOCK SUMMARY")
        print(f"{'='*60}")
        for a, hr_val in results.items():
            status = f"{hr_val:.1f}%" if hr_val is not None else "FAILED"
            print(f"  {a:<12} Hit Ratio: {status}")

    else:
        trainer = LSTMTrainer(arg)
        trainer.train()
