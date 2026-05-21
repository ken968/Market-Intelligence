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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
def build_model(seq_len: int, n_features: int, asset_key: str) -> tf.keras.Model:
    """
    Build LSTM model from config['model_arch'] (per-asset architecture from Minggu 2).

    Architecture profiles (set in utils/config.py):
        Gold (stable):       units=[64, 32],    dropout=0.20, attention=False
        BTC (volatile):      units=[128,64,32], dropout=0.30, attention=True
        NVDA/TSLA (high β):  units=[128, 64],   dropout=0.35, attention=True
        SPY/DIA (index):     units=[64, 32],    dropout=0.20, attention=False

    Output: 1 neuron (unnormalized % change — StandardScaler handles scaling)
    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input as KInput, LSTM, Dense, Dropout,
        MultiHeadAttention, LayerNormalization, Flatten
    )
    from tensorflow.keras.regularizers import l2 as keras_l2

    # Read from config; fall back to safe defaults if key missing
    asset_cfg = ASSETS.get(asset_key, {})
    arch      = asset_cfg.get('model_arch', {'units': [100, 50], 'dropout': 0.3, 'attention': False})
    units     = arch.get('units',    [100, 50])
    dropout   = arch.get('dropout',  0.3)
    use_attn  = arch.get('attention', False)

    print(f"  Model arch: units={units}, dropout={dropout}, attention={use_attn}")

    inp = KInput(shape=(seq_len, n_features))
    x   = inp

    for i, u in enumerate(units):
        is_last    = (i == len(units) - 1)
        return_seq = (not is_last) or use_attn   # keep sequence for attention or next LSTM
        x = LSTM(u, return_sequences=return_seq, kernel_regularizer=keras_l2(0.001))(x)
        x = Dropout(dropout)(x)

        # Self-Attention after first LSTM layer (if enabled)
        if use_attn and i == 0:
            attn_out = MultiHeadAttention(num_heads=4, key_dim=max(1, u // 4))(x, x)
            attn_out = LayerNormalization()(attn_out + x)   # residual
            x = attn_out if not is_last else Flatten()(attn_out)
        elif is_last and use_attn:
            x = Flatten()(x)   # flatten sequence dim after last attended LSTM

    x   = Dense(max(16, units[-1] // 2), kernel_regularizer=keras_l2(0.001))(x)
    out = Dense(1)(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


# ─────────────────────────────────────────────────────────────────────────────
# CORE TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def train_lstm_pct(asset_key: str) -> dict:
    asset_key = asset_key.lower()
    print(f"\n{'='*60}")
    print(f" LSTM Pct-Change Training — {asset_key.upper()}")
    print(f"{'='*60}")

    if asset_key not in ASSETS:
        print(f"Error: Unknown asset '{asset_key}'. Available: {list(ASSETS.keys())}")
        return {}

    config   = ASSETS[asset_key]
    data_file = config['data_file']
    seq_len   = config.get('sequence_length', 60)

    # ── Determine price column ──────────────────────────────────────────────
    price_col = None
    for candidate in config['features']:
        if candidate in ['Gold', 'BTC'] or candidate in STOCK_TICKERS:
            price_col = candidate
            break
    if price_col is None:
        price_col = config['features'][0]

    # ── Load data ────────────────────────────────────────────────────────────
    if not os.path.exists(data_file):
        print(f"Error: Data file not found: {data_file}")
        return {}

    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    df = df.sort_index()

    # Fill missing features with 0
    for feat in config['features']:
        if feat not in df.columns:
            print(f"  Warning: '{feat}' missing, filling with 0")
            df[feat] = 0

    features = config['features']
    print(f"  Dataset: {len(df)} samples | {len(features)} features")
    print(f"  Price column: {price_col} | Sequence: {seq_len} days | Horizon: {HORIZON_DAYS} days")

    # ── Compute target: 7-day forward % change ───────────────────────────────
    df['_pct_target'] = (
        df[price_col].shift(-HORIZON_DAYS) - df[price_col]
    ) / df[price_col]

    # Drop last HORIZON_DAYS rows (no future price known)
    df = df.dropna(subset=['_pct_target'])

    raw_features = df[features].ffill().fillna(0).values   # (N, n_features)
    raw_target   = df['_pct_target'].values                  # (N,)

    # ── Scaler 1: Feature scaler (normalize all input features) ─────────────
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = feature_scaler.fit_transform(raw_features)

    # ── Scaler 2: Target scaler (standardize % change values) ───────────────
    # StandardScaler is better here than MinMax because % change can be negative
    # and its distribution is approximately normal (not bounded like prices)
    target_scaler = StandardScaler()
    scaled_target = target_scaler.fit_transform(raw_target.reshape(-1, 1)).flatten()

    # ── Build sequences: X = last seq_len rows of features, y = target ──────
    X_all, y_all = [], []
    for t in range(seq_len, len(scaled_features)):
        X_all.append(scaled_features[t - seq_len: t, :])
        y_all.append(scaled_target[t])   # target at time t (aligned)

    X_all = np.array(X_all)   # (N-seq_len, seq_len, n_features)
    y_all = np.array(y_all)   # (N-seq_len,)

    # ── 80/20 chronological split ────────────────────────────────────────────
    split_idx  = int(len(X_all) * 0.80)
    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]

    print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Build and train model ────────────────────────────────────────────────
    model = build_model(seq_len, X_train.shape[2], asset_key)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=7, min_lr=1e-6, verbose=1),
    ]

    print("\n  Training...")
    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=2,
    )

    # ── Evaluate ─────────────────────────────────────────────────────────────
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    # Inverse transform back to actual % change values
    y_pred_pct = target_scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)).flatten()
    y_true_pct = target_scaler.inverse_transform(
        y_test.reshape(-1, 1)).flatten()

    hr     = hit_ratio(y_pred_pct, y_true_pct)
    rmse   = np.sqrt(mean_squared_error(y_true_pct, y_pred_pct))

    print(f"\n{'='*60}")
    print(f"  [RESULT] {asset_key.upper()} LSTM Pct-Change Model")
    print(f"{'='*60}")
    print(f"  Test Hit Ratio : {hr:.1f}%")
    print(f"  Test RMSE      : {rmse:.6f}  (in % change units)")
    print(f"  Mean pred      : {y_pred_pct.mean():.4f}  True mean: {y_true_pct.mean():.4f}")

    # ── Save model and scalers ────────────────────────────────────────────────
    os.makedirs('models', exist_ok=True)

    # Model file — kept at the same path config expects
    # This ensures predictor.py can still load it without changes
    model_path = config['model_file']
    model.save(model_path)

    # Feature scaler — saved at the path config expects
    feat_scaler_path = config['scaler_file']
    with open(feat_scaler_path, 'wb') as f:
        pickle.dump(feature_scaler, f)

    # Target scaler — NEW file, used by Ridge stacker
    target_scaler_path = config['scaler_file'].replace('.pkl', '_target.pkl')
    with open(target_scaler_path, 'wb') as f:
        pickle.dump(target_scaler, f)

    # Training metadata for Ridge stacker and monitoring
    meta = {
        'asset': asset_key,
        'price_col': price_col,
        'features': features,
        'horizon_days': HORIZON_DAYS,
        'sequence_length': seq_len,
        'target_type': 'pct_change_7d',
        'hit_ratio_test': hr,
        'rmse_test': rmse,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'target_scaler_path': target_scaler_path,
    }
    meta_path = config['scaler_file'].replace('.pkl', '_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=4)

    print(f"\n  Model:          {model_path}")
    print(f"  Feature scaler: {feat_scaler_path}")
    print(f"  Target scaler:  {target_scaler_path}")
    print(f"  Metadata:       {meta_path}")

    return meta


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/train_lstm_pct.py [asset | all]")
        print("       asset: gold, btc, spy, qqq, aapl, msft, ...")
        print("       all  : train all assets sequentially")
        sys.exit(1)

    arg = sys.argv[1].lower()

    if arg == 'all':
        print(f"Training {len(ALL_ASSETS)} assets: {ALL_ASSETS}")
        results = {}
        for a in ALL_ASSETS:
            try:
                r = train_lstm_pct(a)
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

    else:
        train_lstm_pct(arg)
