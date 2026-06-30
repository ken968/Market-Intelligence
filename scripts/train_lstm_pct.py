import os
import sys

# Force UTF-8 output on Windows terminals (CP1252 chokes on box-drawing/arrow chars)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
XAUUSD Multi-Asset Terminal
Phase 7: The Hybrid High-Alpha Path (Independent Models A & B)

Step 0 (Multi-Window Quorum):
  - Save all 5 window models (not just the best model)
  - Save OOS predictions per window → Stacker retraining data
  - Generate model_registry.json → centralized directory for predictor_engine
"""
import gc
import json
import numpy as np
import pandas as pd
import warnings
import joblib
from datetime import datetime, timezone

# Fix import path when running script directly (outside project root)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

import tensorflow as tf
tf.random.set_seed(42)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input as KInput, LSTM, Dense, Dropout,
    MultiHeadAttention, LayerNormalization, Flatten
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2 as keras_l2

from utils.config import ASSETS, STOCK_TICKERS

# Only these 5 core assets are trained in Phase 7 (Beta-Scaling strategy)
CORE_ASSETS = ['gold', 'btc', 'spy', 'qqq', 'dia']

# Asset-specific EWMA decay for Rolling OOS IC weighting at inference
# BTC: λ=0.85 (half-life ~4d) — crypto needs faster regime adaptation
# Gold: λ=0.94 (half-life ~11d) — macro-driven, needs stability
# Indices: λ=0.90-0.92 — between BTC and Gold
EWMA_LAMBDA = {
    'gold': 0.94,
    'btc':  0.85,
    'spy':  0.92,
    'qqq':  0.90,
    'dia':  0.93,
}


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────
def hit_ratio(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float((np.sign(y_pred) == np.sign(y_true)).mean()) * 100.0

def naive_hit_ratio(y_true: np.ndarray) -> float:
    p_pos = float((y_true > 0).mean())
    return max(p_pos, 1.0 - p_pos) * 100.0

def skill_score(hr: float, naive_hr: float) -> float:
    if naive_hr >= 100.0:
        return 0.0
    return (hr - naive_hr) / (100.0 - naive_hr) * 100.0

def information_coefficient(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    from scipy import stats
    if len(y_pred) < 3:
        return 0.0
    corr, _ = stats.spearmanr(y_pred, y_true)
    return float(corr) if not np.isnan(corr) else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTION: Directional MSE
# ─────────────────────────────────────────────────────────────────────────────
def directional_mse(y_true, y_pred):
    """
    Directional Penalty Loss — v3 (Smooth Gradient / Anti-Collapse).
    
    Problem with tf.sign(): gradient is 0 or undefined at 0,
    causing LSTM saturation and model collapse (constant predictions).
    
    Fix: use relu(-y_true * y_pred) as the directional term.
    - relu(-y*p) > 0 ONLY when y and p have opposite signs (wrong direction)
    - Has smooth, well-defined gradients everywhere
    - MSE term alone forces model away from predicting 0
    """
    mse         = tf.reduce_mean(tf.square(y_true - y_pred))
    dir_penalty = tf.reduce_mean(tf.nn.relu(-y_true * y_pred))
    return mse + dir_penalty


# ─────────────────────────────────────────────────────────────────────────────
# TRAINER CLASS
# ─────────────────────────────────────────────────────────────────────────────
class LSTMTrainer:
    def __init__(self, asset_key: str):
        self.asset_key = asset_key.lower()
        if self.asset_key not in ASSETS:
            raise ValueError(f"Unknown asset '{self.asset_key}'. Available: {list(ASSETS.keys())}")
        self.config = ASSETS[self.asset_key]
        self.data_file = self.config['data_file']
        self.seq_len = self.config.get('sequence_length', 90)

    def _get_price_col(self) -> str:
        """Finds the primary price column from config features."""
        for candidate in self.config['features']:
            if candidate in ['Gold', 'BTC'] or candidate in STOCK_TICKERS:
                return candidate
        return self.config['features'][0]

    def _get_available_features(self, df: pd.DataFrame) -> list:
        """Validates that all required features exist in the dataframe. Raises ValueError if missing."""
        all_feats = self.config['features']
        missing = [f for f in all_feats if f not in df.columns]
        if missing:
            raise ValueError(f"CRITICAL ERROR: Missing required features in dataset: {missing}. Training aborted to prevent data corruption.")
        return all_feats

    def build_model(self, seq_len: int, n_features: int, out_dim: int) -> tf.keras.Model:
        """Builds LSTM model from asset config architecture."""
        arch     = self.config.get('model_arch', {'units': [100, 50], 'dropout': 0.3, 'attention': False})
        units    = arch.get('units', [100, 50])
        dropout  = arch.get('dropout', 0.3)
        use_attn = arch.get('attention', False)

        inp = KInput(shape=(seq_len, n_features))
        x = inp
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

        x   = Dense(max(16, units[-1] // 2), kernel_regularizer=keras_l2(0.001), activation='relu')(x)
        out = Dense(out_dim)(x)

        model = Model(inputs=inp, outputs=out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
            loss=directional_mse,
            metrics=['mae']
        )
        return model

    def train_walk_forward(
        self,
        df_base: pd.DataFrame,
        horizons: list,
        model_name: str,
        registry: dict,
    ) -> dict:
        """
        Trains a multi-output LSTM with 5-window Walk-Forward Cross Validation.

        Langkah 0 changes:
          - ALL 5 window models are saved to disk (not just the best)
          - OOS predictions + actuals from each window's validation set are saved
            → These are used later to retrain Ridge Stacker without data leakage
          - model_registry dict is updated with metadata for each saved window
          - collapse detection still applies (marks collapsed window in registry)

        Scalers are fit per-window on training data only (no look-ahead bias).
        """
        print(f"\n  --- Training {model_name} (Horizons: {horizons}) Walk-Forward CV ---")
        price_col = self._get_price_col()
        features  = self._get_available_features(df_base)

        df = df_base.copy()

        # Preserve date index for OOS predictions alignment
        df_dates = df.index  # DatetimeIndex

        # Build multi-output targets
        target_cols = []
        for h in horizons:
            col = f'_pct_target_{h}'
            df[col] = (df[price_col].shift(-h) - df[price_col]) / df[price_col]
            target_cols.append(col)

        df = df.dropna(subset=target_cols)

        if len(df) < self.seq_len + 100:
            print("  [Skip] Not enough rows to train. Requires seq_len + 100 minimum.")
            return {}

        raw_features = df[features].ffill().fillna(0).values   # (N, n_features)
        raw_targets  = df[target_cols].values                  # (N, n_horizons)
        # Date index aligned with X_raw (each row = end date of its seq window)
        date_index   = df.index[self.seq_len:]                 # (N_samples,)

        # Build sliding windows (no scaling yet — done per window to prevent leakage)
        X_raw, y_raw = [], []
        for t in range(self.seq_len, len(raw_features)):
            X_raw.append(raw_features[t - self.seq_len: t, :])
            y_raw.append(raw_targets[t])
        X_raw = np.array(X_raw)  # (N_samples, seq_len, n_features)
        y_raw = np.array(y_raw)  # (N_samples, n_horizons)

        n_windows   = 5
        window_size = len(X_raw) // (n_windows + 1)

        _arch = self.config.get('model_arch', {'units': [100, 50], 'dropout': 0.3, 'attention': False})
        print(f"    Model arch: units={_arch.get('units')}, dropout={_arch.get('dropout')}, "
              f"attention={_arch.get('attention')}, out_dim={len(horizons)}")

        # Paths for all-window saves
        base_model_path  = self.config['model_file'].replace('.keras', f'_{model_name.lower()}')
        base_scaler_path = self.config['scaler_file'].replace('.pkl', f'_{model_name.lower()}')
        os.makedirs(os.path.dirname(base_model_path), exist_ok=True)

        best_val_loss   = float('inf')
        best_window_idx = -1
        last_model      = None
        last_scaler_key = None
        window_results  = []  # collect per-window metadata

        for w in range(n_windows):
            train_end = (w + 1) * window_size
            test_end  = (w + 2) * window_size if w < n_windows - 1 else len(X_raw)

            gap       = max(horizons)
            split_idx = train_end - gap

            if split_idx <= 0:
                print(f"    Window {w+1}/5 - skipped (insufficient gap buffer)")
                continue

            X_train_raw = X_raw[:split_idx]
            y_train_raw = y_raw[:split_idx]
            X_val_raw   = X_raw[train_end:test_end]
            y_val_raw   = y_raw[train_end:test_end]
            dates_val   = date_index[train_end:test_end]

            # ── Scaler fit ONLY on training window ───────────────────────────
            n_features_dim = X_train_raw.shape[2]
            feature_scaler = MinMaxScaler(feature_range=(0, 1))
            feature_scaler.fit(X_train_raw.reshape(-1, n_features_dim))

            X_train = feature_scaler.transform(
                X_train_raw.reshape(-1, n_features_dim)
            ).reshape(X_train_raw.shape)

            if len(X_val_raw) > 0:
                X_val = feature_scaler.transform(
                    X_val_raw.reshape(-1, n_features_dim)
                ).reshape(X_val_raw.shape)
            else:
                X_val = X_train[-10:]

            target_scaler = StandardScaler()
            y_train = target_scaler.fit_transform(y_train_raw)
            y_val   = target_scaler.transform(y_val_raw) if len(y_val_raw) > 0 else y_train[-10:]

            # ── Train ─────────────────────────────────────────────────────────
            model = self.build_model(self.seq_len, X_train.shape[2], len(horizons))
            cb = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0),
            ]
            model.fit(
                X_train, y_train,
                epochs=100, batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=cb, verbose=0
            )

            val_loss = model.evaluate(X_val, y_val, verbose=0)
            if isinstance(val_loss, (list, tuple)):
                val_loss = val_loss[0]
            print(f"    Window {w+1}/5 - Val Loss: {val_loss:.4f}")

            # ── Collapse detection per window ─────────────────────────────────
            collapsed = False
            if len(X_val_raw) >= 10:
                _chk_x    = feature_scaler.transform(
                    X_val_raw[:10].reshape(-1, n_features_dim)
                ).reshape(X_val_raw[:10].shape)
                _chk_pred = model.predict(_chk_x, verbose=0)
                if np.std(_chk_pred) < 1e-6:
                    collapsed = True
                    print(f"    Window {w+1}/5 - ⚠ COLLAPSED (constant output, flagged in registry)")

            # ── Generate OOS predictions for this window (for Stacker training) ─
            oos_preds_unscaled = None
            window_metrics     = {}
            if len(X_val_raw) > 0 and not collapsed:
                preds_scaled       = model.predict(X_val, verbose=0)
                oos_preds_unscaled = target_scaler.inverse_transform(preds_scaled)  # (N_val, n_horizons)

                for i, h in enumerate(horizons):
                    y_p = oos_preds_unscaled[:, i]
                    y_t = y_val_raw[:, i]
                    window_metrics[f'hr_{h}d']   = float(hit_ratio(y_p, y_t))
                    window_metrics[f'rmse_{h}d']  = float(np.sqrt(np.mean((y_p - y_t) ** 2)))
                    window_metrics[f'ic_{h}d']    = float(information_coefficient(y_p, y_t))
                    window_metrics[f'ss_{h}d']    = float(skill_score(
                        window_metrics[f'hr_{h}d'], naive_hit_ratio(y_t)
                    ))

            # ── Save this window's model & scaler ─────────────────────────────
            w_model_path  = f"{base_model_path}_w{w+1}.keras"
            w_scaler_path = f"{base_scaler_path}_w{w+1}.pkl"
            model.save(w_model_path)
            joblib.dump({'feature_scaler': feature_scaler, 'target_scaler': target_scaler}, w_scaler_path)

            # ── Save OOS data bundle for Stacker training (OOS-only, no leakage) ─
            w_oos_path = f"{base_model_path}_w{w+1}_oos.pkl"
            joblib.dump({
                'predictions':  oos_preds_unscaled,          # (N_val, n_horizons) or None
                'actuals':      y_val_raw,                    # (N_val, n_horizons)
                'dates':        np.array(dates_val, dtype=str),  # ISO date strings
                'horizons':     horizons,
                'collapsed':    collapsed,
                'val_loss':     float(val_loss),
                'metrics':      window_metrics,
            }, w_oos_path)

            # ── Register window metadata ──────────────────────────────────────
            w_key = f"{self.asset_key}_{model_name.lower()}_w{w+1}"
            registry[w_key] = {
                'asset':        self.asset_key,
                'model_group':  model_name,       # 'Model_A' or 'Model_B'
                'window':       w + 1,
                'horizons':     horizons,
                'model_path':   w_model_path,
                'scaler_path':  w_scaler_path,
                'oos_path':     w_oos_path,
                'val_loss':     float(val_loss),
                'collapsed':    collapsed,
                'metrics':      window_metrics,
                'ewma_lambda':  EWMA_LAMBDA.get(self.asset_key, 0.92),
                'train_rows':   int(split_idx),
                'val_rows':     int(len(X_val_raw)),
            }
            window_results.append((w, val_loss, collapsed, model, feature_scaler, target_scaler, X_val_raw, y_val_raw))

            last_model      = model
            last_scaler_key = w_key

            if val_loss < best_val_loss and not collapsed:
                best_val_loss   = val_loss
                best_window_idx = w

        if not window_results:
            print("  [Error] No valid windows trained. Aborting.")
            return {}

        # ── Determine best window (collapse-aware) ────────────────────────────
        if best_window_idx < 0:
            # All windows collapsed — fall back to last window
            print("  [Warning] All windows collapsed. Using last window as fallback.")
            best_window_idx = window_results[-1][0]

        best_w, _, _, best_model, best_fs, best_ts, best_X_val, best_y_val = window_results[best_window_idx]
        best_w_key = f"{self.asset_key}_{model_name.lower()}_w{best_w+1}"

        # Mark best window in registry
        registry[best_w_key]['is_best_window'] = True
        for wr in window_results:
            k = f"{self.asset_key}_{model_name.lower()}_w{wr[0]+1}"
            if k != best_w_key:
                registry[k]['is_best_window'] = False

        # ── Also save the best model under the canonical legacy path ──────────
        # (keeps backward compatibility with any code that loads the old path)
        out_model_file  = self.config['model_file'].replace('.keras', f'_{model_name.lower()}.keras')
        out_scaler_file = self.config['scaler_file'].replace('.pkl', f'_{model_name.lower()}.pkl')
        best_model.save(out_model_file)
        joblib.dump({'feature_scaler': best_fs, 'target_scaler': best_ts}, out_scaler_file)
        print(f"  [Saved] Best window (W{best_w+1}) -> {out_model_file}")
        print(f"  [Saved] All {len(window_results)} window models saved individually. (Use model_registry.json to enumerate)")

        # ── Final Evaluation on best window's OOS set ─────────────────────────
        n_features_dim = best_X_val.shape[2]
        eval_X_scaled  = best_fs.transform(
            best_X_val.reshape(-1, n_features_dim)
        ).reshape(best_X_val.shape)

        preds          = best_model.predict(eval_X_scaled, verbose=0)
        unscaled_preds = best_ts.inverse_transform(preds)

        print(f"  {'─'*46}")
        print(f"  {'Horizon':<10} {'RMSE':>8} {'Hit Ratio':>10} {'Naive HR':>9} {'Skill Score':>12} {'IC':>7}")
        print(f"  {'─'*46}")

        metrics = {}
        for i, h in enumerate(horizons):
            y_p = unscaled_preds[:, i]
            y_t = best_y_val[:, i]

            rmse     = float(np.sqrt(np.mean((y_p - y_t) ** 2)))
            hr       = hit_ratio(y_p, y_t)
            naive_hr = naive_hit_ratio(y_t)
            ss       = skill_score(hr, naive_hr)
            ic       = information_coefficient(y_p, y_t)

            metrics[f'hr_{h}d']   = hr
            metrics[f'rmse_{h}d'] = rmse
            metrics[f'ic_{h}d']   = ic
            metrics[f'ss_{h}d']   = ss

            flag = ''
            if hr < 45.0:
                flag = ' ⚠ LOW'
            elif ic < 0:
                flag = ' ⚠ NEG-IC'

            print(f"  {h}D{' '*8} {rmse:>8.4f} {hr:>9.1f}% {naive_hr:>8.1f}% {ss:>11.1f}% {ic:>7.3f}{flag}")

        print(f"  {'─'*46}")
        return metrics


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def run_training(target_asset: str = 'all'):
    if target_asset == 'all':
        assets_to_train = CORE_ASSETS
    else:
        target_asset = target_asset.lower()
        if target_asset not in ASSETS:
            print(f"Error: Unknown asset '{target_asset}'. Available: {list(ASSETS.keys())}")
            return
        assets_to_train = [target_asset]

    # ── Shared registry: accumulates all window metadata across all assets ────
    registry_path = os.path.join('models', 'model_registry.json')
    registry = {}
    if os.path.exists(registry_path):
        try:
            import json
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        except Exception as e:
            print(f"  [Warning] Could not load existing registry: {e}")
    for asset in assets_to_train:
        sep = '=' * 50
        print(f"\n{sep}\nTraining Asset: {asset.upper()}\n{sep}")
        try:
            trainer = LSTMTrainer(asset)
            data_path = trainer.data_file

            # Phase 7 Upgrade: Use DuckDB as Primary Data Store (CSV as fallback)
            df = None
            try:
                from utils.data_store import MarketDataStore
                store = MarketDataStore()
                table_name = os.path.splitext(os.path.basename(data_path))[0].lower()
                df = store.read_table(table_name, format='pandas')
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                df = df.sort_index()
                print(f"  [DB] Loaded data from DuckDB table: {table_name}")
            except Exception as e:
                print(f"  [DB Warning] Failed to load from DuckDB ({e}). Falling back to CSV.")
                if os.path.exists(data_path):
                    df = pd.read_csv(data_path, index_col=0, parse_dates=True).sort_index()
                else:
                    print(f"  [Skip] Both DuckDB and CSV data not found for {asset}.")
                    continue

            print(f"  Data loaded: {len(df)} rows | seq_len={trainer.seq_len}")

            # Model A: Short-Term Specialist (1D, 7D, 14D)
            trainer.train_walk_forward(df, [1, 7, 14], 'Model_A', registry)

            # Model B: Macro Specialist (30D, 90D)
            trainer.train_walk_forward(df, [30, 90], 'Model_B', registry)

        except Exception as e:
            import traceback
            print(f"  [Error] Training {asset}: {e}")
            traceback.print_exc()

    # ── Write model_registry.json ─────────────────────────────────────────────
    registry_path = os.path.join('models', 'model_registry.json')
    os.makedirs('models', exist_ok=True)
    
    existing_assets = registry.get('_meta', {}).get('assets_trained', [])
    updated_assets = list(set(existing_assets + assets_to_train))
    
    registry['_meta'] = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'phase': 7,
        'assets_trained': updated_assets,
        'n_windows': 5,
        'description': (
            'Quorum-Based Multi-Window Registry. '
            'All 5 walk-forward window models saved per asset per model group. '
            'OOS predictions bundled for Ridge Stacker retraining without data leakage.'
        ),
    }
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2, default=str)
    print(f"\n  [Registry] model_registry.json written -> {os.path.abspath(registry_path)}")
    print(f"  [Registry] Total windows saved: {sum(1 for k in registry if not k.startswith('_'))}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train Dual-LSTM Models (Phase 7: Model A & B)')
    parser.add_argument(
        'asset', type=str, nargs='?', default='all',
        help='Asset to train: gold | btc | spy | qqq | dia | all'
    )
    args = parser.parse_args()
    run_training(args.asset)
