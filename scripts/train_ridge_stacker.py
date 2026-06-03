"""
Ridge Dual-Head Meta-Learner (Stacker) — Upgraded Ensemble Layer
=================================================================
Perubahan dari versi sebelumnya (Ridge tunggal):

    SEBELUM: 1 model Ridge → meminimalkan MSE (RMSE) saja
    SEKARANG: 2 head secara bersamaan:
        Head 1 - Direction Head (LogisticRegressionCV)
            → Memprediksi ARAH (UP/DOWN), mengoptimasi Hit Ratio
            → Menggunakan class_weight='balanced' untuk menangani imbalance
        Head 2 - Magnitude Head (HuberRegressor)
            → Memprediksi BESAR % CHANGE, tahan terhadap outlier
            → Huber loss: tidak menghukum outlier berlebihan seperti MSE

    Output akhir digabungkan:
        final_signal = direction_prob * magnitude
        Positif  → prediksi naik (dengan keyakinan direction_prob)
        Negatif  → prediksi turun (dengan keyakinan 1-direction_prob)

    Keunggulan Dual-Head vs Ridge tunggal:
        - Tidak ada trade-off antara Hit Ratio vs RMSE
        - Direction head murni fokus ke akurasi arah
        - Magnitude head tidak "takut" LSTM yang sesekali outlier
        - Lebih robust di berbagai kondisi market

Usage:
    python scripts/train_ridge_stacker.py gold
    python scripts/train_ridge_stacker.py btc
    python scripts/train_ridge_stacker.py spy
    python scripts/train_ridge_stacker.py       ← train semua

Output:
    models/{asset}_stacker_direction.pkl   ← LogisticRegressionCV
    models/{asset}_stacker_magnitude.pkl   ← HuberRegressor
    models/{asset}_stacker_meta_scaler.pkl ← StandardScaler untuk meta-features
    models/{asset}_stacker_meta.json       ← koefisien & metrics
    reports/stacker_{asset}_backtest.json  ← perbandingan LSTM vs XGB vs Ensemble
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pickle

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'   # Force fully deterministic TF CPU ops
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Reproducibility ──────────────────────────────────────────────────────────
import random as _random
_random.seed(42)
import numpy as np
np.random.seed(42)

from sklearn.linear_model import LogisticRegressionCV, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import xgboost as xgb
from utils.config import ASSETS

HORIZON_DAYS = 7


def load_asset_data(asset_key: str, config: dict) -> pd.DataFrame:
    """
    Load data for asset from DuckDB, falling back to CSV.
    """
    table_name = f"{asset_key.lower()}_global_insights"
    try:
        from utils.data_store import MarketDataStore
        store = MarketDataStore()
        df = store.read_table(table_name, format='pandas')
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        df = df.sort_index()
        return df
    except Exception as e:
        print(f"  [Data] DuckDB read failed for '{table_name}' ({e}). Falling back to CSV.")
        df = pd.read_csv(config['data_file'], index_col=0, parse_dates=True)
        return df.sort_index()



# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Get LSTM predictions (requires train_lstm_pct.py to have been run)
# ─────────────────────────────────────────────────────────────────────────────
def get_lstm_pct_predictions(asset_key: str, df_test: pd.DataFrame) -> np.ndarray:
    """
    Load LSTM pct-change model and predict 7D % change for each row in df_test.
    Returns zeros if model not found (falls back gracefully).
    """
    config = ASSETS[asset_key]
    feat_scaler_path   = config['scaler_file']
    target_scaler_path = config['scaler_file'].replace('.pkl', '_target.pkl')
    model_path         = config['model_file']

    files_needed = [model_path, feat_scaler_path, target_scaler_path]
    if not all(os.path.exists(p) for p in files_needed):
        missing = [p for p in files_needed if not os.path.exists(p)]
        print(f"  [LSTM] Files missing: {missing}")
        print(f"  [LSTM] Run: python scripts/train_lstm_pct.py {asset_key}")
        print(f"  [LSTM] Using zeros as fallback.")
        return np.zeros(len(df_test))

    try:
        from tensorflow.keras.models import load_model as keras_load
        lstm_model = keras_load(model_path)
        with open(feat_scaler_path, 'rb') as f:
            feature_scaler = pickle.load(f)
        with open(target_scaler_path, 'rb') as f:
            target_scaler = pickle.load(f)
    except Exception as e:
        print(f"  [LSTM] Load failed: {e}. Using zeros.")
        return np.zeros(len(df_test))

    seq_len  = config.get('sequence_length', 60)
    features = [f for f in config['features'] if f in df_test.columns]

    # Load full dataset for building lookback windows
    df_full = load_asset_data(asset_key, config)
    df_full = df_full[[f for f in features if f in df_full.columns]].ffill().fillna(0)

    preds = []
    for date in df_test.index:
        try:
            pos = df_full.index.get_loc(date)
        except KeyError:
            preds.append(0.0)
            continue

        if pos < seq_len:
            preds.append(0.0)
            continue

        window = df_full.iloc[pos - seq_len: pos].values
        if window.shape[0] < seq_len:
            preds.append(0.0)
            continue

        try:
            w_scaled = feature_scaler.transform(window)
            X = w_scaled.reshape(1, seq_len, -1)
            p_scaled = lstm_model.predict(X, verbose=0)[0, 0]
            pct = target_scaler.inverse_transform([[p_scaled]])[0, 0]
            preds.append(float(pct))
        except Exception:
            preds.append(0.0)

    return np.array(preds)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Get XGBoost predictions
# ─────────────────────────────────────────────────────────────────────────────
def get_xgb_pct_predictions(asset_key: str, df_test: pd.DataFrame) -> np.ndarray:
    """
    Load XGBoost macro model and predict 7D % change.

    Auto-generates lagged features (e.g. CPI_MoM_lag3) on-the-fly if they are
    required by the saved model but missing from df_test. This ensures backward
    compatibility when the XGBoost model was trained with lag features that are
    not persisted in the CSV.
    """
    model_path   = f'models/{asset_key}_xgb_macro.json'
    scaler_path  = f'models/{asset_key}_xgb_scaler.pkl'
    feature_path = f'models/{asset_key}_xgb_features.json'

    if not all(os.path.exists(p) for p in [model_path, scaler_path, feature_path]):
        print(f"  [XGB] Model files missing for {asset_key}. Using zeros.")
        return np.zeros(len(df_test))

    model = xgb.XGBRegressor()
    model.load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(feature_path, 'r') as f:
        meta = json.load(f)

    required_features = meta['features']

    # ── Auto-generate lag features missing from df_test ──────────────────────
    # Detect pattern: <base_col>_lag<N>  (e.g. CPI_MoM_lag3, NFP_Change_lag6)
    import re
    df_work = df_test.copy()
    for feat in required_features:
        if feat not in df_work.columns:
            match = re.match(r'^(.+)_lag(\d+)$', feat)
            if match:
                base_col, lag_n = match.group(1), int(match.group(2))
                if base_col in df_work.columns:
                    df_work[feat] = df_work[base_col].shift(lag_n).fillna(0)
                    print(f"  [XGB] Auto-generated lag feature: {feat} from {base_col} (lag={lag_n})")
                else:
                    df_work[feat] = 0.0
                    print(f"  [XGB] Lag base col '{base_col}' not in data — filling {feat} with 0")
            else:
                df_work[feat] = 0.0
                print(f"  [XGB] Feature '{feat}' not in data — filling with 0")

    # Now all required features should be present
    missing_still = [f for f in required_features if f not in df_work.columns]
    if missing_still:
        print(f"  [XGB] WARNING: still missing {missing_still} — filling with 0")
        for f in missing_still:
            df_work[f] = 0.0

    X = df_work[required_features].fillna(0).values

    # Final safety check against scaler's expected feature count
    if hasattr(scaler, 'n_features_in_') and X.shape[1] != scaler.n_features_in_:
        print(f"  [XGB] Feature count mismatch: data={X.shape[1]}, scaler={scaler.n_features_in_}")
        print(f"  [XGB] Retrain XGBoost to fix permanently: python scripts/train_xgboost_macro.py {asset_key}")
        # Pad or truncate to match
        if X.shape[1] < scaler.n_features_in_:
            pad = np.zeros((X.shape[0], scaler.n_features_in_ - X.shape[1]))
            X   = np.hstack([X, pad])
        else:
            X = X[:, :scaler.n_features_in_]

    return model.predict(scaler.transform(X))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Train Dual-Head Stacker
# ─────────────────────────────────────────────────────────────────────────────
def train_dual_head_stacker(asset_key: str) -> dict:
    asset_key = asset_key.lower()
    print(f"\n{'='*60}")
    print(f" Dual-Head Stacker — {asset_key.upper()}")
    print(f"{'='*60}")

    if asset_key not in ASSETS:
        print(f"Error: Unknown asset '{asset_key}'")
        return {}

    config = ASSETS[asset_key]
    price_col = None
    for c in config['features']:
        if c in ['Gold', 'BTC', 'SPY', 'QQQ', 'DIA', 'AAPL', 'MSFT',
                 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'TSM']:
            price_col = c
            break
    if price_col is None:
        price_col = config['features'][0]

    # Load data, compute target
    df = load_asset_data(asset_key, config)
    df['_target'] = (df[price_col].shift(-HORIZON_DAYS) - df[price_col]) / df[price_col]
    df = df.dropna(subset=['_target'])

    # 80% train | 20% test — stacker is trained/evaluated on the test split
    split_idx = int(len(df) * 0.80)
    df_test   = df.iloc[split_idx:].copy()

    print(f"Stacker data: {len(df_test)} samples (20% unseen test period)")
    print(f"  {df_test.index[0].date()} to {df_test.index[-1].date()}")

    # ── Get base model predictions ──────────────────────────────────────────
    print("\nGenerating LSTM predictions...")
    lstm_preds = get_lstm_pct_predictions(asset_key, df_test)

    print("Generating XGBoost predictions...")
    xgb_preds  = get_xgb_pct_predictions(asset_key, df_test)

    # ── Build meta-feature matrix ────────────────────────────────────────────
    # Context features: regime indicators that help stacker decide WHEN to trust each model
    # Phase 3 features included so stacker learns regime-awareness (VIX fear, coupling, anomaly)
    ctx_features = ['VIX', 'GK_Vol_21d', 'Sentiment', 'Sentiment_Std',
                    'YieldCurve_10Y2Y', 'DXY',
                    # Phase 3 dynamic regime features
                    'vix_percentile_252d', 'return_zscore_90d',
                    'roll_corr_spy_90d', 'roll_corr_dxy_90d', 'roll_corr_qqq_90d']
    ctx_avail = [f for f in ctx_features if f in df_test.columns]

    meta_df = pd.DataFrame({
        'lstm_pred': lstm_preds,
        'xgb_pred':  xgb_preds,
    }, index=df_test.index)
    for f in ctx_avail:
        meta_df[f] = df_test[f].fillna(0).values

    y_pct = df_test['_target'].values

    # Direction labels: 1 = price goes UP, 0 = price goes DOWN
    y_dir = (y_pct > 0).astype(int)

    # ── 70% for stacker training, 30% for stacker evaluation ────────────────
    split2 = int(len(meta_df) * 0.70)
    X_tr, X_ev = meta_df.iloc[:split2].values,  meta_df.iloc[split2:].values
    y_pct_tr, y_pct_ev = y_pct[:split2], y_pct[split2:]
    y_dir_tr,  y_dir_ev  = y_dir[:split2],  y_dir[split2:]

    # ── StandardScaler for meta-features ────────────────────────────────────
    meta_scaler = StandardScaler()
    X_tr_sc = meta_scaler.fit_transform(X_tr)
    X_ev_sc = meta_scaler.transform(X_ev)

    # ════════════════════════════════════════════════════════════════════════
    #  HEAD 1: Direction Head — LogisticRegressionCV
    #  Optimizes for directional accuracy (hit ratio)
    #  class_weight='balanced': compensates if UP/DOWN class is imbalanced
    # ════════════════════════════════════════════════════════════════════════
    dir_head = LogisticRegressionCV(
        Cs=[0.01, 0.1, 1.0, 10.0, 100.0],
        cv=5,
        max_iter=500,
        class_weight='balanced',   # key: handles UP/DOWN imbalance
        scoring='accuracy',
        random_state=42,
    )
    dir_head.fit(X_tr_sc, y_dir_tr)

    # Probability of UP direction (class 1)
    dir_prob_train = dir_head.predict_proba(X_tr_sc)[:, 1]
    dir_prob_eval  = dir_head.predict_proba(X_ev_sc)[:, 1]

    dir_pred_train = (dir_prob_train > 0.5).astype(int)
    dir_pred_eval  = (dir_prob_eval  > 0.5).astype(int)

    hr_dir_train = accuracy_score(y_dir_tr, dir_pred_train) * 100
    hr_dir_eval  = accuracy_score(y_dir_ev,  dir_pred_eval) * 100

    # ════════════════════════════════════════════════════════════════════════
    #  HEAD 2: Magnitude Head — HuberRegressor
    #  Optimizes for % change magnitude, robust to outliers
    #  epsilon=1.35: outliers > 1.35 std get linear (not quadratic) penalty
    # ════════════════════════════════════════════════════════════════════════
    mag_head = HuberRegressor(
        epsilon=1.35,   # Standard Huber threshold
        alpha=0.001,    # L2 regularization
        max_iter=500,
    )
    mag_head.fit(X_tr_sc, y_pct_tr)

    mag_pred_train = mag_head.predict(X_tr_sc)
    mag_pred_eval  = mag_head.predict(X_ev_sc)
    rmse_mag_eval  = np.sqrt(mean_squared_error(y_pct_ev, mag_pred_eval))

    # ════════════════════════════════════════════════════════════════════════
    #  COMBINED OUTPUT: Direction × Magnitude
    #  Convert direction probability to signed signal:
    #    dir_prob > 0.5 → positive signal (bullish)
    #    dir_prob < 0.5 → negative signal (bearish)
    #  Multiply by magnitude for the final % change estimate
    # ════════════════════════════════════════════════════════════════════════
    def combine(dir_prob, mag_pred):
        # dir_signal: 1.0 at fully bullish, -1.0 at fully bearish
        dir_signal = (dir_prob - 0.5) * 2.0
        # Use magnitude's absolute value with direction from logistic
        return dir_signal * np.abs(mag_pred)

    combined_train = combine(dir_prob_train, mag_pred_train)
    combined_eval  = combine(dir_prob_eval,  mag_pred_eval)

    hr_combined_train = float((np.sign(combined_train) == np.sign(y_pct_tr)).mean()) * 100
    hr_combined_eval  = float((np.sign(combined_eval)  == np.sign(y_pct_ev)).mean()) * 100
    rmse_combined     = np.sqrt(mean_squared_error(y_pct_ev, combined_eval))

    # ── Baselines (individual models on eval period) ─────────────────────────
    hr_lstm_only = float((np.sign(X_ev[:, 0]) == np.sign(y_pct_ev)).mean()) * 100
    hr_xgb_only  = float((np.sign(X_ev[:, 1]) == np.sign(y_pct_ev)).mean()) * 100

    # ── Print results ────────────────────────────────────────────────────────
    feature_names = list(meta_df.columns)
    print(f"\n{'='*60}")
    print(f"  DUAL-HEAD ENSEMBLE — {asset_key.upper()} ({HORIZON_DAYS}D horizon)")
    print(f"{'='*60}")
    print(f"  LSTM only           Hit Ratio: {hr_lstm_only:.1f}%")
    print(f"  XGBoost only        Hit Ratio: {hr_xgb_only:.1f}%")
    print(f"  Direction Head      Hit Ratio: {hr_dir_eval:.1f}%  (optimized for direction)")
    print(f"  Magnitude Head      RMSE:      {rmse_mag_eval:.6f}  (optimized for magnitude)")
    print(f"  Combined (D×M)      Hit Ratio: {hr_combined_eval:.1f}%  | RMSE: {rmse_combined:.6f}")
    print(f"\n  Direction Head best_C: {dir_head.C_[0]:.4f}")
    print(f"\n  Direction Head coefficients (what drives UP/DOWN call):")
    for name, coef in zip(feature_names, dir_head.coef_[0]):
        bar_len = min(int(abs(coef) * 10), 30)
        bar = ('+' if coef > 0 else '-') * bar_len
        print(f"    {name:<22} {coef:+.4f}  {bar}")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    dir_path  = f'models/{asset_key}_stacker_direction.pkl'
    mag_path  = f'models/{asset_key}_stacker_magnitude.pkl'
    scl_path  = f'models/{asset_key}_stacker_meta_scaler.pkl'
    meta_path = f'models/{asset_key}_stacker_meta.json'
    rpt_path  = f'reports/stacker_{asset_key}_backtest.json'

    with open(dir_path, 'wb') as f: pickle.dump(dir_head, f)
    with open(mag_path, 'wb') as f: pickle.dump(mag_head, f)
    with open(scl_path, 'wb') as f: pickle.dump(meta_scaler, f)

    result = {
        'asset': asset_key,
        'horizon_days': HORIZON_DAYS,
        'feature_names': feature_names,
        'hit_ratio_lstm': hr_lstm_only,
        'hit_ratio_xgb': hr_xgb_only,
        'hit_ratio_direction_head': hr_dir_eval,
        'rmse_magnitude_head': rmse_mag_eval,
        'hit_ratio_combined': hr_combined_eval,
        'rmse_combined': rmse_combined,
        'best_C_direction': float(dir_head.C_[0]),
        'direction_coefs': dict(zip(feature_names, dir_head.coef_[0].tolist())),
    }

    with open(meta_path, 'w') as f: json.dump(result, f, indent=4)
    with open(rpt_path,  'w') as f: json.dump(result, f, indent=4)

    print(f"\n  Direction model: {dir_path}")
    print(f"  Magnitude model: {mag_path}")
    print(f"  Meta scaler:     {scl_path}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) > 1:
        train_dual_head_stacker(sys.argv[1])
    else:
        print("Training Dual-Head Stacker for Gold, BTC, and SPY...")
        for asset in ['gold', 'btc', 'spy']:
            try:
                train_dual_head_stacker(asset)
            except Exception as e:
                print(f"Error on {asset}: {e}")
                import traceback; traceback.print_exc()
