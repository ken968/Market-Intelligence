"""
Ridge Meta-Learner (Stacker) — Final Ensemble Layer
=====================================================
Role in the ensemble architecture:
    Combines predictions from LSTM and XGBoost into a single, optimized
    final prediction. The Ridge learns WHEN to trust each model more.

    [LSTM % change]  ─┐
    [XGBoost % change]─┤──► Ridge Stacker ──► Final % change prediction
    [VIX]            ─┤     (learns optimal
    [GK_Vol_21d]     ─┤      weights per
    [Sentiment]       ┘      market regime)

Why Ridge (not another neural net or XGBoost)?
    - Ridge is a linear model: final_pred = w1*lstm + w2*xgb + w3*vix + ...
    - It's interpretable: you can print the coefficients and understand
      "in this data, LSTM was weighted 0.6x and XGBoost 0.4x"
    - It's robust: linear combination of models reduces variance (ensemble effect)
    - It won't overfit on 571 test samples (unlike a neural net would)
    - L2 regularization (alpha) prevents any single model from dominating

Training strategy:
    - Uses the 20% TEST SET from LSTM and XGBoost backtests as Ridge training data
    - Rationale: both models trained on 80%, so their predictions on the
      remaining 20% are "out-of-sample" → no data leakage
    - Ridge trains on these out-of-sample predictions → generalizes to future data

Usage:
    python scripts/train_ridge_stacker.py gold
    python scripts/train_ridge_stacker.py btc
    python scripts/train_ridge_stacker.py spy

Output:
    models/{asset}_ridge_stacker.pkl    <- stacker model
    models/{asset}_ridge_meta.json      <- coefficients & feature names
    reports/ridge_{asset}_backtest.json <- performance metrics
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pickle

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import xgboost as xgb
from utils.config import ASSETS
from utils.predictor import AssetPredictor

HORIZON_DAYS = 7


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Get LSTM predictions on test period
# ─────────────────────────────────────────────────────────────────────────────

def get_lstm_pct_change_predictions(asset_key: str, df_test: pd.DataFrame,
                                     price_col: str) -> np.ndarray:
    """
    Load the trained LSTM model and generate 7-day % change predictions
    for each row in df_test.

    Strategy: Predict price for t+7 using the last (sequence_length) days,
    then compute (predicted - actual_today) / actual_today.
    """
    predictor = AssetPredictor(asset_key)
    if not predictor.load_model():
        print(f"  Warning: LSTM model for {asset_key} could not be loaded. Using zeros.")
        return np.zeros(len(df_test))

    config = ASSETS[asset_key]
    seq_len = config.get('sequence_length', 60)
    features = [f for f in config['features'] if f in df_test.columns]

    # We need the full dataframe (train + test) for lookback sequences
    full_data_file = config['data_file']
    df_full = pd.read_csv(full_data_file, index_col=0, parse_dates=True).sort_index()
    df_full = df_full[[f for f in features if f in df_full.columns]].ffill().fillna(0)

    predictions = []
    test_indices = df_test.index

    for i, date in enumerate(test_indices):
        try:
            pos = df_full.index.get_loc(date)
        except KeyError:
            predictions.append(0.0)
            continue

        if pos < seq_len:
            predictions.append(0.0)
            continue

        # Build input sequence: last seq_len rows before this date
        window = df_full.iloc[pos - seq_len: pos]
        if len(window) < seq_len:
            predictions.append(0.0)
            continue

        # Use predictor's scaler to normalize
        try:
            data_scaled = predictor.scaler.transform(window.values)
            X = data_scaled.reshape(1, seq_len, -1)
            pred_scaled = predictor.model.predict(X, verbose=0)[0, 0]

            # Inverse transform: predict_scaled → price
            inv = np.zeros((1, len(features)))
            inv[0, 0] = pred_scaled
            pred_price = predictor.scaler.inverse_transform(inv)[0, 0]

            current_price = float(df_full.iloc[pos][price_col])
            pct_change = (pred_price - current_price) / current_price if current_price != 0 else 0.0
            predictions.append(pct_change)

        except Exception:
            predictions.append(0.0)

    return np.array(predictions)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Get XGBoost predictions on test period
# ─────────────────────────────────────────────────────────────────────────────

def get_xgb_pct_change_predictions(asset_key: str, df_test: pd.DataFrame) -> np.ndarray:
    """
    Load the trained XGBoost macro model and generate 7-day % change predictions.
    """
    model_path = f'models/{asset_key}_xgb_macro.json'
    scaler_path = f'models/{asset_key}_xgb_scaler.pkl'
    feature_path = f'models/{asset_key}_xgb_features.json'

    if not all(os.path.exists(p) for p in [model_path, scaler_path, feature_path]):
        print(f"  Warning: XGBoost model files for {asset_key} not found. Using zeros.")
        return np.zeros(len(df_test))

    model = xgb.XGBRegressor()
    model.load_model(model_path)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    with open(feature_path, 'r') as f:
        meta = json.load(f)

    features = [f for f in meta['features'] if f in df_test.columns]
    X = df_test[features].fillna(0).values
    X_scaled = scaler.transform(X)

    return model.predict(X_scaled)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Build Ridge training matrix and train
# ─────────────────────────────────────────────────────────────────────────────

def train_ridge_stacker(asset_key: str):
    asset_key = asset_key.lower()
    print(f"\n{'='*60}")
    print(f" Ridge Meta-Learner — {asset_key.upper()}")
    print(f"{'='*60}")

    if asset_key not in ASSETS:
        print(f"Error: Unknown asset {asset_key}")
        return

    config = ASSETS[asset_key]
    data_file = config['data_file']
    price_col = [c for c in config['features'] if c in
                 ['Gold', 'BTC', 'SPY', 'QQQ', 'DIA', 'AAPL', 'MSFT',
                  'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'TSM']][0]

    df = pd.read_csv(data_file, index_col=0, parse_dates=True).sort_index()

    # Compute 7-day forward % change (same target as XGBoost)
    df['target_pct_7d'] = (
        df[price_col].shift(-HORIZON_DAYS) - df[price_col]
    ) / df[price_col]

    # 80/20 split — Ridge trains on predictions from the 20% test portion
    split_idx = int(len(df) * 0.80)
    df_test = df.iloc[split_idx:].dropna(subset=['target_pct_7d'])

    print(f"Ridge training set: {len(df_test)} samples (the 20% unseen test period)")
    print(f"Date range: {df_test.index[0].date()} to {df_test.index[-1].date()}")

    # --- Get predictions from both base models ---
    print("\nGenerating LSTM predictions on test period...")
    lstm_preds = get_lstm_pct_change_predictions(asset_key, df_test, price_col)

    print("Generating XGBoost predictions on test period...")
    xgb_preds = get_xgb_pct_change_predictions(asset_key, df_test)

    # --- Context features for regime-awareness ---
    ctx_features = ['VIX', 'GK_Vol_21d', 'Sentiment', 'Sentiment_Std',
                    'YieldCurve_10Y2Y', 'DXY']
    ctx_available = [f for f in ctx_features if f in df_test.columns]

    # Build meta-feature matrix
    meta_X = pd.DataFrame({
        'lstm_pred': lstm_preds,
        'xgb_pred': xgb_preds,
    }, index=df_test.index)

    for feat in ctx_available:
        meta_X[feat] = df_test[feat].fillna(0).values

    y = df_test['target_pct_7d'].values

    # --- Further 70/30 split within the test set for Ridge eval ---
    # (70% to train Ridge, 30% to evaluate the full ensemble)
    ridge_split = int(len(meta_X) * 0.70)
    X_ridge_train = meta_X.iloc[:ridge_split].values
    y_ridge_train = y[:ridge_split]
    X_ridge_test  = meta_X.iloc[ridge_split:].values
    y_ridge_test  = y[ridge_split:]

    print(f"\nRidge train: {len(X_ridge_train)} | Ridge eval: {len(X_ridge_test)}")

    # StandardScaler for Ridge input
    meta_scaler = StandardScaler()
    X_train_scaled = meta_scaler.fit_transform(X_ridge_train)
    X_test_scaled  = meta_scaler.transform(X_ridge_test)

    # RidgeCV: auto-selects best alpha from the list via cross-validation
    # alpha controls L2 regularization strength
    ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
    ridge.fit(X_train_scaled, y_ridge_train)

    # --- Evaluation ---
    y_pred_train = ridge.predict(X_train_scaled)
    y_pred_test  = ridge.predict(X_test_scaled)

    def hit_ratio(preds, actuals):
        return float((np.sign(preds) == np.sign(actuals)).mean()) * 100.0

    hr_train = hit_ratio(y_pred_train, y_ridge_train)
    hr_test  = hit_ratio(y_pred_test,  y_ridge_test)
    rmse_test = np.sqrt(mean_squared_error(y_ridge_test, y_pred_test))

    # LSTM-only and XGBoost-only baselines on same eval period
    hr_lstm_only = hit_ratio(X_ridge_test[:, 0], y_ridge_test)
    hr_xgb_only  = hit_ratio(X_ridge_test[:, 1], y_ridge_test)

    print(f"\n{'='*60}")
    print(f" ENSEMBLE COMPARISON ({asset_key.upper()}, {HORIZON_DAYS}D horizon)")
    print(f"{'='*60}")
    print(f"  LSTM only       Hit Ratio: {hr_lstm_only:.1f}%")
    print(f"  XGBoost only    Hit Ratio: {hr_xgb_only:.1f}%")
    print(f"  Ridge Stacker   Hit Ratio: {hr_test:.1f}%  | RMSE: {rmse_test:.6f}")
    print(f"  Best alpha: {ridge.alpha_}")

    # Print coefficients
    feature_names = list(meta_X.columns)
    print(f"\n  Ridge Coefficients (model weights):")
    for name, coef in zip(feature_names, ridge.coef_):
        direction = "+" if coef > 0 else "-"
        bar = '|' * min(int(abs(coef) * 200), 40)
        print(f"    {name:<22} {coef:+.4f}  {bar}")

    # --- Save ---
    os.makedirs('models', exist_ok=True)
    stacker_path = f'models/{asset_key}_ridge_stacker.pkl'
    scaler_path  = f'models/{asset_key}_ridge_meta_scaler.pkl'

    with open(stacker_path, 'wb') as f:
        pickle.dump(ridge, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(meta_scaler, f)

    meta_info = {
        'asset': asset_key,
        'feature_names': feature_names,
        'coefficients': dict(zip(feature_names, ridge.coef_.tolist())),
        'intercept': float(ridge.intercept_),
        'best_alpha': float(ridge.alpha_),
        'horizon_days': HORIZON_DAYS,
        'hit_ratio_lstm': hr_lstm_only,
        'hit_ratio_xgb': hr_xgb_only,
        'hit_ratio_ensemble': hr_test,
        'rmse_test': rmse_test,
    }
    with open(f'models/{asset_key}_ridge_meta.json', 'w') as f:
        json.dump(meta_info, f, indent=4)

    os.makedirs('reports', exist_ok=True)
    with open(f'reports/ridge_{asset_key}_backtest.json', 'w') as f:
        json.dump(meta_info, f, indent=4)

    print(f"\n  Stacker saved: {stacker_path}")
    print(f"  Meta info:     models/{asset_key}_ridge_meta.json")
    return meta_info


if __name__ == '__main__':
    if len(sys.argv) > 1:
        train_ridge_stacker(sys.argv[1])
    else:
        print("Training Ridge Stacker for Gold, BTC, and SPY...")
        for asset in ['gold', 'btc', 'spy']:
            try:
                train_ridge_stacker(asset)
            except Exception as e:
                print(f"Error on {asset}: {e}")
                import traceback; traceback.print_exc()
