"""
XGBoost Macro Model — Alpha Engine Component
============================================
Role in the ensemble:
    Reads tabular FRED macro indicators and outputs % price change prediction.
    Designed to capture NON-LINEAR interactions between macro variables that
    LSTM misses because LSTM is optimized for sequential temporal patterns,
    not tabular cross-feature relationships.

Algorithm:
    XGBoost = eXtreme Gradient Boosting
    Base learner: CART (Classification and Regression Trees)
    Boosting: sequential trees, each correcting residuals of the previous
    Split criterion: gradient/hessian-based gain (NOT Information Gain or Gain Ratio)

Usage:
    python scripts/train_xgboost_macro.py gold
    python scripts/train_xgboost_macro.py btc
    python scripts/train_xgboost_macro.py spy

Output:
    models/{asset}_xgb_macro.json       <- trained model
    reports/xgb_{asset}_backtest.json   <- performance metrics
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pickle

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import xgboost as xgb
except ImportError:
    print("Error: xgboost not installed. Run: pip install xgboost")
    sys.exit(1)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from utils.config import ASSETS

# ─────────────────────────────────────────────────────────────────────────────
# MACRO FEATURES — These are the only features fed to XGBoost.
# XGBoost reads macro data. LSTM reads price sequences.
# They are separate models with different inputs, same output unit (% change).
# ─────────────────────────────────────────────────────────────────────────────
MACRO_FEATURES = [
    # Market macro (daily)
    'DXY', 'VIX', 'Yield_10Y', 'Oil_Price',

    # FRED monthly indicators
    'CPI_MoM', 'PPI_MoM', 'PCE_MoM', 'NFP_Change',
    'M2_MoM', 'M2_YoY', 'YieldCurve_10Y2Y',
    'Yield_10Y_Rate', 'Breakeven_5Y5Y',
    'M2_Liquidity_Spike', 'MacroEvent_Flag',

    # Lagged features — note: add_lagged_macro_features generates these names
    'CPI_MoM_lag3', 'CPI_MoM_lag6', 'NFP_Change_lag3',

    # Sentiment
    'Sentiment', 'Sentiment_Std',

    # GK Volatility regime
    'GK_Vol_21d',
]

# Asset-specific hyperparameter profiles.
# Gold and SPY have ~2800 samples and noisier macro-to-price relationship.
# BTC is more macro-sensitive and has more data, so tighter regularization is fine.
ASSET_PARAMS = {
    'gold': {
        'n_estimators': 300,
        'max_depth': 5,          # Deeper trees allowed — more expressiveness
        'learning_rate': 0.05,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'reg_alpha': 0.05,       # Lighter L1
        'reg_lambda': 0.5,       # Lighter L2
        'min_child_weight': 3,   # Allow splits with fewer samples
        'gamma': 0.01,           # Very small gain threshold
    },
    'btc': {
        'n_estimators': 500,
        'max_depth': 4,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'min_child_weight': 5,
        'gamma': 0.1,
    },
    'default': {  # For stocks (SPY, QQQ, etc.)
        'n_estimators': 300,
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'reg_alpha': 0.05,
        'reg_lambda': 0.5,
        'min_child_weight': 3,
        'gamma': 0.01,
    },
}

# Target horizon: 7-day forward % change
HORIZON_DAYS = 7

def load_and_prepare(asset_key: str):

    """
    Load CSV, compute target (7-day forward % change), select macro features.

    Target definition:
        y = (price[t+7] - price[t]) / price[t]
    This is % change from today to 7 days from now.
    All models in the ensemble predict the SAME unit for consistency.
    """
    if asset_key not in ASSETS:
        raise ValueError(f"Unknown asset: {asset_key}. Available: {list(ASSETS.keys())}")

    config = ASSETS[asset_key]
    data_file = config['data_file']
    price_col = [c for c in config['features'] if c in
                 ['Gold', 'BTC', 'SPY', 'QQQ', 'DIA', 'AAPL', 'MSFT',
                  'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'TSM']][0]

    print(f"Loading data from: {data_file}")
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    df = df.sort_index()

    # Compute 7-day forward % change (the target)
    df['target_pct_change'] = (
        df[price_col].shift(-HORIZON_DAYS) - df[price_col]
    ) / df[price_col]

    # Select available macro features
    available = [f for f in MACRO_FEATURES if f in df.columns]
    missing = [f for f in MACRO_FEATURES if f not in df.columns]
    if missing:
        print(f"  Warning: Missing features (will skip): {missing}")

    # Drop rows where target is NaN (last HORIZON_DAYS rows)
    df = df[available + ['target_pct_change']].dropna()

    print(f"  Dataset: {len(df)} samples | {len(available)} macro features")
    print(f"  Target: {HORIZON_DAYS}-day forward % change")
    print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")

    return df, available, price_col


def get_hit_ratio(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Directional accuracy: % of predictions with correct sign."""
    if len(y_pred) < 2:
        return 0.0
    correct = np.sign(y_pred) == np.sign(y_true)
    return float(correct.mean()) * 100.0


def train_xgboost_macro(asset_key: str):
    asset_key = asset_key.lower()
    print(f"\n{'='*60}")
    print(f" XGBoost Macro Model — {asset_key.upper()}")
    print(f"{'='*60}")

    df, features, price_col = load_and_prepare(asset_key)

    X = df[features].values
    y = df['target_pct_change'].values

    # 80/20 chronological split (NOT random — time series order must be preserved)
    split_idx = int(len(X) * 0.80)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\nTrain: {len(X_train)} samples | Test: {len(X_test)} samples")

    # Standardize features (XGBoost doesn't require this, but improves stability
    # when features have very different scales, e.g., NFP=50000 vs M2_MoM=0.01)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Asset-adaptive hyperparameters (see ASSET_PARAMS at top of file)
    params = ASSET_PARAMS.get(asset_key, ASSET_PARAMS['default'])
    model = xgb.XGBRegressor(
        **params,
        objective='reg:squarederror',
        eval_metric='rmse',
        early_stopping_rounds=40,
        random_state=42,
        verbosity=0
    )

    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )

    # Evaluate
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test  = model.predict(X_test_scaled)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test  = np.sqrt(mean_squared_error(y_test, y_pred_test))
    hit_train  = get_hit_ratio(y_pred_train, y_train)
    hit_test   = get_hit_ratio(y_pred_test, y_test)

    print(f"\n{'='*60}")
    print(f" RESULTS — XGBoost Macro ({asset_key.upper()}, {HORIZON_DAYS}D target)")
    print(f"{'='*60}")
    print(f"  Train Hit Ratio: {hit_train:.1f}% | RMSE: {rmse_train:.6f}")
    print(f"  Test  Hit Ratio: {hit_test:.1f}%  | RMSE: {rmse_test:.6f}")
    print(f"  Best iteration:  {model.best_iteration}")

    # Feature importance (top 10)
    importance = model.feature_importances_
    feat_imp = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 10 Most Important Macro Features:")
    for feat, imp in feat_imp[:10]:
        bar = '|' * int(imp * 200)
        print(f"    {feat:<25} {imp:.4f} {bar}")

    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    model_path  = f'models/{asset_key}_xgb_macro.json'
    scaler_path = f'models/{asset_key}_xgb_scaler.pkl'

    model.save_model(model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Save feature list (critical for inference — must match training order)
    feature_path = f'models/{asset_key}_xgb_features.json'
    with open(feature_path, 'w') as f:
        json.dump({'features': features, 'horizon_days': HORIZON_DAYS}, f, indent=2)

    # Save backtest metrics
    os.makedirs('reports', exist_ok=True)
    metrics = {
        'asset': asset_key,
        'model_type': 'xgboost_macro',
        'horizon_days': HORIZON_DAYS,
        'hit_ratio_train': hit_train,
        'hit_ratio_test': hit_test,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'n_features': len(features),
        'best_iteration': model.best_iteration,
        'top_features': [{'feature': f, 'importance': float(i)} for f, i in feat_imp[:10]]
    }
    with open(f'reports/xgb_{asset_key}_backtest.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"\n  Model saved: {model_path}")
    print(f"  Metrics saved: reports/xgb_{asset_key}_backtest.json")
    return metrics


if __name__ == '__main__':
    if len(sys.argv) > 1:
        result = train_xgboost_macro(sys.argv[1])
    else:
        # Train all major assets
        print("No asset specified. Training Gold, BTC, and SPY...")
        for asset in ['gold', 'btc', 'spy']:
            try:
                train_xgboost_macro(asset)
            except Exception as e:
                print(f"Error training {asset}: {e}")
