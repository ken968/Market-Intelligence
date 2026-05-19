"""
Inject lagged macro features into all existing *_global_insights.csv files.

This is a one-time migration script that adds:
  CPI_MoM_lag3, CPI_MoM_lag6
  PPI_MoM_lag3, PPI_MoM_lag6
  NFP_Change_lag3, NFP_Change_lag6
  M2_MoM_lag3,   M2_MoM_lag6
  YieldCurve_10Y2Y_lag3, YieldCurve_10Y2Y_lag6
  ... (all MONTHLY_INDICATORS)

Run once after updating this script, then re-train XGBoost and Stacker.

Usage:
    py scripts/inject_lag_features.py
"""

import os
import sys
import glob
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feature_engineering import add_lagged_macro_features, MONTHLY_INDICATORS

DATA_DIR = "data"
PATTERN  = os.path.join(DATA_DIR, "*_global_insights.csv")


def inject_lags(filepath: str, lags_months: list = [3, 6]) -> None:
    """Add lagged columns to a CSV if not already present."""
    df = pd.read_csv(filepath)

    # Check which lag columns are already present
    expected_lags = []
    for col in MONTHLY_INDICATORS:
        if col in df.columns:
            for lag in lags_months:
                expected_lags.append(f"{col}_lag{lag}")

    already_present = [c for c in expected_lags if c in df.columns]
    missing_lags    = [c for c in expected_lags if c not in df.columns]

    if not missing_lags:
        print(f"  SKIP  {os.path.basename(filepath)} — all {len(already_present)} lag columns already present")
        return

    print(f"  ADDING {len(missing_lags)} lag columns to {os.path.basename(filepath)}")

    # Preserve Date column as-is; apply lags on numeric part
    df_with_lags = add_lagged_macro_features(df, lags_months=lags_months)

    # Only write back the newly added columns to avoid overwriting anything
    new_cols = [c for c in df_with_lags.columns if c not in df.columns]
    for col in new_cols:
        df[col] = df_with_lags[col].values

    df.to_csv(filepath, index=False)
    print(f"    Saved {len(new_cols)} new cols: {new_cols[:5]}{'...' if len(new_cols)>5 else ''}")


def main():
    csv_files = sorted(glob.glob(PATTERN))
    if not csv_files:
        print(f"No CSV files found matching: {PATTERN}")
        return

    print(f"Found {len(csv_files)} *_global_insights.csv files")
    print(f"Injecting lags: [3, 6] months per MONTHLY_INDICATOR\n")

    for fp in csv_files:
        try:
            inject_lags(fp)
        except Exception as e:
            print(f"  ERROR {os.path.basename(fp)}: {e}")

    print("\nDone. Now re-train XGBoost and Stacker:")
    print("  py scripts/train_xgboost_macro.py")
    print("  py scripts/train_ridge_stacker.py")


if __name__ == "__main__":
    main()
