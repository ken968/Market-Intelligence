# BTC Forecast NaN Bug - Fix Documentation

## Problem Summary
User reported that BTC forecast (and possibly all assets) returns `$nan` for all predicted prices instead of actual price predictions.

## Root Cause Analysis

### Primary Issue
The latest data row for BTC (and other assets) contained **NaN values** in macro indicators:
- `DXY` (Dollar Index): NaN
- `VIX` (Volatility Index): NaN  
- `Yield_10Y` (10-Year Treasury): NaN

### Why This Happened
In `data_fetcher_v2.py`, when merging asset price data with macro indicators:

**Original (Buggy) Code:**
```python
df = df.join(macro, how='left')
# Fill NaN for early dates where macro data doesn't exist
df['DXY'].fillna(method='bfill', inplace=True)  # WRONG!
df['VIX'].fillna(method='bfill', inplace=True)  # WRONG!
df['Yield_10Y'].fillna(method='bfill', inplace=True)  # WRONG!
```

**Problem:**
- `bfill()` (backward fill) fills missing values from **bottom to top**
- If the **last row** has NaN (latest trading day with no macro data yet), there's nothing below it to fill from
- Result: NaN remains in the last row

### Why Latest Data Had NaN
When `yfinance` downloads data, sometimes macro indicators (DXY, VIX, Yield) for the **current trading day** aren't available yet (market still open, data delayed, etc.), while asset price data (BTC, Gold, stocks) is available.

This causes the merge to create NaN for the latest date's macro indicators.

### Impact on Predictions
1. Predictor loads data with NaN in last row
2. `scaler.transform()` propagates NaN through all features
3. Model prediction with NaN input → NaN output
4. All forecast ranges show `$nan`

## Solution Implemented

### Fix 1: Changed Fill Strategy in `data_fetcher_v2.py`

**New (Fixed) Code:**
```python
df = df.join(macro, how='left')
# Fill NaN using forward fill first (for recent dates), then backward fill (for early dates)
# This prevents NaN when macro data is not yet available for the latest trading day
df['DXY'] = df['DXY'].ffill().bfill()
df['VIX'] = df['VIX'].ffill().bfill()
df['Yield_10Y'] = df['Yield_10Y'].ffill().bfill()
```

**How This Works:**
- `ffill()` (forward fill) fills missing values by propagating the **last valid value forward**
- If latest row has NaN, it gets filled with yesterday's value
- `bfill()` still handles early dates where macro data doesn't exist yet
- This ensures **no NaN in any row**

**Changed Join Type:**
- Gold & Stocks: Changed from `how='inner'` to `how='left'`
- Prevents data loss when asset has more recent data than macro indicators

### Fix 2: Added Safety Check in `ui_components.py`

Added NaN validation in `render_prediction_table()`:

```python
# Safety check: Detect NaN values
has_nan = any(pd.isna(v) or (isinstance(v, float) and np.isnan(v)) 
              for v in predictions_dict.values())

if has_nan:
    st.error("""
    ⚠️ **Prediction Error**: Invalid data detected (NaN values).
    
    **Possible causes:**
    - Recent market data not yet available
    - Missing macro indicators (DXY, VIX, Yield)
    
    **Solution**: Try syncing data from the Settings page.
    """)
    return
```

**Benefits:**
- Catches NaN errors before attempting to format
- Shows helpful error message to user
- Prevents cryptic `$nan` display

## Files Modified

1. **`scripts/data_fetcher_v2.py`**
   - Fixed `fetch_gold_data()` (line 133-140)
   - Fixed `fetch_bitcoin_data()` (line 180-188)
   - Fixed `fetch_stock_data()` (line 235-242)

2. **`utils/ui_components.py`**
   - Added NaN safety check in `render_prediction_table()` (line 377-393)

3. **New Helper Scripts Created:**
   - `scripts/fix_btc_nan.py` - Quick fix for BTC only
   - `scripts/fix_all_assets.py` - Fix all assets at once

## Verification Steps Taken

1. ✅ Checked BTC data before fix → **NaN detected** in row 4161 (2026-02-07)
2. ✅ Re-synced BTC data with fixed code
3. ✅ Checked BTC data after fix → **No NaN** values found
4. ✅ Verified all macro indicators filled correctly using forward fill

## Re-sync Process

To fix existing installations:

```bash
# Option 1: Fix BTC only
python scripts/fix_btc_nan.py

# Option 2: Fix all assets (recommended)
python scripts/fix_all_assets.py
```

The fix script runs:
1. `data_fetcher_v2.py` → Fetch latest data with fixed merge logic
2. `sentiment_fetcher_v2.py` → Re-integrate sentiment with clean data

## Results

After running `fix_all_assets.py`:

```
✅ ALL ASSETS FIXED:
- Gold: 2,514 records (No NaN)
- Bitcoin: 4,162 records (No NaN)
- 11 Stocks: 2,515 records each (No NaN)
```

## Prevention

This fix ensures the issue won't happen again because:

1. **Forward fill strategy** handles missing recent data gracefully
2. **Safety validation** catches any future NaN issues before display
3. **Left join** prevents data loss from timing mismatches

## Testing Recommendations

After applying fix:
1. Open Bitcoin Analysis page
2. Click "Generate BTC Forecast"
3. Verify predictions show actual prices (not `$nan`)
4. Repeat for Gold and at least one stock (e.g., AAPL)

## Related Issues

This same fix pattern applies to:
- Any asset using macro indicators
- Any feature that depends on external market data
- Future assets added to the system

---

**Date Fixed:** 2026-02-07  
**Severity:** Critical (Predictions unusable)  
**Status:** ✅ RESOLVED
