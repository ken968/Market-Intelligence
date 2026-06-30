from utils.predictor import AssetPredictor
import json

try:
    p = AssetPredictor('btc')
    f = p.get_multi_range_forecast()
    
    print("=== BTC FORECAST VALIDATION ===")
    for k, v in f.items():
        if k == 'Current':
            print(f"Current: {v:,.2f}")
        elif k == 'ceo_context':
            ctx = v or {}
            print(f"CEO Fallback: {ctx.get('is_fallback', True)}, Headlines used: {ctx.get('headlines_used', 0)}")
        elif isinstance(v, dict) and 'price' in v:
            price = v.get('price', 0)
            label = v.get('confidence', {}).get('label', 'N/A')
            print(f"  {k}: price={price:,.2f}, confidence={label}")
    print()
    print("=== ALL HORIZONS PRESENT ===")
    for hz in ['1 Day', '1 Week', '2 Weeks', '1 Month', '3 Months']:
        present = hz in f
        has_series = isinstance(f.get(hz), dict) and len(f.get(hz, {}).get('series', [])) > 0
        print(f"  {hz}: present={present}, has_series={has_series}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
