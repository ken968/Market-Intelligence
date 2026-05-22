import sys
import os
sys.path.append(os.path.abspath('.'))
from utils.predictor import AssetPredictor
import traceback

for asset in ['gold', 'btc', 'spy']:
    print(f"\n--- Testing Ensemble Forecast for {asset.upper()} ---")
    p = AssetPredictor(asset)
    try:
        res = p.ensemble_forecast()
        print("Result:", res)
    except Exception as e:
        print("Outer exception caught:")
        traceback.print_exc()
