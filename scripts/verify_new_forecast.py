import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.predictor import get_forecast_dataframe, batch_predict_tomorrow

def verify_predictions():
    print("="*60)
    print("VERIFICATION OF NEW FORECAST MODELS")
    print("="*60)
    
    test_assets = ['gold', 'btc', 'spy']
    
    # 1. Test 1-day batch prediction
    print("\n--- Testing 1-Day Forecast ---")
    results = batch_predict_tomorrow(test_assets)
    for asset, result in results.items():
        if 'error' in result:
            print(f"[ERROR] {asset.upper()}: {result['error']}")
        else:
            print(f"[OK] {asset.upper()}: Current={result['current']:.2f}, Tomorrow={result['predicted']:.2f}")

    # 2. Test Multi-range forecast for BTC (most changed features)
    print("\n--- Testing Multi-Range Forecast (BTC) ---")
    try:
        df = get_forecast_dataframe('btc')
        print(df.to_string(index=False))
        print("\n[OK] Multi-range forecast generated successfully.")
    except Exception as e:
        print(f"[ERROR] BTC Multi-range forecast failed: {e}")

if __name__ == "__main__":
    verify_predictions()
