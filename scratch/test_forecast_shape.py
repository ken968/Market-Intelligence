import sys
import os

sys.path.append(r"d:\Market-Intelligence")
from utils.predictor import AssetPredictor

def main():
    p = AssetPredictor('btc')
    forecasts = p.get_multi_range_forecast()
    
    for key in ['1 Week', '1 Month']:
        print(f"\n--- {key} ---")
        data = forecasts[key]
        print(f"Price: {data['price']}")
        print(f"Baseline Price: {data['baseline_price']}")
        print(f"Contextual Series (first 5): {data['series'][:5]} ... (last 5): {data['series'][-5:]}")
        print(f"Baseline Series (first 5): {data['baseline_series'][:5]} ... (last 5): {data['baseline_series'][-5:]}")
        
        # Check if the series is exactly a straight line (i.e. constant differences)
        diffs = [data['series'][i+1] - data['series'][i] for i in range(len(data['series'])-1)]
        is_straight = all(abs(diffs[i] - diffs[0]) < 1e-6 for i in range(len(diffs)))
        print(f"Contextual is straight line: {is_straight}")

if __name__ == '__main__':
    main()
