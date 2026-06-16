import os, sys
sys.path.append(os.path.dirname(__file__))
from utils.predictor import AssetPredictor

def test():
    p = AssetPredictor('btc')
    f = p.get_multi_range_forecast()
    print("BTC Phase 7 Uncertainty:", f['1 Week']['phase7_uncertainty'])

if __name__ == '__main__':
    test()
