"""Quick audit test for forecast system"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Redirect output to file
log_file = open('test_forecast_results.txt', 'w', encoding='utf-8')
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()
sys.stdout = Tee(sys.stdout, log_file)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils.predictor import AssetPredictor

def test_asset(key, name):
    print(f"\n{'='*50}")
    print(f" {name} FORECAST TEST")
    print(f"{'='*50}")
    
    try:
        p = AssetPredictor(key)
        
        # Tomorrow
        t = p.predict_tomorrow()
        print(f"Current:   ${t['current']:,.2f}")
        print(f"Tomorrow:  ${t['predicted']:,.2f} ({t['pct_change']:+.2f}%)")
        
        # 1 Week
        w = p.predict_week()
        print(f"1 Week:    ${w['predicted']:,.2f} ({w['pct_change']:+.2f}%)")
        
        # Multi-range
        f = p.get_multi_range_forecast()
        current = f['Current']
        print(f"\nMulti-Range Forecast (from ${current:,.2f}):")
        for k, v in f.items():
            if k == 'Current':
                continue
            price = v['price']
            change = ((price - current) / current) * 100
            confidence = v['confidence']['label']
            print(f"  {k:10s}: ${price:,.2f} ({change:+.1f}%) [{confidence}]")
        
        # Sanity checks
        print(f"\n--- Sanity Checks ---")
        issues = []
        
        if abs(t['pct_change']) > 5:
            issues.append(f"WARNING: Tomorrow prediction {t['pct_change']:+.2f}% exceeds 5% daily limit")
        
        if abs(w['pct_change']) > 15:
            issues.append(f"WARNING: 1-week prediction {w['pct_change']:+.2f}% exceeds 15% weekly limit")
        
        for k, v in f.items():
            if k == 'Current':
                continue
            change_pct = ((v['price'] - current) / current) * 100
            if abs(change_pct) > 100:
                issues.append(f"WARNING: {k} prediction {change_pct:+.1f}% exceeds 100% change")
        
        for k, v in f.items():
            if k == 'Current':
                continue
            if v['price'] <= 0:
                issues.append(f"BUG: {k} prediction is ${v['price']:,.2f} (negative/zero!)")
        
        if issues:
            for issue in issues:
                print(f"  [!] {issue}")
        else:
            print(f"  [OK] All sanity checks passed!")
            
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()

test_asset('gold', 'GOLD')
test_asset('btc', 'BITCOIN')
test_asset('spy', 'SPY (S&P 500)')

print(f"\n{'='*50}")
print(" AUDIT COMPLETE")
print(f"{'='*50}")
