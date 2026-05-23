"""
Model Health Monitor
Analyzes the counterfactual_log.jsonl to calculate hit ratios for recent forecasts
and determine if models need retraining.
"""

import os
import json
import argparse
from datetime import datetime

LOG_PATH = 'data/counterfactual_log.jsonl'
HEALTH_OUTPUT = 'data/model_health.json'

def evaluate_health():
    if not os.path.exists(LOG_PATH):
        print(f"Log file not found at {LOG_PATH}")
        return {}

    records_by_asset = {}
    
    with open(LOG_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line)
                # We evaluate based on the 7-day horizon forecasts that have been resolved
                if r.get('actual_price') is not None and r.get('steps') == 7:
                    asset = r['asset']
                    if asset not in records_by_asset:
                        records_by_asset[asset] = []
                    records_by_asset[asset].append(r)
            except Exception:
                pass

    health_status = {}

    for asset, records in records_by_asset.items():
        # Sort by forecast date, get last 20
        records = sorted(records, key=lambda x: x['forecast_date'])
        recent = records[-20:]
        
        if len(recent) < 5:
            health_status[asset] = {
                'status': 'INITIALIZING',
                'hit_ratio': 0.0,
                'resolved_count': len(recent),
                'message': 'Need at least 5 resolved 7-day forecasts to evaluate health.'
            }
            continue

        hits = sum(1 for r in recent if r.get('contextual_hit') is True)
        hit_ratio = hits / len(recent)
        
        status = 'HEALTHY'
        if hit_ratio < 0.35:
            status = 'DEGRADED'
        elif hit_ratio < 0.40:
            status = 'WARNING'

        health_status[asset] = {
            'status': status,
            'hit_ratio': round(hit_ratio, 3),
            'resolved_count': len(recent),
            'message': f'Hit ratio is {hit_ratio:.1%} over the last {len(recent)} resolved 7-day forecasts.'
        }

    # Save to JSON
    os.makedirs('data', exist_ok=True)
    with open(HEALTH_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump({
            'last_checked': datetime.now().isoformat(),
            'assets': health_status
        }, f, indent=4)

    return health_status

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--auto-retrain', action='store_true', help='Automatically trigger retraining for DEGRADED assets')
    args = parser.parse_args()

    print("Running Model Health Monitor...")
    status = evaluate_health()
    
    if not status:
        print("No resolved 7-day forecasts found. Run some forecasts and wait for their target dates.")
    
    for asset, data in status.items():
        print(f"[{asset.upper()}] Status: {data['status']} | Hit Ratio: {data['hit_ratio']:.1%} ({data['resolved_count']} records)")
        
        if args.auto_retrain and data['status'] == 'DEGRADED':
            print(f"  -> AUTO-RETRAIN triggered for {asset.upper()}!")
            import subprocess
            subprocess.run(['.venv\\Scripts\\python.exe', 'scripts/train_lstm_pct.py', asset])
            subprocess.run(['.venv\\Scripts\\python.exe', 'scripts/train_ridge_stacker.py', asset])
