from utils.predictor import batch_predict_week

assets = ['gold', 'btc', 'spy']
results = batch_predict_week(assets)
for key, r in results.items():
    if 'error' not in r:
        lstm = r.get('lstm_signal', 0) * 100
        xgb  = r.get('xgb_signal', 0) * 100
        conf = r.get('direction_prob', 0.5) * 100
        pct  = r.get('pct_change', 0)
        ens  = r.get('has_ensemble', False)
        print(f"{key.upper():6s} | Predicted {pct:+.2f}% | LSTM {lstm:+.2f}% | XGB {xgb:+.2f}% | Conf {conf:.0f}% | Ensemble: {ens}")
    else:
        print(f"{key.upper():6s} | ERROR: {r['error']}")
