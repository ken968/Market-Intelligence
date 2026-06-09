import re

file_path = 'd:/Market-Intelligence/utils/predictor_engine.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Delete recursive_forecast
content = re.sub(r'    def recursive_forecast\(.*?\n(    def ensemble_forecast)', r'\1', content, flags=re.DOTALL)

# Replace pct_chain_forecast
new_pct_chain = '''    def _get_lstm_ab_forecasts(self) -> dict:
        import joblib
        import pandas as pd
        import os
        from tensorflow.keras.models import load_model
        
        # Load Model A and B
        res = {1: 0.0, 7: 0.0, 14: 0.0, 30: 0.0, 90: 0.0}
        
        try:
            data_path = self.config['data_file']
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            features = self.config['features']
            seq_len = self.config.get('sequence_length', 90)
            raw_features = df[features].ffill().fillna(0).values[-seq_len:]
            
            # Model A
            ma_path = self.config['model_file'].replace('.keras', '_model_a.keras')
            sa_path = self.config['scaler_file'].replace('.pkl', '_model_a.pkl')
            if os.path.exists(ma_path) and os.path.exists(sa_path):
                model_a = load_model(ma_path)
                scalers = joblib.load(sa_path)
                scaled_x = scalers['feature_scaler'].transform(raw_features)
                pred_a = model_a.predict(scaled_x[np.newaxis, :, :], verbose=0)
                pred_a = scalers['target_scaler'].inverse_transform(pred_a)[0]
                res[1], res[7], res[14] = pred_a[0], pred_a[1], pred_a[2]
                
            # Model B
            mb_path = self.config['model_file'].replace('.keras', '_model_b.keras')
            sb_path = self.config['scaler_file'].replace('.pkl', '_model_b.pkl')
            if os.path.exists(mb_path) and os.path.exists(sb_path):
                model_b = load_model(mb_path)
                scalers = joblib.load(sb_path)
                scaled_x = scalers['feature_scaler'].transform(raw_features)
                pred_b = model_b.predict(scaled_x[np.newaxis, :, :], verbose=0)
                pred_b = scalers['target_scaler'].inverse_transform(pred_b)[0]
                res[30], res[90] = pred_b[0], pred_b[1]
                
        except Exception as e:
            print(f"Warning: Could not get Model A/B forecasts: {e}")
            
        return res

    def pct_chain_forecast(self, steps: int, ceo_drift_multiplier: float = 1.0, ensemble_7d_pct: float = None) -> list:
        current_price = self.data_handler.get_latest_price()
        
        # 1. Get raw predictions from Model A & B
        ab_preds = self._get_lstm_ab_forecasts()
        
        # 2. Get Ensemble 7D Anchor (3-Layer Causal Hierarchy)
        if ensemble_7d_pct is None:
            try:
                ens = self.ensemble_forecast()
                ensemble_7d_pct = ens.get('pct_change_7d', ab_preds[7])
            except:
                ensemble_7d_pct = ab_preds[7]
                
        # 3. Apply CEO Drift
        dm = max(0.85, min(1.15, ceo_drift_multiplier))
        adjusted_7d_pct = ensemble_7d_pct * dm
        
        # 4. Calculate Alpha Shift (Difference between Ensemble 7D and Raw LSTM 7D)
        alpha_shift = adjusted_7d_pct - ab_preds[7]
        
        # 5. Apply Alpha Shift to all anchor points to maintain the curve shape but shift it
        x_anchors = [0, 1, 7, 14, 30, 90]
        y_anchors = [
            0.0,
            ab_preds[1] + (alpha_shift * (1/7)),
            adjusted_7d_pct,
            ab_preds[14] + (alpha_shift * (14/7)),
            ab_preds[30] + (alpha_shift * (30/7)),
            ab_preds[90] + (alpha_shift * (90/7))
        ]
        
        # Filter anchors up to steps
        x_filtered = [x for x in x_anchors if x <= steps]
        y_filtered = y_anchors[:len(x_filtered)]
        
        # If steps is not in x_anchors, add it using linear extrapolation
        if steps not in x_filtered:
            x_filtered.append(steps)
            if len(y_filtered) >= 2:
                slope = (y_filtered[-1] - y_filtered[-2]) / (x_filtered[-2] - x_filtered[-3] + 1e-5)
                y_filtered.append(y_filtered[-1] + slope * (steps - x_filtered[-2]))
            else:
                y_filtered.append(y_filtered[-1])

        # 6. Spline Interpolation
        from scipy.interpolate import interp1d
        try:
            f_spline = interp1d(x_filtered, y_filtered, kind='quadratic', fill_value="extrapolate")
            daily_pcts = f_spline(np.arange(1, steps + 1))
        except:
            f_spline = interp1d(x_filtered, y_filtered, kind='linear', fill_value="extrapolate")
            daily_pcts = f_spline(np.arange(1, steps + 1))
            
        prices = [current_price * (1.0 + float(pct)) for pct in daily_pcts]
        return prices'''

content = re.sub(r'    def pct_chain_forecast\(.*?(?=    def predict_chain)', new_pct_chain + '\n\n', content, flags=re.DOTALL)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print('Updated predictor_engine.py')
