import re

file_path = 'd:/Market-Intelligence/utils/signal_generator.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace VIX > 35.0 with Dynamic 99th Percentile or Asset-Specific Crash (-3Z)
new_ood_logic = '''        vix = ctx_values.get('VIX', 0.0)
        vix_pct = ctx_values.get('vix_percentile_252d', 0.0)
        return_zscore = ctx_values.get('return_zscore_90d', 0.0)
        
        # Phase 7 OOD Logic: Dynamic VIX 99th percentile or Asset-Specific -3 Sigma Crash
        if vix_pct > 0.99 or return_zscore < -3.0 or return_zscore > 3.0:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'weight': 0.0,
                'reason': 'OOD_CIRCUIT_BREAKER',
                'explanation': f'Extreme Market Event Detected. VIX Percentile: {vix_pct:.2f}, Asset Z-Score: {return_zscore:.2f}. Algorithm is enforcing a hard HOLD to protect capital.'
            }'''

# Replace the existing OOD logic in generate_final_signal
content = re.sub(r'        vix = ctx_values\.get\(\'VIX\', 0\.0\).*?\}', new_ood_logic, content, flags=re.DOTALL)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print('Updated signal_generator.py')
