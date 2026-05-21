import pandas as pd
import numpy as np
import pickle
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN info messages
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# 1. Load Data
if not os.path.exists('data/gold_global_insights.csv'):
    print("Error: data/gold_global_insights.csv not found")
    exit(1)

df = pd.read_csv('data/gold_global_insights.csv')
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.config import ASSETS
    features  = ASSETS['gold']['features']
    arch      = ASSETS['gold'].get('model_arch', {'units': [100, 50], 'dropout': 0.3, 'attention': False})
    seq_len   = ASSETS['gold'].get('sequence_length', 60)
except ImportError:
    features  = ['Gold', 'DXY', 'VIX', 'Yield_10Y', 'Oil_Price',
                 'CPI_MoM', 'PPI_MoM', 'PCE_MoM', 'NFP_Change',
                 'YieldCurve_10Y2Y', 'M2_MoM', 'M2_YoY', 'Yield_10Y_Rate',
                 'Breakeven_5Y5Y', 'M2_Liquidity_Spike', 'MacroEvent_Flag',
                 'Sentiment', 'EMA_90']
    arch      = {'units': [100, 50], 'dropout': 0.3, 'attention': False}
    seq_len   = 60

missing = [f for f in features if f not in df.columns]
if missing:
    print(f"Missing features: {missing} — filling with 0 for robustness")
    for f in missing:
        df[f] = 0

data = df[features].values

# 2. Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Save scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 3. Data Preparation
prediction_days = seq_len
x_train, y_train = [], []

if len(scaled_data) <= prediction_days:
    print("Not enough data to train")
    exit(1)

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, :])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Dynamic Model Architecture (reads from config['model_arch'])
# ─────────────────────────────────────────────────────────────────────────────
def build_lstm_model(input_shape, arch: dict):
    """
    Build LSTM model from architecture config dict.

    arch keys:
        units    : list of int — LSTM units per layer (e.g. [64, 32])
        dropout  : float — dropout rate applied after each LSTM layer
        attention: bool — if True, add MultiHeadAttention after first LSTM layer

    Example configs:
        Gold (stable):  {'units': [64, 32],      'dropout': 0.20, 'attention': False}
        BTC (volatile): {'units': [128, 64, 32],  'dropout': 0.30, 'attention': True}
        NVDA (volatile):{'units': [128, 64],      'dropout': 0.35, 'attention': True}
        SPY (index):    {'units': [64, 32],       'dropout': 0.20, 'attention': False}
    """
    units     = arch.get('units', [100, 50])
    dropout   = arch.get('dropout', 0.3)
    attention = arch.get('attention', False)

    inp = Input(shape=input_shape)
    x   = inp

    for i, u in enumerate(units):
        is_last = (i == len(units) - 1)
        # All but the last LSTM return sequences (for attention or next LSTM)
        return_seq = (not is_last) or attention
        x = LSTM(units=u, return_sequences=return_seq, kernel_regularizer=l2(0.001))(x)
        x = Dropout(dropout)(x)

        # Self-Attention after first LSTM layer (if enabled)
        if attention and i == 0:
            attn_out = MultiHeadAttention(num_heads=4, key_dim=u // 4)(x, x)
            attn_out = LayerNormalization()(attn_out + x)   # Residual connection
            # If this is also the last layer, flatten; else pass through
            if is_last:
                x = Flatten()(attn_out)
            else:
                x = attn_out
        elif is_last and attention:
            # Last layer after attention was applied: flatten sequence
            x = Flatten()(x)

    # Dense head
    x = Dense(units=max(16, units[-1] // 2), kernel_regularizer=l2(0.001))(x)
    out = Dense(units=1)(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


print(f"\nBuilding LSTM model — arch: units={arch['units']}, dropout={arch['dropout']}, attention={arch['attention']}")
model = build_lstm_model((x_train.shape[1], x_train.shape[2]), arch)
model.summary()

# ─────────────────────────────────────────────────────────────────────────────
# 5. Callbacks
# ─────────────────────────────────────────────────────────────────────────────
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=1e-6,
    verbose=1
)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Training
# ─────────────────────────────────────────────────────────────────────────────
print("\nStarting Gold LSTM training...")
history = model.fit(
    x_train, y_train,
    epochs=75,
    batch_size=32,
    validation_split=0.1,
    verbose=1,
    callbacks=[early_stop, reduce_lr]
)

# 7. Save Model
model.save('models/gold_ultimate_model.keras')
print("\nModel saved as 'models/gold_ultimate_model.keras'")
print(f"Architecture: units={arch['units']}, dropout={arch['dropout']}, attention={arch['attention']}")