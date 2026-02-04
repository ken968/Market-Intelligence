import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import os
import subprocess
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# --- CONFIG PAGE ---
st.set_page_config(
    page_title="XAUUSD Global Insight AI", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- PREMIUM CSS SYSTEM ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@400;600;700&display=swap');

    :root {
        --bg-deep: #0B101B;
        --bg-surface: #151B28;
        --accent: #C5A059;
        --accent-muted: rgba(197, 160, 89, 0.2);
        --border: #232D3F;
        --text-pri: #E2E8F0;
        --text-sec: #94A3B8;
        --success: #00C076;
        --danger: #FF4D4D;
    }

    /* Global Reset */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 95% !important;
    }
    
    .stApp {
        background-color: var(--bg-deep);
        color: var(--text-pri);
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif !important;
        color: var(--text-pri);
        font-weight: 600 !important;
    }

    /* Header Styling */
    .terminal-header {
        border-bottom: 1px solid var(--border);
        padding-bottom: 1rem;
        margin-bottom: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Metric Cards - Industrial */
    .metric-card {
        background: var(--bg-surface);
        border: 1px solid var(--border);
        padding: 1.25rem;
        border-radius: 4px; /* Minimalist sharp corners */
        position: relative;
        overflow: hidden;
    }
    .metric-card-lbl {
        font-size: 0.75rem;
        color: var(--text-sec);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .metric-card-val {
        font-size: 1.75rem;
        font-weight: 700;
        font-family: 'Outfit', sans-serif;
        color: #FFFFFF;
    }
    .metric-card-delta {
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }
    .up { color: var(--success); }
    .down { color: var(--danger); }

    /* News Cards */
    .news-item {
        background: var(--bg-surface);
        border-bottom: 1px solid var(--border);
        padding: 1rem 0;
        margin-bottom: 0.5rem;
    }
    .news-item:last-child { border-bottom: none; }
    .news-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-pri);
        text-decoration: none !important;
        display: block;
        line-height: 1.4;
    }
    .news-title:hover { color: var(--accent); }
    .news-meta {
        font-size: 0.75rem;
        color: var(--text-sec);
        margin-top: 0.4rem;
    }

    /* Streamlit Component Overrides */
    .stButton>button {
        background: var(--accent) !important;
        color: #000 !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        border: none !important;
        height: 2.8rem;
        transition: opacity 0.2s;
    }
    .stButton>button:hover {
        opacity: 0.9;
    }
    
    .stTable {
        background: var(--bg-surface);
        border-radius: 4px;
    }
    
    /* Sidebar Fixes */
    section[data-testid="stSidebar"] {
        background-color: #080C14 !important;
        border-right: 1px solid var(--border);
    }
    
    .stPlotlyChart {
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)



# Helper untuk Load Model
@st.cache_resource
def get_model():
    if os.path.exists('gold_ultimate_model.h5'):
        return load_model('gold_ultimate_model.h5')
    return None

def run_sync():
    with st.spinner('Menarik data pasar terbaru (yfinance)...'):
        subprocess.run([".venv\\Scripts\\python.exe", "data_fetcher.py"], capture_output=True)
    with st.spinner('Menganalisis sentimen berita global...'):
        subprocess.run([".venv\\Scripts\\python.exe", "sentiment_fetcher.py"], capture_output=True)
    st.success("Sinkronisasi Berhasil!")
    st.rerun()

def run_training():
    import re
    def strip_ansi(text):
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    with st.status("Training AI Model (Deep Learning)...", expanded=True) as status:
        st.write("Initializing data pipelines...")
        process = subprocess.Popen([".venv\\Scripts\\python.exe", "train_ultimate.py"], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT, 
                                 text=True,
                                 bufsize=1)
        
        while True:
            line = process.stdout.readline()
            if not line: break
            clean_line = strip_ansi(line.strip())
            if clean_line:
                st.text(clean_line)
            
        process.wait()
        status.update(label="Training Complete!", state="complete", expanded=False)
    st.success("AI Model has been successfully updated with latest market dynamics.")
    st.rerun()

# --- SIDEBAR ---
st.sidebar.title("Control Panel")
if st.sidebar.button("Sync Global Data & Sentiment"):
    run_sync()

st.sidebar.markdown("### AI Model Management")
if st.sidebar.button("Re-Train AI Model (New Data)"):
    run_training()

st.sidebar.markdown("---")
st.sidebar.info("""
**AI Insights Components:**
- Gold Price (GC=F)
- DXY (Dollar Index)
- VIX (Volatility Index)
- TNX (US 10Y Yield)
- News Sentiment (NLP)
""")

# --- MAIN PAGE ---
st.markdown("""
    <div class="terminal-header">
        <div>
            <h1 style="margin:0; font-size:1.5rem; color:var(--accent);">XAUUSD CORE TERMINAL</h1>
            <div style="font-size:0.8rem; color:var(--text-sec);">AI Intelligence • Real-time Market Data • Sentiment Analysis</div>
        </div>
        <div style="text-align:right;">
            <div style="font-size:0.75rem; color:var(--text-sec); text-transform:uppercase;">System Status</div>
            <div style="font-size:0.9rem; color:var(--success); font-weight:600;">ACTIVE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Load Data
if not os.path.exists('gold_global_insights.csv'):
    st.warning("Data belum tersedia. Silakan klik 'Sync Global Data' di sidebar.")
    st.stop()

df = pd.read_csv('gold_global_insights.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Metrics Row
latest = df.iloc[-1]
prev = df.iloc[-2]

def format_delta(curr, prev):
    delta = curr - prev
    return f"{delta:+.2f}"

def render_metric(label, value, delta):
    is_up = "+" in delta
    d_cls = "up" if is_up else "down"
    d_sym = "▲" if is_up else "▼"
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-lbl">{label}</div>
            <div class="metric-card-val">{value}</div>
            <div class="metric-card-delta {d_cls}">{d_sym} {delta}</div>
        </div>
        """, unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    render_metric("Gold Price", f"${latest['Gold']:,.2f}", format_delta(latest['Gold'], prev['Gold']))
with col2:
    render_metric("DXY Index", f"{latest['DXY']:.2f}", format_delta(latest['DXY'], prev['DXY']))
with col3:
    render_metric("VIX Index", f"{latest['VIX']:.2f}", format_delta(latest['VIX'], prev['VIX']))
with col4:
    render_metric("Sentiment", f"{latest['Sentiment']:.2f}", format_delta(latest['Sentiment'], prev['Sentiment']))


# Main Chart Section
st.markdown("#### Market Performance & Global Intercorrelations")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['Gold'], name='Gold Price', line=dict(color='#FFD700', width=2)))
fig.add_trace(go.Scatter(x=df['Date'], y=df['DXY']*20, name='DXY (Scaled)', line=dict(color='#4b6bff', width=1, dash='dot'))) # Scaled for visibility
fig.update_layout(template="plotly_dark", height=500, margin=dict(l=20, r=20, t=20, b=20))
st.plotly_chart(fig, use_container_width=True)

# Prediction & News Layout
left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown("### Multi-Range AI Prediction")
    if st.button('Execute Forecast (1D - 1Y)'):
        model = get_model()
        if model is None:
            st.error("Model 'gold_ultimate_model.h5' tidak ditemukan. Silakan jalankan training terlebih dahulu.")
        else:
            with st.spinner('AI sedang memodelkan masa depan...'):
                import pickle
                features = ['Gold', 'DXY', 'VIX', 'Yield_10Y', 'Sentiment']
                
                # Load Scaler yang sudah dilatih (PENTING!)
                if os.path.exists('scaler.pkl'):
                    with open('scaler.pkl', 'rb') as f:
                        scaler = pickle.load(f)
                    # Jika ada data baru yang melebihi range scaler lama, 
                    # kita transform saja (bukan fit) agar model tidak bingung.
                    scaled_data = scaler.transform(df[features])
                else:
                    st.warning("Perhatian: 'scaler.pkl' tidak ditemukan. Menggunakan scaler darurat (Prediksi mungkin tidak akurat). Silakan re-train model.")
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_data = scaler.fit_transform(df[features])
                
                # Recursive Prediction Logic
                def predict_future(steps):
                    current_batch = scaled_data[-60:].reshape(1, 60, 5)
                    predictions = []
                    
                    temp_data = current_batch.copy()
                    
                    for _ in range(steps):
                        # Predict next step
                        res = model.predict(temp_data, verbose=0)
                        # res is (1, 1)
                        
                        # Create new frame
                        # For simplicity, we assume other features stay constant or follow trend
                        # (Ideally the model should predict all 5, but here we only predict gold)
                        last_frame = temp_data[0, -1, :].copy()
                        last_frame[0] = res[0, 0] # Update only Gold price
                        
                        # Shift batch and add new frame
                        new_batch = np.append(temp_data[:, 1:, :], [[last_frame]], axis=1)
                        temp_data = new_batch
                        predictions.append(res[0,0])
                    
                    return predictions

                # Ranges
                ranges = {
                    "Besok (1D)": 1,
                    "1 Minggu (5D)": 5,
                    "1 Bulan (21D)": 21,
                    "6 Bulan (126D)": 126,
                    "1 Tahun (252D)": 252
                }
                
                results = {}
                for label, steps in ranges.items():
                    preds = predict_future(steps)
                    # Inverse only the last prediction
                    dummy = np.zeros((1, 5))
                    dummy[0, 0] = preds[-1]
                    val = scaler.inverse_transform(dummy)[0, 0]
                    results[label] = val
                
                # Table Display
                res_df = pd.DataFrame(list(results.items()), columns=["Timeframe", "Predicted Price"])
                st.table(res_df.style.format({"Predicted Price": "${:,.2f}"}))
                
                # Highlight Tomorrow
                st.success(f"**Prediksi Besok: ${results['Besok (1D)']:,.2f}**")
                
                # Warning for long term
                if results['1 Tahun (252D)'] > 0:
                    st.warning("Prediksi jangka panjang (>1 bulan) bersifat spekulatif dan berbasis rekursif.")

with right_col:
    st.markdown("### News Insights")
    if os.path.exists('latest_news.json'):
        with open('latest_news.json', 'r') as f:
            news = json.load(f)
        
        for art in news[:6]:
            score = art['sentiment']
            s_class = "up" if score > 0.1 else ("down" if score < -0.1 else "sentiment-neu")
            s_label = "POS" if score > 0.1 else ("NEG" if score < -0.1 else "NEU")
            
            st.markdown(f"""
            <div class="news-item">
                <a href="{art['url']}" target="_blank" class="news-title">{art['title']}</a>
                <div class="news-meta">
                    {art['date']} • <span class="{s_class}">{s_label} ({score:.2f})</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Berita tidak ditemukan. Klik 'Sync Global Data' untuk memuat berita.")

st.markdown("---")
st.caption("Disclaimer: Aplikasi ini adalah proyek edukasi AI. Bukan nasihat keuangan. Investasi emas memiliki risiko tinggi.")