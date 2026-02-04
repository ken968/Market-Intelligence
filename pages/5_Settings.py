"""
Settings & Control Panel
Data synchronization and model training interface
"""

import streamlit as st
import subprocess
import os
import sys
import re
from utils.config import get_asset_status, get_all_stock_tickers, ASSETS
from utils.ui_components import (
    inject_custom_css, render_page_header, render_status_badge,
    show_loading_message, show_success_message, show_error_message
)

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="Settings | XAUUSD Terminal",
    page_icon="",
    layout="wide"
)

inject_custom_css()

# ==================== HELPER FUNCTIONS ====================

def strip_ansi(text):
    """Remove ANSI escape codes from text"""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def run_command(command, description):
    """
    Run subprocess command and stream output
    
    Args:
        command (list): Command to run
        description (str): Description for status display
    
    Returns:
        bool: Success status
    """
    try:
        with st.status(description, expanded=True) as status:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            output_lines = []
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                clean_line = strip_ansi(line.strip())
                if clean_line:
                    st.text(clean_line)
                    output_lines.append(clean_line)
            
            process.wait()
            
            if process.returncode == 0:
                status.update(label=f"{description} - Complete ", state="complete")
                return True
            else:
                status.update(label=f"{description} - Failed ❌", state="error")
                return False
    
    except Exception as e:
        show_error_message(f"Error running command: {e}")
        return False

# ==================== MAIN CONTENT ====================

render_page_header(
    icon="",
    title="Settings & Control Panel",
    subtitle="Manage data synchronization and train AI models"
)

# ==================== SYSTEM STATUS ====================

st.markdown("###  System Status")

status = get_asset_status()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("####  Gold")
    if status['gold']['data']:
        render_status_badge('success', 'Data ')
    else:
        render_status_badge('danger', 'Data ')
    
    if status['gold']['model']:
        render_status_badge('success', 'Model ')
    else:
        render_status_badge('warning', 'Model ')

with col2:
    st.markdown("####  Bitcoin")
    if status['btc']['data']:
        render_status_badge('success', 'Data ')
    else:
        render_status_badge('danger', 'Data ')
    
    if status['btc']['model']:
        render_status_badge('success', 'Model ')
    else:
        render_status_badge('warning', 'Model ')

with col3:
    st.markdown("####  US Stocks")
    stocks_data = sum(1 for t in get_all_stock_tickers() if status[t.lower()]['data'])
    stocks_models = sum(1 for t in get_all_stock_tickers() if status[t.lower()]['model'])
    total_stocks = len(get_all_stock_tickers())
    
    if stocks_data == total_stocks:
        render_status_badge('success', f'Data: {stocks_data}/{total_stocks} ')
    elif stocks_data > 0:
        render_status_badge('warning', f'Data: {stocks_data}/{total_stocks}')
    else:
        render_status_badge('danger', f'Data: {stocks_data}/{total_stocks} ')
    
    if stocks_models == total_stocks:
        render_status_badge('success', f'Models: {stocks_models}/{total_stocks} ')
    elif stocks_models > 0:
        render_status_badge('warning', f'Models: {stocks_models}/{total_stocks}')
    else:
        render_status_badge('warning', f'Models: {stocks_models}/{total_stocks} ')

st.markdown("---")

# ==================== DATA SYNC SECTION ====================

st.markdown("### 📥 Data Synchronization")

st.info("""
**Data Sources:**
- Market prices from Yahoo Finance (10 years for Gold/Stocks, full history for Bitcoin)
- Macro indicators: DXY, VIX, US 10Y Treasury Yield
- News sentiment from NewsAPI (Bloomberg, Reuters, WSJ, CNBC, etc.)
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Quick Sync")
    
    if st.button("🔄 Sync All Assets (Market Data)", use_container_width=True):
        python_exe = sys.executable
        success = run_command(
            [python_exe, "data_fetcher_v2.py"],
            "Fetching market data for all assets..."
        )
        if success:
            show_success_message("All market data synchronized!")
            st.rerun()
    
    if st.button("📰 Sync Sentiment (All Assets)", use_container_width=True):
        python_exe = sys.executable
        success = run_command(
            [python_exe, "sentiment_fetcher_v2.py", "all"],
            "Analyzing news sentiment for all assets..."
        )
        if success:
            show_success_message("Sentiment analysis complete!")
            st.rerun()

with col2:
    st.markdown("#### Individual Asset Sync")
    
    asset_choice = st.selectbox(
        "Select asset to sync",
        ["Gold", "Bitcoin", "Stocks", "SPY", "AAPL", "NVDA", "TSLA"]
    )
    
    asset_map = {
        "Gold": "gold",
        "Bitcoin": "btc",
        "Stocks": "stocks"
    }
    
    asset_key = asset_map.get(asset_choice, asset_choice.lower())
    
    if st.button(f"Sync {asset_choice} Data & Sentiment", use_container_width=True):
        python_exe = sys.executable
        
        # Data fetch
        if asset_choice in ["Gold", "Bitcoin", "Stocks"]:
            success1 = run_command(
                [python_exe, "data_fetcher_v2.py", asset_key],
                f"Fetching {asset_choice} market data..."
            )
        else:
            success1 = run_command(
                [python_exe, "data_fetcher_v2.py", asset_choice.upper()],
                f"Fetching {asset_choice} market data..."
            )
        
        # Sentiment fetch
        success2 = run_command(
            [python_exe, "sentiment_fetcher_v2.py", asset_key],
            f"Analyzing {asset_choice} sentiment..."
        )
        
        if success1 and success2:
            show_success_message(f"{asset_choice} fully synchronized!")
            st.rerun()

st.markdown("---")

# ==================== MODEL TRAINING SECTION ====================

st.markdown("### 🤖 AI Model Training")

st.warning("""
 **Training Notes:**
- Gold model: ~2-3 minutes (30 epochs, 60-day window)
- Bitcoin model: ~3-5 minutes (50 epochs, 90-day window)
- Stock model (each): ~2-3 minutes (30 epochs, 60-day window)
- **All 11 stocks**: ~25-35 minutes total

Ensure data is synced before training!
""")

tab1, tab2, tab3 = st.tabs(["Quick Train", "Individual Assets", "Batch Training"])

with tab1:
    st.markdown("####  Quick Train Core Assets")
    st.info("Train Gold, Bitcoin, and SPY (S&P 500 index)")
    
    if st.button("Train Core Assets (Gold + BTC + SPY)", use_container_width=True):
        python_exe = sys.executable
        
        # Gold
        run_command([python_exe, "train_ultimate.py"], "Training Gold model...")
        
        # Bitcoin
        run_command([python_exe, "train_btc.py"], "Training Bitcoin model...")
        
        # SPY
        run_command([python_exe, "train_stocks.py", "SPY"], "Training SPY model...")
        
        show_success_message("Core assets trained successfully!")
        st.rerun()

with tab2:
    st.markdown("####  Train Individual Asset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Train Gold Model", use_container_width=True):
            if not status['gold']['data']:
                show_error_message("Gold data not available. Sync data first!")
            else:
                python_exe = sys.executable
                success = run_command(
                    [python_exe, "train_ultimate.py"],
                    "Training Gold AI model..."
                )
                if success:
                    show_success_message("Gold model trained!")
                    st.rerun()
    
    with col2:
        if st.button("Train Bitcoin Model", use_container_width=True):
            if not status['btc']['data']:
                show_error_message("Bitcoin data not available. Sync data first!")
            else:
                python_exe = sys.executable
                success = run_command(
                    [python_exe, "train_btc.py"],
                    "Training Bitcoin AI model..."
                )
                if success:
                    show_success_message("Bitcoin model trained!")
                    st.rerun()
    
    st.markdown("---")
    st.markdown("####  Train Individual Stock")
    
    stock_tickers = get_all_stock_tickers()
    selected_stock = st.selectbox("Select stock to train", stock_tickers)
    
    if st.button(f"Train {selected_stock} Model", use_container_width=True):
        if not status[selected_stock.lower()]['data']:
            show_error_message(f"{selected_stock} data not available. Sync data first!")
        else:
            python_exe = sys.executable
            success = run_command(
                [python_exe, "train_stocks.py", selected_stock],
                f"Training {selected_stock} AI model..."
            )
            if success:
                show_success_message(f"{selected_stock} model trained!")
                st.rerun()

with tab3:
    st.markdown("####  Train All 11 Stocks")
    st.warning(" This will take 25-35 minutes. Keep this page open!")
    
    stock_list = get_all_stock_tickers()
    st.info(f"**Will train:** {', '.join(stock_list)}")
    
    if st.button(" Train All Stocks (Batch)", use_container_width=True, type="primary"):
        # Check if data exists
        missing_data = [t for t in stock_list if not status[t.lower()]['data']]
        
        if missing_data:
            show_error_message(f"Missing data for: {', '.join(missing_data)}. Sync first!")
        else:
            python_exe = sys.executable
            success = run_command(
                [python_exe, "train_stocks.py", "ALL"],
                "Batch training all stock models (this will take a while)..."
            )
            
            if success:
                show_success_message("All stock models trained successfully! 🎉")
                st.rerun()

st.markdown("---")

# ==================== MAINTENANCE SECTION ====================

st.markdown("### 🛠️ Maintenance")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Clear Cache")
    if st.button("🗑️ Clear Streamlit Cache", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        show_success_message("Cache cleared!")

with col2:
    st.markdown("#### System Info")
    st.info(f"""
    **Python:** {sys.version.split()[0]}  
    **Working Dir:** {os.getcwd()}  
    **Models:** {sum(1 for s in status.values() if s['model'])} trained  
    **Data Files:** {sum(1 for s in status.values() if s['data'])} available
    """)

# ==================== ADVANCED SETTINGS ====================

with st.expander(" Advanced Settings"):
    st.markdown("#### Hyperparameter Info")
    st.code("""
Gold Model:
- LSTM Units: [100, 50, 25]
- Dropout: 0.2
- Epochs: 30
- Sequence Length: 60 days

Bitcoin Model:
- LSTM Units: [128, 64, 32]
- Dropout: 0.3
- Epochs: 50
- Sequence Length: 90 days

Stock Models:
- LSTM Units: [100, 50, 25]
- Dropout: 0.2
- Epochs: 30
- Sequence Length: 60 days
    """)
    
    st.warning("Modifying hyperparameters requires editing training scripts directly.")

# ==================== FOOTER ====================

st.markdown("---")
st.caption("""
 **Tips:**
1. Always sync data before training models
2. Re-train models weekly to incorporate latest market data
3. Bitcoin model needs full history (2009+) for optimal accuracy
4. Check system status before running predictions
""")
