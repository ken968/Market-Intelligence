#!/usr/bin/env python3
"""
XAUUSD Multi-Asset Terminal - Quick Setup Script
Automated setup for first-time deployment
"""

import os
import sys
import subprocess
import time

def print_header(text):
    """Print section header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def run_command(command, description):
    """Run command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        print(f" {description} - Complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed!")
        print(f"Error: {e.stderr}")
        return False

def check_dependencies():
    """Check if Python and pip are available"""
    print_header("Checking Dependencies")
    
    # Check Python
    try:
        python_version = sys.version.split()[0]
        print(f" Python {python_version} detected")
        
        major, minor = python_version.split('.')[:2]
        if int(major) < 3 or (int(major) == 3 and int(minor) < 8):
            print("❌ Python 3.8 or higher required!")
            return False
    except:
        print("❌ Python not found!")
        return False
    
    # Check pip
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      capture_output=True, check=True)
        print(" pip is available")
    except:
        print("❌ pip not found!")
        return False
    
    return True

def setup_venv():
    """Create and activate virtual environment"""
    print_header("Setting Up Virtual Environment")
    
    if os.path.exists('.venv'):
        print("  Virtual environment already exists. Skipping creation.")
        return True
    
    return run_command(
        f"{sys.executable} -m venv .venv",
        "Creating virtual environment"
    )

def install_requirements():
    """Install Python dependencies"""
    print_header("Installing Dependencies")
    
    # Get pip path
    if sys.platform == "win32":
        pip_exe = ".venv\\Scripts\\pip.exe"
    else:
        pip_exe = ".venv/bin/pip"
    
    if not os.path.exists(pip_exe):
        pip_exe = sys.executable + " -m pip"
    
    return run_command(
        f"{pip_exe} install -r requirements.txt",
        "Installing Python packages"
    )

def create_utils_init():
    """Ensure utils/__init__.py exists"""
    print_header("Setting Up Utils Package")
    
    os.makedirs('utils', exist_ok=True)
    
    init_file = 'utils/__init__.py'
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write('"""Utils package for XAUUSD Terminal"""\n')
        print(" Created utils/__init__.py")
    else:
        print(" utils/__init__.py already exists")
    
    return True

def fetch_data():
    """Download market data"""
    print_header("Fetching Market Data")
    
    # Get python path
    if sys.platform == "win32":
        python_exe = ".venv\\Scripts\\python.exe"
    else:
        python_exe = ".venv/bin/python"
    
    if not os.path.exists(python_exe):
        python_exe = sys.executable
    
    print("\n📥 Downloading historical data (this may take 5-10 minutes)...")
    print("   - Gold: 10 years")
    print("   - Bitcoin: Full history (2009-present)")
    print("   - Stocks: 10 years for 11 tickers")
    
    return run_command(
        f"{python_exe} data_fetcher_v2.py",
        "Downloading market data"
    )

def fetch_sentiment():
    """Analyze news sentiment"""
    print_header("Analyzing News Sentiment")
    
    if sys.platform == "win32":
        python_exe = ".venv\\Scripts\\python.exe"
    else:
        python_exe = ".venv/bin/python"
    
    if not os.path.exists(python_exe):
        python_exe = sys.executable
    
    print("\n📰 Analyzing news from Bloomberg, Reuters, WSJ, etc.")
    
    return run_command(
        f"{python_exe} sentiment_fetcher_v2.py all",
        "Sentiment analysis"
    )

def train_core_models():
    """Train Gold, Bitcoin, and SPY models"""
    print_header("Training Core AI Models")
    
    if sys.platform == "win32":
        python_exe = ".venv\\Scripts\\python.exe"
    else:
        python_exe = ".venv/bin/python"
    
    if not os.path.exists(python_exe):
        python_exe = sys.executable
    
    print("\n🤖 Training 3 core models (~10 minutes total):")
    print("   1. Gold (XAUUSD)")
    print("   2. Bitcoin (BTC)")
    print("   3. S&P 500 (SPY)")
    
    success = True
    
    # Train Gold
    print("\n--- Training Gold Model ---")
    success &= run_command(f"{python_exe} train_ultimate.py", "Gold model training")
    
    # Train Bitcoin
    print("\n--- Training Bitcoin Model ---")
    success &= run_command(f"{python_exe} train_btc.py", "Bitcoin model training")
    
    # Train SPY
    print("\n--- Training SPY Model ---")
    success &= run_command(f"{python_exe} train_stocks.py SPY", "SPY model training")
    
    return success

def main():
    """Main setup flow"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   XAUUSD MULTI-ASSET TERMINAL - QUICK SETUP               ║
    ║   AI-Powered Market Intelligence Platform                 ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    print("\n This script will:")
    print("   1. Check dependencies")
    print("   2. Create virtual environment")
    print("   3. Install Python packages")
    print("   4. Download market data")
    print("   5. Analyze news sentiment")
    print("   6. Train core AI models (Gold, BTC, SPY)")
    
    print("\n  Estimated time: 20-30 minutes")
    print(" You can train all 11 stocks later from Settings page")
    
    input("\nPress Enter to start setup... ")
    
    start_time = time.time()
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Setup failed: Missing dependencies")
        sys.exit(1)
    
    # Step 2: Setup venv
    if not setup_venv():
        print("\n❌ Setup failed: Could not create virtual environment")
        sys.exit(1)
    
    # Step 3: Install requirements
    if not install_requirements():
        print("\n❌ Setup failed: Could not install dependencies")
        sys.exit(1)
    
    # Step 4: Create utils package
    if not create_utils_init():
        print("\n❌ Setup failed: Could not create utils package")
        sys.exit(1)
    
    # Step 5: Fetch data
    if not fetch_data():
        print("\n  Warning: Data fetch failed. You can retry manually.")
        print("   Run: python data_fetcher_v2.py")
    
    # Step 6: Fetch sentiment
    if not fetch_sentiment():
        print("\n  Warning: Sentiment analysis failed. You can retry manually.")
        print("   Run: python sentiment_fetcher_v2.py all")
    
    # Step 7: Train models
    print("\n  Model training will now begin (~10 minutes)")
    response = input("Train core models now? (y/n): ").lower()
    
    if response == 'y':
        if not train_core_models():
            print("\n  Warning: Model training had issues. Check errors above.")
    else:
        print("\n  Skipping model training. Run later from Settings page.")
    
    # Summary
    elapsed = time.time() - start_time
    
    print_header("Setup Complete! 🎉")
    print(f"\n  Total time: {elapsed/60:.1f} minutes")
    print("\n Next steps:")
    print("   1. Activate virtual environment:")
    if sys.platform == "win32":
        print("      .venv\\Scripts\\activate")
    else:
        print("      source .venv/bin/activate")
    print("\n   2. Launch the app:")
    print("      streamlit run app.py")
    print("\n   3. Open browser at: http://localhost:8501")
    print("\n📚 Read DEPLOYMENT_GUIDE.md for more details")
    print("📖 Check README.md for full documentation")
    
    print("\n" + "="*60)
    print("  Thank you for using XAUUSD Multi-Asset Terminal!")
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Setup interrupted by user. Run again to continue.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        print("Please report this issue on GitHub")
        sys.exit(1)
