"""
Quick fix script to re-sync BTC data and fix NaN issue
"""
import subprocess
import sys

print("=" * 60)
print("FIX SCRIPT: Re-syncing BTC data to fix NaN issue")
print("=" * 60)

# Step 1: Re-fetch BTC macro data (will use the fixed data_fetcher_v2.py)
print("\n[STEP 1/2] Re-fetching BTC macro data...")
result = subprocess.run([
    sys.executable, 
    "scripts/data_fetcher_v2.py", 
    "btc"
], capture_output=True, text=True)

print(result.stdout)
if result.returncode != 0:
    print("ERROR:", result.stderr)
    sys.exit(1)

# Step 2: Re-integrate sentiment with fixed macro data
print("\n[STEP 2/2] Re-integrating sentiment data...")
result = subprocess.run([
    sys.executable,
    "scripts/sentiment_fetcher_v2.py",
    "btc"
], capture_output=True, text=True)

print(result.stdout)
if result.returncode != 0:
    print("ERROR:", result.stderr)
    sys.exit(1)

print("\n" + "=" * 60)
print("SUCCESS! BTC data fixed.")
print("=" * 60)
print("\nNow you can run the forecast again from the UI.")
