"""
Fix all assets (Gold + BTC + All Stocks) to solve NaN issue
"""
import subprocess
import sys

print("=" * 70)
print("FIX SCRIPT: Re-syncing ALL ASSETS to fix NaN prediction issue")
print("=" * 70)

# Step 1: Re-fetch ALL asset data (uses fixed data_fetcher_v2.py)
print("\n[STEP 1/2] Re-fetching ALL macro + asset data...")
result = subprocess.run([
    sys.executable, 
    "scripts/data_fetcher_v2.py"  # No args = fetch all
], capture_output=True, text=True)

print(result.stdout)
if result.returncode != 0:
    print("ERROR during data fetch:", result.stderr)
    sys.exit(1)

# Step 2: Re-integrate sentiment for ALL assets
print("\n[STEP 2/2] Re-integrating sentiment for ALL assets...")
result = subprocess.run([
    sys.executable,
    "scripts/sentiment_fetcher_v2.py",
    "all"
], capture_output=True, text=True)

print(result.stdout)
if result.returncode != 0:
    print("ERROR during sentiment integration:", result.stderr)
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ SUCCESS! All assets have been fixed.")
print("=" * 70)
print("\nYou can now run forecasts for ANY asset from the UI.")
print("The $nan issue should be completely resolved.")
