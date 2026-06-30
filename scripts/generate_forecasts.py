import os
import sys
import argparse
from datetime import datetime
import json

# Adjust path to find utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import ASSETS, STOCK_TICKERS
from utils.predictor import AssetPredictor
from utils.data_store import MarketDataStore

def generate_forecasts(asset_key: str):
    """
    Simulate the UI's prediction trigger for a specific asset to generate
    and log the 7-day (and other) forecasts to counterfactual_log.jsonl.
    """
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Generating forecast for {asset_key.upper()}...")
    
    predictor = AssetPredictor(asset_key)
    
    # Check if model exists before trying to predict
    model_path = f"models/{asset_key}_ultimate_model.keras"
    if not os.path.exists(model_path) and asset_key in ['gold', 'btc']:
        # Check horizon model
        if not os.path.exists(f"models/{asset_key}_model_7d.keras"):
             print(f"  -> Skipping: No model found for {asset_key.upper()}")
             return
    elif asset_key not in ['gold', 'btc']:
        if not os.path.exists(f"models/{asset_key}_model_7d.keras"):
             print(f"  -> Skipping: No model found for {asset_key.upper()}")
             return

    # To trigger the Counterfactual Logger, we need to pass headlines.
    # We will fetch the latest cached headlines for this asset.
    news_file = f"data/latest_news_{asset_key}.json"
    news_headlines = []
    published_at_list = []
    
    if os.path.exists(news_file):
        try:
            with open(news_file, 'r', encoding='utf-8') as f:
                news_data = json.load(f)
                
                # Support both dict with 'items' or direct list of articles
                items_list = news_data.get('items', []) if isinstance(news_data, dict) else (news_data if isinstance(news_data, list) else [])
                
                for item in items_list:
                    # Use LLM-generated headline if available, else raw title
                    headline = item.get('llm_headline') or item.get('headline') or item.get('title')
                    if headline:
                        news_headlines.append(headline)
                        
                        # Also collect published_at or date
                        date_str = item.get('published_at') or item.get('date')
                        if date_str:
                            published_at_list.append(date_str)
                            
        except Exception as e:
            print(f"  -> Warning: Failed to load news for {asset_key}: {e}")
    
    if not news_headlines:
        print(f"  -> Warning: No headlines found for {asset_key.upper()}. LLM CEO and Counterfactual Logging may be bypassed.")
    else:
        print(f"  -> Loaded {len(news_headlines)} cached headlines.")

    # Call get_multi_range_forecast just like the UI does.
    # This automatically triggers `log_forecast` internally if headlines exist and LLM succeeds.
    try:
        results = predictor.get_multi_range_forecast(headlines=news_headlines)
        
        # Print summary for key horizons
        for hz in ['1 Week', '1 Month', '3 Months']:
            if hz in results and 'error' not in results[hz]:
                pred_price = results[hz]['price']
                pct_change = ((pred_price - results['Current']) / results['Current']) * 100
                print(f"  -> Forecast successful: {hz} Target: {pred_price:.2f} ({pct_change:+.2f}%)")
            else:
                 print(f"  -> {hz} prediction was unavailable.")
                 
    except Exception as e:
        print(f"  -> Error generating forecast for {asset_key.upper()}: {e}")

import time

def main():
    parser = argparse.ArgumentParser(description="Automated Forecast Generation & Logging")
    parser.add_argument('asset', nargs='?', default='all', help="Asset ticker (e.g., btc, spy) or 'all'")
    args = parser.parse_args()

    # Get target assets
    if args.asset.lower() == 'all':
        # Default changed to essential macro assets to save API tokens & time
        assets_to_run = ['btc', 'gold', 'spy']
    elif args.asset.lower() == 'everything':
        assets_to_run = ['gold', 'btc'] + [t.lower() for t in STOCK_TICKERS.keys()]
    else:
        assets_to_run = [a.strip() for a in args.asset.lower().split(',')]

    print(f"Starting Automated Forecast Generation for {len(assets_to_run)} asset(s)...")
    
    success_count = 0
    for asset in assets_to_run:
        if asset in ASSETS:
            generate_forecasts(asset)
            success_count += 1
            # Rate limiting protection: Gemini Free Tier allows 15 RPM (1 request every 4 seconds)
            # Sleep for 5 seconds between assets to stay under the limit.
            if len(assets_to_run) > 1:
                print("  -> Sleeping 5s to respect Gemini API rate limits...")
                time.sleep(5)
        else:
            print(f"Unknown asset: {asset}")
            
    print(f"\nCompleted forecast generation for {success_count} assets.")

if __name__ == '__main__':
    main()
