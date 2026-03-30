"""
On-Chain BTC Data Fetcher
Fetches key on-chain indicators from CryptoQuant API or Blockchain.com API.
"""

import os
import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict

try:
    from sentiment_sources.base_fetcher import BaseSentimentFetcher
except ImportError:
    from base_fetcher import BaseSentimentFetcher

from dotenv import load_dotenv
load_dotenv()


class OnChainFetcher(BaseSentimentFetcher):
    """
    Fetches on-chain data for Bitcoin.
    Uses CryptoQuant API (preferred) or falls back to Blockchain.com free API.
    """
    
    def __init__(self):
        super().__init__("On-Chain Data")
        self.cryptoquant_api_key = os.getenv("CRYPTOQUANT_API_KEY")
        self.cryptoquant_endpoint = os.getenv("CRYPTOQUANT_ENDPOINT", "https://api.cryptoquant.com/v1")

    def fetch_news(self, asset: str, days: int = 30) -> List[Dict]:
        """
        Since this is the sentiment fetcher pipeline, we treat on-chain bullish/bearish 
        metrics as 'sentiment'. For robust model training, we map on-chain 
        to a normalized -1.0 to 1.0 score.
        Only applies to BTC.
        """
        asset_lower = asset.lower()
        if asset_lower not in ['btc', 'bitcoin']:
            return []
            
        articles = []
        
        # In a real rigorous setup, we would fetch Exchange Netflow, Active Addresses, etc.
        # Fallback to Blockchain.com API for hash rate as a proxy if CryptoQuant fails/lacks endpoint details
        has_cq_data = False
        if self.cryptoquant_api_key:
            has_cq_data = self._fetch_cryptoquant(articles, days)
                
        if not has_cq_data:
            self._fetch_blockchain_com(articles, days)
            
        return articles
        
    def _fetch_cryptoquant(self, articles: List[Dict], days: int) -> bool:
        """
        Attempt to fetch CryptoQuant Exchange Net Flow.
        Negative net flow (outflow) = bullish (1.0). Positive net flow = bearish (-1.0).
        """
        try:
            # Endpoint for Exchange Netflow (Total) for BTC
            # Using v1 docs format
            url = f"{self.cryptoquant_endpoint}/btc/exchange-flows/netflow?limit={days}"
            headers = {
                "Authorization": f"Bearer {self.cryptoquant_api_key}"
            }
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'result' in data and 'data' in data['result']:
                    items = data['result']['data']
                    for item in items:
                        # item usually has 'date' and 'netflow'
                        date_str = item.get('date', datetime.utcnow().strftime('%Y-%m-%d'))
                        if 'T' in date_str:
                            date_str = date_str.split('T')[0]
                        netflow = float(item.get('netflow', 0))
                        
                        # Normalize netflow: rough heuristic
                        # High positive netflow (deposits) is bearish (-1)
                        # High negative netflow (withdrawals) is bullish (+1)
                        # We clip robustly
                        sentiment = max(-1.0, min(1.0, -netflow / 10000.0))
                        
                        articles.append({
                            'title': f"CryptoQuant Netflow: {netflow:,.2f} BTC",
                            'url': 'https://cryptoquant.com/',
                            'date': date_str,
                            'sentiment': sentiment,
                            'source': 'CryptoQuant'
                        })
                    return True
            else:
                print(f"CryptoQuant API Error {response.status_code}: {response.text}")
                return False
        except Exception as e:
            print(f"Error calling CryptoQuant: {e}")
            return False

    def _fetch_blockchain_com(self, articles: List[Dict], days: int):
        """
        Fallback: fetch Hash Rate or Transaction Volume from blockchain.com.
        """
        try:
            # Fetch estimated transaction volume USD
            url = f"https://api.blockchain.info/charts/estimated-transaction-volume-usd?timespan={days}days&format=json"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                values = data.get('values', [])
                
                # We need to normalize against recent mean
                vols = [float(v['y']) for v in values]
                if not vols:
                    return
                mean_vol = sum(vols) / len(vols) if vols else 1
                
                for item in values:
                    timestamp = int(item['x'])
                    date_str = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d')
                    vol = float(item['y'])
                    
                    # Sentiment proxy: above average volume = bullish (>0), below = bearish (<0)
                    sentiment = (vol - mean_vol) / mean_vol
                    sentiment = max(-1.0, min(1.0, sentiment)) # clip
                    
                    articles.append({
                        'title': f"BTC Network Transfer Vol: ${vol/1e9:.2f}B",
                        'url': 'https://www.blockchain.com/explorer/charts',
                        'date': date_str,
                        'sentiment': sentiment,
                        'source': 'Blockchain.com'
                    })
        except Exception as e:
            print(f"Error fetching blockchain.com data: {e}")

if __name__ == "__main__":
    fetcher = OnChainFetcher()
    data = fetcher.fetch_news('btc', 7)
    for d in data:
        print(f"{d['date']}: {d['sentiment']} -> {d['title']}")
