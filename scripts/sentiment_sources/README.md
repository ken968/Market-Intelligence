# Sentiment Sources Package

This package provides multiple free sentiment data sources to replace NewsAPI.

## Available Sources

1. **Yahoo RSS** (`yahoo_rss.py`)
   - Unlimited, no API key required
   - Fetches from Yahoo Finance RSS feeds

2. **Finnhub** (`finnhub_fetcher.py`)
   - Free tier: 60 calls/minute
   - Requires API key (free registration)

3. **Alpha Vantage** (`alpha_vantage_fetcher.py`)
   - Free tier: 500 calls/day
   - Requires API key (free registration)

## Usage

```python
from scripts.sentiment_sources.aggregator import SentimentAggregator

aggregator = SentimentAggregator()
sentiment_data = aggregator.fetch_all('AAPL', days=30)
```
