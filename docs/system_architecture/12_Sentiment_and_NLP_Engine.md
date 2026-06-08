# 12. Sentiment and NLP Engine Architecture

## 1. Overview
The Sentiment and NLP Engine is responsible for ingesting unstructured text data from multiple sources (news APIs, RSS feeds, Twitter/X, specialized financial sites) and converting them into structured, quantitative signals that the forecasting models can consume.

## 2. Core Upgrades: From Lexicon to Deep Learning
The system has been completely overhauled from using rule-based/lexicon-based analyzers (like TextBlob and VADER) to a deep-learning-based architecture using **FinBERT**.

### Why FinBERT?
- **Financial Context Understanding**: Traditional NLP models interpret words like "cut" or "bear" in general contexts. FinBERT is pre-trained on a massive corpus of financial text, corporate reports, and analyst notes. It correctly identifies "rate cut" as potentially bullish for equities, and "bearish divergence" as a negative technical signal.
- **Noise Reduction**: In the modern media landscape, headlines are often clickbait. FinBERT evaluates the nuanced relationship between the headline and the summary (where available) to extract genuine economic sentiment.

## 3. The Sentiment Aggregation Pipeline

The pipeline operates in 3 distinct stages:

### Stage 1: Multi-Source Fetching
The system queries up to 8 distinct sources to prevent single-source bias:
- **NewsAPI** (General Macro News)
- **Finnhub** (Asset-specific News)
- **Yahoo Finance RSS** (Retail & Market News)
- **AlphaVantage** (Market Sentiment & News)
- **Twitter/X** (Retail Sentiment & Viral Trends)

### Stage 2: FinBERT Processing
Each piece of text is tokenized and passed through the local FinBERT model. The model outputs a probability distribution across 3 classes: `[Positive, Negative, Neutral]`.
A composite score is calculated:
`Score = Prob(Positive) - Prob(Negative)`
This yields a bounded sentiment score between -1.0 and +1.0.

### Stage 3: Z-Score Normalization
Raw sentiment scores are highly volatile. A score of +0.5 today might mean little if the 30-day average is +0.6. The engine computes a **Rolling Z-Score**:
`Z_Score = (Today's Sentiment - 30_Day_Mean) / 30_Day_StdDev`
This ensures the ML models only react to *statistically significant* sentiment shifts, ignoring baseline market chatter.

## 4. Integration with Forecasting Models
The normalized Sentiment Z-Score is injected into the machine learning pipeline in two ways:
1. **As a direct feature** into the XGBoost Stacker.
2. **As a temporal sequence feature** into the LSTM, allowing the neural network to learn the lag effect between a news event and price action (e.g., news takes 2-3 days to be fully priced in).
