# MARKET INTELLIGENCE SYSTEM: COMPREHENSIVE ARCHITECTURE & RESEARCH WHITEPAPER

## PART 1: SYSTEM ONTOLOGY, CAUSAL HIERARCHY, AND HIGH-LEVEL ARCHITECTURE

## 1. SYSTEM ONTOLOGY AND CORE TAXONOMY

The Market Intelligence Terminal is designed as a production-grade, multi-asset class predictive and analytical framework. Rather than acting as a high-frequency execution system or a simplistic rules-based technical indicator, the terminal serves as a Hierarchical Macro Intelligence & Quantitative Simulation Engine. The core asset scope focuses on Gold (XAUUSD), Bitcoin (BTCUSD), and major US Equity indices and selected large-cap equities (S&P 500 ETF, Nasdaq 100 ETF, Dow Jones ETF, Apple, Microsoft, Alphabet, Amazon, NVIDIA, Meta, Tesla, and TSMC).

The fundamental ontology of the system is the synthesis of high-dimensional, multi-frequency, asynchronous datasets into stationary, probabilistically-weighted expected returns. Speculative financial assets exhibit high non-stationarity, meaning their underlying probability distributions (mean, variance) drift over time. Standard machine learning models that attempt to predict absolute price levels ($P_t$) suffer from severe geometric compounding errors, variance loss, and generalization failure.

To resolve this, the system's objective function targets the Expected Percentage Return over a specified horizon:

$$
y_t = \frac{\text{Price}_{t + H} - \text{Price}_t}{\text{Price}_t}
$$

This formulation ensures that the target variable remains stationary, enabling the base learners and the meta-ensemble to model structural market cycles, regime transitions, and directional probabilities rather than attempting to memorize price boundaries.

## 2. THREE-LAYER CAUSAL HIERARCHY ARCHITECTURE

The system is structured as a strict three-layer hierarchy, segregating concerns and separating deterministic data modeling from speculative overlay:

### A. Layer 1: Worker Layer (Base Learners)

The Worker Layer contains specialized models that learn isolated dimensions of market pricing:

*   **Worker 1 (Temporal Sequence Model)**: A Deep Recurrent Neural Network (LSTM) augmented with Multi-Head Self-Attention. It operates on a 60-day (or 90-day for Bitcoin) sequence of OHLCV prices, technical indicators, atrocious short-term alternative metrics. It is optimized to detect autoregressive trends, momentum anomalies, and localized volatility patterns.
*   **Worker 2 (Regime Detection Model)**: An Extreme Gradient Boosting (XGBoost) model trained on macroeconomic inputs, monetary policy spreads, and institutional commitments. XGBoost operates on tabular cross-sectional data, learning non-linear, multi-variable decision boundaries that LSTMs cannot resolve.

### B. Layer 2: Manager Layer (Dual-Head Ensemble Stacker)

Traditional ensembles average the output of individual base models. This approach fails during regime shifts where one model's signals are completely dominated by noise. The Manager Layer employs a Dual-Head Stacker Meta-Learner (Logistic Regression CV and Huber Regressor). The Stacker ingests the predictions of the Worker models along with live macro-context features (e.g., VIX, Garman-Klass realized volatility, DXY, US 10Y-2Y Yield Curve Spread) and maps them to:

*   **Head 1 (Directional Head)**: A logistic classifier predicting binary sign probability.
*   **Head 2 (Magnitude Head)**: A Huber regressor predicting return magnitude.

The output of these heads is combined to yield a single volatility-adjusted expected return.

### C. Layer 3: CEO Layer (LLM Strategic Override)

The CEO Layer is a contextual bias injection engine utilizing a Large Language Model (Gemini) to evaluate qualitative variables (geopolitics, monetary news, policy shifts) that quantitative models cannot perceive. The LLM processes raw weekly news timeline feeds and produces a structural drift multiplier and bias vector. This overlay acts as a soft constraint, shifting the mean-reversion anchor of the forecast trajectory without overriding the mathematical foundations of the base learners.

## 3. LLM INGESTION PIPELINE AND CAUSAL HIERARCHY

The CEO Layer executes a rigorous multi-stage pipeline to convert unstructured news feeds into a structured numeric injection vector:

### A. Ingestion and Filtering

Raw news headlines are processed through a strict temporal filter with a 168-hour (7-day) staleness cutoff. This window ensures that the model only considers narratives actively influencing the current trading regime. Chronological sorting is enforced (oldest to newest) to let the LLM trace narrative progression and evolution.

### B. Cosine Similarity Deduplication

Financial feeds contain redundant headlines. To prevent narrative over-weighting, the pipeline applies a lightweight, bag-of-words cosine similarity check. Headlines with a similarity score exceeding a threshold of 0.85 are discarded, keeping only the first unique instance of a news event.

### C. Causal Hierarchy Prompting

The prompt forces the LLM to analyze the data according to a strict macroeconomic causal chain, reasoning in the following order:

1.  **Supply Shock Severity**: Disruptions in physical commodity supply or network infrastructure.
2.  **Geopolitical Stress**: Tensions, wars, sanctions, trade barriers, or diplomatic conflicts.
3.  **Monetary Policy Hawkishness**: Central bank rate trajectories and quantitative tightening/easing plans.
4.  **Risk Appetite**: Broad institutional and retail willingness to hold risky assets.
5.  **Market Sentiment**: The downstream public retail mood and short-term media narrative.

This structure is mathematically grounded in economic transmission theory: upstream physical supply shocks and geopolitical developments drive central bank policy actions, which in turn adjust risk appetite, ultimately manifesting as retail sentiment. The LLM is prohibited from letting downstream sentiment affect upstream supply/geopolitical scores.

### D. Few-Shot Anchor Calibration

To prevent scale drift, the system prompt contains historical calibration anchors. The LLM evaluates new timeline feeds relative to these fixed historical reference points:

*   **Feb 2022 (Russia-Ukraine commodities shock)**: Supply Shock=0.90, Geopolitical Stress=0.95, Monetary policy=0.60, Risk Appetite=0.10, Market Sentiment=0.15.
*   **Sep 2024 (Fed emergency rate cuts)**: Supply Shock=0.10, Geopolitical Stress=0.20, Monetary policy=0.05, Risk Appetite=0.75, Market Sentiment=0.70.
*   **Mar 2023 (SVB bank run)**: Supply Shock=0.15, Geopolitical Stress=0.25, Monetary policy=0.30, Risk Appetite=0.15, Market Sentiment=0.20.

## 4. POST-LLM MATHEMATICAL PROCESSING

Raw outputs from an LLM are prone to multi-collinearity and scale instability. The system applies two post-processing mathematical steps:

### A. Gram-Schmidt Orthogonalization

To ensure that downstream categories do not double-count upstream factors, the system orthogonalizes the raw score vector. The causal order defines the basis. Let the raw scores be represented by a vector $S = [s_1, s_2, s_3, s_4, s_5]$ corresponding to the 5 categories. The orthogonalized basis vectors $B = [b_1, b_2, b_3, b_4, b_5]$ are computed as:

$$
b_1 = s_1 \\
b_i = s_i - \sum_{j < i} (\text{Proj}_{b_j} (s_i))
$$

Where the projection is defined as:

$$
\text{Proj}_u (v) = \frac{(u \cdot v)}{(u \cdot u + \epsilon)} \cdot u
$$

This ensures that the score for category $i$ only reflects variance that has not already been explained by categories $1$ to $i-1$. The resulting vector is clipped back to $[0.0, 1.0]$.

### B. ZCA Whitening

To prevent any single score category from dominating the LSTM drift calculation due to scale discrepancies, the orthogonalized vector undergoes Zero-phase Component Analysis (ZCA) whitening. The vector is standardized to zero mean and unit variance, and then rescaled to the $[0.0, 1.0]$ domain, stabilizing the output before it is passed to the drift engine.

## 5. CEO DRIFT MULTIPLIER AND BIAS VECTOR INJECTION

The processed LLM scores are mapped into a single drift multiplier, $\delta$, which directly scales the forecast returns. The formula utilizes asset-specific sensitivities:

*   **Gold (Geopolitical and inflation hedge)**:
    $\text{raw bias} = 0.40 \cdot \text{Geopolitics} + 0.30 \cdot \text{Supply} + 0.20 \cdot (1 - \text{Monetary}) + 0.10 \cdot \text{Risk}$
*   **Bitcoin (Risk-on and high-beta alternative asset)**:
    $\text{raw bias} = 0.40 \cdot \text{Risk} + 0.30 \cdot \text{Sentiment} + 0.20 \cdot (1 - \text{Monetary}) + 0.10 \cdot (1 - \text{Geopolitics})$
*   **Oil**:
    $\text{raw bias} = 0.50 \cdot \text{Supply} + 0.30 \cdot \text{Geopolitics} + 0.20 \cdot \text{Risk}$
*   **Equities / General Stocks**:
    $\text{raw bias} = 0.35 \cdot \text{Risk} + 0.30 \cdot (1 - \text{Monetary}) + 0.20 \cdot \text{Sentiment} + 0.15 \cdot (1 - \text{Geopolitics})$

The raw bias is mapped linearly into a drift multiplier:

$$
\delta = 0.90 + \text{raw bias} \cdot 0.20
$$

This locks the drift multiplier strictly within $[0.90, 1.10]$, allowing the CEO Layer to bias the expected return up or down by a maximum of 10% based on qualitative macroeconomic developments. If the Gemini API call fails, the engine triggers a Zero-Vector Fallback, setting the bias vector to a neutral 0.5 and $\delta$ to 1.0, reverting to pure quantitative inference.