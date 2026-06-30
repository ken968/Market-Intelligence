# 12. Sentiment and NLP Engine Architecture
Status: FINAL TECHNICAL SPECIFICATION
Date: 2026-06-08

## 1. Overview
The *Sentiment and NLP Engine* module is responsible for converting unstructured text data (macro news, RSS feeds, X/Twitter) into *structured quantitative tensors* ready to be injected into the LSTM and XGBoost Stacker. This module has been migrated from a *rule-based lexicon* (VADER/TextBlob) to a Deep Learning architecture (Transformer).

## 2. FinBERT Core Architecture
The system does not use generic NLP models. It implements `ProsusAI/finbert`, a derivative of the BERT (Bidirectional Encoder Representations from Transformers) architecture specifically fine-tuned on the TRC2 (Thomson Reuters Text Research Collection) and Financial PhraseBank datasets.

### 2.1. Tokenization & Forward Pass
- **Input**: A collection of headlines `H = {h_1, h_2, ..., h_n}`
- **Tokenization**: BertTokenizer(vocab_size=30522) splits the text into token ids, attention masks, and token type ids.
- **Model Output**: The final layer logits (Dense) produce a vector $[l_p, l_{neg}, l_{neu}]$ for each headline.
- **Softmax Activation**: Probabilities are mapped as:
  $$ P(C=c_i | h) = \frac{\exp(l_i)}{\sum_{j} \exp(l_j)} $$
  Where $C \in \{Positive, Negative, Neutral\}$.

### 2.2. Composite Sentiment Formulation
Aggregate probabilities are calculated into a single scalar value $\mu_S \in [-1, 1]$:
$$ \mu_{S, t} = \frac{1}{N} \sum_{i=1}^{N} \left[ P(Positive | h_i) - P(Negative | h_i) \right] $$
Primary advantage: Headlines classified as *Neutral* with high probability will not drastically affect the expected value.

## 3. Z-Score Normalization (Regime Contextualization)
The raw $\mu_{S, t}$ values suffer from autocorrelation issues (common in *bear market* regimes where news is persistently negative). Machine Learning models do not require absolute values, but rather divergence from the trend.

### 3.1. Rolling Z-Score Computation
$$ Z_{S, t} = \frac{\mu_{S, t} - E_{30}[\mu_S]}{\sqrt{Var_{30}(\mu_S) + \epsilon}} $$
- $E_{30}$ is the 30-day *moving average*.
- $\epsilon = 1e-6$ to prevent *division by zero* during market holidays.
- **Output**: Vector $Z_S \sim N(0, 1)$. A value of $|Z_S| > 1.96$ indicates a *statistically significant sentiment shock* (95% CI).

## 4. Sentiment Volatility (Uncertainty Proxy)
Instead of discarding the variance from the daily aggregation, the system extracts the intra-day sentiment standard deviation as an independent volatility signal:
$$ \sigma_{S, t} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (S_i - \mu_{S, t})^2} $$
- **Purpose**: A high $\sigma_{S, t}$ indicates that the day contains highly polarized news (e.g., "Fed raises interest rates" vs "Unemployment data improves").
- **Injection**: $\sigma_{S, t}$ is used as an independent feature in `manager_anchor.py` to widen the *confidence interval* of the classification probabilities.

## 5. Tensor Injection Workflow
- In `scripts/sentiment_fetcher_v2.py`: Text is fetched, inferred via CPU/GPU, aggregated, and saved as `Sentiment` and `Sentiment_Std` in the DuckDB `macro_indicators` table.
- In `utils/layers/worker_lstm.py`: This vector enters as the input matrix $X_t$, where the `MultiHeadAttention` layer calculates the Attention Score between $Z_{S, t}$ and *Price Momentum*.
