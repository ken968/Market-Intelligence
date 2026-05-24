# MARKET INTELLIGENCE SYSTEM: COMPREHENSIVE ARCHITECTURE & RESEARCH WHITEPAPER

## PART 8: MULTI-FACTOR TRADING SIGNAL GENERATION & RISK CALIBRATION

### 1. CONSOLIDATED SIGNAL METRICS

The `SignalGenerator` class synthesizes quantitative model outputs, macroeconomic indicators, sentiment indexes, and technical indicators into a single, actionable trading signal (BUY, SELL, or HOLD). Financial models that rely only on a single signal source are vulnerable to failure when market conditions shift. By combining multiple independent factors, the signal generator ensures that the final trading recommendation is supported by both macro regimes and micro price action.

The output dictionary returned by `generate_signal()` contains:

*   **Consolidated Signal**: BUY, SELL, or HOLD.
*   **Confidence**: A float score between 0.0 and 1.0.
*   **Reasons**: A list of qualitative and quantitative attributions explaining the decision.
*   **Factors**: Sub-scores for each input category.
*   **Risk Parameters**: Entry price, target price, and stop-loss levels.

### 2. MULTI-FACTOR WEIGHTING MATRIX

The signal generator scores and aggregates four categories of market data:

#### A. AI Forecast Score (40% Weight)

The primary driver of the signal is the 1-week expected return from the Dual-Head Stacker. The raw percent change is normalized to a 0-1 scale:

$$ \text{Score} = \frac{\text{Abs}(\text{Pct\_Change\_7D})}{10} $$

If the predicted return is positive, it contributes to a BUY score; if negative, to a SELL score. The confidence of this factor is scaled by the dynamic hit ratio retrieved from backtest reports.

#### B. Macroeconomic Regime Score (25% to 30% Weight)

This factor evaluates macro-liquidity indicators.

*   **Gold**: Evaluates DXY, VIX, and CME FedWatch decisions. DXY < 105 and VIX > 15 are bullish; DXY > 108 and VIX < 12 are bearish.
*   **Bitcoin**: Evaluates DXY and Fed policy. DXY < 104 is bullish; DXY > 107 is bearish.
*   **Equities**: Evaluates VIX and interest rates. VIX < 18 is bullish; VIX > 25 is bearish.

#### C. Alternative Sentiment Score (20% Weight)

This factor combines retail interest and fear indexes.

*   **Google Trends**: The slope of search volume for key terms (e.g., "buy gold", "bitcoin price"). A rising trend slope contributes to bullish sentiment.
*   **Fear & Greed Index**: Ingested daily. Values below 20 (extreme fear) are treated contrarian-bullish, while values above 80 (extreme greed) are treated contrarian-bearish.

#### D. Technical Price Action Score (15% to 20% Weight)

This factor evaluates standard momentum indicators:

*   **EMA_90 Crossover**: Close price above the 90-day Exponential Moving Average indicates a bullish trend.
*   **RSI_14**: Relative Strength Index values below 30 are oversold (bullish trigger), while values above 70 are overbought (bearish trigger).

### 3. INTEGRATION OF ALTERNATIVE SOURCES

The signal engine integrates two specialized fetchers to enrich its alternative data scores:

#### A. CME Fed Watch Rate Probability Parser

The `FedWatchFetcher` scrapes CME Group futures pricing to extract target interest rate probabilities for upcoming FOMC meetings.

*   **Dovish Score**: Sum of probabilities for interest rate cuts or halts.
*   **Hawkish Score**: Sum of probabilities for interest rate hikes.

If the Dovish Score exceeds 60%, the engine registers a dovish stance. This stance is mapped directly as:

*   **Gold**: **BULLISH** (Lower interest rates reduce non-yielding asset opportunity cost).
*   **Bitcoin**: **BULLISH** (Easier financial conditions fuel risk asset appreciation).
*   **Stocks**: **BULLISH** (Lower discount rates expand DCF multiples, except during recessions).

#### B. Google Trends Search Interest Normalization

The `GoogleTrendsFetcher` queries search volume indexes. Because raw interest index values are noisy, the fetcher calculates the rolling change:

$$ \text{Slope} = \frac{\text{Interest}_t - \text{Mean}(\text{Interest}_{t-7:t})}{\text{Std}(\text{Interest}_{t-7:t})} $$

A positive slope indicates accelerating public retail interest, signaling a momentum expansion.

### 4. RISK PARAMETERS CALIBRATION

When a BUY or SELL signal is triggered, the engine calculates risk thresholds:

*   **Entry Price**: Set to the latest actual closing price ($t=0$).
*   **Target Price**: Derived from the 7-day expected return from the Dual-Head Stacker:

    $$ \text{Target\_Price} = \text{Entry\_Price} \times (1 + \text{Stacker\_Return\_7D}) $$

*   **Stop-Loss**: Computed dynamically based on Garman-Klass volatility ($\sigma_{\text{GK}}$). The stop-loss is placed at a distance proportional to two standard deviations of 7-day price movement:

    $$ \text{Stop\_Loss\_Distance} = \text{Entry\_Price} \times (2 \times \sigma_{\text{GK}} \times \sqrt{7/252}) $$

    For BUY signals:
    $$ \text{Stop\_Loss} = \text{Entry\_Price} - \text{Stop\_Loss\_Distance} $$

    For SELL signals:
    $$ \text{Stop\_Loss} = \text{Entry\_Price} + \text{Stop\_Loss\_Distance} $$

This ensures that stop-loss boundaries expand during volatile regimes and tighten during quiet regimes, protecting capital from noise.