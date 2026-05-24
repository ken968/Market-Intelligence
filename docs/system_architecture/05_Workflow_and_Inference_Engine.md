# MARKET INTELLIGENCE SYSTEM: COMPREHENSIVE ARCHITECTURE & RESEARCH WHITEPAPER

## PART 5: PRODUCTION INFERENCE ENGINE, TRAJECTORY SMOOTHING, & SRE ROADMAP

## 1. DAILY INFERENCE EXECUTION PIPELINE (STATE-MACHINE)
The production inference loop is orchestrated by `utils/predictor_engine.py` and runs on a daily schedule. The execution path follows a strict, sequential state machine to ensure data integrity and prevent runtime crashes:

-   **State 1: Data Ingestion and Validation**
    The engine fetches the latest closing prices (`t=0`) and computes daily technical indicators (e.g., `EMA_90`, realized volatility).
-   **State 2: Macroeconomic and COT Synchronization**
    The engine queries DuckDB OLAP tables to extract historical macroeconomic indicators and COT reports. It joins these datasets and computes rolling features (such as the 3-Year Rolling COT Index and the Smart Money Sentiment Gap) on the full historical dataset to guarantee valid rolling window calculations before extracting the final `t=0` row.
-   **State 3: Base Learner Forward Pass (Worker Inference)**
    The scaled 60-day or 90-day feature window is passed to the LSTM models (Layer 1, Worker 1) to generate sequence predictions. Concurrently, the `t=0` tabular macro vector is passed to the XGBoost regressors (Layer 1, Worker 2) to generate macro predictions.
-   **State 4: Meta-Learner Stacking (Manager Inference)**
    The base model return predictions and live context features (`VIX`, `DXY`, Garman-Klass volatility, sentiment metrics) are assembled into a meta-vector. This vector is standardized and fed into the Dual-Head Stacker to output directional probability and return magnitude.
-   **State 5: CEO Layer Integration (Speculative Override)**
    If news headlines are available and the Gemini API key is valid, the CEO Layer processes the chronologically-sorted, deduplicated news timeline. It outputs a drift multiplier, delta, and a bias vector. The drift multiplier scales the expected return, while the bias vector is logged for explainability.
-   **State 6: Trajectory Generation and Monte Carlo Simulation**
    The final expected returns are projected across a 90-day horizon, and 500 paths are simulated to construct the probability cloud. The results are logged to the database and passed to the Streamlit UI.

## 2. TRAJECTORY GENERATION AND SMOOTHING
A major challenge is displaying a daily 90-day forecast trajectory that is mathematically consistent with the model's direct predictions. The engine solves this through anchor-point interpolation:
-   The engine runs direct predictions for the target horizons $H = [1, 7, 14, 30, 90]$ days using the corresponding horizon models.
-   This yields a set of discrete price points:
    $$y_H = [ \text{Price}_0, \text{Price}_1, \text{Price}_7, \text{Price}_{14}, \text{Price}_{30}, \text{Price}_{90} ]$$
-   To construct a smooth daily path without running ninety individual models (which would introduce noise and horizon decoherence), the engine interpolates these anchor points using a daily evaluation space ($x_{eval} = 1 \text{ to } 90$) via the `numpy.interp` function. This produces a smooth, natural trajectory for display on Streamlit charts.

## 3. POWER-LAW MOMENTUM DECAY
If a specific horizon model (e.g., 90-day LSTM) is missing or fails to load, the engine falls back to projecting the 7-day Dual-Head Stacker prediction across the 90-day trajectory. Simply extrapolating a 7-day expected return linearly (e.g., multiplying a 1% weekly return by 12.8 to get a 90-day return) violates the Efficient Market Hypothesis (EMH), which states that speculative asset returns revert to the mean over longer horizons due to market efficiency and volume exhaustion.

To model this, the engine applies a Power-Law Momentum Decay equation:
$$R_i = R_{anchor} \cdot \left( \frac{i}{7.0} \right)^\alpha$$
Where:
-   $R_i$: Projected cumulative return on day $i$.
-   $R_{anchor}$: The 7-day expected return from the Dual-Head Stacker.
-   $\alpha$: The decay exponent, calibrated to 0.65.
Because $\alpha < 1$, the curve exhibits momentum clustering in the short term (rapid initial return scaling) but experiences asymptotic decay in the long term, simulating the exhaustion of buying or selling pressure.

## 4. GEOMETRIC BROWNIAN MOTION MONTE CARLO SIMULATION
To project probability clouds (P10 to P90 bands) around the expected trajectory, the engine runs 500 simulated paths using Geometric Brownian Motion (GBM):
$$S_{t+1} = S_t \cdot \exp\left( \left(\mu - 0.5 \sigma^2\right) dt + \sigma \sqrt{dt} \cdot \epsilon \right)$$
Where:
-   $S_t$: Asset price at step $t$.
-   $\mu$: Expected drift rate derived from the contextual forecast trajectory.
-   $\sigma$: Daily realized volatility.
-   $\epsilon$: Random variable drawn from a standard normal distribution $N(0, 1)$.
-   $dt$: Time step delta (1/252 for daily trading cycles).

Crucially, rather than relying on historical residual errors (which are backward-looking), the daily volatility $\sigma$ is derived from live options market implied volatility (IV):
-   **BTC**: Sourced from the Deribit Implied Volatility Index (`DVOL`), representing the 30-day forward implied volatility of Bitcoin option contracts.
-   **Gold/Equities**: Sourced from the CBOE Volatility Index (`VIX`), representing the implied volatility of S&P 500 index options.
The annual IV is converted to daily volatility:
$$\sigma_{daily} = \frac{IV_{annual}}{\sqrt{252}}$$
Furthermore, the volatility is dynamically adjusted based on the CEO (LLM) confidence score. If the LLM confidence is high (confidence $>= 0.7$), it signals high narrative clarity, and the engine narrows the probability cloud:
$$\sigma_{adjusted} = \sigma_{daily} \cdot \max(0.50, 1.0 - 0.40 \cdot \text{Confidence}_{CEO})$$
This reduces the width of the Monte Carlo fan chart, visually signaling high-conviction regimes.

## 5. RETRAINING CYCLE AND MODEL HEALTH MONITORING
To detect performance degradation due to regime shifts or model drift, the terminal implements a continuous feedback loop:
-   **Counterfactual Logging**: Every daily prediction series, baseline series, and associated macro feature vector is logged to `counterfactual_log.jsonl`.
-   **Accuracy Evaluation**: The script `scripts/model_monitor.py` runs on a weekly schedule. It matches historical predictions (where the 7-day target date has elapsed) against actual realized market close prices.
-   **Directional Hit Ratio**: The monitor computes the percentage of correct directional predictions over a rolling window of the last 20 predictions.
-   **Health Classification**:
    -   **Healthy** (Hit Ratio $>= 40\%$): The models are performing within normal statistical parameters for financial time series.
    -   **Warning** (Hit Ratio $< 40\%$): Performance is degrading. The Streamlit UI displays a warning banner.
    -   **Degraded** (Hit Ratio $< 35\%$): Severe model degradation. The UI displays an alert and provides the necessary terminal commands to trigger retraining.

## 6. FUTURE ROADMAP (PHASE 6 & 7): CAPITAL ALLOCATION AND RISK MANAGEMENT
The final phase of the Market Intelligence Terminal plans the integration of a capital allocation and risk management layer:

### A. Dynamic Kelly Criterion Position Sizing
The Stacker's Direction Head outputs the raw probability of a positive return, $P(\text{Up})$. This probability will be fed directly into the Kelly Criterion formula to determine optimal portfolio leverage:
$$f^* = P(\text{Up}) - \frac{1 - P(\text{Up})}{B}$$
Where:
-   $f^*$: Optimal fraction of capital to allocate to the trade.
-   $B$: Payout ratio (Expected win magnitude divided by expected loss magnitude).
This transitions the terminal from a directional signaling tool into an active, risk-adjusted portfolio optimization engine.

### B. Volatility-Adjusted Stop-Loss Calibration
Instead of using static percentage stop-losses (e.g., 2% for all trades), the system will calibrate exit thresholds based on the 21-day Garman-Klass realized volatility. High-volatility regimes will automatically widen stop-losses to prevent premature liquidation due to noise, while low-volatility regimes will tighten stops to protect capital, aligning portfolio exposure with real-time liquidity conditions.