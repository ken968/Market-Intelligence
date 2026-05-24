# PART 4: ENSEMBLE MODELING & DUAL-HEAD META-LEARNING ARCHITECTURE

## 1. The Fallacy of Recursive Prediction in Financial Time Series
A major pitfall in time-series forecasting is the reliance on Recursive Rollout models. In a recursive architecture, a single model is trained to predict the return at step `t+1`. To generate predictions for longer horizons (e.g., `t+2` to `t+90`), the model's output at `t+1` is appended to the input sequence, and the model is run iteratively.

While mathematically simple, this approach fails in high-noise financial regimes due to:

### A. Exponential Error Propagation
Any prediction error at step `t+1` acts as an artificial input anomaly for step `t+2`. The forecast error compounds exponentially with the prediction horizon:
$$ Error_t = O(e^t) $$

### B. Convergence to the Mean
Because financial time series have a low signal-to-noise ratio, recursive feedforward loops quickly lose variance. The predicted trajectory dampens and flatlines, converging directly to the historical mean.

To resolve this, the system implements a Direct Multi-Step Forecasting paradigm. The engine trains five independent, non-recursive models for each asset, each optimized for a specific target horizon: 1 Day, 7 Days, 14 Days, 30 Days, and 90 Days. Each model directly maps the input feature matrix at time `t` to the expected return at time `t+H`:
$$ y_{t+H} = f_H(X_t) $$
This eliminates recursive error propagation and preserves trajectory variance.

## 2. Base Learners Design
The Worker Layer uses two independent models to capture different dimensional structures of financial data:

### A. Worker 1: LSTM with Multi-Head Self-Attention
The LSTM (Long Short-Term Memory) recurrent network is designed to process temporal sequence data.
*   **Input Formatting**: The model processes sequences of length `T = 60` (Gold and equities) or `T = 90` (Bitcoin) containing daily closing prices, volumes, and technical features.
*   **Sequence Processing**: The network architecture contains stacked LSTM layers (units: `[64, 32]` for Gold; `[128, 64, 32]` for Bitcoin) with L2 activity regularization (`0.001`) to prevent overfitting.
*   **Multi-Head Self-Attention**: For volatile assets (such as Bitcoin, NVIDIA, and Tesla), a Multi-Head Self-Attention layer is placed after the first LSTM layer. This mechanism computes attention weights across the time steps:
    $$ Attention(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) * V $$
    This allows the network to learn temporal dependencies across specific historical events (e.g., matching a liquidity event 45 days ago to today's price action) rather than relying only on the final state vector.

### B. Worker 2: Extreme Gradient Boosting (XGBoost)
XGBoost is a decision-tree ensemble based on gradient boosting algorithms.
*   **Tabular Modeling**: While LSTM learns sequence-dependent relationships, XGBoost models tabular macro-regime interactions. It builds sequential trees where each tree corrects the residual errors of the previous ones.
*   **Regularized Split Selection**: The model optimizes an objective function containing L1 regularization (`reg_alpha`) and L2 regularization (`reg_lambda`) to penalize tree complexity:
    $$ Obj = \sum_i L(y_i, \hat{y}_i) + \sum_j \left( \gamma \cdot T_j + 0.5 \cdot \lambda \cdot w_j^2 + \alpha \cdot w_j \right) $$
    Splits are chosen to maximize the gradient/hessian-based gain, rather than simple information gain. Asset-specific configurations (e.g., Gold: `max_depth=5`, `reg_alpha=0.05`, `reg_lambda=0.5`; Bitcoin: `max_depth=4`, `reg_alpha=0.1`, `reg_lambda=1.0`) are tuned to prevent overfitting to historical macroeconomic cycles.

## 3. Target Standardization and Scaler Separation
Financial assets have vastly different nominal values and volatilities (e.g., Gold trading at `$2,400/oz` vs Bitcoin at `$60,000`). To ensure model stability, the data preparation pipeline enforces two strict rules:

### A. Feature vs Target Scaler Isolation
Features and targets are scaled independently. Input features are normalized using `MinMaxScaler` to map values into `[0, 1]`, preserving relative feature structures. Targets (expected returns) are standardized using `StandardScaler` to have zero mean and unit variance.

### B. Target Leakage Prevention
Using a single scaler for both features and targets causes look-ahead leakage. The future target value is present in the feature scaling matrix during fitting. The system saves target scaling parameters in a separate file (`{asset}_scaler_target.pkl`). This ensures that during inference, the raw target return predictions can be inverse-transformed back to percentage returns without leaking future price structures.

## 4. The Dual-Head Stacker Meta-Learner
Traditional ensembles use simple weighted averages to combine model predictions. However, in regime-shifting markets, base model performance is highly variable. The Manager Layer uses a Dual-Head Stacker Meta-Learner (trained on a chronological 20% hold-out validation set) to resolve this:

### A. Meta-Feature Matrix
The input to the Stacker is a meta-feature vector containing:
$$ X_{\text{meta}} = [ \text{lstm\_pred}, \text{xgb\_pred}, \text{VIX}, \text{GK\_Vol\_21d}, \text{Sentiment}, \text{Sentiment\_Std}, \text{YieldCurve\_10Y2Y}, \text{DXY} ] $$
This matrix is normalized using a `Meta StandardScaler` before entering the heads.

### B. Head 1: Direction Head (LogisticRegressionCV)
The Direction Head acts as a probabilistic classifier. It is trained to predict the binary direction of the return (`y_dir = 1` if return `> 0`, else `0`) using L2 penalty selection (cross-validated across `Cs=[0.01, 0.1, 1.0, 10.0, 100.0]`) and 5-fold cross-validation. The model minimizes Binary Cross-Entropy Loss:
$$ Loss = - \frac{1}{N} \sum_i \left[ y_{\text{dir},i} \ln(p_i) + (1 - y_{\text{dir},i}) \ln(1 - p_i) \right] $$
Crucially, the Direction Head sets `class_weight='balanced'` to correct for any historical upward bias in assets, focusing strictly on directional accuracy (Hit Ratio).

### C. Head 2: Magnitude Head (HuberRegressor)
The Magnitude Head predicts the absolute percentage return. Standard ordinary least squares (OLS) regression minimizes mean squared error (MSE), which heavily penalizes large errors. In financial markets, tail-risk events (e.g., sudden market crashes) dominate OLS training, causing the model to overfit to outliers. The Magnitude Head uses a Huber Loss function (`epsilon=1.35`, L2 regularization=`0.001`):
$$ Loss = \begin{cases} 0.5 \cdot (y - \hat{y})^2 & \text{for } |y - \hat{y}| \le \epsilon \\ \epsilon \cdot (|y - \hat{y}| - 0.5 \cdot \epsilon) & \text{for } |y - \hat{y}| > \epsilon \end{cases} $$
This penalizes outliers linearly rather than quadratically, rendering the magnitude estimation robust to market spikes.

### D. Combined Output Synthesis
The predictions of the two heads are combined to generate the final expected return signal:
$$ \text{Final\_Return} = (2 \cdot P(\text{Up}) - 1) \cdot |\text{Expected\_Magnitude}| $$
Where `P(Up)` is the probability output of the Direction Head, and `Expected_Magnitude` is the output of the Huber Regressor. This design resolves "Hesitation Bias" (where conflicting base models cause the ensemble to output a timid near-zero prediction) by forcing a directional choice scaled by historical volatility expectations.