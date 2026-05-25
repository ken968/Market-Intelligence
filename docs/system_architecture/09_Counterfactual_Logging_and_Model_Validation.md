# PART 9: COUNTERFACTUAL BACKTESTING & MODEL ACCURACY SCORES

## 1. THE PRINCIPLE OF COUNTERFACTUAL TESTING
In machine learning systems that incorporate human inputs or LLM speculative overlays, evaluating model performance is challenging. If a model performs well, it is difficult to determine if the accuracy was driven by the underlying quantitative features or by the LLM override. Conversely, if the model performs poorly, it is unclear if the AI override corrupted an otherwise accurate base model.

The system resolves this through **Counterfactual Testing**. During daily inference, the engine calculates and logs two separate forecasts simultaneously:
- **Baseline Forecast**: The raw prediction from the base models and the Dual-Head Stacker (Worker and Manager layers only, no LLM bias).
- **Contextual Forecast**: The final prediction containing the CEO Layer's drift multiplier and bias vector override.

By comparing the performance of these two parallel trajectories over time, the system can statistically prove whether the **CEO Layer** adds value (improves the **Hit Ratio**) or acts as noise.

## 2. PARALLEL FORECAST LOGGING PIPELINE
The `CounterfactualLogger` class manages the persistence of forecast records.
- **Storage**: Records are appended as JSON lines to the file `data/counterfactual_log.jsonl`.
- **Duplicate Prevention**: Streamlit runs in a stateless, reactive loop; reloading the page or switching tabs re-runs the entire prediction script. To prevent writing duplicate records for the same day, the logger checks the existing log lines. If a record matches the same asset, forecast date, and forecast horizon steps, the logger overwrites the existing entry in-place instead of appending a new line.

A logged record contains the following schema:
- `logged_at`: UTC timestamp.
- `forecast_date`: The date index (YYYY-MM-DD) when the forecast was executed.
- `asset`: The target asset key.
- `steps`: The forecast horizon (e.g., 7 days).
- `baseline_final`: The predicted price at the horizon endpoint without LLM bias.
- `contextual_final`: The predicted price at the horizon endpoint with LLM bias.
- `baseline_series`: The daily 90-day baseline price trajectory.
- `contextual_series`: The daily 90-day contextual price trajectory.
- `llm_scores`: The raw score vector returned by the Gemini API.
- `actual_price`: Placeholder for the realized close price (resolved later).
- `baseline_hit`: Placeholder for directional accuracy (resolved later).
- `contextual_hit`: Placeholder for directional accuracy (resolved later).

## 3. AUTOMATIC OUTCOME RESOLUTION
During the daily data synchronization process, the system automatically executes `auto_resolve_all_outcomes()`.
- The function scans the log file to identify any unresolved records (where `actual_price` is `null`).
- It calculates the target date:
  $$ \text{Target Date} = \text{Forecast Date} + \text{Steps} $$
- It checks if the target date is present in the updated historical database. If the target date has occurred, the engine retrieves the realized close price from the database:
  $$ P_{\text{actual}} = \text{Close price at Target Date} $$
- The starting price ($P_0$) is extracted from the first element of the logged baseline series. The directional changes are evaluated:
  $$ \text{Actual Direction} = P_{\text{actual}} > P_0 $$
  $$ \text{Baseline Direction} = P_{\text{baseline final}} > P_0 $$
  $$ \text{Contextual Direction} = P_{\text{contextual final}} > P_0 $$
- The hit results are resolved:
  $$ \text{Baseline Hit} = ( \text{Baseline Direction} == \text{Actual Direction} ) $$
  $$ \text{Contextual Hit} = ( \text{Contextual Direction} == \text{Actual Direction} ) $$
The record is updated, and the results are written back to `counterfactual_log.jsonl`.

Once resolved records accumulate, the dashboard displays the comparative **Hit Ratio**:
$$ \text{Hit Ratio} = \frac{\text{Hits}}{\text{Total Resolved}} \times 100 $$
The delta (Contextual Hit Ratio minus Baseline Hit Ratio) confirms the value-add of the **CEO Layer**.

## 4. WALK-FORWARD BACKTEST ENGINE FRAMEWORK
Prior to production deployment, models are evaluated using a historical backtest framework in `scripts/backtest_engine.py`.
- **Chronological Split**: Traditional cross-validation randomizes samples, which introduces look-ahead leakage in time series. The backtest engine enforces a chronological 80/20 train/test split. The model is trained on the first 80% of the historical timeline and evaluated on the final 20% of unseen data.
- **Performance Metrics**: The engine calculates the directional accuracy (Hit Ratio) and **Root Mean Squared Error (RMSE)** in percent change space:
  $$ \text{RMSE} = \sqrt{ \text{Mean}( ( y_{\text{realized}} - y_{\text{predicted}} )^2 ) } $$
- **Stacker vs Base Performance**: The backtest engine evaluates the LSTM, XGBoost, and the Stacker Ensemble separately, saving the comparative metrics to `reports/stacker_{asset}_backtest.json`.