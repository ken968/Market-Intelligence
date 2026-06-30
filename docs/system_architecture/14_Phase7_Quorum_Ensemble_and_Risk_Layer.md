# 14. Phase 7: Quorum Ensemble and Risk Layer

## Overview

The Market Intelligence system has evolved to **Phase 7**, shifting from a static Manager Stacker (Ridge/Huber Regressor) to a dynamic **Quorum Inference Ensemble** managed by the **Risk Layer**. This architecture fundamentally addresses non-stationarity in financial time-series by using continuous rolling windows and uncertainty-based weighting.

## 1. Dual-Model Architecture

Instead of relying on a single LSTM or XGBoost model, Phase 7 trains two distinct model groups for each asset:

*   **Model A:** Evaluates the market from a fundamental/macro perspective.
*   **Model B:** Evaluates the market focusing on short-term price action and volatility dynamics.

These models are trained across **multiple chronological rolling windows** (W1, W2, W3...).

## 2. Model Registry & Out-of-Sample Selection

The core of the ensemble is the `model_registry.json`. During the training phase (`train_lstm_pct.py`), the system evaluates every model window on an Out-Of-Sample (OOS) validation set.
The system automatically tags the model window with the best Information Coefficient (IC) as `is_best_window = True`.

During live inference, the Predictor Engine only loads the "best" window for Model A and Model B.

## 3. Risk Layer: Quorum Inference

The `RiskLayer` (`utils/layers/risk_layer.py`) replaces the legacy Manager Layer. It performs **Quorum Inference** by blending the predictions of Model A and Model B.

### EWMA Information Coefficient (IC) Weighting
Models are not weighted equally. The Risk Layer looks at their historical out-of-sample IC scores.
*   Model with higher IC gets more weight.
*   If both models agree (Quorum), the signal is reinforced.
*   If they disagree, the resulting prediction is muted, reflecting uncertainty.

### Epistemic and Aleatoric Uncertainty
The Risk Layer calculates uncertainty through two vectors:
1.  **Epistemic Uncertainty:** Measured via Monte Carlo (MC) Dropout during inference. If the model is unsure about the current data distribution, dropout passes will have high variance.
2.  **Cross-Window Disagreement:** The variance between Model A and Model B's predictions.

Total Uncertainty = $\sqrt{\text{Epistemic}^2 + \text{Cross-Window}^2}$

## 4. Kelly Criterion Shrinkage

Financial forecasting must account for risk. The Risk Layer uses the calculated total uncertainty to apply **Kelly Shrinkage**.
*   A base Kelly fraction (e.g., 0.25 or 25% max allocation) is defined.
*   As Total Uncertainty increases, the Kelly fraction is severely shrunk (penalized).
*   This directly reduces the reported "Confidence Score" in the UI. A 90% directional probability with massive uncertainty will result in a "Low" confidence rating.

## 5. 90-Day Monte Carlo Fan Charts

The system outputs 5 distinct horizons natively: **1 Day, 1 Week, 2 Weeks, 1 Month, and 3 Months (90 Days).**

To visualize this, the Predictor Engine constructs a **90-Day Probability Cloud (Fan Chart)**:
1.  The 5 discrete prediction points are interpolated to form a continuous 90-day base trajectory.
2.  The LLM (CEO Layer) narrative bias is applied to create the "Contextual Trajectory".
3.  **Real-Time Volatility Injection:** The system fetches live VIX (for equities/gold) or DVOL (for BTC).
4.  500 Monte Carlo random walks are generated around the contextual trajectory, using the real-time volatility as the daily standard deviation.
5.  The 10th and 90th percentiles of these walks form the Fan Chart bounds, providing a realistic visualization of potential market variance over the next quarter.
