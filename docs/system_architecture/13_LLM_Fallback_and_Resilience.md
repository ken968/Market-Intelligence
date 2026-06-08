# 13. Multi-LLM Fallback and System Resilience

## 1. Overview
The Market Intelligence system relies heavily on Large Language Models (LLMs) via the `CEO Layer` to synthesize quantitative signals, perform regime classification, and generate explainable narratives. However, third-party APIs are subject to rate limits, server outages, and deprecations. To guarantee 100% uptime for the automated forecasting pipeline, the system employs a Multi-LLM Hierarchical Fallback mechanism.

## 2. Hierarchical Fallback Strategy

The system evaluates and routes requests through a cascading list of providers until a successful response is generated:

### Level 1: Primary Provider (Google Gemini)
- **Models Used:** `gemini-pro-latest`, `gemini-1.5-pro`
- **Why:** Best-in-class context window and robust reasoning for complex financial data.
- **Failover Trigger:** Rate limits (HTTP 429), API exhaustion, or Server Errors (HTTP 500+).

### Level 2: Secondary Provider (Deepseek via Groq / OpenRouter)
- **Models Used:** `Deepseek R1`
- **Why:** High reasoning capability at significantly lower latency. Groq provides ultra-fast inference, while OpenRouter serves as an infrastructural backup if Groq is down.

### Level 3: Tertiary Provider (Meta Llama via Groq / OpenRouter)
- **Models Used:** `Llama 3.3 70B`
- **Why:** Open-weights standard with excellent instruction-following capabilities. Used only if both Gemini and Deepseek pipelines fail.

## 3. Zero-Vector Injection (Failsafe)
If the entire API ecosystem is unreachable (e.g., loss of external internet connection, or catastrophic multi-provider outage), the system triggers the **Zero-Vector Failsafe**.

### How it works:
Instead of halting execution and leaving the UI empty, the system bypasses the CEO Layer entirely. It injects a `0.0` matrix into the macroeconomic bias vector. The inference engine then falls back strictly to the **Quantitative Layer** (LSTM + XGBoost Stacker).
- **Effect:** The forecast is generated purely based on historical price action and technical indicators, without any qualitative macro-economic adjustment. The UI displays a warning that the model is running in "Offline Quant Mode."

## 4. Normalization Layer (Z-Score Calibration)
Because different LLMs have different "personalities" (e.g., Deepseek might consistently score bullish sentiment higher than Gemini), the system cannot feed raw LLM scores into the LSTM.
- **Per-Model Calibration:** The system tracks the historical mean and standard deviation of each model's output.
- **Rescaling:** Before injection, the raw score is normalized into a standard Z-score specific to the provider that generated it. This ensures that an LLM failover event does not cause a sudden, artificial jump in the predicted price path.
