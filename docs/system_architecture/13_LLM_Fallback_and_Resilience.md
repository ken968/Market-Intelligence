# 13. Multi-LLM Fallback and System Resilience
Status: FINAL TECHNICAL SPECIFICATION
Date: 2026-06-08

## 1. Overview
The system cannot run optimally without the CEO Layer (Contextual Narrative Injection) which relies on Large Language Models (LLM). However, third-party APIs have probabilistic reliability (Rate Limits, HTTP 502 Bad Gateway). This document specifies the *State Machine* for a multi-layered *Failover* system that guarantees 99.9% uptime SLA on the inference algorithm.

## 2. API Cascade State Machine
If the `analyze_news_context()` function in `utils/llm_manager.py` encounters a network exception or invalid JSON format, it will perform an exponential cascade:

### 2.1. Node 1: Gemini 2.0 Flash / Pro
- **Endpoint**: `google.generativeai.GenerativeModel('gemini-2.0-flash')`
- **Purpose**: Massive *context window* and best financial reasoning capabilities (Primary Router).
- **Retry Mechanism**: $WaitTime = \min(2^k, 8)$ seconds, max 3 retries (HTTP 429). If it fails, the system switches to Node 2.

### 2.2. Node 2: Deepseek R1 (via Groq & OpenRouter)
- **Endpoint**: `groq.client.chat.completions.create(model='deepseek-r1-distill-llama-70b')`
- **Purpose**: Extremely low *inference latency* (~300 tok/sec) via Groq TPU/LPU. If Groq is *down*, the system rotates to the OpenRouter *base_path* URL with the `Bearer $OPENROUTER_KEY` credential header.
- **Output Characteristics**: Requires specific *Regex stripping* to remove `<think>` tags from the Chain-of-Thought before JSON extraction.

### 2.3. Node 3: Meta Llama 3.3 70B
- **Endpoint**: Same as above (Groq/OpenRouter).
- **Purpose**: The best *open-weights* standard. This model strictly adheres to pure JSON format (`response_format={"type": "json_object"}`), even though its macro reasoning capabilities are slightly below Gemini/Deepseek.

## 3. Quorum Normalization Layer (Cross-LLM Calibration)
Due to differences in neural network topologies, LLM A might always output an average volatility value of 0.6 while LLM B averages 0.4.

### Z-Score Calibration Algorithm per Provider
The system maintains a *State Dictionary* in DuckDB for each LLM provider:
$$ \mu_{LLM\_A}, \sigma_{LLM\_A} $$

When LLM $i$ responds with a raw value $V_i$, the value is calibrated (Z-Score) before being converted into a multiplier.
$$ Z_{val} = \frac{V_i - \mu_{LLM\_i}}{\sigma_{LLM\_i}} $$
It is then standardized back to the anchor distribution (e.g., Gemini). This prevents discontinuities (*artificial jumps*) in tomorrow's prediction line purely because the responding model changed from Gemini to Llama.

## 4. Zero-Vector Injection (Offline Quant Failsafe)
Extreme condition: None of the Nodes (1-3) return a response (Total internet blackout or no valid keys).
- **Trigger**: The final catcher in the `try/except Exception as e` block.
- **Injection Routine**:
  1. The `is_fallback = True` variable is flagged to the UI.
  2. The function returns a mock dictionary:
     ```python
     {
         'narrative': 'SYSTEM OFFLINE - RUNNING IN PURE QUANTITATIVE MODE',
         'drift_multiplier': 1.0,  # Identity scalar, does not change target value
         'bias_vector': np.zeros(5), # Zero vector, eliminates contextual bias
         'confidence': 0.5
     }
     ```
- **Inference Mathematics**: The output from the *Manager Layer* (Ridge Stacker), which should be $\hat{Y}_{final} = \hat{Y}_{stacker} \times Drift$, now becomes $\hat{Y}_{final} = \hat{Y}_{stacker} \times 1.0$.
- **Impact**: The system continues to generate forecasts 100% on time based on pure historical price data (LSTM) and local quantitative sentiment (FinBERT), without requiring external narrative intelligence.
