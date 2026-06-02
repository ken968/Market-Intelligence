# [VISION] Market Intelligence Terminal v2.0
**Beyond Prediction: Moving to Agentic Causal Intelligence**

## 1. Pillars of the Grand Vision

### A. Bayesian Causal Inference (Probabilistic CEO Layer)
- **Current**: CEO Layer generates a fixed bias vector and a drift multiplier.
- **Upgrade**: Implement a **Hierarchical Bayesian model** to represent the Causal Chain. Instead of a single number, the CEO Layer provides a *probability distribution* of market regimes.
- **Why**: This allows the system to express "I am 70% sure of a dovish shift, but there's a 30% tail risk of a hawkish spike," allowing the Manager Layer to adjust risk management (Kelly sizing) more dynamically.

### B. Autonomous Scenario Stress-Testing ("What-If" Agent)
- **Concept**: An agentic wrapper that autonomously generates "Black Swan" and "Grey Swan" scenarios based on current geopolitical tensions.
- **Feature**: A "Causal Simulator" UI. You (the user) can input: *"What if the Fed holds rates but oil spikes to $120 due to Suez closure?"*
- **Outcome**: The system propagates this shock through the features, predicting the re-alignment of Gold, BTC, and S&P 500 under that specific causal path.

### C. Financial Knowledge Graph (KG) Integration
- **Concept**: Build a Knowledge Graph connecting macro entities (Fed, NFP, OPEC, ECB, Tech Earnings).
- **Benefit**: Instead of treating features like DXY and Yield as independent, the system understands their underlying structural relationships. If "NFP is high", the KG knows this pressures "Fed Hawkishness" which increases "Yields" and "DXY", eventually depressing "Gold".

---

## 2. Infrastructure Refinement (Production Hardening)

### D. Agentic RL for Position Sizing
- Use **Reinforcement Learning** to optimize the `Half-Kelly` multiplier. The RL agent learns from the `model_monitor.py` data to be more aggressive in "High-Conviction" regimes and more defensive when "OOD Anomaly Gate" is frequent.

### E. Unified Observability Dashboard
- A dedicated page to monitor the "Health" of every data connection (FRED, NewsAPI, CME Scraper) in real-time, with automatic alerts to Telegram/Discord.

---

## 3. Initial Setup Revision (Bootstrap Guide)

To address the "Empty Data" issue after cloning:

1. **Bootstrap Script**: Create `scripts/setup_system.py` which:
    - Verifies all entries in `.env`.
    - Automatically runs `fred_fetcher.py`, `cot_fetcher.py`, and `data_fetcher_v2.py`.
    - Triggers the full training pipeline for all assets.
2. **Mock-to-Prod Toggle**: A global configuration to switch from "Mock Data" (for local testing) to "Real API Data" once keys are provided.
