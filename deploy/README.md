---
title: ReturnIQ
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# ReturnIQ — Causal Intelligence for Marketplace Returns

**Deployment Status**: Interactive Telemetry Dashboard
This interface provides real-time access to high-fidelity **pre-computed** causal insights. The heavy-duty analytical pipeline (processing 3.7M reviews via EconML & DistilBERT) is executed offline to ensure zero-latency reporting for executive users.

## Key Capabilities
1. **Isolated Impact (CATE)**: Measures the true causal effect of seller quality on returns.
2. **Key Findings**: Five plain-English insights with real numbers from the analysis.
3. **Policy Simulator**: Forecasts how marketplace quality shifts impact prevented returns.
4. **Statistical Proof**: Validates findings using Double ML (XGBoost) and Sensitivity Analysis.
5. **Methodology**: Full identification strategy walkthrough with model card.

## Technical Stack
- **Engine**: EconML / CausalForestDML (Double ML)
- **NLP**: DistilBERT / Mistral-7B
- **Visuals**: Plotly / Streamlit

Built by Jainish Patel.
