# ReturnIQ — Causal Intelligence for Marketplace Returns

[![CI](https://github.com/Jainishpatel1019/ReturnIQ/actions/workflows/ci.yml/badge.svg)](https://github.com/Jainishpatel1019/ReturnIQ/actions)
[![HuggingFace Space](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Space-blue)](https://huggingface.co/spaces/Jainishp1019/seller-intelligence-platform)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

**Why standard analytics fails:** Most platforms treat return rates as a simple performance metric. But if a seller moves high-risk goods (e.g., Electronics), they're penalized for their category, not their conduct.

**ReturnIQ** is a causal-first platform that uses **Double Machine Learning (DML)** to isolate the true **Conditional Average Treatment Effect (CATE)** of seller operations, de-biasing findings from market-level noise.

---

## 🏗️ Production Architecture
*This project implements a decoupled "Offline-to-Online" architecture used in real-world ML engineering.*

1.  **Causal Engine (Offline)**:
    *   **Data Scale**: Processed 3.7M Amazon reviews at scale using a DuckDB analytical warehouse.
    *   **Feature Engineering**: Natural Language Processing (DistilBERT) to measure listing-to-review semantic alignment.
    *   **Estimation**: Executed `CausalForestDML` (EconML) with 200-iteration bootstrap for high-dimensional de-biasing.
    *   **Reasoning**: Batch-precomputed executive summaries for 1,000+ sellers using Mistral-7B.

2.  **Telemetry Dashboard (Live Demo)**:
    *   **Capability**: High-fidelity glassmorphism interface for executive reporting.
    *   **Data**: Navigates the finalized causal intelligence results for a representative 5,000-seller sample.
    *   **Sync**: The live demo provides zero-latency access to pre-computed causal rankings and SHAP explanations.

---

## 📊 Key Results
- **Operational Lift**: Operations drive **41%** of return rate variance in our sample.
- **Accuracy**: 1.4x enhancement in risk ranking precision compared to random choice.
- **Rigor**: Validated via Placebo Shuffle tests ($p \approx 0.38$) and AUUC ranking benchmarks.

---

## 🚀 Repository Navigation
- `src/`: The heart of the platform. Causal estimation, NLP pipelines, and variance decomposition logic.
- `streamlit_app/`: The visual interface and Glassmorphism design system.
- `tests/`: Integrated test suite for statistical range check and data integrity.

## 👤 Author
**Jainish Patel**  
[GitHub](https://github.com/Jainishpatel1019)  ·  [HuggingFace](https://huggingface.co/Jainishp1019)
