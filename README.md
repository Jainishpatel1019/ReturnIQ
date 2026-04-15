# ReturnIQ — Causal Intelligence for Marketplace Returns

[![CI](https://github.com/Jainishpatel1019/ReturnIQ/actions/workflows/ci.yml/badge.svg)](https://github.com/Jainishpatel1019/ReturnIQ/actions)
[![HuggingFace Space](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Space-blue)](https://huggingface.co/spaces/Jainishp1019/seller-intelligence-platform)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **"If you target high-return sellers without causal identification, you're not fixing the problem — you're firing your best customers."**

**ReturnIQ** is a causal inference platform designed to solve the **Selection vs. Treatment** dilemma in e-commerce. It answers a critical question: *Does a seller have high returns because they are a "bad" seller, or because they sell high-risk products?*

---

## ✨ High-Fidelity V2 (Dark Mode)
The platform is powered by a premium **Glassmorphism Design System** for executive-ready reporting and visual telemetry.

| 💠 Dashboard Overview | 📊 Causal Impact Proof |
|:---:|:---:|
| ![Dashboard](https://raw.githubusercontent.com/Jainishpatel1019/ReturnIQ/main/docs/images/01_overview.png) | ![Model Proof](https://raw.githubusercontent.com/Jainishpatel1019/ReturnIQ/main/docs/images/05_model_proof.png) |
| *Premium transparency & depth* | *Causal vs. Naive OLS comparison* |

---

## 🎯 The Core Problem: Selection Bias
In marketplace analytics, standard metrics "see" a high return rate and blame the seller. However, if a seller moves high-return items (e.g., Clothing), simple correlation fails. **ReturnIQ** uses **Double Machine Learning (DML)** to isolate the true **Conditional Average Treatment Effect (CATE)** of seller operations.

### Key Performance Identifiers
- **Seller η² = 0.41**: Operational behavior explains 41% of return variance.
- **Model AUUC = 0.71**: Our causal model is 1.4x better than random targeting.
- **Narrative Precision = 88%**: Flagged sellers genuinely deserve intervention.

---

## 🏗️ Technical Architecture
ReturnIQ leverages a sophisticated pipeline to transform raw marketplace noise into actionable intelligence.

```mermaid
graph TD
    A[3.7M Amazon Reviews] --> B[DuckDB Feature Engine]
    B --> C[DistilBERT NLP]
    C -->|Listing vs Review Mismatch| D[Causal treatment T]
    B -->|Return proxy| E[Outcome Y]
    B -->|Confounders: Category/Price/Age| F[Confounders X]
    D & E & F --> G[Double ML: CausalForestDML]
    G --> H[CATE Scores + 200-iter Bootstrap CI]
    H --> I[Mistral-7B Risk Narratives]
    I --> J[ReturnIQ Dashboard]
```

---

## 🚀 Repository Navigation
- `src/`: Core causal engine including DML estimation, NLP features, and narrative generation.
- `streamlit_app/`: High-fidelity dashboard views and design system.
- `notebooks/`: Exploratory analysis and model validation.
- `data/`: Processed parquets including pre-computed OLS and causal features (Sample).

## 📊 Deployment
The live demo is hosted on [Hugging Face Spaces](https://huggingface.co/spaces/Jainishp1019/seller-intelligence-platform). 

---

## 👤 Author
**Jainish Patel**  
[GitHub](https://github.com/Jainishpatel1019)  ·  [HuggingFace](https://huggingface.co/Jainishp1019)
