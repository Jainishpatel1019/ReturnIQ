# ReturnIQ: Causal Intelligence for Marketplace Integrity

[![Pipeline Status](https://github.com/Jainishpatel1019/ReturnIQ/actions/workflows/ci.yml/badge.svg)](https://github.com/Jainishpatel1019/ReturnIQ/actions)
[![Demo](https://img.shields.io/badge/Demo-HuggingFace-blue)](https://huggingface.co/spaces/Jainishp1019/seller-intelligence-platform)

**ReturnIQ** is a production-grade causal inference terminal designed to isolate the true operational drivers of product returns across high-volume marketplaces. By leveraging **Double Machine Learning (DML)** and NLP-based feature engineering, we move beyond vanilla prediction to identify the *causal* impact of seller quality on customer dissatisfaction.

---

### 1. The Problem
Marketplace return rates are often treated as a prediction task, failing to account for confounding variables (price tiers, seasonality, category bias). A seller might have a high return rate simply because they sell electronics, not because of poor operations. ReturnIQ solves this by estimating the **Conditional Average Treatment Effect (CATE)**.

---

### 2. Live Demo
![ReturnIQ Platform Demo](https://raw.githubusercontent.com/Jainishpatel1019/ReturnIQ/main/assets/demo.gif)
> *Note: If GIF is not visible, visit the [Live HuggingFace Space](https://huggingface.co/spaces/Jainishp1019/seller-intelligence-platform).*

---

### 3. Key Findings (Pipeline Verified)
| Metric | Value | Proof |
| :--- | :--- | :--- |
| **Returns Preventable** | 19.6% | Top 15% risk-targeted reduction |
| **Operational Lift** | 1.31x | Targeting efficiency vs. random |
| **Model AUUC** | 0.345 | Uplift curve validation |
| **Placebo p-value** | 0.38 | PASS: No noise artifacts |

---

### 4. Technical Methodology
We implement a multi-stage **Offline-to-Online** pipeline:
1. **NLP Ground Truth**: DistilBERT-based scorers categorize 3.7M reviews into functional vs. accidental returns.
2. **Causal Estimation**: EconML's `CausalForestDML` isolates the treatment effect (Seller Quality) from 12+ marketplace confounders.
3. **Robustness**: 200-iteration bootstrap for 95% Confidence Intervals + Placebo robustness tests.

---

### 5. How to Reproduce
ReturnIQ is designed for total reproducibility.

```bash
# 1. Clone & Setup
git clone https://github.com/Jainishpatel1019/ReturnIQ.git
pip install -r requirements.txt

# 2. Run Reproducibility Pipeline (Notebooks)
# Open and run notebooks/01 to 06 sequentially.
# This will generate: data/processed/final_dashboard_data.parquet
# And: models/causal_forest.pkl

# 3. Launch Local Dashboard
streamlit run streamlit_app/app.py
```

---

### 6. Architecture & Limitations
**Architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md) for data flow diagrams.

**Limitations**:
- **Observational Data**: Subject to the Unconfoundedness assumption.
- **Proxy Labels**: Return signals are inferred from 1-2 star reviews (functional dissatisfaction).
- **Static Baseline**: Causal effects are estimated on the 2023 Amazon Review Corpus.

---

### 7. The Stack
- **Engine**: EconML (Double ML), XGBoost
- **NLP**: Transformers (DistilBERT base), Sentence-Transformers
- **Data**: DuckDB, PyArrow
- **Deploy**: Streamlit, Plotly, HuggingFace Spaces
