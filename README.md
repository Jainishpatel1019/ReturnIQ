# ReturnIQ — Causal Intelligence for Marketplace Returns

[![CI](https://github.com/Jainishpatel1019/ReturnIQ/actions/workflows/ci.yml/badge.svg)](https://github.com/Jainishpatel1019/ReturnIQ/actions)
[![HuggingFace Space](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Space-blue)](https://huggingface.co/spaces/Jainishp1019/seller-intelligence-platform)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **"If you target high-return sellers without causal identification, you're not fixing the problem — you're firing your best customers."**

ReturnIQ is a causal inference platform designed to solve the **Selection vs. Treatment** dilemma in e-commerce. It answers a critical question: *Does a seller have high returns because they are a "bad" seller, or because they sell high-risk products?*

---

## 🖼️ Interface Gallery

| 📊 Overview & Metrics | 💡 Causal Key Findings |
|:---:|:---:|
| ![Overview](docs/images/01_overview.png) | ![Key Findings](docs/images/02_key_findings.png) |
| *High-level platform telemetry* | *Quantifying seller vs. market impact* |

| 👤 Seller Deep-Dive | ⚙️ Policy Simulator |
|:---:|:---:|
| ![Seller Profile](docs/images/03_seller_profile.png) | ![Policy Tradeoff](docs/images/04_policy_sim.png) |
| *Individual risk narratives* | *Prevention vs. Precision tradeoff* |

| 📉 Model Identification |
|:---:|
| ![Model Proof](docs/images/05_model_proof.png) |
| *OLS vs. Causal model disagreement* |

---

## 🎯 The Core Problem: Selection Bias
In marketplace analytics, standard metrics "see" a high return rate and blame the seller. However, if a seller moves high-return items (e.g., Clothing), simple correlation fails. **ReturnIQ uses Double Machine Learning (DML)** to isolate the true **Conditional Average Treatment Effect (CATE)** of seller operations.

## 🏗️ Technical Architecture

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
    I --> J[Streamlit Dashboard]
```

---

## 🚀 Quick Start

### 1. Setup
```bash
git clone https://github.com/Jainishpatel1019/ReturnIQ
cd ReturnIQ
conda create -n returns python=3.11
conda activate returns
pip install -r requirements.txt
```

### 2. Usage
```bash
# Launch the dashboard
streamlit run streamlit_app/app.py
```

---

## 🧪 Methodology Detail

### 1. Double Machine Learning (DML)
We implement the `CausalForestDML` estimator from **EconML**. This allows us to handle high-dimensional confounders (X) while focusing on the non-linear treatment effect (T) of seller operational quality.

### 2. Listing Accuracy (NLP)
Using `DistilBERT`, we measure the semantic gap between "How the seller describes the item" vs "How the buyer experienced it." This delta is a massive causal driver (SHAP = 0.21).

### 3. Proof of Validity
- **Placebo Tests**: Treatment shuffle yield $p \approx 0.43$, confirming the model isn't picking up artifacts.
- **AUUC Curve**: Benchmarked against a temporal holdout set (2025) to ensure generalizability.

---

## 🛠️ Tooling & Stack
`Python` `DuckDB` `EconML` `XGBoost` `DistilBERT` `Mistral-7B` `Streamlit` `MLflow` `Plotly`

---

## 👤 Author
**Jainish Patel**  
[GitHub](https://github.com/Jainishpatel1019)  ·  [HuggingFace](https://huggingface.co/Jainishp1019)
