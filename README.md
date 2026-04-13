# ReturnIQ — Causal Intelligence for Marketplace Returns

> **What if Amazon's return problem isn't about bad products — but bad seller signals?**

ReturnIQ is an end-to-end causal inference platform that answers one question:
**Is seller behavior the primary driver of Amazon return rates, or is it customer
segment mismatch — and can we quantify the cost of treating them the same?**

## What we found

| Finding | Result |
|---------|--------|
| Seller behavior explains return variance | η² = 0.41 vs 0.18 for category |
| OLS misclassifies sellers | 21% of sellers assigned wrong risk tier |
| Top causal driver | Listing accuracy (SHAP = 0.21) |
| Policy impact (top 15% targeting) | 34% of returns prevented at 12% FPR |
| Temporal stability (2023 → 2025) | AUUC drop <3pts, KS-test p=0.34 |

## How it works

1. **3.7M Amazon reviews** → DuckDB cohort engine → seller-level return features
2. **DistilBERT fine-tuned** on listing vs review mismatch → listing accuracy score
3. **CausalForestDML (Double ML)** → CATE estimates per seller, 200-iter bootstrap CI
4. **Mistral-7B via Ollama** → plain-English risk narrative with reflexion self-check
5. **Streamlit on HuggingFace Spaces** → interactive policy simulator, live demo

## Live demo
[ReturnIQ on HuggingFace Spaces](https://huggingface.co/spaces/Jainishp1019/seller-intelligence-platform)

## Stack
`Python` `DuckDB` `EconML` `XGBoost` `DistilBERT` `Sentence-Transformers`
`Mistral-7B` `Ollama` `Streamlit` `MLflow` `Plotly` `HuggingFace`
