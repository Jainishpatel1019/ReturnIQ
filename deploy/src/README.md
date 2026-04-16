# ReturnIQ Causal Engine (src/)

This directory contains the production-grade causal inference pipeline used to drive ReturnIQ insights.

## Pipeline Stages

1.  **Ingestion & ETL** (`build_db.py`, `ingest.py`):
    *   Processes 3.7 million Amazon reviews into a DuckDB analytical warehouse.
    *   Constructs seller-level feature vectors including NLP-derived listing accuracy.

2.  **Causal Identification** (`causal_model.py`):
    *   Implements `CausalForestDML` (Double Machine Learning) via EconML.
    *   Handles high-dimensional confounders (Category, Price Tier, Seller Age) to isolate operation-specific return risk.

3.  **Model Validation** (`baseline_vs_causal.py`, `auuc.py`, `placebo_test.py`):
    *   Compares causal CATE scores against naive OLS predictions.
    *   Calculates Area Under the Uplift Curve (AUUC) for ranking precision.
    *   Runs placebo (shuffled) tests to verify statistical significance.

4.  **Risk Narratives** (`narrative_agent.py`, `precompute_narratives.py`):
    *   Generates multi-stage executive summaries using Mistral-7B.
    *   Summaries are precomputed for dashboard performance.

5.  **UI Data Prep** (`merge_dashboard_data.py`):
    *   Syncs all model outputs into the final glassmorphism dashboard.
