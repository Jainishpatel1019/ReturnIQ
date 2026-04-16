# Model Card: ReturnIQ CATE Estimator

### 1. General Info
- **Type**: CausalForestDML (Double Machine Learning)
- **Framework**: EconML / XGBoost
- **Objective**: Estimate the Heterogeneous Treatment Effect (CATE) of seller operational quality on product return rates.

### 2. Dataset
- **Primary Source**: 3.7M Amazon Reviews (2023 Cohort)
- **Ground Truth**: Proxy Return Rate derived from 1-2 star reviews indicating product failure/dissatisfaction.
- **Sample Scale**: Aggregated to 1,542 unique sellers for dashboard telemetry.

### 3. Identification Strategy
We address **Observed Confounding** through Double ML:
- **Treatment (T)**: Normalized seller quality score.
- **Outcome (Y)**: Proxy return rate.
- **Confounders (X)**: Price tier, category, competitor density, and buyer volume.
- **Model Support**: Positivity confirmed across the distribution through overlap checks.

### 4. Limitations & Biases
- **Unobserved Confounding**: While we control for broad market signals, latent variables (e.g., external marketing spend) may persist.
- **Proxy Bias**: 1-2 star reviews are a high-fidelity signal for functional failure but do not capture "change of mind" returns without failure.
- **Temporal Cutoff**: Analysis is fixed to the 2023 review corpus.
