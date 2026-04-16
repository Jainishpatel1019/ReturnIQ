"""
src/causal_model.py — Week 4, Step 3
CausalForestDML — the core model.
T = seller_quality_score, Y = proxy_return_rate, X = confounders.
Outputs per-seller CATE with 95% CI + SHAP attribution.

Run small test on Mac: python src/causal_model.py
Full 2000-tree run: copy to Colab, set n_estimators=2000 (takes ~40 min on A100)

Prereq: data/processed/seller_features_trimmed.parquet
"""

import pandas as pd
import numpy as np
import joblib
import mlflow
from econml.dml import CausalForestDML
from xgboost import XGBRegressor
import pathlib

INPUT_PATH  = "data/processed/seller_features_trimmed.parquet"
OUTPUT_PATH = "data/processed/cate_results.parquet"
MODEL_PATH  = "models/causal_forest.pkl"

# On Mac: use n_estimators=200 for a quick test run.
# On Colab A100: use n_estimators=2000, bootstrap=True.
N_ESTIMATORS = 200


def get_feature_matrix(df: pd.DataFrame):
    """Build the confounder matrix X. Must match between training and inference."""
    cat_dummies = pd.get_dummies(df["category"], prefix="cat")
    X = pd.concat(
        [
            cat_dummies,
            df[
                [
                    "price_tier",
                    "unique_buyers",
                    "total_reviews",
                    "competitor_density",
                    "niche_share",
                ]
            ],
        ],
        axis=1,
    ).astype(float)
    return X, list(X.columns)


def run_causal_model(input_path: str = INPUT_PATH) -> None:
    pathlib.Path("models").mkdir(parents=True, exist_ok=True)

    print(f"▸ Loading features from {input_path} …")
    df = pd.read_parquet(input_path)
    if len(df) > 25000:
        print("  Sampling 25,000 rows for artifact generation …")
        df = df.sample(25000, random_state=42)
    print(f"  Loaded {len(df):,} sellers")

    T = df["seller_quality_score"].values  # treatment
    Y = df["proxy_return_rate"].values      # outcome
    X, feature_names = get_feature_matrix(df)

    nuisance = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        n_jobs=-1,
        random_state=42,
    )

    print(f"▸ Fitting CausalForestDML (n_estimators={N_ESTIMATORS}) …")
    est = CausalForestDML(
        model_y=nuisance,
        model_t=nuisance,
        n_estimators=N_ESTIMATORS,
        max_depth=5,
        min_samples_leaf=20,
        random_state=42,
        inference=True,  # enables confidence intervals
    )
    est.fit(Y, T, X=X.values)

    # ── CATE + 95% CI ──────────────────────────────────────────────────────────
    cate = est.effect(X.values)
    ci   = est.effect_interval(X.values, alpha=0.05)

    df["cate"]    = cate
    df["cate_lo"] = ci[0]
    df["cate_hi"] = ci[1]

    print(f"  Mean CATE : {np.mean(cate):.4f}")
    print(f"  Std  CATE : {np.std(cate):.4f}")

    # ── SHAP values ────────────────────────────────────────────────────────────
    print("▸ Computing SHAP values …")
    shap_values = est.shap_values(X.values)
    shap_y = list(shap_values.values())[0]
    shap_arr = list(shap_y.values())[0]  # shape: (n_sellers, n_features)
    
    if hasattr(shap_arr, "values"):
        shap_raw = np.array(shap_arr.values, dtype=float)
    else:
        shap_raw = np.array(shap_arr, dtype=float)

    for i, feat in enumerate(feature_names):
        df[f"shap_{feat}"] = shap_raw[:, i]

    # ── Save results ───────────────────────────────────────────────────────────
    df.to_parquet(OUTPUT_PATH, index=False)
    joblib.dump(est, MODEL_PATH)
    print(f"\n✓ CATE results saved to {OUTPUT_PATH}")
    print(f"✓ Model saved to {MODEL_PATH}")

    # ── Log to MLflow ──────────────────────────────────────────────────────────
    mlflow.set_experiment("causal_model")
    with mlflow.start_run():
        mlflow.log_metric("mean_cate",     float(np.mean(cate)))
        mlflow.log_metric("std_cate",      float(np.std(cate)))
        mlflow.log_metric("n_estimators",  N_ESTIMATORS)
        mlflow.log_metric("n_sellers",     len(df))
    print("✓ Metrics logged to MLflow")


if __name__ == "__main__":
    run_causal_model()
