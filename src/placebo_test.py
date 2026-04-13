"""
src/placebo_test.py — Week 5, Step 1
Placebo falsification test.
Re-runs CausalForestDML with random noise as the outcome Y.
If it finds a significant effect → confounders are wrong.
Target: ≥90% of CIs should contain zero.

Run: python src/placebo_test.py
Prereq: data/processed/seller_features_trimmed.parquet
"""

import pandas as pd
import numpy as np
from econml.dml import CausalForestDML
from xgboost import XGBRegressor

INPUT_PATH = "data/processed/seller_features_trimmed.parquet"


def get_feature_matrix(df):
    cat_dummies = pd.get_dummies(df["category"], prefix="cat")
    return pd.concat(
        [cat_dummies, df[["price_tier", "unique_buyers", "total_reviews"]]],
        axis=1,
    ).values


def main():
    print(f"▸ Loading features from {INPUT_PATH} …")
    df = pd.read_parquet(INPUT_PATH)
    print(f"  Loaded {len(df):,} sellers")

    T = df["seller_quality_score"].values
    X = get_feature_matrix(df)

    # Pure noise outcome — seller quality should have NO causal effect on this
    rng = np.random.RandomState(0)
    Y_fake = rng.normal(0, 1, size=len(df))

    print("▸ Fitting placebo CausalForestDML (Y = random noise) …")
    est_placebo = CausalForestDML(
        model_y=XGBRegressor(n_estimators=100, random_state=42),
        model_t=XGBRegressor(n_estimators=100, random_state=42),
        n_estimators=200,
        inference=True,
        random_state=42,
    )
    est_placebo.fit(Y_fake, T, X=X)

    cate_fake = est_placebo.effect(X)
    ci_fake   = est_placebo.effect_interval(X, alpha=0.05)

    # Check: CIs should mostly contain 0 for a placebo
    contains_zero = (
        (ci_fake[0] <= 0) & (ci_fake[1] >= 0)
    ).mean()

    print("\n── Placebo Test ────────────────────────────────────────")
    print(f"  CIs containing 0 : {contains_zero:.1%}  (target: ≥ 90%)")
    print(f"  Mean CATE placebo : {np.mean(cate_fake):.4f}  (should be ≈ 0)")
    print(f"  Std  CATE placebo : {np.std(cate_fake):.4f}")
    print("────────────────────────────────────────────────────────")

    if contains_zero >= 0.90:
        print("  ✓ PASS — placebo test passed. Model is not picking up spurious effects.")
        print("    Proceed to full causal estimation with confidence.")
    elif contains_zero >= 0.85:
        print("  ⚠  MARGINAL — 85-90% is borderline. Consider adding more confounders.")
    else:
        print("  ✗ FAIL — <85% of CIs contain zero.")
        print("    Your confounders are likely wrong. Go back to causal_model.py")
        print("    and add more X variables (e.g. listing quality, seller age).")

    return {"contains_zero_pct": float(contains_zero)}


if __name__ == "__main__":
    main()
