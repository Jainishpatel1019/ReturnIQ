"""
src/auuc.py — Week 5, Step 2
AUUC (Area Under the Uplift Curve) / Qini metric.
Compares causal model vs OLS baseline vs random.
The AUUC number goes directly in the resume bullet.

Run: python src/auuc.py
Prereq: data/processed/cate_results.parquet + data/processed/ols_predictions.parquet
"""

import pandas as pd
import numpy as np
import pathlib

CATE_PATH = "data/processed/cate_results.parquet"
OLS_PATH  = "data/processed/ols_predictions.parquet"


def auuc_score(df: pd.DataFrame, score_col: str, outcome_col: str) -> float:
    """Compute AUUC (Qini curve area vs random baseline)."""
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    n = len(df)
    cum_outcome  = np.cumsum(df[outcome_col]) / n
    random_baseline = np.linspace(0, df[outcome_col].mean(), n)
    auuc = float(np.trapz(cum_outcome - random_baseline) / n)
    return auuc


def main():
    print(f"▸ Loading CATE results from {CATE_PATH} …")
    df_cate = pd.read_parquet(CATE_PATH)

    auuc_model = auuc_score(df_cate, "cate", "proxy_return_rate")

    # Load OLS predictions if available
    auuc_ols = None
    if pathlib.Path(OLS_PATH).exists():
        df_ols = pd.read_parquet(OLS_PATH)
        # Merge on matching seller_id to avoid cartesian product crash
        merged = df_cate[["seller_id", "proxy_return_rate", "cate"]].merge(
            df_ols[["seller_id", "ols_pred"]],
            on="seller_id",
            how="inner",
        )
        if len(merged) > 0:
            if len(merged) > 100_000:
                merged = merged.sample(100_000, random_state=42)
            auuc_ols = auuc_score(merged, "ols_pred", "proxy_return_rate")

    print("\n── AUUC Results ────────────────────────────────────────")
    print(f"  AUUC causal model : {auuc_model:.4f}")
    if auuc_ols is not None:
        print(f"  AUUC OLS baseline : {auuc_ols:.4f}")
        lift = (auuc_model - auuc_ols) / abs(auuc_ols) * 100 if auuc_ols != 0 else float("inf")
        print(f"  Lift over OLS     : +{lift:.1f}%")
    print("  AUUC random       : 0.0000")
    print("────────────────────────────────────────────────────────")

    if auuc_model >= 0.68:
        print(f"  ✓ AUUC {auuc_model:.4f} is in the expected range [0.68–0.75].")
        print("    Record this number in your resume bullet!")
    else:
        print(f"  ⚠  AUUC {auuc_model:.4f} is below 0.68 — consider more estimators.")

    return {
        "auuc_model": auuc_model,
        "auuc_ols":   auuc_ols,
    }


if __name__ == "__main__":
    main()
