"""
src/ols_baseline.py — Week 2, Step 3
OLS baseline + Breusch-Pagan heteroskedasticity test.
p < 0.05 confirms heteroskedasticity → justifies the causal ML approach over naive OLS.
Saves OLS predictions for the "baseline vs causal" Streamlit view.

Run: python src/ols_baseline.py
Prereq: cohort_stats table must exist in DuckDB
"""

import duckdb
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tools import add_constant

DB_PATH = "data/processed/returns.duckdb"


def run_ols_baseline(db_path: str = DB_PATH) -> dict:
    print(f"▸ Loading cohort_stats from {db_path} …")
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute("""
        SELECT
            seller_id,
            proxy_return_rate,
            total_reviews,
            rating_variance,
            unique_buyers,
            CASE category WHEN 'Electronics' THEN 1 ELSE 0 END AS is_electronics
        FROM cohort_stats
    """).df().dropna()
    con.close()
    print(f"  Loaded {len(df):,} sellers")

    # ── OLS fit ────────────────────────────────────────────────────────────────
    feature_cols = ["total_reviews", "rating_variance", "unique_buyers", "is_electronics"]
    X = add_constant(df[feature_cols])
    y = df["proxy_return_rate"]

    print("▸ Fitting OLS …")
    ols_fit = OLS(y, X).fit()

    # ── Breusch-Pagan test ─────────────────────────────────────────────────────
    bp_stat, bp_pval, _, _ = het_breuschpagan(ols_fit.resid, X)

    print("\n── OLS Baseline ────────────────────────────────────────")
    print(f"  R²                  = {ols_fit.rsquared:.4f}")
    print(f"  Adj. R²             = {ols_fit.rsquared_adj:.4f}")
    print(f"  Breusch-Pagan stat  = {bp_stat:.2f}")
    print(f"  Breusch-Pagan p     = {bp_pval:.4e}")
    print("────────────────────────────────────────────────────────")

    if bp_pval < 0.05:
        print("  ✓ p < 0.05 — heteroskedasticity CONFIRMED.")
        print("    OLS assumptions violated → proceed to CausalForestDML.")
    else:
        print("  ⚠  p ≥ 0.05 — heteroskedasticity not detected in this sample.")
        print("    OLS may be adequate, but causal model still preferred for CATE.")

    # ── Save predictions for Streamlit View 3 ─────────────────────────────────
    df["ols_pred"] = ols_fit.fittedvalues
    out_path = "data/processed/ols_predictions.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n✓ OLS predictions saved to {out_path}")

    return {
        "r2": ols_fit.rsquared,
        "bp_stat": bp_stat,
        "bp_pval": bp_pval,
    }


if __name__ == "__main__":
    run_ols_baseline()
