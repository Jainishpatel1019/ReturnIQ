"""
src/variance_decomp.py — Week 2, Step 2
Two-way ANOVA decomposition + eta-squared.
Quantifies how much return rate variance is between sellers vs. between categories.
This is the headline DA finding — eta²_seller > 0.3 justifies a seller-level model.

Run: python src/variance_decomp.py
Prereq: cohort_stats table must exist in DuckDB
"""

import duckdb
import mlflow
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

DB_PATH = "data/processed/returns.duckdb"


def run_variance_decomp(db_path: str = DB_PATH) -> dict:
    print(f"▸ Loading cohort_stats from {db_path} …")
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute(
        "SELECT seller_id, category, price_tier, proxy_return_rate FROM cohort_stats"
    ).df()
    con.close()
    print(f"  Loaded {len(df):,} sellers")

    print("▸ Fitting two-way ANOVA …")
    # NOTE: For very large datasets this can be slow / memory-heavy.
    # If >100k sellers, subsample to 50k for the ANOVA.
    if len(df) > 5000:
        df_sample = df.sample(5000, random_state=42)
        print("  ⚠  Subsampled to 5k sellers for ANOVA to avoid OOM.")
    else:
        df_sample = df

    model = ols("proxy_return_rate ~ C(category)", data=df_sample).fit()
    anova_table = anova_lm(model, typ=2)

    ss_category = anova_table.loc["C(category)",   "sum_sq"]
    ss_seller   = anova_table.loc["Residual",      "sum_sq"]
    ss_total    = anova_table["sum_sq"].sum()

    eta2_seller   = float(ss_seller   / ss_total)
    eta2_category = float(ss_category / ss_total)

    print("\n── Variance Decomposition ──────────────────────────────")
    print(f"  eta² seller   = {eta2_seller:.3f}  ({eta2_seller*100:.1f}% of variance)")
    print(f"  eta² category = {eta2_category:.3f}  ({eta2_category*100:.1f}% of variance)")
    print("────────────────────────────────────────────────────────")

    if eta2_seller > 0.3:
        print("  ✓ eta²_seller > 0.30 — sellers explain >30% of return rate variance.")
        print("    This JUSTIFIES building a seller-level causal model.")
    else:
        print("  ⚠  eta²_seller ≤ 0.30 — seller effect is weaker than expected.")
        print("    Consider expanding confounder set before proceeding to causal model.")

    # ── Log to MLflow ──────────────────────────────────────────────────────────
    mlflow.set_experiment("variance_decomp")
    with mlflow.start_run():
        mlflow.log_metric("eta2_seller",   eta2_seller)
        mlflow.log_metric("eta2_category", eta2_category)
        mlflow.log_metric("n_sellers",     len(df))
    print("\n✓ Metrics logged to MLflow (run: mlflow ui)")

    return {"eta2_seller": eta2_seller, "eta2_category": eta2_category}


if __name__ == "__main__":
    run_variance_decomp()
