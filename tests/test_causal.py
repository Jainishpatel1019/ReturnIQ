"""
tests/test_causal.py — Week 8 pytest suite
Run: pytest tests/ -v
"""

import pytest
import pandas as pd
import pathlib


# ── Fixtures ───────────────────────────────────────────────────────────────────
CATE_PATH  = pathlib.Path("data/processed/cate_results.parquet")
FINAL_PATH = pathlib.Path("data/processed/final_dashboard_data.parquet")
OLS_PATH   = pathlib.Path("data/processed/ols_predictions.parquet")
SELLER_PATH = pathlib.Path("data/processed/seller_features_trimmed.parquet")


def skip_if_missing(path: pathlib.Path):
    return pytest.mark.skipif(
        not path.exists(),
        reason=f"{path} not yet generated — run the pipeline first.",
    )


# ── Week 1 checks ──────────────────────────────────────────────────────────────
@skip_if_missing(pathlib.Path("data/processed/returns.duckdb"))
def test_duckdb_exists():
    import duckdb
    con = duckdb.connect("data/processed/returns.duckdb", read_only=True)
    tables = con.execute("SHOW TABLES").fetchall()
    table_names = [t[0] for t in tables]
    assert "reviews" in table_names,  "reviews table missing"
    assert "products" in table_names, "products table missing"
    assert "sellers" in table_names,  "sellers table missing"
    con.close()


# ── Week 4 checks ──────────────────────────────────────────────────────────────
@skip_if_missing(SELLER_PATH)
def test_seller_quality_score_range():
    df = pd.read_parquet(SELLER_PATH)
    assert "seller_quality_score" in df.columns, "seller_quality_score column missing"
    assert df["seller_quality_score"].between(0, 1).all(), \
        "seller_quality_score values outside [0, 1]"
    assert df["seller_quality_score"].std() > 0.01, \
        "seller_quality_score has no variance — positivity violation!"


@skip_if_missing(SELLER_PATH)
def test_proxy_return_rate_range():
    df = pd.read_parquet(SELLER_PATH)
    assert "proxy_return_rate" in df.columns
    assert (df["proxy_return_rate"] >= 0).all(), "Negative proxy_return_rate found"
    assert (df["proxy_return_rate"] <= 1).all(), "proxy_return_rate > 1 found"


# ── Week 4 CATE checks ─────────────────────────────────────────────────────────
@skip_if_missing(CATE_PATH)
def test_cate_results_exist():
    df = pd.read_parquet(CATE_PATH)
    assert "cate"    in df.columns, "cate column missing"
    assert "cate_lo" in df.columns, "cate_lo column missing"
    assert "cate_hi" in df.columns, "cate_hi column missing"
    assert df["cate"].notna().all(), "NaN values in cate column"


@skip_if_missing(CATE_PATH)
def test_ci_ordering():
    """Lower CI bound must be ≤ CATE ≤ upper CI bound for all sellers."""
    df = pd.read_parquet(CATE_PATH)
    assert (df["cate_lo"] <= df["cate"]).all(), "cate_lo > cate for some sellers"
    assert (df["cate"]    <= df["cate_hi"]).all(), "cate > cate_hi for some sellers"


@skip_if_missing(CATE_PATH)
def test_shap_columns_present():
    df = pd.read_parquet(CATE_PATH)
    shap_cols = [c for c in df.columns if c.startswith("shap_")]
    assert len(shap_cols) >= 1, "No shap_ columns found in CATE results"


# ── Week 6 narrative checks ────────────────────────────────────────────────────
@skip_if_missing(FINAL_PATH)
def test_narrative_not_empty():
    df = pd.read_parquet(FINAL_PATH)
    assert "narrative" in df.columns, "narrative column missing"
    # At least 80% of narratives should be non-trivial (>50 chars)
    pct_ok = df["narrative"].str.len().gt(50).mean()
    assert pct_ok >= 0.8, f"Only {pct_ok:.1%} of narratives have >50 chars"


@skip_if_missing(FINAL_PATH)
def test_narrative_no_error_strings():
    df = pd.read_parquet(FINAL_PATH)
    error_mask = df["narrative"].str.startswith("[Error", na=False)
    pct_errors = error_mask.mean()
    assert pct_errors < 0.05, \
        f"{pct_errors:.1%} of narratives are error strings — check Ollama connection"


# ── Week 2 OLS checks ─────────────────────────────────────────────────────────
@skip_if_missing(OLS_PATH)
def test_ols_predictions_exist():
    df = pd.read_parquet(OLS_PATH)
    assert "ols_pred" in df.columns, "ols_pred column missing"
    assert df["ols_pred"].notna().all(), "NaN values in ols_pred"
