"""
src/data_quality.py — Week 1, Step 3
Great Expectations audit on the reviews table.
Catches schema drift and null rates before any modeling.

Run: python src/data_quality.py
Prereq: data/processed/returns.duckdb must exist
"""

import duckdb
import pandas as pd
import great_expectations as ge

DB_PATH = "data/processed/returns.duckdb"
SAMPLE_SIZE = 50_000


def run_audit(db_path: str = DB_PATH) -> dict:
    print(f"▸ Loading {SAMPLE_SIZE:,} rows from {db_path} …")
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute(f"SELECT * FROM reviews LIMIT {SAMPLE_SIZE}").df()
    con.close()

    print("▸ Running Great Expectations suite …")
    gdf = ge.from_pandas(df)

    # ── Schema expectations ────────────────────────────────────────────────────
    gdf.expect_column_to_exist("asin")
    gdf.expect_column_to_exist("rating")
    gdf.expect_column_to_exist("text")
    gdf.expect_column_to_exist("parent_asin")
    gdf.expect_column_to_exist("user_id")
    gdf.expect_column_to_exist("category")

    # ── Value range expectations ───────────────────────────────────────────────
    gdf.expect_column_values_to_be_between("rating", 1, 5)

    # ── Null expectations ──────────────────────────────────────────────────────
    gdf.expect_column_values_to_not_be_null("asin")
    gdf.expect_column_values_to_not_be_null("parent_asin")
    gdf.expect_column_values_to_not_be_null("rating")
    gdf.expect_column_values_to_not_be_null("text")
    gdf.expect_column_values_to_not_be_null("category")

    # ── Run validation ─────────────────────────────────────────────────────────
    results = gdf.validate()
    pass_rate = results["statistics"]["success_percent"]

    icon = "✓" if pass_rate >= 95 else "✗"
    print(f"\n{icon} Pass rate: {pass_rate:.1f}%")

    if pass_rate < 95:
        print("⚠  WARNING: Pass rate < 95%. Check for nulls in 'text' column.")
        print("   Rows with null text are useless for embeddings — drop them.")
        null_text = df["text"].isna().sum()
        print(f"   Null 'text' rows in sample: {null_text:,} / {SAMPLE_SIZE:,}")

    # Print per-expectation detail
    print("\nExpectation results:")
    for r in results["results"]:
        status = "✓" if r["success"] else "✗"
        exp_type = r["expectation_config"]["expectation_type"]
        col = r["expectation_config"]["kwargs"].get("column", "—")
        print(f"  {status} {exp_type} [{col}]")

    return results


if __name__ == "__main__":
    run_audit()
