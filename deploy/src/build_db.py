"""
src/build_db.py — Week 1, Step 2
Builds DuckDB database from raw Parquet files.
Creates 3 tables: reviews, products, sellers.

Run: python src/build_db.py
Prereq: data/raw/*.parquet must exist (run ingest.py first)
"""

import duckdb
import pathlib

DB_PATH = "data/processed/returns.duckdb"
RAW_GLOB = "data/raw/*.parquet"

pathlib.Path("data/processed").mkdir(parents=True, exist_ok=True)


def build(db_path: str = DB_PATH) -> None:
    print(f"▸ Connecting to {db_path} …")
    con = duckdb.connect(db_path)

    # ── reviews table ──────────────────────────────────────────────────────────
    print("  Building reviews table …")
    con.execute("""
        CREATE OR REPLACE TABLE reviews AS
        SELECT
            *,
            CAST(timestamp AS BIGINT) AS ts_unix
        FROM read_parquet('data/raw/*.parquet')
    """)
    n_reviews = con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    print(f"  ✓ reviews: {n_reviews:,} rows")

    # ── products table ─────────────────────────────────────────────────────────
    print("  Building products table …")
    con.execute("""
        CREATE OR REPLACE TABLE products AS
        SELECT
            parent_asin,
            category,
            COUNT(*)       AS review_count,
            AVG(rating)    AS avg_rating,
            MIN(ts_unix)   AS first_review_ts
        FROM reviews
        GROUP BY 1, 2
    """)
    n_products = con.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    print(f"  ✓ products: {n_products:,} rows")

    # ── sellers table ──────────────────────────────────────────────────────────
    # Proxy: parent_asin ≈ seller unit (no direct seller_id in public dataset)
    # proxy_return_rate = fraction of reviews with rating ≤ 2
    print("  Building sellers table …")
    con.execute("""
        CREATE OR REPLACE TABLE sellers AS
        SELECT
            parent_asin                                              AS seller_id,
            category,
            COUNT(*)                                                 AS total_reviews,
            AVG(CASE WHEN rating <= 2 THEN 1.0 ELSE 0.0 END)        AS proxy_return_rate,
            STDDEV(rating)                                           AS rating_variance,
            COUNT(DISTINCT user_id)                                  AS unique_buyers
        FROM reviews
        GROUP BY 1, 2
    """)
    n_sellers = con.execute("SELECT COUNT(*) FROM sellers").fetchone()[0]
    print(f"  ✓ sellers: {n_sellers:,} rows")

    con.close()
    print(f"\n✓ DuckDB built at {db_path}")


if __name__ == "__main__":
    build()
