"""
src/cohorts.py — Week 2, Step 1
Builds cohort_stats table using DuckDB window functions.
This is the main analytics table every downstream module reads from.

Run: python src/cohorts.py
Prereq: sellers table must exist in DuckDB
"""

import duckdb

DB_PATH = "data/processed/returns.duckdb"


def build_cohorts(db_path: str = DB_PATH) -> None:
    print(f"▸ Building cohort_stats in {db_path} …")
    con = duckdb.connect(db_path)

    con.execute("""
        CREATE OR REPLACE TABLE cohort_stats AS
        WITH base AS (
            SELECT
                s.seller_id,
                s.category,
                NTILE(5) OVER (PARTITION BY s.category ORDER BY p.avg_rating) AS price_tier,
                s.proxy_return_rate,
                s.total_reviews,
                s.rating_variance,
                s.unique_buyers
            FROM sellers s
            JOIN products p ON s.seller_id = p.parent_asin
        ),
        ranked AS (
            SELECT
                *,
                -- NEW: Competition Density (How crowded is this price/category bucket?)
                COUNT(*) OVER (PARTITION BY category, price_tier) AS competitor_density,
                -- NEW: Market Share Proxy (Seller reviews vs category total)
                CAST(total_reviews AS FLOAT) / SUM(total_reviews) OVER (PARTITION BY category) AS niche_share,
                AVG(proxy_return_rate) OVER (PARTITION BY category)                   AS cat_avg_return,
                AVG(proxy_return_rate) OVER (PARTITION BY price_tier)                 AS tier_avg_return,
                RANK() OVER (PARTITION BY category ORDER BY proxy_return_rate DESC)   AS return_rank
            FROM base
        )
        SELECT * FROM ranked
    """)

    n = con.execute("SELECT COUNT(*) FROM cohort_stats").fetchone()[0]
    print(f"  ✓ cohort_stats: {n:,} sellers")

    # Quick sanity check
    sample = con.execute("""
        SELECT category, 
               COUNT(*) as n_sellers,
               AVG(proxy_return_rate) as avg_return_rate,
               MIN(price_tier) as min_tier,
               MAX(price_tier) as max_tier
        FROM cohort_stats
        GROUP BY category
        ORDER BY 1
    """).df()
    print("\nCohort summary by category:")
    print(sample.to_string(index=False))

    con.close()
    print("\n✓ cohort_stats built.")


if __name__ == "__main__":
    build_cohorts()
