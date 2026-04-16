"""
src/seller_features.py — Week 4, Step 1
Builds the seller quality score (Treatment variable T).
Components: fulfillment speed proxy + listing accuracy (embedding cosine consistency).
Saves seller_features.parquet for use by causal_model.py.

Run: python src/seller_features.py
Prereq: cohort_stats in DuckDB + embeddings Parquet files in data/embeddings/
"""

import duckdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pathlib

DB_PATH = "data/processed/returns.duckdb"
EMB_DIR = pathlib.Path("data/embeddings")


def cosine_consistency(vecs) -> float:
    """Mean pairwise cosine similarity within a group — high = consistent reviews = accurate listing."""
    v = np.stack(vecs)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / (norms + 1e-9)
    sim_matrix = v @ v.T
    indices = np.triu_indices(len(v), k=1)
    return float(sim_matrix[indices].mean()) if len(v) > 1 else 1.0


def build_seller_features(db_path: str = DB_PATH) -> None:
    print(f"▸ Loading cohort_stats from {db_path} …")
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute("SELECT * FROM cohort_stats").df()
    con.close()
    print(f"  Loaded {len(df):,} sellers")

    # ── Component 1: Fulfillment speed proxy ───────────────────────────────────
    # Lower rating_variance = more consistent seller = better fulfillment
    df["fulfill_score"] = 1 - MinMaxScaler().fit_transform(df[["rating_variance"]])

    # ── Component 2: Listing accuracy (embedding cosine consistency) ───────────
    emb_files = list(EMB_DIR.glob("*_emb.parquet"))
    if emb_files:
        print(f"▸ Loading embeddings from {EMB_DIR} ({len(emb_files)} files) …")
        emb_df = pd.concat([pd.read_parquet(f) for f in emb_files], ignore_index=True)
        print(f"  Loaded {len(emb_df):,} review embeddings")

        print("▸ Computing listing accuracy (cosine consistency per seller) …")
        # NOTE: This can take ~10 min for 5M reviews — run once and cache
        listing_acc = (
            emb_df.groupby("parent_asin")["emb"]
            .apply(cosine_consistency)
            .reset_index()
        )
        listing_acc.columns = ["seller_id", "listing_acc"]
        df = df.merge(listing_acc, on="seller_id", how="left")
        df["listing_acc"] = df["listing_acc"].fillna(df["listing_acc"].median())
    else:
        print("  ⚠  No embedding files found in data/embeddings/ — using rating_variance proxy.")
        print("     Run Week 3 Colab notebook first to generate embeddings.")
        # Fallback: use normalized total_reviews as listing_acc proxy (more reviews = more 'proven' listing)
        df["listing_acc"] = MinMaxScaler().fit_transform(
            df[["total_reviews"]]
        ).flatten()

    # ── Composite quality score (equal-weight) ─────────────────────────────────
    scaler = MinMaxScaler()
    df[["fulfill_score_n", "listing_acc_n"]] = scaler.fit_transform(
        df[["fulfill_score", "listing_acc"]]
    )
    df["seller_quality_score"] = df[["fulfill_score_n", "listing_acc_n"]].mean(axis=1)

    # ── Save ───────────────────────────────────────────────────────────────────
    out_path = "data/processed/seller_features.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n✓ Seller features saved to {out_path}")
    print(df[["seller_id", "seller_quality_score", "proxy_return_rate"]].describe())


if __name__ == "__main__":
    build_seller_features()
