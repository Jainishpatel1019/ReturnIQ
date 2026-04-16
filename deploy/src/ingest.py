"""
src/ingest.py — Parallel Version
Streams the 3 target categories in parallel using Joblib.
Saves Parquet files to data/raw/.
"""

from datasets import load_dataset
import pandas as pd
import random
import pathlib
import time
from joblib import Parallel, delayed

CATS = [
    "Electronics",
    "Clothing_Shoes_and_Jewelry",
    "Home_and_Kitchen",
]

COLS = [
    "asin",
    "user_id",
    "rating",
    "text",
    "timestamp",
    "parent_asin",
    "verified_purchase",
]

TARGET = 1_250_000
OUT_DIR = pathlib.Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def reservoir_sample(ds, target: int, cols: list[str], category: str) -> pd.DataFrame:
    reservoir = []
    t0 = time.time()

    for i, row in enumerate(ds):
        record = {k: row.get(k) for k in cols}
        record["category"] = category

        if i < target:
            reservoir.append(record)
        else:
            j = random.randint(0, i)
            if j < target:
                reservoir[j] = record

        if i % 250_000 == 0 and i > 0:
            elapsed = time.time() - t0
            print(f"  [{category}] streamed {i:,} rows in {elapsed:.0f}s …")

    return pd.DataFrame(reservoir)


def process_cat(cat):
    random.seed(42)  # Local seed per process
    out_path = OUT_DIR / f"{cat}.parquet"
    if out_path.exists():
        print(f"⏭  {cat}.parquet already exists — skipping.")
        return

    print(f"\n▸ Streaming {cat} (Parallel Mode) …")
    ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_review_{cat}",
        split="full",
        streaming=True,
        trust_remote_code=True,
    )

    df = reservoir_sample(ds, TARGET, COLS, cat)
    df.to_parquet(out_path, index=False)
    print(f"✓ Saved {len(df):,} rows → {out_path}")


def main():
    print(f"🚀 Starting parallel ingestion of {len(CATS)} categories...")
    t_start = time.time()
    Parallel(n_jobs=len(CATS))(delayed(process_cat)(cat) for cat in CATS)
    t_end = time.time()
    
    # ── Final Report ───────────────────────────────────────────────────────────
    total_size = sum(f.stat().st_size for f in OUT_DIR.glob("*.parquet")) / (1024**2)
    elapsed = (t_end - t_start) / 60
    
    print("\n✅ All categories completed!")
    print(f"   Total Footprint : {total_size:.2f} MB")
    print(f"   Total Time      : {elapsed:.1f} minutes")
    print(f"   Avg Throughput  : {total_size / (elapsed * 60):.2f} MB/s")


if __name__ == "__main__":
    main()
