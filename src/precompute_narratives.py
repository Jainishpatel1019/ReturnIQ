"""
src/precompute_narratives.py — Week 6, Step 2
Batch-generates Mistral narratives for all sellers.
Saves to data/processed/final_dashboard_data.parquet.
Streamlit app reads from this file — zero live inference at demo time.

Run: python src/precompute_narratives.py
     (or with a sample: python src/precompute_narratives.py --sample 5000)
Prereq: ollama serve & AND data/processed/cate_results.parquet
"""

import pandas as pd
import argparse
from tqdm import tqdm
from narrative_agent import generate_narrative

INPUT_PATH  = "data/processed/cate_results.parquet"
OUTPUT_PATH = "data/processed/final_dashboard_data.parquet"


def main(sample: int | None = None):
    print(f"▸ Loading CATE results from {INPUT_PATH} …")
    df = pd.read_parquet(INPUT_PATH)

    if sample:
        df = df.sample(min(sample, len(df)), random_state=42).reset_index(drop=True)
        print(f"  Using random sample of {len(df):,} sellers (passed --sample {sample})")
    else:
        print(f"  Processing all {len(df):,} sellers")

    print("▸ Generating narratives via Mistral-7B (2-pass reflexion) …")
    narratives = []
    failed = 0
    for seller_id in tqdm(df["seller_id"], desc="Generating"):
        try:
            narrative = generate_narrative(seller_id, df)
            narratives.append(narrative)
        except Exception as e:
            narratives.append(f"[Error generating narrative: {e}]")
            failed += 1

    df["narrative"] = narratives
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"\n✓ Narratives saved to {OUTPUT_PATH}")
    print(f"  Total: {len(df):,}  Failed: {failed:,}")
    if failed > 0:
        print("  ⚠  Some narratives failed. Check that Ollama is running: ollama serve &")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute seller narratives")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Use a random sample of N sellers (omit for all sellers)",
    )
    args = parser.parse_args()
    main(sample=args.sample)
