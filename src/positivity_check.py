"""
src/positivity_check.py — Week 4, Step 2
Positivity check: ensure treatment (quality score) has sufficient overlap across
confounders. Trims top/bottom 1% of treatment distribution.
MUST run before causal_model.py.

Run: python src/positivity_check.py
Prereq: data/processed/seller_features.parquet
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless — saves PNG without needing a display
import matplotlib.pyplot as plt

INPUT_PATH = "data/processed/seller_features.parquet"
TRIMMED_PATH = "data/processed/seller_features_trimmed.parquet"
PLOT_PATH = "data/processed/positivity_check.png"

CATS = [
    "Electronics",
    "Clothing_Shoes_and_Jewelry",
    "Home_and_Kitchen",
    "Sports_and_Outdoors",
]


def main():
    print(f"▸ Loading seller features from {INPUT_PATH} …")
    df = pd.read_parquet(INPUT_PATH)
    print(f"  Loaded {len(df):,} sellers")

    # ── Plot distribution per category ─────────────────────────────────────────
    available_cats = [c for c in CATS if c in df["category"].unique()]
    fig, axes = plt.subplots(
        1, len(available_cats), figsize=(5 * len(available_cats), 4)
    )
    if len(available_cats) == 1:
        axes = [axes]

    for ax, cat in zip(axes, available_cats):
        g = df[df["category"] == cat]
        ax.hist(g["seller_quality_score"], bins=40, alpha=0.8)
        ax.set_title(cat[:20], fontsize=9)
        ax.set_xlabel("Quality score")
        ax.set_ylabel("Count")

        # Detect clustering (positivity violation)
        q10 = g["seller_quality_score"].quantile(0.10)
        q90 = g["seller_quality_score"].quantile(0.90)
        ax.axvline(q10, color="red", linestyle="--", alpha=0.5, label="p10/p90")
        ax.axvline(q90, color="red", linestyle="--", alpha=0.5)

        if q90 - q10 < 0.1:
            ax.set_title(f"⚠ {cat[:15]} — LOW VARIANCE", color="red", fontsize=9)
            print(f"  ⚠ WARNING: {cat} — treatment has very low variance (q90-q10={q90-q10:.3f}).")
            print("             Consider EXCLUDING this category from causal estimation.")

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"\n✓ Positivity check plot saved to {PLOT_PATH}")

    # ── Trim extremes ──────────────────────────────────────────────────────────
    p1  = df["seller_quality_score"].quantile(0.01)
    p99 = df["seller_quality_score"].quantile(0.99)
    df_trimmed = df[
        (df["seller_quality_score"] >= p1) & (df["seller_quality_score"] <= p99)
    ].copy()

    df_trimmed.to_parquet(TRIMMED_PATH, index=False)
    pct_kept = len(df_trimmed) / len(df) * 100
    print(f"✓ Trimmed to [{p1:.3f}, {p99:.3f}]: {len(df_trimmed):,} / {len(df):,} sellers ({pct_kept:.1f}%)")
    print(f"✓ Saved trimmed features to {TRIMMED_PATH}")


if __name__ == "__main__":
    main()
