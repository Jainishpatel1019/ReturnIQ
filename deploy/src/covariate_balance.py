"""
src/covariate_balance.py — Week 5, Step 4
Covariate Balance (Love Plot): Proof of Unbiasedness.
Calculates Standardized Mean Difference (SMD) for all confounders 
across "High Quality" vs "Low Quality" sellers.
Target: SMD < 0.1 for most features (proves independence).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pathlib

INPUT_PATH = "data/processed/seller_features_trimmed.parquet"
PLOT_PATH  = "data/processed/covariate_balance.png"

def calculate_smd(df, feature, treatment_col):
    """Calculate Standardized Mean Difference for a feature."""
    treat = df[df[treatment_col] > df[treatment_col].median()][feature]
    ctrl  = df[df[treatment_col] <= df[treatment_col].median()][feature]
    
    m1, m2 = treat.mean(), ctrl.mean()
    v1, v2 = treat.var(), ctrl.var()
    
    denom = np.sqrt((v1 + v2) / 2)
    return (m1 - m2) / denom if denom > 0 else 0

def main():
    print(f"▸ Loading features from {INPUT_PATH} …")
    if not pathlib.Path(INPUT_PATH).exists():
        print("  ✗ File missing. Run seller_features.py first.")
        return
        
    df = pd.read_parquet(INPUT_PATH)
    
    # Define confounders to check balance for
    confounders = [
        "total_reviews",
        "unique_buyers",
        "price_tier",
    ]
    # Add category dummies
    cat_dummies = pd.get_dummies(df["category"], prefix="cat")
    df_bal = pd.concat([df, cat_dummies], axis=1)
    confounders += list(cat_dummies.columns)

    smds = {}
    for feat in confounders:
        smds[feat] = calculate_smd(df_bal, feat, "seller_quality_score")

    # ── Plotting ───────────────────────────────────────────────────────────────
    smd_df = pd.Series(smds).abs().sort_values()
    
    plt.figure(figsize=(10, 6))
    colors = ["#3ddc84" if val < 0.1 else "#ff6b6b" for val in smd_df.values]
    plt.barh(smd_df.index, smd_df.values, color=colors, alpha=0.8)
    plt.axvline(0.1, color="white", linestyle="--", label="Balance Threshold (0.1)")
    
    plt.title("Covariate Balance (Love Plot): Proof of Unbiasedness", fontsize=14, color="white")
    plt.xlabel("Absolute Standardized Mean Difference (SMD)", color="white")
    plt.xticks(color="white")
    plt.yticks(color="white")
    plt.grid(axis='x', alpha=0.2)
    plt.legend()
    
    # Dark theme styling
    plt.gca().set_facecolor('#0d1117')
    plt.gcf().set_facecolor('#0d1117')
    
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, facecolor='#0d1117')
    print(f"\n✓ Covariate balance plot saved to {PLOT_PATH}")
    
    # ── Report summaries ───────────────────────────────────────────────────────
    balanced = sum(1 for v in smds.values() if abs(v) < 0.1)
    print(f"  Balanced features: {balanced} / {len(smds)} (SMD < 0.1)")

if __name__ == "__main__":
    main()
