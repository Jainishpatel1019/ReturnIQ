"""
src/diff_in_diff.py — Week 5, Step 4
Difference-in-Differences (DiD) validation.
Tests if a policy change (simulated Jan 2024 quality crackdown) 
successfully reduced return rates for the treated group.
Uses PanelOLS with Entity and Time Fixed Effects.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
import pathlib

DB_PATH = "data/processed/returns.duckdb"
PLOT_PATH = "data/processed/did_validation.png"

def run_did():
    feature_path = "data/processed/seller_features.parquet"
    print(f"▸ Loading panel data from {feature_path} …")
    if not pathlib.Path(feature_path).exists():
        print("  ✗ Feature file missing. Run seller_features.py first.")
        return
    
    df = pd.read_parquet(feature_path)
    
    # Filter to necessary columns and simulate Treated group
    df["treated"] = (df["seller_quality_score"] > 0.6).astype(int)
    
    # Expand into 2 time periods (Year=0 and Year=1)
    df_pre = df.copy()
    df_pre["year"] = 0
    df_pre["post"] = 0
    
    df_post = df.copy()
    df_post["year"] = 1
    df_post["post"] = 1
    # Simulate a 15% reduction in returns for treated sellers in the post period
    df_post.loc[df_post["treated"] == 1, "proxy_return_rate"] *= 0.85
    
    df_panel = pd.concat([df_pre, df_post])
    df_panel["treat_post"] = df_panel["treated"] * df_panel["post"]
    
    # Set multi-index for PanelOLS
    df_panel = df_panel.set_index(["seller_id", "year"])
    
    print("▸ Fitting PanelOLS (DiD with Entity & Time Fixed Effects) …")
    mod = PanelOLS.from_formula(
        "proxy_return_rate ~ treat_post + EntityEffects + TimeEffects", 
        data=df_panel
    )
    res = mod.fit()
    
    print("\n── Diff-in-Diff Validation ─────────────────────────────")
    print(f"  DiD Coefficient : {res.params['treat_post']:.4f}")
    print(f"  t-stat          : {res.tstats['treat_post']:.2f}")
    print(f"  p-value         : {res.pvalues['treat_post']:.4e}")
    print("────────────────────────────────────────────────────────")
    
    if res.pvalues['treat_post'] < 0.05:
        print("  ✓ SUCCESS — Policy interaction is significant (p < 0.05).")
        print(f"    Sellers in the treatment group saw a {abs(res.params['treat_post']*100):.1f}% drop in returns.")
    
    # ── Quick visualization ────────────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    df_plot = df_panel.reset_index()
    summary = df_plot.groupby(["year", "treated"])["proxy_return_rate"].mean().unstack()
    
    plt.plot(summary.index, summary[0], 'o-', label="Control Group", color="#6b738a")
    plt.plot(summary.index, summary[1], 's-', label="Treated Group", color="#3ddc84")
    plt.axvline(0.5, color="white", linestyle="--", alpha=0.5, label="Policy Change")
    
    plt.title("DiD Validation: Effect of Platform Quality Policy", color="white")
    plt.xticks([0, 1], ["2023 (Pre)", "2024 (Post)"], color="white")
    plt.yticks(color="white")
    plt.legend()
    plt.gca().set_facecolor('#0d1117')
    plt.gcf().set_facecolor('#0d1117')
    plt.savefig(PLOT_PATH, facecolor='#0d1117')
    print(f"\n✓ DiD plot saved to {PLOT_PATH}")

if __name__ == "__main__":
    run_did()
