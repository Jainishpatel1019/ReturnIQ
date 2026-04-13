"""
src/narrative_agent.py — Week 6, Step 1
Mistral-7B reflexion agent via Ollama.
Generates 3-sentence seller risk narratives grounded in SHAP feature ordering.
Uses 2-pass reflexion: if narrative misidentifies the top driver, it self-corrects.

Prereq: ollama serve & (in a separate terminal)
Run: python src/narrative_agent.py  (tests with a single seller)
"""

import requests
import pandas as pd
import numpy as np

OLLAMA_URL = "http://localhost:11434/api/generate"


def call_mistral(prompt: str, timeout: int = 60) -> str:
    """Call local Mistral-7B via Ollama API."""
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": "mistral", "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()["response"]
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama. Start it with: ollama serve &\n"
            "Then pull the model: ollama pull mistral"
        )


def get_shap_ranking(seller_row: pd.Series) -> list[str]:
    """Return SHAP feature names sorted by absolute contribution (highest first)."""
    shap_cols = [c for c in seller_row.index if c.startswith("shap_")]
    vals = seller_row[shap_cols].abs().sort_values(ascending=False)
    return [c.replace("shap_", "") for c in vals.index]


def extract_top_driver_from_narrative(narrative: str, features: list[str]) -> str | None:
    """Find which feature the narrative claims is the top driver."""
    narrative_lower = narrative.lower()
    for feat in features:
        if feat.replace("_", " ") in narrative_lower:
            return feat
    return None


def generate_narrative(seller_id: str, df: pd.DataFrame) -> str:
    """Generate a grounded 3-sentence risk narrative for a seller.
    Uses 2-pass reflexion to ensure the top SHAP driver is correctly identified.
    """
    row = df[df["seller_id"] == seller_id].iloc[0]
    shap_ranking = get_shap_ranking(row)

    cate_val = float(row["cate"])
    risk_label = (
        "HIGH risk" if cate_val > 0.15 else
        "MEDIUM risk" if cate_val > 0.05 else
        "LOW risk"
    )

    prompt = f"""You are a data analyst at an e-commerce marketplace.
Given seller metrics, write a precise 3-sentence plain-English risk narrative.

Seller ID: {seller_id}
Risk level (causal effect on return rate): {cate_val:.3f} → {risk_label}
Top SHAP drivers in order of importance: {shap_ranking[:3]}
Category: {row['category']}
Proxy return rate: {float(row['proxy_return_rate']):.3f}

Write exactly 3 sentences:
1. State the return risk level and what it means for the marketplace.
2. Identify the PRIMARY driver (must be {shap_ranking[0]}) and explain why it matters.
3. Give one specific, actionable recommendation to reduce return risk.
Be concise and data-driven. Do not mention seller IDs directly."""

    # Pass 1: Generate
    narrative = call_mistral(prompt)

    # Reflexion check: does narrative mention the correct top driver?
    claimed_driver = extract_top_driver_from_narrative(narrative, shap_ranking)
    if claimed_driver != shap_ranking[0]:
        # Pass 2: Correct
        fix_prompt = f"""{prompt}

Your previous response incorrectly emphasized "{claimed_driver or 'an unrecognized feature'}" as the primary driver.
The data clearly shows "{shap_ranking[0]}" has the HIGHEST SHAP magnitude and is the PRIMARY driver.
Rewrite the narrative, making sure "{shap_ranking[0]}" is explicitly identified as the main driver in sentence 2."""
        narrative = call_mistral(fix_prompt)

    return narrative.strip()


if __name__ == "__main__":
    import pathlib

    CATE_PATH = "data/processed/cate_results.parquet"

    if not pathlib.Path(CATE_PATH).exists():
        print(f"⚠  {CATE_PATH} not found. Run causal_model.py first.")
    else:
        df = pd.read_parquet(CATE_PATH)
        # Test with the seller with highest CATE
        test_seller = df.nlargest(1, "cate")["seller_id"].iloc[0]
        print(f"▸ Testing narrative agent on seller: {test_seller}")
        try:
            narrative = generate_narrative(test_seller, df)
            print(f"\nGenerated narrative:\n{'─' * 60}\n{narrative}\n{'─' * 60}")
        except RuntimeError as e:
            print(f"✗ {e}")
