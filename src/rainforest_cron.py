"""
src/rainforest_cron.py — Week 1, Step 4
Fetches live seller/product data from Rainforest API.
100 req/month free tier → ~5 sellers/day for 20 days.
Saves JSON responses to data/raw/rainforest/<date>/

Schedule via crontab (9am daily):
  crontab -e
  0 9 * * * cd ~/projects/Seller\ Project && source ~/envs/returns/bin/activate && python src/rainforest_cron.py

Run manually: python src/rainforest_cron.py
"""

import requests
import json
import os
import datetime
import pathlib
from dotenv import load_dotenv

load_dotenv()

RF_KEY = os.getenv("RAINFOREST_API_KEY")
RF_BASE = "https://api.rainforestapi.com/request"

# Seed ASINs — top Electronics products. Extend this list over 20 days.
ASINS_TO_COLLECT = [
    "B08N5WRWNW",  # Echo Dot 4th gen
    "B07XJ8C8F7",  # Fire TV Stick
    "B09B8YWXDF",  # AirPods 3rd gen
    "B09G9BL5CP",  # Apple Watch SE
    "B08F7N3R34",  # Kindle Paperwhite
]

REQUESTS_PER_DAY = 5  # conservative — 5 × 20 days = 100 total


def fetch_product(asin: str) -> dict:
    """Fetch product + seller data from Rainforest API."""
    if not RF_KEY or RF_KEY == "your_rainforest_api_key_here":
        raise ValueError(
            "RAINFOREST_API_KEY not set. Add it to your .env file.\n"
            "Get a free key at: https://www.rainforestapi.com/"
        )

    r = requests.get(
        RF_BASE,
        params={
            "api_key": RF_KEY,
            "type": "product",
            "asin": asin,
            "amazon_domain": "amazon.com",
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def main():
    today = datetime.date.today().isoformat()
    out_dir = pathlib.Path(f"data/raw/rainforest/{today}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"▸ Collecting {REQUESTS_PER_DAY} ASINs for {today} …")
    collected = 0

    for asin in ASINS_TO_COLLECT[:REQUESTS_PER_DAY]:
        out_file = out_dir / f"{asin}.json"
        if out_file.exists():
            print(f"  ⏭  {asin} already collected today — skipping.")
            continue

        try:
            data = fetch_product(asin)
            out_file.write_text(json.dumps(data, indent=2))
            name = data.get("product", {}).get("title", "unknown")[:60]
            rating = data.get("product", {}).get("rating", "?")
            print(f"  ✓ {asin} — {name} (rating: {rating})")
            collected += 1
        except Exception as e:
            print(f"  ✗ {asin} — Error: {e}")

    print(f"\n✓ Done. {collected} new records saved to {out_dir}")
    print(f"  Daily API budget remaining: ~{100 - collected} requests")


if __name__ == "__main__":
    main()
