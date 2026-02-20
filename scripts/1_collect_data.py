"""
Script 1: Data Collection
=========================
Downloads the WFP / HDX Sri Lanka Food Prices dataset and extracts
all rice-related price records.

Source: WFP Price Database via Humanitarian Data Exchange (HDX)
URL   : https://data.humdata.org/dataset/wfp-food-prices-for-sri-lanka
License: CC BY-IGO (free for academic and research use)

Rice commodities available in this dataset:
  - Rice (red nadu)    → main locally-grown variety
  - Rice (white)       → generic white rice
  - Rice (medium grain)→ medium-grain variety
"""

import os
import requests
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

RAW_ALL_CSV  = os.path.join(RAW_DIR, "wfp_food_prices_lka.csv")
RAW_RICE_CSV = os.path.join(RAW_DIR, "rice_prices_raw.csv")

# ─── Dataset URL ──────────────────────────────────────────────────────────────
# Direct CSV download from HDX (WFP Sri Lanka Food Prices)
HDX_CSV_URL = (
    "https://data.humdata.org/dataset/0298c598-d312-4771-b564-f4ac4d831f05"
    "/resource/3638f0d6-9969-48cf-a919-1d879d037ec6/download"
)


def download_raw_data():
    """Download the full WFP Sri Lanka food-price CSV from HDX."""
    if os.path.exists(RAW_ALL_CSV):
        print(f"[INFO] Raw file already exists: {RAW_ALL_CSV}")
        print("[INFO] Delete it manually if you want to re-download.")
        return

    print("[DOWNLOAD] Fetching WFP Sri Lanka Food Prices dataset …")
    print(f"  Source : {HDX_CSV_URL}")

    headers = {"User-Agent": "Mozilla/5.0 (ML-Assignment Research)"}
    response = requests.get(HDX_CSV_URL, headers=headers, timeout=60)
    response.raise_for_status()

    with open(RAW_ALL_CSV, "wb") as f:
        f.write(response.content)

    size_kb = os.path.getsize(RAW_ALL_CSV) / 1024
    print(f"[OK] Saved to: {RAW_ALL_CSV}  ({size_kb:.1f} KB)")


def extract_rice_data():
    """Filter the full dataset to keep only rice commodities."""
    print("\n[FILTER] Extracting rice-related records …")
    df = pd.read_csv(RAW_ALL_CSV, low_memory=False)

    print(f"  Total rows in full dataset  : {len(df):,}")
    print(f"  Unique commodities found    : {df['commodity'].nunique()}")
    print(f"  All commodities:\n  {sorted(df['commodity'].unique())}")

    # Keep rows where commodity name contains 'rice' (case-insensitive)
    rice_mask = df["commodity"].str.lower().str.contains("rice", na=False)
    rice_df   = df[rice_mask].copy()

    print(f"\n  Rice rows extracted         : {len(rice_df):,}")
    print(f"  Rice varieties found        : {sorted(rice_df['commodity'].unique())}")
    print(f"  Date range                  : {rice_df['date'].min()} → {rice_df['date'].max()}")
    print(f"  Markets (provinces/cities)  : {sorted(rice_df['market'].unique())}")

    rice_df.to_csv(RAW_RICE_CSV, index=False)
    print(f"\n[OK] Raw rice dataset saved to: {RAW_RICE_CSV}")
    return rice_df


def show_sample(rice_df: pd.DataFrame):
    """Print a formatted preview of the extracted rice data."""
    print("\n" + "=" * 70)
    print("  SAMPLE — First 10 rows of extracted rice dataset")
    print("=" * 70)
    cols = ["date", "admin1", "market", "commodity", "pricetype", "currency", "price", "usdprice"]
    print(rice_df[cols].head(10).to_string(index=False))

    print("\n" + "=" * 70)
    print("  RICE PRICE SUMMARY (LKR per KG)")
    print("=" * 70)
    summary = (
        rice_df.groupby("commodity")["price"]
        .agg(["count", "min", "mean", "max"])
        .rename(columns={"count": "records", "min": "min_LKR",
                         "mean": "avg_LKR", "max": "max_LKR"})
        .round(2)
    )
    print(summary.to_string())
    print("=" * 70)


if __name__ == "__main__":
    download_raw_data()
    rice_df = extract_rice_data()
    show_sample(rice_df)
    print("\n[DONE] Step 1 complete. Run scripts/2_preprocess.py next.")
