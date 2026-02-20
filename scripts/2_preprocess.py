"""
Script 2: Data Preprocessing & Feature Engineering
===================================================
Transforms the raw rice-price data into a clean, feature-rich
ML-ready dataset.

Features engineered:
  ─ Temporal      : year, month, quarter, season (Yala/Maha)
  ─ Lags          : price 1, 2, 3 months ago (autoregressive signals)
  ─ Rolling stats : 3-month & 6-month rolling mean / std
  ─ Macro         : approximate USD/LKR exchange rate per month
  ─ Crisis flag   : binary indicator for 2022 Sri Lanka economic crisis
  ─ Variety enc.  : one-hot encoding for rice variety
  ─ Province enc. : one-hot encoding for province (admin1)

Target variable: price_lkr (retail rice price per KG)
"""

import os
import numpy as np
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_RICE_CSV  = os.path.join(BASE_DIR, "data", "raw",       "rice_prices_raw.csv")
PROC_CSV      = os.path.join(BASE_DIR, "data", "processed", "rice_prices_processed.csv")
os.makedirs(os.path.join(BASE_DIR, "data", "processed"), exist_ok=True)


# ─── Approximate Monthly USD/LKR Exchange Rates ───────────────────────────────
# Compiled from CBSL Annual Reports & Central Bank Statistical Digest
# (publicly available at cbsl.gov.lk)
# Key: "YYYY-MM", Value: approximate mid-rate LKR per 1 USD
USD_LKR_RATES = {
    # 2004–2010 baseline period
    **{f"200{y}-{m:02d}": v for y, m, v in [
        (4, 1, 103.5), (4, 6, 103.2), (4, 12, 105.0),
        (5, 1, 103.0), (5, 6, 101.5), (5, 12, 102.8),
        (6, 1, 103.2), (6, 6, 103.5), (6, 12, 107.6),
        (7, 1, 110.5), (7, 6, 111.0), (7, 12, 111.0),
        (8, 1, 108.5), (8, 6, 108.3), (8, 12, 113.0),
        (9, 1, 113.5), (9, 6, 114.0), (9, 12, 114.9),
    ]},
    # 2010–2019 stable growth period
    "2010-01": 114.0, "2010-06": 113.0, "2010-12": 111.0,
    "2011-01": 111.0, "2011-06": 110.2, "2011-12": 113.9,
    "2012-01": 114.0, "2012-06": 133.7, "2012-12": 127.5,
    "2013-01": 127.8, "2013-06": 130.8, "2013-12": 130.5,
    "2014-01": 130.7, "2014-06": 130.2, "2014-12": 131.1,
    "2015-01": 133.2, "2015-06": 136.3, "2015-12": 144.1,
    "2016-01": 144.4, "2016-06": 146.9, "2016-12": 149.8,
    "2017-01": 151.5, "2017-06": 153.5, "2017-12": 153.7,
    "2018-01": 155.0, "2018-06": 160.8, "2018-12": 182.5,
    "2019-01": 182.7, "2019-06": 177.2, "2019-12": 181.6,
    # 2020 COVID disruption
    "2020-01": 182.0, "2020-02": 183.0, "2020-03": 193.7,
    "2020-04": 190.5, "2020-05": 188.8, "2020-06": 186.7,
    "2020-07": 185.4, "2020-08": 185.5, "2020-09": 185.6,
    "2020-10": 185.8, "2020-11": 185.9, "2020-12": 187.5,
    # 2021 pre-crisis tightening
    "2021-01": 188.0, "2021-02": 188.5, "2021-03": 189.0,
    "2021-04": 190.0, "2021-05": 197.0, "2021-06": 200.0,
    "2021-07": 200.5, "2021-08": 201.0, "2021-09": 202.0,
    "2021-10": 202.5, "2021-11": 203.0, "2021-12": 200.5,
    # 2022 CRISIS — massive depreciation
    "2022-01": 202.0, "2022-02": 203.0, "2022-03": 232.0,
    "2022-04": 301.0, "2022-05": 349.0, "2022-06": 358.0,
    "2022-07": 358.0, "2022-08": 363.0, "2022-09": 363.0,
    "2022-10": 365.0, "2022-11": 362.0, "2022-12": 363.0,
    # 2023 stabilisation
    "2023-01": 360.0, "2023-02": 357.0, "2023-03": 325.0,
    "2023-04": 320.0, "2023-05": 315.0, "2023-06": 310.0,
    "2023-07": 316.0, "2023-08": 320.0, "2023-09": 325.0,
    "2023-10": 323.0, "2023-11": 322.0, "2023-12": 320.0,
    # 2024 recovery
    "2024-01": 315.0, "2024-02": 310.0, "2024-03": 305.0,
    "2024-04": 300.0, "2024-05": 303.0, "2024-06": 304.0,
    "2024-07": 304.0, "2024-08": 299.0, "2024-09": 298.0,
    "2024-10": 298.0, "2024-11": 297.0, "2024-12": 297.0,
    # 2025 (partial)
    "2025-01": 299.0, "2025-02": 300.0, "2025-03": 300.0,
    "2025-04": 300.0, "2025-05": 300.0, "2025-06": 300.0,
    "2025-07": 301.0, "2025-08": 302.0,
}


# ─── CCPI Food Inflation (Year-on-Year %) ─────────────────────────────────────
# Source: Department of Census and Statistics, CBSL Monthly Bulletins
FOOD_INFLATION_YOY = {
    # 2019
    "2019-01": 1.5,  "2019-02": 1.8,  "2019-03": 2.0,
    "2019-04": 2.5,  "2019-05": 2.8,  "2019-06": 3.0,
    "2019-07": 3.5,  "2019-08": 4.0,  "2019-09": 4.5,
    "2019-10": 5.0,  "2019-11": 5.5,  "2019-12": 6.3,
    # 2020
    "2020-01": 12.4, "2020-02": 11.0, "2020-03": 10.5,
    "2020-04": 9.5,  "2020-05": 8.5,  "2020-06": 9.0,
    "2020-07": 10.9, "2020-08": 12.3, "2020-09": 11.5,
    "2020-10": 10.0, "2020-11": 9.5,  "2020-12": 9.0,
    # 2021
    "2021-01": 8.5,  "2021-02": 8.0,  "2021-03": 8.5,
    "2021-04": 9.0,  "2021-05": 9.5,  "2021-06": 10.0,
    "2021-07": 10.5, "2021-08": 11.0, "2021-09": 11.5,
    "2021-10": 12.8, "2021-11": 17.5, "2021-12": 22.0,
    # 2022 — PEAK CRISIS
    "2022-01": 25.0, "2022-02": 30.0, "2022-03": 45.0,
    "2022-04": 58.0, "2022-05": 70.0, "2022-06": 80.0,
    "2022-07": 90.9, "2022-08": 93.7, "2022-09": 94.9,
    "2022-10": 85.6, "2022-11": 73.7, "2022-12": 64.4,
    # 2023 — Rapid stabilisation
    "2023-01": 60.1, "2023-02": 54.4, "2023-03": 47.6,
    "2023-04": 30.6, "2023-05": 21.5, "2023-06": 4.1,
    "2023-07": -1.4, "2023-08": -4.8, "2023-09": -5.2,
    "2023-10": -5.2, "2023-11": -3.6, "2023-12": 0.3,
    # 2024 — Normalisation
    "2024-01": 3.3,  "2024-02": 3.5,  "2024-03": 3.8,
    "2024-04": 2.9,  "2024-05": 0.0,  "2024-06": 0.5,
    "2024-07": 0.7,  "2024-08": 0.8,  "2024-09": 0.9,
    "2024-10": 0.8,  "2024-11": 0.6,  "2024-12": 0.8,
    # 2025 (partial)
    "2025-01": 1.0,  "2025-02": 1.2,  "2025-03": 1.3,
    "2025-04": 1.5,  "2025-05": 1.4,  "2025-06": 1.3,
    "2025-07": 1.2,  "2025-08": 1.1,
}


def load_and_clean(path: str) -> pd.DataFrame:
    """Load raw rice CSV, drop unwanted columns, parse dates."""
    print("[1/6] Loading raw rice data …")
    df = pd.read_csv(path, low_memory=False)

    # Parse date and create period key
    df["date"]       = pd.to_datetime(df["date"])
    df["year"]       = df["date"].dt.year
    df["month"]      = df["date"].dt.month
    df["period_key"] = df["date"].dt.strftime("%Y-%m")

    # Keep only retail prices (drop wholesale if any)
    df = df[df["pricetype"].str.lower() == "retail"].copy()

    # Drop columns unused for ML
    drop_cols = ["market_id", "latitude", "longitude", "commodity_id",
                 "priceflag", "currency", "category", "unit"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Rename for clarity
    df.rename(columns={
        "admin1": "province",
        "admin2": "district",
        "price":  "price_lkr",
    }, inplace=True)

    # Remove obvious outliers (price = 0 or extreme spike > 10x median)
    median_price = df["price_lkr"].median()
    df = df[(df["price_lkr"] > 0) & (df["price_lkr"] < median_price * 10)]

    print(f"   Rows after cleaning : {len(df):,}")
    print(f"   Date range          : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"   Rice varieties      : {sorted(df['commodity'].dropna().astype(str).unique())}")
    print(f"   Provinces           : {sorted(df['province'].dropna().astype(str).unique())}")
    return df


def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach USD/LKR rate and food inflation using period_key."""
    print("[2/6] Adding macro-economic features …")

    df["usd_lkr_rate"]       = df["period_key"].map(USD_LKR_RATES)
    df["food_inflation_yoy"]  = df["period_key"].map(FOOD_INFLATION_YOY)

    # Forward-fill any gaps (missing months get nearest prior value)
    df.sort_values("date", inplace=True)
    df["usd_lkr_rate"]      = df["usd_lkr_rate"].ffill().bfill()
    df["food_inflation_yoy"] = df["food_inflation_yoy"].ffill()
    # For early years (pre-2019) not in lookup, fill with 0 (stable pre-crisis baseline)
    df["food_inflation_yoy"] = df["food_inflation_yoy"].fillna(0.0)

    print(f"   USD/LKR nulls       : {df['usd_lkr_rate'].isna().sum()}")
    print(f"   Inflation nulls     : {df['food_inflation_yoy'].isna().sum()}")
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and agricultural-season features."""
    print("[3/6] Engineering temporal features …")

    df["quarter"] = df["date"].dt.quarter

    # Sri Lanka has two paddy seasons:
    #   Maha (major) : Oct → Mar  → season = 0
    #   Yala (minor) : Apr → Sep  → season = 1
    df["agri_season"] = (df["month"].between(4, 9)).astype(int)

    # Crisis period flag (Apr 2022 – Jun 2023)
    df["crisis_period"] = (
        (df["date"] >= "2022-04-01") & (df["date"] <= "2023-06-30")
    ).astype(int)

    # COVID lockdown flag (Mar 2020 – Sep 2020)
    df["covid_lockdown"] = (
        (df["date"] >= "2020-03-01") & (df["date"] <= "2020-09-30")
    ).astype(int)

    # Month sin/cos encoding for cyclical seasonality
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (province, commodity) group, compute:
      - Lag 1, 2, 3 months of price
      - 3-month and 6-month rolling mean
      - 3-month rolling std (volatility)
      - Month-over-month price change %
    """
    print("[4/6] Computing lag & rolling features …")
    df = df.sort_values(["province", "commodity", "date"]).copy()

    group_cols = ["province", "commodity"]

    for lag in [1, 2, 3]:
        df[f"price_lag_{lag}m"] = (
            df.groupby(group_cols)["price_lkr"]
            .shift(lag)
        )

    df["price_roll3m_mean"] = (
        df.groupby(group_cols)["price_lkr"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=2).mean())
    )
    df["price_roll6m_mean"] = (
        df.groupby(group_cols)["price_lkr"]
        .transform(lambda x: x.shift(1).rolling(6, min_periods=3).mean())
    )
    df["price_roll3m_std"] = (
        df.groupby(group_cols)["price_lkr"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=2).std())
    )

    df["price_mom_pct"] = (
        df.groupby(group_cols)["price_lkr"]
        .pct_change() * 100
    ).round(2)

    lag_cols = ["price_lag_1m", "price_lag_2m", "price_lag_3m",
                "price_roll3m_mean", "price_roll6m_mean", "price_roll3m_std"]
    rows_before = len(df)
    df.dropna(subset=lag_cols, inplace=True)
    print(f"   Rows dropped (insufficient history) : {rows_before - len(df):,}")
    print(f"   Final rows with lag features        : {len(df):,}")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode commodity variety and province."""
    print("[5/6] Encoding categorical variables …")
    df = pd.get_dummies(df, columns=["commodity", "province"],
                        prefix=["rice", "prov"], drop_first=False)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def save(df: pd.DataFrame):
    """Save the final processed dataset."""
    print("[6/6] Saving processed dataset …")
    df.drop(columns=["date", "period_key", "pricetype", "usdprice"],
            errors="ignore", inplace=True)
    df.to_csv(PROC_CSV, index=False)
    print(f"\n[OK] Processed dataset saved to: {PROC_CSV}")
    print(f"     Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\n     Columns:\n     {list(df.columns)}")


if __name__ == "__main__":
    df = load_and_clean(RAW_RICE_CSV)
    df = add_macro_features(df)
    df = add_temporal_features(df)
    df = add_lag_and_rolling_features(df)
    df = encode_categoricals(df)
    save(df)
    print("\n[DONE] Step 2 complete. Run scripts/3_eda.py next.")
