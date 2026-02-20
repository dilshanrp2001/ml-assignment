"""
Script 3: Exploratory Data Analysis (EDA)
==========================================
Produces 6 publication-quality plots saved to outputs/eda/
  1. Rice price trend over time (by variety)
  2. Price distribution by rice variety (box plot)
  3. Average price by province (bar chart)
  4. Correlation heatmap of numeric features
  5. Price vs USD/LKR exchange rate (scatter)
  6. Monthly seasonality — average price by month
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_CSV  = os.path.join(BASE_DIR, "data", "processed", "rice_prices_processed.csv")
OUT_DIR   = os.path.join(BASE_DIR, "outputs", "eda")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#3a3d4d",
    "text.color":       "#e0e0e0",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#a0a0b0",
    "ytick.color":      "#a0a0b0",
    "grid.color":       "#2a2d3d",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.5,
    "font.family":      "sans-serif",
    "font.size":        11,
})

PALETTE  = ["#6c63ff", "#ff6584", "#43e97b", "#f7971e", "#2980b9"]
CRISIS_COLOR = "#ff4444"

# ─── Load ─────────────────────────────────────────────────────────────────────
print("[EDA] Loading processed dataset …")
df = pd.read_csv(PROC_CSV)

# Reconstruct useful raw columns from the OHE
variety_cols = [c for c in df.columns if c.startswith("rice_")]
prov_cols    = [c for c in df.columns if c.startswith("prov_")]

df["variety"]  = df[variety_cols].idxmax(axis=1).str.replace("rice_", "", regex=False)
df["province"] = df[prov_cols].idxmax(axis=1).str.replace("prov_", "", regex=False)
df["date"]     = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str))

print(f"   Rows: {len(df):,}  |  Date: {df['year'].min()} – {df['year'].max()}")
print(f"   Varieties : {sorted(df['variety'].unique())}")
print(f"   Provinces : {sorted(df['province'].unique())}")


# ─── Plot 1: Price Trend Over Time ────────────────────────────────────────────
print("\n[1/6] Price trend over time …")
fig, ax = plt.subplots(figsize=(14, 5))

for i, var in enumerate(sorted(df["variety"].unique())):
    sub = df[df["variety"] == var].groupby("date")["price_lkr"].mean()
    ax.plot(sub.index, sub.values, label=var, color=PALETTE[i % len(PALETTE)],
            linewidth=1.8, alpha=0.9)

# Crisis shading
ax.axvspan(pd.Timestamp("2022-04-01"), pd.Timestamp("2023-06-30"),
           alpha=0.15, color=CRISIS_COLOR, label="2022 Economic Crisis")
ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-09-30"),
           alpha=0.12, color="#f7971e", label="COVID Lockdown")

ax.set_title("Sri Lanka Rice Retail Price Trend (LKR/KG) — 2004 to 2025",
             fontsize=13, fontweight="bold", pad=14)
ax.set_xlabel("Date")
ax.set_ylabel("Average Retail Price (LKR/KG)")
ax.legend(fontsize=9, loc="upper left", framealpha=0.3)
ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "1_price_trend.png"), dpi=150)
plt.close()
print("   Saved → 1_price_trend.png")


# ─── Plot 2: Price Distribution by Variety (Box Plot) ────────────────────────
print("[2/6] Price distribution by variety …")
fig, ax = plt.subplots(figsize=(10, 5))

order = df.groupby("variety")["price_lkr"].median().sort_values().index
bp = ax.boxplot(
    [df[df["variety"] == v]["price_lkr"].values for v in order],
    labels=[v.replace("Rice ", "") for v in order],
    patch_artist=True,
    medianprops=dict(color="white", linewidth=2),
    whiskerprops=dict(color="#a0a0b0"),
    capprops=dict(color="#a0a0b0"),
    flierprops=dict(marker="o", color="#ff6584", alpha=0.3, markersize=3),
)
for patch, color in zip(bp["boxes"], PALETTE):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_title("Price Distribution by Rice Variety (LKR/KG)", fontsize=13,
             fontweight="bold", pad=14)
ax.set_xlabel("Rice Variety")
ax.set_ylabel("Retail Price (LKR/KG)")
ax.grid(True, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "2_price_by_variety.png"), dpi=150)
plt.close()
print("   Saved → 2_price_by_variety.png")


# ─── Plot 3: Average Price by Province (Horizontal Bar) ──────────────────────
print("[3/6] Average price by province …")
fig, ax = plt.subplots(figsize=(10, 5))

prov_avg = df.groupby("province")["price_lkr"].mean().sort_values(ascending=True)
bars = ax.barh(prov_avg.index, prov_avg.values, color=PALETTE[0], alpha=0.8, height=0.6)

# Color bars by value (gradient)
max_val = prov_avg.max()
for bar, val in zip(bars, prov_avg.values):
    ratio = val / max_val
    bar.set_color(plt.cm.cool(ratio))  # type: ignore
    ax.text(val + 2, bar.get_y() + bar.get_height() / 2,
            f"LKR {val:.0f}", va="center", fontsize=9, color="#e0e0e0")

ax.set_title("Average Rice Retail Price by Province (LKR/KG)", fontsize=13,
             fontweight="bold", pad=14)
ax.set_xlabel("Average Retail Price (LKR/KG)")
ax.set_ylabel("Province")
ax.grid(True, axis="x")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "3_price_by_province.png"), dpi=150)
plt.close()
print("   Saved → 3_price_by_province.png")


# ─── Plot 4: Correlation Heatmap ─────────────────────────────────────────────
print("[4/6] Correlation heatmap …")
numeric_cols = [
    "price_lkr", "year", "month", "usd_lkr_rate", "food_inflation_yoy",
    "price_lag_1m", "price_lag_2m", "price_lag_3m",
    "price_roll3m_mean", "price_roll6m_mean", "price_roll3m_std",
    "price_mom_pct", "agri_season", "crisis_period", "covid_lockdown",
]
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm_r",
    center=0, linewidths=0.4, linecolor="#0f1117",
    ax=ax, annot_kws={"size": 8},
    cbar_kws={"shrink": 0.8},
)
ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold", pad=14)
ax.tick_params(axis="x", rotation=45, labelsize=8)
ax.tick_params(axis="y", rotation=0,  labelsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "4_correlation_heatmap.png"), dpi=150)
plt.close()
print("   Saved → 4_correlation_heatmap.png")


# ─── Plot 5: Price vs USD/LKR Rate ───────────────────────────────────────────
print("[5/6] Price vs USD/LKR rate …")
fig, ax = plt.subplots(figsize=(10, 5))

sc = ax.scatter(
    df["usd_lkr_rate"], df["price_lkr"],
    c=df["year"], cmap="plasma", alpha=0.45, s=20, edgecolors="none"
)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Year", color="#e0e0e0")
cbar.ax.yaxis.set_tick_params(color="#e0e0e0")

ax.set_title("Rice Retail Price vs USD/LKR Exchange Rate", fontsize=13,
             fontweight="bold", pad=14)
ax.set_xlabel("USD/LKR Exchange Rate")
ax.set_ylabel("Retail Price (LKR/KG)")
ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "5_price_vs_usdlkr.png"), dpi=150)
plt.close()
print("   Saved → 5_price_vs_usdlkr.png")


# ─── Plot 6: Monthly Seasonality ─────────────────────────────────────────────
print("[6/6] Monthly seasonality …")
month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

fig, ax = plt.subplots(figsize=(10, 5))
monthly = df.groupby("month")["price_lkr"].mean()
ax.bar(monthly.index, monthly.values, color=PALETTE[2], alpha=0.75, width=0.6)
ax.plot(monthly.index, monthly.values, color="white", linewidth=1.8,
        marker="o", markersize=5)

# Shade Maha / Yala seasons
ax.axvspan(0.5, 3.5,  alpha=0.08, color="#6c63ff", label="Maha harvest (Jan–Mar)")
ax.axvspan(9.5, 12.5, alpha=0.08, color="#6c63ff")
ax.axvspan(3.5, 9.5,  alpha=0.08, color="#f7971e", label="Yala season (Apr–Sep)")

ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names)
ax.set_title("Average Rice Price by Month — Seasonal Pattern", fontsize=13,
             fontweight="bold", pad=14)
ax.set_xlabel("Month")
ax.set_ylabel("Average Retail Price (LKR/KG)")
ax.legend(fontsize=9, framealpha=0.3)
ax.grid(True, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "6_seasonality.png"), dpi=150)
plt.close()
print("   Saved → 6_seasonality.png")


print(f"\n[DONE] All 6 EDA plots saved to: {OUT_DIR}")
