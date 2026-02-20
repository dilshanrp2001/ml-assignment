"""
Creates a clean, well-labelled CSV of the collected rice price data.
Saves to: data/raw/sri_lanka_rice_prices_collected.csv
"""
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'raw', 'rice_prices_raw.csv'), low_memory=False)

# Rename columns to clear, human-readable names
df_clean = df.rename(columns={
    'date':       'Date',
    'admin1':     'Province',
    'admin2':     'District',
    'market':     'Market',
    'commodity':  'Rice_Variety',
    'pricetype':  'Price_Type',
    'currency':   'Currency',
    'price':      'Price_LKR_per_KG',
    'usdprice':   'Price_USD_per_KG',
})

# Keep only the useful columns
df_clean = df_clean[[
    'Date', 'Province', 'District', 'Market',
    'Rice_Variety', 'Price_Type',
    'Currency', 'Price_LKR_per_KG', 'Price_USD_per_KG'
]]

# Sort by date and province
df_clean['Date'] = pd.to_datetime(df_clean['Date'])
df_clean = df_clean.sort_values(['Date', 'Province', 'Rice_Variety']).reset_index(drop=True)
df_clean['Date'] = df_clean['Date'].dt.strftime('%Y-%m-%d')

# Save
out_path = os.path.join(BASE_DIR, 'data', 'raw', 'sri_lanka_rice_prices_collected.csv')
df_clean.to_csv(out_path, index=False)

print(f"Saved: {out_path}")
print(f"Shape: {df_clean.shape[0]:,} rows x {df_clean.shape[1]} columns")
print()
print("="*65)
print("  DATASET PREVIEW (first 15 rows)")
print("="*65)
print(df_clean.head(15).to_string(index=False))

print()
print("="*65)
print("  STATISTICS")
print("="*65)
print(f"  Date range      : {df_clean['Date'].min()}  ->  {df_clean['Date'].max()}")
print(f"  Rice varieties  : {sorted(df_clean['Rice_Variety'].dropna().astype(str).unique())}")
print(f"  Provinces       : {sorted(df_clean['Province'].dropna().astype(str).unique())}")
print(f"  Total records   : {len(df_clean):,}")
print()
print("  Price summary per variety (LKR/KG):")
summary = df_clean.groupby('Rice_Variety')['Price_LKR_per_KG'].agg(
    Records='count', Min='min', Mean='mean', Max='max'
).round(2)
print(summary.to_string())
