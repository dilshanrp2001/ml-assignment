import pandas as pd

raw = pd.read_csv('data/raw/wfp_food_prices_lka.csv', low_memory=False)
print('=== FULL DATASET ===')
print(f'Total rows    : {len(raw):,}')
print(f'Date range    : {raw["date"].min()} -> {raw["date"].max()}')
print(f'All commodities:')
for c in sorted(raw['commodity'].dropna().astype(str).unique()):
    print(f'  - {c}')

print()
rice = pd.read_csv('data/raw/rice_prices_raw.csv', low_memory=False)
print('=== RICE-ONLY DATASET ===')
print(f'Total rows    : {len(rice):,}')
print(f'Date range    : {rice["date"].min()} -> {rice["date"].max()}')
print(f'Rice varieties: {sorted(rice["commodity"].dropna().astype(str).unique())}')
print(f'Provinces     : {sorted(rice["admin1"].dropna().astype(str).unique())}')
print(f'Markets       : {sorted(rice["market"].dropna().astype(str).unique())}')
print()
print('--- Price summary (LKR per KG) by rice variety ---')
print(rice.groupby('commodity')['price'].agg(['count','min','mean','max']).round(2).to_string())
print()
print('--- Records per year ---')
rice['year'] = pd.to_datetime(rice['date']).dt.year
print(rice.groupby('year').size().reset_index(name='records').to_string(index=False))
print()
print(f'TOTAL RICE RECORDS: {len(rice):,}')
