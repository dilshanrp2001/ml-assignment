import numpy as np
import pandas as pd

# ── Constants ───────────────────────────────────────────────────────────────
VARIETIES = ["Rice (long grain)", "Rice (medium grain)", "Rice (red nadu)", "Rice (red)", "Rice (white)"]
PROVINCES = ["Central", "Eastern", "North Central", "North Western", "Northern",
             "Sabaragamuwa", "Southern", "Uva", "Western"]

def build_feature_vector(feature_list, year, month, province, variety,
                         lag1, lag2, lag3, roll3, roll6, roll3std,
                         usd_lkr, inflation, crisis, covid):
    """
    Construct a single-row feature array matching the training schema.
    Used by both FastAPI and Streamlit.
    """
    row = {f: 0 for f in feature_list}

    row["year"]              = year
    row["month"]             = month
    row["usd_lkr_rate"]      = usd_lkr
    row["food_inflation_yoy"]= inflation
    row["quarter"]           = (month - 1) // 3 + 1
    row["agri_season"]       = 1 if 4 <= month <= 9 else 0
    row["crisis_period"]     = int(crisis)
    row["covid_lockdown"]    = int(covid)
    row["month_sin"]         = np.sin(2 * np.pi * month / 12)
    row["month_cos"]         = np.cos(2 * np.pi * month / 12)
    row["price_lag_1m"]      = lag1
    row["price_lag_2m"]      = lag2
    row["price_lag_3m"]      = lag3
    row["price_roll3m_mean"] = roll3
    row["price_roll6m_mean"] = roll6
    row["price_roll3m_std"]  = roll3std
    row["price_mom_pct"]     = ((lag1 - lag2) / max(lag2, 1)) * 100

    rice_key = f"rice_{variety}"
    prov_key = f"prov_{province}"
    
    if rice_key in row: row[rice_key] = 1
    if prov_key in row: row[prov_key] = 1

    # Return as numeric array in correct order
    return np.array([row[f] for f in feature_list]).reshape(1, -1)
