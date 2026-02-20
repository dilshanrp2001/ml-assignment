"""Page 3 — Price Prediction"""
import os, pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

BASE   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DARK   = "#0f1117"
PURPLE = "#6c63ff"
GREEN  = "#43e97b"

# ── USD/LKR lookup (same as preprocessing) ───────────────────────────────────
USD_LKR = {
    2004:103.5,2005:102.0,2006:104.0,2007:111.0,2008:110.5,
    2009:114.5,2010:113.0,2011:111.0,2012:128.0,2013:129.8,
    2014:130.8,2015:138.0,2016:147.0,2017:153.0,2018:165.0,
    2019:181.0,2020:186.0,2021:198.0,2022:332.0,2023:320.0,
    2024:305.0,2025:300.0,
}
INFLATION = {
    2004:0.0,2005:0.0,2006:0.0,2007:0.0,2008:0.0,2009:0.0,
    2010:0.0,2011:0.0,2012:0.0,2013:0.0,2014:0.0,2015:0.0,
    2016:0.0,2017:0.0,2018:0.0,2019:3.5,2020:10.0,2021:11.0,
    2022:65.0,2023:20.0,2024:2.0,2025:1.2,
}

VARIETIES  = ["Rice (long grain)","Rice (medium grain)","Rice (red nadu)","Rice (red)","Rice (white)"]
PROVINCES  = ["Central","Eastern","North Central","North Western","Northern",
              "Sabaragamuwa","Southern","Uva","Western"]

@st.cache_resource
def load_model():
    with open(os.path.join(BASE,"outputs","models","xgb_model.pkl"),"rb") as f:
        return pickle.load(f)

@st.cache_data
def load_processed():
    return pd.read_csv(os.path.join(BASE,"data","processed","rice_prices_processed.csv"))


def build_feature_vector(bundle, year, month, province, variety,
                         lag1, lag2, lag3, roll3, roll6, roll3std,
                         usd_lkr, inflation, crisis, covid):
    """Construct a single-row feature array matching the training schema."""
    features = bundle["features"]
    row = {f: 0 for f in features}

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

    return np.array([row[f] for f in features]).reshape(1, -1)


def show():
    bundle = load_model()
    df_proc = load_processed()

    h1, h2 = st.columns([1, 8])
    with h1:
        logo_path = os.path.join(BASE, "app", "assets", "logo.png")
        if os.path.exists(logo_path):
            st.image(logo_path, width=70)
        else:
            st.markdown("<h1 style='font-size:2rem;margin:0;'>🤖</h1>", unsafe_allow_html=True)
    with h2:
        st.markdown("""
        <h1 style='background:linear-gradient(90deg,#6c63ff,#43e97b);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   font-size:2rem;font-weight:800;margin:0;'>Price Prediction</h1>
        <p style='color:#8888aa;margin-top:0;'>Enter market conditions to predict retail rice price.</p>
        """, unsafe_allow_html=True)

    # ── Input Form ────────────────────────────────────────────────────────────
    with st.form("predict_form"):
        st.markdown("#### 📋 Market Conditions")
        c1, c2, c3 = st.columns(3)
        year     = c1.number_input("Year",  min_value=2004, max_value=2030, value=2025)
        month    = c2.selectbox("Month", list(range(1,13)),
                                format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                                                       "Jul","Aug","Sep","Oct","Nov","Dec"][m-1],
                                index=0)
        province = c3.selectbox("Province", PROVINCES, index=PROVINCES.index("Western"))

        c4, c5 = st.columns(2)
        variety  = c4.selectbox("Rice Variety", VARIETIES, index=2)
        usd_lkr  = c5.number_input("USD/LKR Exchange Rate",
                                    min_value=90.0, max_value=600.0,
                                    value=float(USD_LKR.get(year, 300.0)), step=1.0)

        st.markdown("#### 📊 Historical Price Data (LKR/KG)")
        p1, p2, p3 = st.columns(3)
        lag1 = p1.number_input("Last month's price",   min_value=0.0, value=220.0, step=1.0)
        lag2 = p2.number_input("2 months ago price",   min_value=0.0, value=215.0, step=1.0)
        lag3 = p3.number_input("3 months ago price",   min_value=0.0, value=210.0, step=1.0)

        roll3    = (lag1 + lag2 + lag3) / 3
        roll6    = roll3 * 0.97
        roll3std = abs(lag1 - lag3) / 2
        inflation = float(INFLATION.get(year, 2.0))

        crisis = year == 2022 or (year == 2023 and month <= 6)
        covid  = year == 2020 and 3 <= month <= 9

        st.markdown("---")
        submitted = st.form_submit_button("🔮 Predict Price", use_container_width=True)

    if submitted:
        X = build_feature_vector(bundle, year, month, province, variety,
                                 lag1, lag2, lag3, roll3, roll6, roll3std,
                                 usd_lkr, inflation, crisis, covid)
        pred = bundle["model"].predict(X)[0]

        # ── Result display ─────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        color = "#ff6584" if pred > 300 else GREEN if pred < 150 else PURPLE
        tier  = "⚠️ High" if pred > 300 else "✅ Normal" if pred < 200 else "🟡 Elevated"

        r1, r2, r3 = st.columns([2,1,1])
        r1.markdown(f"""
        <div style='background:linear-gradient(135deg,rgba(108,99,255,0.15),rgba(67,233,123,0.08));
                    border:2px solid {color}55;border-radius:16px;
                    padding:2rem;text-align:center;'>
            <div style='font-size:.95rem;color:#8888aa;margin-bottom:.5rem;'>
                Predicted Retail Price for <b style='color:#c8c4ff;'>{variety}</b>
                in <b style='color:#c8c4ff;'>{province}</b> Province
            </div>
            <div style='font-size:3.5rem;font-weight:800;color:{color};'>
                LKR {pred:.2f}
            </div>
            <div style='font-size:.9rem;color:#8888aa;margin-top:.4rem;'>per KG &nbsp;|&nbsp;
                <span style='color:{color};font-weight:600;'>{tier}</span>
            </div>
        </div>""", unsafe_allow_html=True)

        r2.markdown(f"""
        <div style='background:rgba(255,255,255,0.04);border:1px solid #2a2d3e;
                    border-radius:12px;padding:1.2rem;text-align:center;height:100%;'>
            <div style='font-size:.8rem;color:#8888aa;'>USD Equivalent</div>
            <div style='font-size:1.6rem;font-weight:700;color:#43e97b;'>
                $ {pred/usd_lkr:.3f}
            </div>
            <div style='font-size:.75rem;color:#555870;'>per KG</div>
        </div>""", unsafe_allow_html=True)

        r3.markdown(f"""
        <div style='background:rgba(255,255,255,0.04);border:1px solid #2a2d3e;
                    border-radius:12px;padding:1.2rem;text-align:center;height:100%;'>
            <div style='font-size:.8rem;color:#8888aa;'>vs Last Month</div>
            <div style='font-size:1.6rem;font-weight:700;
                        color:{"#ff6584" if pred>lag1 else "#43e97b"};'>
                {"▲" if pred>lag1 else "▼"} {abs(pred-lag1):.1f}
            </div>
            <div style='font-size:.75rem;color:#555870;'>LKR change</div>
        </div>""", unsafe_allow_html=True)

        # ── Gauge chart ────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode   = "gauge+number+delta",
            value  = pred,
            delta  = {"reference": lag1, "valueformat":".1f",
                      "increasing":{"color":"#ff6584"},"decreasing":{"color":"#43e97b"}},
            number = {"suffix":" LKR/KG","font":{"size":32,"color":"#c8c4ff"}},
            gauge  = {
                "axis": {"range":[0,500],"tickcolor":"#8888aa"},
                "bar":  {"color": color},
                "bgcolor": "rgba(26,29,46,0.8)",
                "steps": [
                    {"range":[0,150],   "color":"rgba(67,233,123,0.15)"},
                    {"range":[150,300], "color":"rgba(108,99,255,0.15)"},
                    {"range":[300,500], "color":"rgba(255,101,132,0.15)"},
                ],
                "threshold":{"line":{"color":"white","width":2},"value":pred},
            },
            title={"text":f"Price Gauge — {month}/{year}","font":{"color":"#8888aa","size":14}},
        ))
        fig.update_layout(paper_bgcolor=DARK, font_color="#ccccdd",
                          height=280, margin=dict(l=30,r=30,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

        # ── Context ────────────────────────────────────────────────────────
        if crisis:
            st.warning("⚠️ **Crisis period detected** (Apr 2022 – Jun 2023) — prices reflect economic shock conditions.")
        if covid:
            st.info("ℹ️ **COVID lockdown period** (Mar–Sep 2020) — supply chain disruptions may have elevated prices.")
