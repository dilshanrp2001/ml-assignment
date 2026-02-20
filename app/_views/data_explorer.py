"""Page 2 — Data Explorer"""
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DARK   = "#0f1117"
PURPLE = "#6c63ff"
GREEN  = "#43e97b"

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE, "data", "processed", "rice_prices_processed.csv"))
    variety_cols = [c for c in df.columns if c.startswith("rice_")]
    prov_cols    = [c for c in df.columns if c.startswith("prov_")]
    df["variety"]  = df[variety_cols].idxmax(axis=1).str.replace("rice_", "", regex=False)
    df["province"] = df[prov_cols].idxmax(axis=1).str.replace("prov_", "", regex=False)
    df["date"]     = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str))
    return df


def show():
    df = load_data()

    h1, h2 = st.columns([1, 8])
    with h1:
        logo_path = os.path.join(BASE, "app", "assets", "logo.png")
        if os.path.exists(logo_path):
            st.image(logo_path, width=70)
        else:
            st.markdown("<h1 style='font-size:2rem;margin:0;'>📊</h1>", unsafe_allow_html=True)
    with h2:
        st.markdown("""
        <h1 style='background:linear-gradient(90deg,#6c63ff,#43e97b);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   font-size:2rem;font-weight:800;margin:0;'>Data Explorer</h1>
        <p style='color:#8888aa;margin-top:0;'>Filter and explore the Sri Lanka rice price dataset.</p>
        """, unsafe_allow_html=True)

    # ── Filters ───────────────────────────────────────────────────────────────
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)

        varieties  = sorted(df["variety"].unique())
        provinces  = sorted(df["province"].unique())
        year_range = st.sidebar.slider("", int(df["year"].min()),
                                       int(df["year"].max()),
                                       (2015, 2025))

        sel_var  = f1.multiselect("Rice Variety", varieties, default=varieties[:2])
        sel_prov = f2.multiselect("Province", provinces, default=["Western", "Northern"])
        year_min, year_max = f3.slider("Year Range",
                                        int(df["year"].min()), int(df["year"].max()),
                                        (2015, 2025))
        st.markdown("</div>", unsafe_allow_html=True)

    mask = (
        df["variety"].isin(sel_var if sel_var else varieties) &
        df["province"].isin(sel_prov if sel_prov else provinces) &
        df["year"].between(year_min, year_max)
    )
    fdf = df[mask].copy()

    # ── Stats row ─────────────────────────────────────────────────────────────
    s1, s2, s3, s4 = st.columns(4)
    stats = [
        (f"{len(fdf):,}",           "Records filtered",    PURPLE),
        (f"LKR {fdf['price_lkr'].mean():.0f}", "Mean price", GREEN),
        (f"LKR {fdf['price_lkr'].min():.0f} – {fdf['price_lkr'].max():.0f}", "Price range", "#f7971e"),
        (f"LKR {fdf['price_lkr'].std():.0f}", "Std deviation", "#ff6584"),
    ]
    for col,(val,lbl,color) in zip([s1,s2,s3,s4], stats):
        col.markdown(f"""<div style='background:rgba(255,255,255,0.04);border:1px solid {color}33;
            border-radius:10px;padding:1rem;text-align:center;'>
            <div style='font-size:1.3rem;font-weight:700;color:{color};'>{val}</div>
            <div style='font-size:.8rem;color:#8888aa;'>{lbl}</div></div>""",
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Time Series ───────────────────────────────────────────────────────────
    st.markdown("### 📈 Price Over Time")
    trend = fdf.groupby(["date","variety","province"])["price_lkr"].mean().reset_index()
    trend["label"] = trend["variety"] + " · " + trend["province"]
    fig = px.line(trend, x="date", y="price_lkr", color="label",
                  labels={"price_lkr":"Price (LKR/KG)", "date": "", "label":""},
                  color_discrete_sequence=px.colors.sequential.Plasma)
    fig.update_layout(paper_bgcolor=DARK, plot_bgcolor="rgba(26,29,46,0.9)",
                      font_color="#ccccdd", height=340,
                      margin=dict(l=0,r=0,t=10,b=0),
                      xaxis=dict(showgrid=True, gridcolor="#2a2d3e"),
                      yaxis=dict(showgrid=True, gridcolor="#2a2d3e"),
                      legend=dict(bgcolor="rgba(0,0,0,0.3)"))
    st.plotly_chart(fig, use_container_width=True)

    # ── Heatmap & Histogram ───────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 🗓️ Price Heatmap (Month × Year)")
        if len(fdf) > 0:
            hm = fdf.groupby(["year","month"])["price_lkr"].mean().reset_index()
            pivot = hm.pivot(index="month", columns="year", values="price_lkr")
            fig2 = px.imshow(pivot, color_continuous_scale="Viridis",
                             labels=dict(x="Year", y="Month", color="LKR/KG"),
                             aspect="auto")
            fig2.update_layout(paper_bgcolor=DARK, plot_bgcolor=DARK,
                               font_color="#ccccdd", height=300,
                               margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.markdown("### 📊 Price Distribution")
        fig3 = px.histogram(fdf, x="price_lkr", nbins=40, color="variety",
                           color_discrete_sequence=[PURPLE,"#ff6584",GREEN,"#f7971e","#2980b9"],
                           labels={"price_lkr":"Price (LKR/KG)", "variety":"Variety"},
                           barmode="overlay", opacity=0.7)
        fig3.update_layout(paper_bgcolor=DARK, plot_bgcolor="rgba(26,29,46,0.9)",
                           font_color="#ccccdd", height=300,
                           margin=dict(l=0,r=0,t=10,b=0),
                           xaxis=dict(showgrid=True, gridcolor="#2a2d3e"),
                           yaxis=dict(showgrid=True, gridcolor="#2a2d3e"))
        st.plotly_chart(fig3, use_container_width=True)

    # ── Raw Data Table ────────────────────────────────────────────────────────
    st.markdown("### 🗂️ Raw Data")
    show_cols = ["date","year","month","province","variety","price_lkr",
                 "usd_lkr_rate","food_inflation_yoy","price_lag_1m","crisis_period"]
    show_cols = [c for c in show_cols if c in fdf.columns]
    st.dataframe(
        fdf[show_cols].sort_values("date", ascending=False)
                      .reset_index(drop=True)
                      .head(200)
                      .rename(columns={
                          "price_lkr":"Price (LKR/KG)",
                          "usd_lkr_rate":"USD/LKR",
                          "food_inflation_yoy":"Food Inflation %",
                          "price_lag_1m":"Prev Month Price",
                          "crisis_period":"Crisis?",
                      }),
        use_container_width=True, height=320,
    )
