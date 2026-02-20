"""Page 1 — Overview / Home"""
import os, pickle
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE, "data", "processed", "rice_prices_processed.csv"))
    variety_cols = [c for c in df.columns if c.startswith("rice_")]
    prov_cols    = [c for c in df.columns if c.startswith("prov_")]
    df["variety"]  = df[variety_cols].idxmax(axis=1).str.replace("rice_", "", regex=False)
    df["province"] = df[prov_cols].idxmax(axis=1).str.replace("prov_", "", regex=False)
    df["date"]     = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str))
    return df

@st.cache_resource
def load_model():
    path = os.path.join(BASE, "outputs", "models", "xgb_model.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

DARK = "#0f1117"
CARD = "rgba(255,255,255,0.04)"
PURPLE = "#6c63ff"
GREEN  = "#43e97b"

def show():
    df    = load_data()
    bundle = load_model()

    # ── Hero ─────────────────────────────────────────────────────────────────
    h1, h2 = st.columns([1, 6])
    with h1:
        logo_path = os.path.join(BASE, "app", "assets", "logo.png")
        if os.path.exists(logo_path):
            st.image(logo_path, width=100)
        else:
            st.markdown("<h1 style='font-size:3rem;margin:0;'>🌾</h1>", unsafe_allow_html=True)

    with h2:
        st.markdown("""
        <div style='padding:0.5rem 0;'>
            <h1 style='background:linear-gradient(90deg,#6c63ff,#43e97b);
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                       font-size:2.6rem;font-weight:800;margin:0;'>
                Sri Lanka Rice Price Predictor
            </h1>
            <p style='color:#8888aa;font-size:1.05rem;margin-top:.2rem;'>
                ML-powered retail price forecasting across provinces & rice varieties (2004–2025)
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── KPI Cards ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("2,173",  "Training records",       PURPLE),
        ("9",      "Provinces covered",       "#43e97b"),
        ("5",      "Rice varieties",          "#f7971e"),
        (f"{bundle['test_r2']:.4f}", "Model R² score", "#ff6584"),
    ]
    for col, (val, label, color) in zip([c1,c2,c3,c4], kpis):
        col.markdown(f"""
        <div style='background:rgba(255,255,255,0.04);border:1px solid {color}33;
                    border-radius:12px;padding:1.2rem;text-align:center;'>
            <div style='font-size:1.9rem;font-weight:700;color:{color};'>{val}</div>
            <div style='font-size:.82rem;color:#8888aa;margin-top:.3rem;'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Price Trend Chart ─────────────────────────────────────────────────────
    st.markdown("### 📈 Rice Price Trend (2004 – 2025)")
    trend = df.groupby(["date", "variety"])["price_lkr"].mean().reset_index()

    fig = px.line(
        trend, x="date", y="price_lkr", color="variety",
        labels={"price_lkr": "Avg Price (LKR/KG)", "date": "", "variety": "Variety"},
        color_discrete_sequence=[PURPLE, "#ff6584", GREEN, "#f7971e", "#2980b9"],
    )
    # Crisis shading
    fig.add_vrect(x0="2022-04-01", x1="2023-06-30",
                  fillcolor="red", opacity=0.07, line_width=0,
                  annotation_text="2022 Crisis", annotation_position="top left",
                  annotation=dict(font_color="#ff6666", font_size=11))
    fig.add_vrect(x0="2020-03-01", x1="2020-09-30",
                  fillcolor="orange", opacity=0.07, line_width=0,
                  annotation_text="COVID", annotation_position="top left",
                  annotation=dict(font_color="#f7971e", font_size=11))
    fig.update_layout(
        paper_bgcolor=DARK, plot_bgcolor="rgba(26,29,46,0.9)",
        font_color="#ccccdd", height=380,
        legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="#333"),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=True, gridcolor="#2a2d3e"),
        yaxis=dict(showgrid=True, gridcolor="#2a2d3e"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Bottom Row ────────────────────────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.markdown("### 🍚 Price by Rice Variety")
        box_df = df.sort_values("variety")
        fig2   = px.box(
            box_df, x="variety", y="price_lkr",
            color="variety",
            labels={"price_lkr":"Price (LKR/KG)", "variety":""},
            color_discrete_sequence=[PURPLE,"#ff6584",GREEN,"#f7971e","#2980b9"],
        )
        fig2.update_layout(
            paper_bgcolor=DARK, plot_bgcolor="rgba(26,29,46,0.9)",
            font_color="#ccccdd", height=320,
            showlegend=False, margin=dict(l=0,r=0,t=10,b=0),
            xaxis=dict(showgrid=False, tickangle=-20),
            yaxis=dict(showgrid=True, gridcolor="#2a2d3e"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with right:
        st.markdown("### 🗺️ Avg Price by Province")
        prov_avg = df.groupby("province")["price_lkr"].mean().sort_values(ascending=True)
        fig3 = px.bar(
            prov_avg.reset_index(),
            x="price_lkr", y="province", orientation="h",
            labels={"price_lkr":"Avg Price (LKR/KG)", "province":""},
            color="price_lkr",
            color_continuous_scale=["#1a1d2e", PURPLE, GREEN],
        )
        fig3.update_layout(
            paper_bgcolor=DARK, plot_bgcolor="rgba(26,29,46,0.9)",
            font_color="#ccccdd", height=320,
            coloraxis_showscale=False,
            margin=dict(l=0,r=0,t=10,b=0),
            xaxis=dict(showgrid=True, gridcolor="#2a2d3e"),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Model Performance ─────────────────────────────────────────────────────
    st.markdown("### 🤖 Model Performance Summary")
    m1, m2, m3, m4 = st.columns(4)
    perf = [
        ("XGBoost",    "Algorithm"),
        (f"{bundle['test_rmse']:.2f} LKR", "Test RMSE"),
        (f"{bundle['test_r2']:.4f}",        "Test R²"),
        ("~1%",                              "MAPE Error"),
    ]
    colors2 = [PURPLE, "#ff6584", GREEN, "#f7971e"]
    for col,(val,lbl),color in zip([m1,m2,m3,m4], perf, colors2):
        col.markdown(f"""
        <div style='background:rgba(255,255,255,0.04);border:1px solid {color}44;
                    border-radius:10px;padding:1rem;text-align:center;'>
            <div style='font-size:1.4rem;font-weight:700;color:{color};'>{val}</div>
            <div style='font-size:.8rem;color:#8888aa;'>{lbl}</div>
        </div>""", unsafe_allow_html=True)
