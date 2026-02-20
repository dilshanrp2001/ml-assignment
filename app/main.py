"""
Sri Lanka Rice Price Predictor — Streamlit App
===============================================
Run with:  streamlit run app/main.py

Note: page modules live in app/_views/ (NOT pages/)
so Streamlit does NOT auto-generate its own navigation.
"""

import streamlit as st
import os
from PIL import Image

# Load logo
_LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
_logo = Image.open(_LOGO_PATH)

st.set_page_config(
    page_title  = "Rice Price Predictor",
    page_icon   = _logo,
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Dark gradient background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f1117 0%, #1a1d2e 60%, #0f1117 100%);
}
[data-testid="stSidebar"] {
    background: rgba(26, 29, 46, 0.95);
    border-right: 1px solid #2a2d3e;
}

/* Card style */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(108,99,255,0.15), rgba(67,233,123,0.08));
    border: 1px solid rgba(108,99,255,0.3);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #6c63ff; }
.metric-label { font-size: 0.85rem; color: #a0a0b0; margin-top: .2rem; }

/* Badge */
.badge {
    display: inline-block;
    padding: .25rem .65rem;
    border-radius: 20px;
    font-size: .78rem;
    font-weight: 600;
    background: rgba(108,99,255,0.2);
    border: 1px solid rgba(108,99,255,0.4);
    color: #a89bff;
    margin-right: .4rem;
}

/* Headings */
h1 { color: #ffffff !important; }
h2 { color: #c8c4ff !important; }
h3 { color: #a89bff !important; }

/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Navigation ───────────────────────────────────────────────────────
with st.sidebar:
    # Logo image
    col_logo = st.columns([1, 3, 1])[1]
    with col_logo:
        st.image(_logo, width=110)
    st.markdown("""
    <div style='text-align:center; padding: .4rem 0 1.2rem 0;'>
        <div style='font-size:1.15rem; font-weight:700; color:#c8c4ff;'>Rice Price Predictor</div>
        <div style='font-size:0.75rem; color:#6c63ff; margin-top:.3rem;'>Sri Lanka · ML Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠  Overview",
         "📊  Data Explorer",
         "🤖  Predict Price",
         "🔍  Explainability"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:.75rem; color:#555870; padding:.5rem 0;'>
        <b style='color:#6c63ff'>Data source:</b> WFP / HDX<br>
        <b style='color:#6c63ff'>Algorithm:</b> XGBoost Regressor<br>
        <b style='color:#6c63ff'>Coverage:</b> 2004 – 2025<br>
        <b style='color:#6c63ff'>Varieties:</b> 5 · <b style='color:#6c63ff'>Provinces:</b> 9
    </div>
    """, unsafe_allow_html=True)

# ── Route to pages ───────────────────────────────────────────────────────────
if   "Overview"       in page:
    from app._views import overview;       overview.show()
elif "Data Explorer"  in page:
    from app._views import data_explorer;  data_explorer.show()
elif "Predict"        in page:
    from app._views import predict;        predict.show()
elif "Explainability" in page:
    from app._views import explainability; explainability.show()
