"""Page 4 — Explainability (XAI)"""
import os
import streamlit as st
from PIL import Image

BASE   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
XAI    = os.path.join(BASE, "outputs", "explainability")
EVAL   = os.path.join(BASE, "outputs", "evaluation")
DARK   = "#0f1117"
PURPLE = "#6c63ff"

def img(path, caption="", width=None):
    if os.path.exists(path):
        if width:
            st.image(path, caption=caption, width=width)
        else:
            st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Plot not found: {os.path.basename(path)}. Run `5_explainability.py` first.")


def show():
    h1, h2 = st.columns([1, 8])
    with h1:
        logo_path = os.path.join(BASE, "app", "assets", "logo.png")
        if os.path.exists(logo_path):
            st.image(logo_path, width=70)
        else:
            st.markdown("<h1 style='font-size:2rem;margin:0;'>🔍</h1>", unsafe_allow_html=True)
    with h2:
        st.markdown("""
        <h1 style='background:linear-gradient(90deg,#6c63ff,#43e97b);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   font-size:2rem;font-weight:800;margin:0;'>Model Explainability (XAI)</h1>
        <p style='color:#8888aa;margin-top:0;'>
            Interpreting <b style='color:#c8c4ff;'>XGBoost</b> predictions using
            <span style='color:#6c63ff;font-weight:600;'>SHAP</span> and
            <span style='color:#43e97b;font-weight:600;'>Partial Dependence Plots</span>.
        </p>
        """, unsafe_allow_html=True)

    # ── Why XAI? ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='background:rgba(108,99,255,0.08);border-left:3px solid #6c63ff;
                border-radius:8px;padding:1rem 1.2rem;margin-bottom:1.2rem;'>
        <b style='color:#a89bff;'>What is XAI?</b><br>
        <span style='color:#8888aa;font-size:.9rem;'>
        Explainable AI (XAI) techniques help us understand <i>why</i> a model makes a particular
        prediction — not just <i>what</i> it predicts. This is critical for trust, validation,
        and assignment marks!
        </span>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 SHAP Summary",
        "📈 SHAP Dependence",
        "📉 Partial Dependence",
        "📋 Model Evaluation",
    ])

    # ── Tab 1: SHAP Summary ───────────────────────────────────────────────────
    with tab1:
        st.markdown("### SHAP Feature Importance")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Bar Plot — Mean |SHAP|**")
            img(os.path.join(XAI, "2_shap_bar.png"),
                "Average absolute SHAP value per feature")
            st.markdown("""
            <div style='font-size:.85rem;color:#8888aa;'>
                Each bar shows the average impact of that feature across all predictions.
                Longer bar = more influential feature.
            </div>""", unsafe_allow_html=True)

        with col_b:
            st.markdown("**Beeswarm Plot — All Predictions**")
            img(os.path.join(XAI, "1_shap_summary.png"),
                "Each dot = one prediction; colour = feature value")
            st.markdown("""
            <div style='font-size:.85rem;color:#8888aa;'>
                Red dots = high feature value, Blue dots = low value.
                Position on x-axis shows whether it pushes the price up or down.
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 🧠 Key Findings")
        findings = [
            ("🥇 price_lag_1m", "Last month's price is the #1 predictor — rice prices are highly auto-correlated"),
            ("🥈 price_roll3m_mean", "The 3-month rolling average captures the medium-term price trend"),
            ("🥉 usd_lkr_rate", "Exchange rate directly drives import cost → retail price"),
            ("4️⃣ food_inflation_yoy", "Macro inflation is a strong leading indicator during the 2022 crisis"),
            ("5️⃣ crisis_period", "The 2022 crisis flag captures the structural price shock (+3× prices)"),
        ]
        for rank, finding in findings:
            st.markdown(f"""
            <div style='display:flex;gap:.8rem;align-items:start;margin:.6rem 0;'>
                <span style='font-size:1.1rem;'>{rank.split()[0]}</span>
                <div>
                    <b style='color:#c8c4ff;'>{rank.split(" ",1)[1]}</b><br>
                    <span style='font-size:.88rem;color:#8888aa;'>{finding}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Tab 2: SHAP Dependence ────────────────────────────────────────────────
    with tab2:
        st.markdown("### SHAP Dependence Plots")
        st.markdown("""<p style='color:#8888aa;font-size:.9rem;'>
        These plots show how a single feature's value affects the model's prediction.
        Each point is one sample in the test set.</p>""", unsafe_allow_html=True)

        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Top Feature Dependence**")
            img(os.path.join(XAI,"3_shap_dependence_lag1m.png"))
            st.markdown("""<p style='font-size:.83rem;color:#8888aa;'>
            As the previous month's price rises, the
            SHAP value also increases — the model correctly learns
            price momentum (higher past → higher future).</p>""", unsafe_allow_html=True)

        with d2:
            st.markdown("**2nd Feature Dependence**")
            img(os.path.join(XAI,"4_shap_dependence_2nd.png"))
            st.markdown("""<p style='font-size:.83rem;color:#8888aa;'>
            This shows how the second most important feature interacts
            with the model's output, revealing non-linear relationships
            that linear models cannot capture.</p>""", unsafe_allow_html=True)

    # ── Tab 3: PDP ───────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### Partial Dependence Plots (PDP)")
        st.markdown("""<p style='color:#8888aa;font-size:.9rem;'>
        PDPs show the <i>marginal effect</i> of one feature on the predicted price,
        averaging out all other features. They reveal the model's learned relationship.</p>""",
        unsafe_allow_html=True)

        img(os.path.join(XAI,"5_pdp_top3.png"),
            "Partial Dependence for the top 3 most important features")

        st.markdown("""
        <div style='background:rgba(67,233,123,0.08);border-left:3px solid #43e97b;
                    border-radius:8px;padding:1rem 1.2rem;margin-top:1rem;'>
            <b style='color:#43e97b;'>Domain Knowledge Alignment ✅</b><br>
            <span style='color:#8888aa;font-size:.9rem;'>
            The PDP curves align with real-world economics:<br>
            • Higher lagged price → higher current price (momentum effect)<br>
            • Higher USD/LKR rate → higher prices (import cost pass-through)<br>
            • The curves are non-linear — exactly why XGBoost outperforms linear models
            </span>
        </div>""", unsafe_allow_html=True)

    # ── Tab 4: Model Evaluation ───────────────────────────────────────────────
    with tab4:
        st.markdown("### Model Evaluation Plots")

        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**Actual vs Predicted**")
            img(os.path.join(EVAL,"actual_vs_predicted.png"))
            st.markdown("""<p style='font-size:.83rem;color:#8888aa;'>
            Points closely along the diagonal red line = excellent predictions.
            XGBoost achieves R² = 0.9987 on the test set.</p>""", unsafe_allow_html=True)

        with e2:
            st.markdown("**Residual Analysis**")
            img(os.path.join(EVAL,"residuals.png"))
            st.markdown("""<p style='font-size:.83rem;color:#8888aa;'>
            Left: residuals scattered randomly around 0 (no systematic bias).
            Right: residuals are nearly normally distributed — a good sign.</p>""",
            unsafe_allow_html=True)

        st.markdown("**Model Comparison**")
        img(os.path.join(EVAL,"model_comparison.png"))
        st.markdown("""
        <div style='background:rgba(108,99,255,0.08);border-left:3px solid #6c63ff;
                    border-radius:8px;padding:1rem 1.2rem;margin-top:.8rem;'>
            <b style='color:#a89bff;'>Why XGBoost?</b><br>
            <span style='color:#8888aa;font-size:.9rem;'>
            XGBoost achieved <b style='color:#6c63ff;'>RMSE 2.76</b> vs Linear Regression's 7.50
            and Decision Tree's 9.87 — a 63% improvement. It handles non-linearity,
            interactions between features, and is robust to outliers through
            gradient boosting with regularization (L1 + L2).
            </span>
        </div>""", unsafe_allow_html=True)
