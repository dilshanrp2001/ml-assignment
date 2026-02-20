"""
Script 4: Model Training & Evaluation
======================================
Algorithm: XGBoost Regressor
Task     : Predict next-month retail rice price (LKR/KG)
Split    : 70% Train / 15% Validation / 15% Test  (chronological split)

Metrics  : RMSE, MAE, R², MAPE
Baselines: Linear Regression, Decision Tree, Random Forest (for comparison)

Outputs  : outputs/models/xgb_model.pkl
           outputs/evaluation/
             - metrics_comparison.csv
             - actual_vs_predicted.png
             - residuals.png
             - learning_curve.png
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model    import LinearRegression
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor
from sklearn.metrics         import (mean_squared_error,
                                     mean_absolute_error,
                                     r2_score)
from xgboost import XGBRegressor

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_CSV   = os.path.join(BASE_DIR, "data", "processed", "rice_prices_processed.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "outputs", "models")
EVAL_DIR   = os.path.join(BASE_DIR, "outputs", "evaluation")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR,  exist_ok=True)

# ─── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117", "axes.facecolor": "#1a1d27",
    "axes.edgecolor":   "#3a3d4d", "text.color":     "#e0e0e0",
    "axes.labelcolor":  "#e0e0e0", "xtick.color":    "#a0a0b0",
    "ytick.color":      "#a0a0b0", "grid.color":     "#2a2d3d",
    "grid.linestyle":   "--",      "grid.linewidth":  0.5,
    "font.family":      "sans-serif", "font.size":   11,
})
PALETTE = ["#6c63ff", "#ff6584", "#43e97b", "#f7971e"]


# ─── 1. Load & Prepare ────────────────────────────────────────────────────────
print("=" * 60)
print("  STEP 4 — MODEL TRAINING & EVALUATION")
print("=" * 60)
print("\n[1/5] Loading processed dataset …")
df = pd.read_csv(PROC_CSV)

# Drop non-numeric / identifier columns
drop_cols = ["district", "market"]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

TARGET  = "price_lkr"
FEATURES = [c for c in df.columns if c != TARGET]

X = df[FEATURES].values
y = df[TARGET].values

print(f"   Features : {len(FEATURES)}")
print(f"   Samples  : {len(X):,}")
print(f"   Target   : {TARGET}  (mean={y.mean():.1f}, std={y.std():.1f})")


# ─── 2. Chronological Train / Val / Test Split ────────────────────────────────
print("\n[2/5] Splitting data (chronological — no shuffle) …")
n      = len(X)
n_train = int(n * 0.70)
n_val   = int(n * 0.15)
n_test  = n - n_train - n_val

X_train, y_train = X[:n_train],            y[:n_train]
X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
X_test,  y_test  = X[n_train+n_val:],      y[n_train+n_val:]

print(f"   Train : {n_train:,} rows  ({n_train/n*100:.0f}%)")
print(f"   Val   : {n_val:,}  rows  ({n_val/n*100:.0f}%)")
print(f"   Test  : {n_test:,}  rows  ({n_test/n*100:.0f}%)")


# ─── 3. Define & Train Models ─────────────────────────────────────────────────
print("\n[3/5] Training models …")

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree":     DecisionTreeRegressor(max_depth=8, random_state=42),
    "Random Forest":     RandomForestRegressor(n_estimators=100, max_depth=10,
                                               random_state=42, n_jobs=-1),
    "XGBoost":          XGBRegressor(
        n_estimators   = 300,
        learning_rate  = 0.05,
        max_depth      = 6,
        subsample      = 0.8,
        colsample_bytree = 0.8,
        reg_alpha      = 0.1,
        reg_lambda     = 1.0,
        random_state   = 42,
        verbosity      = 0,
        eval_metric    = "rmse",
    ),
}


def evaluate(model, X_tr, y_tr, X_te, y_te, name):
    """Train model and return dict of metrics."""
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    mae  = mean_absolute_error(y_te, pred)
    r2   = r2_score(y_te, pred)
    mape = np.mean(np.abs((y_te - pred) / np.clip(y_te, 1, None))) * 100
    print(f"     {name:<22} RMSE={rmse:7.2f}  MAE={mae:7.2f}  "
          f"R²={r2:.4f}  MAPE={mape:.2f}%")
    return {"Model": name, "RMSE": round(rmse, 2), "MAE": round(mae, 2),
            "R2": round(r2, 4), "MAPE_%": round(mape, 2)}


results  = []
trained  = {}
X_full   = np.vstack([X_train, X_val])
y_full   = np.hstack([y_train, y_val])

print(f"\n   ── Validation Results ──")
for name, model in models.items():
    r = evaluate(model, X_train, y_train, X_val, y_val, name)
    results.append(r)
    trained[name] = model

# Retrain best model (XGBoost) on train+val, evaluate on test
print(f"\n   ── Test Results (XGBoost, trained on train+val) ──")
xgb_final = XGBRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, verbosity=0,
    eval_metric="rmse",
)
xgb_final.fit(X_full, y_full)
test_pred = xgb_final.predict(X_test)

test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
test_mae  = mean_absolute_error(y_test, test_pred)
test_r2   = r2_score(y_test, test_pred)
test_mape = np.mean(np.abs((y_test - test_pred) / np.clip(y_test, 1, None))) * 100
print(f"     {'XGBoost (Final/Test)':<22} RMSE={test_rmse:7.2f}  MAE={test_mae:7.2f}  "
      f"R²={test_r2:.4f}  MAPE={test_mape:.2f}%")


# ─── 4. Save Artifacts ────────────────────────────────────────────────────────
print("\n[4/5] Saving model & metrics …")

# Save trained XGBoost model
model_path = os.path.join(MODEL_DIR, "xgb_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump({"model": xgb_final, "features": FEATURES,
                 "test_rmse": test_rmse, "test_r2": test_r2}, f)
print(f"   Model saved → {model_path}")

# Save metrics comparison
metrics_df = pd.DataFrame(results)
metrics_df.to_csv(os.path.join(EVAL_DIR, "metrics_comparison.csv"), index=False)
print(f"   Metrics saved → outputs/evaluation/metrics_comparison.csv")
print(f"\n   Metrics Table:\n{metrics_df.to_string(index=False)}")


# ─── 5. Evaluation Plots ──────────────────────────────────────────────────────
print("\n[5/5] Generating evaluation plots …")

# Plot A: Actual vs Predicted
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, test_pred, alpha=0.35, s=18, color="#6c63ff", edgecolors="none")
lims = [min(y_test.min(), test_pred.min()), max(y_test.max(), test_pred.max())]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect Prediction")
ax.set_title("XGBoost — Actual vs Predicted Rice Price (Test Set)",
             fontsize=12, fontweight="bold", pad=12)
ax.set_xlabel("Actual Price (LKR/KG)")
ax.set_ylabel("Predicted Price (LKR/KG)")
ax.legend(framealpha=0.3)
ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "actual_vs_predicted.png"), dpi=150)
plt.close()
print("   Saved → actual_vs_predicted.png")

# Plot B: Residual Plot
residuals = y_test - test_pred
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(test_pred, residuals, alpha=0.35, s=18,
                color="#43e97b", edgecolors="none")
axes[0].axhline(0, color="white", linewidth=1.2, linestyle="--")
axes[0].set_title("Residuals vs Predicted", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Predicted Price (LKR/KG)")
axes[0].set_ylabel("Residual (Actual − Predicted)")
axes[0].grid(True)

axes[1].hist(residuals, bins=40, color="#f7971e", alpha=0.75, edgecolor="none")
axes[1].axvline(0, color="white", linewidth=1.5, linestyle="--")
axes[1].set_title("Residual Distribution", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Residual (LKR/KG)")
axes[1].set_ylabel("Frequency")
axes[1].grid(True, axis="y")

plt.suptitle("XGBoost Residual Analysis", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "residuals.png"), dpi=150, bbox_inches="tight")
plt.close()
print("   Saved → residuals.png")

# Plot C: Model Comparison Bar Chart
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = PALETTE[:len(metrics_df)]

for ax, metric, title in zip(
    axes,
    ["RMSE", "R2"],
    ["RMSE (Lower is Better ↓)", "R² Score (Higher is Better ↑)"]
):
    bars = ax.bar(metrics_df["Model"], metrics_df[metric],
                  color=colors, alpha=0.8, width=0.5)
    for bar, val in zip(bars, metrics_df[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.02 if metric == "R2" else 1),
                f"{val:.3f}" if metric == "R2" else f"{val:.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y")

plt.suptitle("Model Comparison — Validation Set", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "model_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("   Saved → model_comparison.png")

print(f"""
{'='*60}
  FINAL RESULTS — XGBoost on Test Set
{'='*60}
  RMSE  : {test_rmse:.2f} LKR/KG
  MAE   : {test_mae:.2f}  LKR/KG
  R²    : {test_r2:.4f}
  MAPE  : {test_mape:.2f}%
{'='*60}
[DONE] Step 4 complete. Run scripts/5_explainability.py next.
""")
