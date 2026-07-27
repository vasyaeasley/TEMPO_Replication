import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, randint, uniform
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

# ==============================================================================
# 1. SETUP & PATH CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

print("🚀 STARTING GOAL 1.3: TEMPO XGBOOST BASELINE REPLICATION 🚀")
print("=" * 75)

# ==============================================================================
# 2. LOAD PRE-SPLIT MASTER NPZ ARCHIVE
# ==============================================================================
possible_files = list(PROCESSED_DIR.glob("*.npz")) + list(BASE_DIR.glob("*.npz"))
data_file = next(
    (
        f
        for f in possible_files
        if "14months" in f.name.lower() or "dataset" in f.name.lower()
    ),
    None,
)

if not data_file:
  raise FileNotFoundError(
      "Could not find epa_point_dataset_14months.npz in data/processed/ or"
      " root!"
  )

print(f"⏳ Loading pre-split archive: {data_file.name}...")
loaded = np.load(data_file, allow_pickle=True)

# Extract feature names safely
feature_names = [str(c) for c in loaded["feature_names"]]
print(f"📋 Detected {len(feature_names)} Covariates: {feature_names}")

# Load Train and Test matrices directly from the archive
X_train = pd.DataFrame(loaded["X_train"], columns=feature_names)
y_train = pd.Series(loaded["y_train"], name="NO2_ppb")

X_test = pd.DataFrame(loaded["X_test"], columns=feature_names)
y_test = pd.Series(loaded["y_test"], name="NO2_ppb")

# Quick data integrity cleanup (drop any accidental NaNs or Infs if present)
train_clean = ~X_train.isna().any(axis=1) & ~y_train.isna()
test_clean = ~X_test.isna().any(axis=1) & ~y_test.isna()

X_train, y_train = X_train[train_clean], y_train[train_clean]
X_test, y_test = X_test[test_clean], y_test[test_clean]

print("-" * 75)
print(
    f"✅ Pre-Split Loaded Successfully!\n"
    f"   * Training Samples : {len(X_train):,}\n"
    f"   * Testing Samples  : {len(X_test):,}\n"
    f"   * Total Covariates : {X_train.shape[1]}"
)
print("=" * 75)

# ==============================================================================
# 3. HYPERPARAMETER TUNING (EXACT TABLE I RANGES FROM PAPER)
# ==============================================================================
print("⚙️ Setting up hyperparameter sweep (Table I specifications)...")
param_distributions = {
    "n_estimators": randint(100, 701),
    "learning_rate": uniform(0.001, 0.499),
    "subsample": uniform(0.2, 0.8),
    "max_depth": randint(1, 12),
    "colsample_bytree": uniform(0.1, 0.9),
    "gamma": uniform(0.0, 0.4),
    "reg_alpha": uniform(0.0, 100.0),
    "reg_lambda": uniform(0.001, 9.999),
}

xgb_base = XGBRegressor(
    objective="reg:squarederror", random_state=42, n_jobs=-1, tree_method="hist"
)

# Use RandomizedSearchCV for fast exploration of the 8D parameter space
search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_distributions,
    n_iter=25,  # 25 candidate sweeps
    scoring="r2",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

print(f"⏳ Executing grid search across {len(X_train):,} training samples...")
search.fit(X_train, y_train)

best_xgb = search.best_estimator_
print("\n🏆 OPTIMAL HYPERPARAMETERS FOUND:")
for k, v in search.best_params_.items():
  print(f"   * {k:18s}: {v:.4f}" if isinstance(v, float) else f"   * {k:18s}: {v}")
print("=" * 75)

# ==============================================================================
# 4. MODEL EVALUATION ON UNSEEN TEST SET
# ==============================================================================
print(f"📊 Evaluating optimal model on {len(X_test):,} unseen test samples...")
y_pred = best_xgb.predict(X_test)

r2_test = r2_score(y_test, y_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
mae_test = mean_absolute_error(y_test, y_pred)

print(f"\n🎯 FINAL TEST PERFORMANCE (TARGET: R² ≈ 0.73):")
print(f"   * Test R² Score : {r2_test:.3f}")
print(f"   * Test RMSE     : {rmse_test:.2f} ppb")
print(f"   * Test MAE      : {mae_test:.2f} ppb")
print("=" * 75)

# ==============================================================================
# 5. PLOT DENSITY CORRELATION SCATTERPLOT (MIRRORS FIGURE 3 IN PAPER)
# ==============================================================================
print("🎨 Generating Figure 3 replication plot (Density Scatterplot)...")
fig, ax = plt.subplots(figsize=(10, 9))

# Subsample for KDE plotting speed if test set is massive (>100k rows takes seconds to KDE)
if len(y_test) > 15000:
  plot_idx = np.random.choice(len(y_test), size=15000, replace=False)
  y_test_plot = y_test.iloc[plot_idx]
  y_pred_plot = y_pred[plot_idx]
else:
  y_test_plot, y_pred_plot = y_test, y_pred

xy = np.vstack([y_test_plot, y_pred_plot])
kde = gaussian_kde(xy)(xy)

# Sort points by density so dense cores plot on top
idx = kde.argsort()
y_test_sorted = y_test_plot.iloc[idx]
y_pred_sorted = y_pred_plot[idx]
kde_sorted = kde[idx]

max_val = np.ceil(max(y_test.max(), y_pred.max()) * 1.05)

sc = ax.scatter(
    y_test_sorted,
    y_pred_sorted,
    c=kde_sorted,
    cmap="viridis",
    s=35,
    alpha=0.8,
    edgecolor="none",
    zorder=2,
)

# 1:1 Perfection Line
ax.plot(
    [0, max_val],
    [0, max_val],
    color="#d62728",
    linestyle="--",
    linewidth=2.2,
    label="1:1 Perfect Prediction Line",
    zorder=3,
)

ax.set_title(
    f"TEMPO Only Test Results\n$R^2 = {r2_test:.2f}$",
    fontsize=16,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel("True Values (ppb)", fontsize=13, fontweight="bold")
ax.set_ylabel("Predicted Values (ppb)", fontsize=13, fontweight="bold")
ax.set_xlim(0, max_val)
ax.set_ylim(0, max_val)
ax.grid(True, linestyle="--", alpha=0.5, color="gray", zorder=0)

stats_box = (
    "Model: XGBoost (TEMPO Covariates)\n"
    f"Test $R^2$:   {r2_test:.3f}\n"
    f"Test RMSE: {rmse_test:.2f} ppb\n"
    f"Test MAE:  {mae_test:.2f} ppb\n"
    f"Test Size: {len(y_test):,} samples"
)
ax.text(
    0.04,
    0.92,
    stats_box,
    transform=ax.transAxes,
    fontsize=10.5,
    verticalalignment="top",
    fontfamily="monospace",
    bbox=dict(
        boxstyle="round,pad=0.6", facecolor="white", edgecolor="gray", alpha=0.95
    ),
    zorder=4,
)

cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.03)
cbar.set_label("Data Point Density (Gaussian KDE)", fontsize=11, fontweight="bold")

out_file = OUTPUT_DIR / "tempo_xgboost_figure3_replication.png"
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"✅ Saved Figure 3 Replicated Scatterplot to: {out_file}")
print("🎯 GOAL 1.3 XGBOOST TRAINING & EVALUATION SUCCESSFULLY FINISHED!")