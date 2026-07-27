import os

# ==============================================================================
# 0. THREAD SAFETY CONTROLS (Must be set before numpy/sklearn imports!)
# Prevents high-core Linux server CPU contention and OpenBLAS core dumps
# ==============================================================================
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
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

data_file = PROCESSED_DIR / "epa_point_dataset_14months_20features.npz"

print("🚀 STARTING STATE-WIDE 20-FEATURE XGBOOST HYPERPARAMETER SWEEP 🚀")
print("=" * 75)

if not data_file.exists():
  raise FileNotFoundError(
      f"Could not locate the 20-feature dataset at: {data_file}"
  )

# ==============================================================================
# 2. LOAD COMPRESSED MASTER ARRAYS
# ==============================================================================
print(f"⏳ Loading 20-feature dataset from: {data_file.name}...")
start_load = time.time()
data = np.load(data_file, allow_pickle=True)

X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]
feature_names = list(data["feature_names"])

print(f"✅ Loaded in {time.time() - start_load:.2f}s!")
print(f"   * Train Matrix: X={X_train.shape} | y={y_train.shape}")
print(f"   * Test Matrix:  X={X_test.shape}  | y={y_test.shape}")
print(f"📋 20 Active Covariates: {feature_names}")
print("=" * 75)

# ==============================================================================
# 3. LIVE LEVEL-3 VERBOSE HYPERPARAMETER SWEEP (WITH PLUME BOOSTING!)
# ==============================================================================
print("⚙️ Configuring Randomized Search & Target Transformations...")

# 1. Log-transform the target to normalize atmospheric right-skewness!
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# 2. Build Plume Gradient Weights: Penalize underpredicting high smog spikes!
# Give moderate spikes (20-40 ppb) 5x weight, and extreme spikes (>40 ppb) 10x weight
sample_weights = np.where(y_train > 20.0, 5.0, 1.0)
sample_weights = np.where(y_train > 40.0, 10.0, sample_weights)

print(
    f"⚖️ Sample Weights Configured | Baseline: {np.sum(sample_weights == 1.0):,}"
    f" | 5x Boost: {np.sum(sample_weights == 5.0):,} | 10x Boost:"
    f" {np.sum(sample_weights == 10.0):,}"
)

param_distributions = {
    "n_estimators": randint(150, 801),
    "learning_rate": uniform(0.005, 0.25),  # Lower max LR to stabilize weighted gradients
    "subsample": uniform(0.4, 0.6),
    "max_depth": randint(3, 14),  # Allow slightly deeper trees to capture plume rules
    "colsample_bytree": uniform(0.2, 0.8),
    "gamma": uniform(0.0, 0.3),
    "reg_alpha": uniform(0.0, 50.0),
    "reg_lambda": uniform(0.001, 5.0),  # Lower max lambda so leaf values aren't over-shrunk
}

xgb_base = XGBRegressor(
    objective="reg:squarederror", random_state=42, n_jobs=1, tree_method="hist"
)

search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_distributions,
    n_iter=30,
    scoring="r2",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=3,
)

print(
    f"⏳ Executing 30-iteration weighted grid search across {len(X_train):,}"
    " samples..."
)
start_train = time.time()

# 3. Pass sample_weight directly into the fitting step!
search.fit(X_train, y_train_log, sample_weight=sample_weights)
train_duration = time.time() - start_train

best_xgb = search.best_estimator_
print("\n🏆 OPTIMAL HYPERPARAMETERS DISCOVERED:")
for k, v in search.best_params_.items():
  print(f"   * {k:18s}: {v:.4f}" if isinstance(v, float) else f"   * {k:18s}: {v}")
print(f"⏱️ Total Hyperparameter Sweep Time: {train_duration:.1f} seconds")
print("=" * 75)

# ==============================================================================
# 4. EVALUATION & FIGURE GENERATION (WITH INVERSE TRANSFORMS)
# ==============================================================================
print(f"📊 Evaluating weighted log-model on {len(X_test):,} unseen test samples...")

# 4. Predict in log-space, then exponentiate back to true ppb!
y_pred_log = best_xgb.predict(X_test)
y_pred = np.expm1(y_pred_log)  # Inverse of log1p: exp(y) - 1

r2_test = r2_score(y_test, y_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
mae_test = mean_absolute_error(y_test, y_pred)

print(f"\n🎯 UPGRADED PLUME-WEIGHTED TEST PERFORMANCE:")
print(f"   * Test R² Score : {r2_test:.3f}")
print(f"   * Test RMSE     : {rmse_test:.2f} ppb")
print(f"   * Test MAE      : {mae_test:.2f} ppb")
print("=" * 75)

print("🎨 Generating Figure 3 replication plot (Gaussian KDE Density)...")
fig, ax = plt.subplots(figsize=(10, 9))

# Subsample for fast, clean scatter plotting if dataset is large
if len(y_test) > 15000:
  plot_idx = np.random.choice(len(y_test), size=15000, replace=False)
  y_test_plot, y_pred_plot = y_test[plot_idx], y_pred[plot_idx]
else:
  y_test_plot, y_pred_plot = y_test, y_pred

xy = np.vstack([y_test_plot, y_pred_plot])
kde = gaussian_kde(xy)(xy)
idx = kde.argsort()
y_test_sorted, y_pred_sorted, kde_sorted = (
    y_test_plot[idx],
    y_pred_plot[idx],
    kde[idx],
)

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
ax.plot(
    [0, max_val],
    [0, max_val],
    color="#d62728",
    linestyle="--",
    linewidth=2.2,
    label="1:1 Perfect Line",
    zorder=3,
)

ax.set_title(
    f"TEMPO 20-Feature XGBoost Test Results\n$R^2 = {r2_test:.2f}$",
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
    "Model: XGBoost (20 Covariates)\n"
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

out_file = OUTPUT_DIR / "tempo_xgboost_20features_results.png"
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"✅ Saved 20-Feature Scatterplot to: {out_file}")
print("🎯 PIPELINE EVALUATION FINISHED!")