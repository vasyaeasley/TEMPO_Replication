import os

# Thread safety controls to prevent server CPU lockups
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# ==============================================================================
# 1. SETUP & PATH CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

data_file = PROCESSED_DIR / "epa_point_dataset_14months_20features.npz"

print("🎯 STARTING TRUE VS. PREDICTED GAUSSIAN KDE DENSITY SCATTERPLOT 🎯")
print("=" * 75)

if not data_file.exists():
  raise FileNotFoundError(f"Could not locate 20-feature dataset at: {data_file}")

# ==============================================================================
# 2. LOAD DATA & TRAIN OPTIMAL XGBOOST DIGITAL TWIN
# ==============================================================================
print("⏳ Loading 20-feature master dataset...")
data = np.load(data_file, allow_pickle=True)

X_full = np.vstack([data["X_train"], data["X_test"]])
y_full = np.concatenate([data["y_train"], data["y_test"]])
g_full = np.concatenate([data["groups_train"], data["groups_test"]])

print("🎯 Recreating 80/20 Literature Domain Split...")
X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
    X_full, y_full, g_full, test_size=0.20, random_state=42
)

print("⏳ Training optimal 20-feature XGBoost model...")
start_train = time.time()
best_xgb = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=673,
    learning_rate=0.0327,
    max_depth=7,
    subsample=0.7100,
    colsample_bytree=0.6610,
    gamma=0.1324,
    reg_alpha=32.5183,
    reg_lambda=7.2963,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)
best_xgb.fit(X_train, y_train)
print(f"✅ Model trained in {time.time() - start_train:.2f}s!")

print("⏳ Generating predictions across unseen test hours...")
y_pred = best_xgb.predict(X_test)

# Calculate formal literature validation metrics
r2_val = r2_score(y_test, y_pred)
rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))
mae_val = mean_absolute_error(y_test, y_pred)
bias_val = np.mean(y_pred - y_test)
n_samples = len(y_test)

print(f"📊 Unseen Test Set Metrics:")
print(f"   * R² Score:   {r2_val:.3f}")
print(f"   * RMSE:       {rmse_val:.2f} ppb")
print(f"   * MAE:        {mae_val:.2f} ppb")
print(f"   * Mean Bias:  {bias_val:.3f} ppb")

# ==============================================================================
# 3. COMPUTE GAUSSIAN KERNEL DENSITY ESTIMATION (KDE)
# ==============================================================================
print(
    "⚙️ Computing Gaussian Kernel Density Estimates across"
    f" {n_samples:,} points..."
)
start_kde = time.time()

# Stack true and predicted arrays for 2D density evaluation
xy = np.vstack([y_test, y_pred])
kde = gaussian_kde(xy)
density_values = kde(xy)

# Sort points by density so dense yellow cores plot cleanly on top of purple background noise!
sort_idx = density_values.argsort()
x_sorted = y_test[sort_idx]
y_sorted = y_pred[sort_idx]
z_sorted = density_values[sort_idx]

print(f"✅ Gaussian KDE computed and sorted in {time.time() - start_kde:.2f}s!")

# ==============================================================================
# 4. RENDER PUBLICATION-GRADE DENSITY SCATTERPLOT
# ==============================================================================
print("🎨 Rendering publication-grade 1:1 density parity plot...")
fig, ax = plt.subplots(figsize=(10, 9))

# Set clean aesthetic bounds (padding slightly above max observed concentration)
max_bound = np.percentile(y_test, 99.9) * 1.15
ax.set_xlim(0, max_bound)
ax.set_ylim(0, max_bound)

# Plot 1:1 Perfect Parity Reference Line
ax.plot(
    [0, max_bound],
    [0, max_bound],
    color="#d62728",
    linestyle="--",
    linewidth=2.8,
    label="1:1 Perfect Parity Line",
    zorder=1,
)

# Render density-colored scatter points using the literature-standard 'viridis' colormap
sc = ax.scatter(
    x_sorted,
    y_sorted,
    c=z_sorted,
    cmap="viridis",
    s=25,
    alpha=0.85,
    edgecolor="none",
    zorder=2,
)

# Format Axes & Titles
ax.set_title(
    f"TEMPO 20-Feature XGBoost Digital Twin Validation\nUnseen Spatial Domain"
    f" Test Results ($R^2 = {r2_val:.3f}$)",
    fontsize=16,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel(
    "Observed Ground Telemetry — True $\text{NO}_2$ (ppb)",
    fontsize=13,
    fontweight="bold",
)
ax.set_ylabel(
    "XGBoost Digital Twin — Predicted $\text{NO}_2$ (ppb)",
    fontsize=13,
    fontweight="bold",
)
ax.grid(True, linestyle="--", alpha=0.4, color="#94a3b8", zorder=0)

# Make axis ticks bold and readable
ax.tick_params(axis="both", which="major", labelsize=11)
for label in ax.get_xticklabels() + ax.get_yticklabels():
  label.set_fontweight("bold")

# Add Master Colorbar
cbar = fig.colorbar(sc, ax=ax, orientation="vertical", shrink=0.88, pad=0.03)
cbar.set_label("Data Point Density (Gaussian KDE)", fontsize=12, fontweight="bold")
cbar.ax.tick_params(labelsize=10)

# --- Add Industry-Standard Statistical Diagnostic Box ---
stats_text = (
    f"Model: XGBoost (20 Covariates)\n"
    f"Test $R^2$:      {r2_val:.3f}\n"
    f"Test RMSE:   {rmse_val:.2f} ppb\n"
    f"Test MAE:    {mae_val:.2f} ppb\n"
    f"Mean Bias:   {bias_val:+.2f} ppb\n"
    f"Test Size:   {n_samples:,} samples"
)

ax.text(
    0.05,
    0.95,
    stats_text,
    transform=ax.transAxes,
    fontsize=11.5,
    fontweight="bold",
    family="monospace",
    verticalalignment="top",
    bbox=dict(
        boxstyle="round,pad=0.6",
        facecolor="white",
        edgecolor="#4b5563",
        alpha=0.92,
        linewidth=1.5,
    ),
    zorder=3,
)

ax.legend(
    loc="lower right",
    frameon=True,
    facecolor="white",
    framealpha=0.95,
    fontsize=11,
    edgecolor="#4b5563",
)

plt.tight_layout()

out_file = OUTPUT_DIR / "tempo_true_vs_pred_density_scatterplot.png"
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"✅ Saved Publication Density Scatterplot to: {out_file}")
print("🎯 DENSITY SCATTERPLOT FINISHED!")