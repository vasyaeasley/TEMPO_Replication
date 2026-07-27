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
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ==============================================================================
# 1. SETUP & PATH CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

data_file = PROCESSED_DIR / "epa_point_dataset_14months_20features.npz"

print("⏱️ STARTING CONTINUOUS TEMPORAL HOLD-OUT EXPERIMENT ⏱️")
print("=" * 75)

if not data_file.exists():
  raise FileNotFoundError(f"Could not locate 20-feature dataset at: {data_file}")

# ==============================================================================
# 2. LOAD MASTER DATA & ISOLATE STATION TIMELINE
# ==============================================================================
print("⏳ Loading 20-feature master dataset...")
data = np.load(data_file, allow_pickle=True)

# Reconstruct full unsplit arrays (natively stored in chronological order per station)
X_full = np.vstack([data["X_train"], data["X_test"]])
y_full = np.concatenate([data["y_train"], data["y_test"]])
g_full = np.concatenate([data["groups_train"], data["groups_test"]])

feature_names = np.array([
    "TEMPO_NO2",
    "blh",
    "traffic",
    "t2m",
    "elev",
    "pop",
    "month_cos",
    "road_density",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "sp",
    "day_of_week_cos",
    "month_sin",
    "is_weekend",
    "u10",
    "v10",
    "d2m",
    "tcc",
    "solar_zenith_angle",
])
if "feature_names" in data:
  feature_names = np.array([str(f) for f in data["feature_names"]])

# Target our high-interest freight corridor: Station 41 (Ontario Route 60 Near-Road)
target_gid = 41
stn_indices = np.where(g_full == target_gid)[0]

print(f"🎯 Isolate timeline for Station Group ID {target_gid} (Ontario Route 60 Freight Corridor)...")
print(f"📋 Total historical daylight hours available for this station: {len(stn_indices):,}")

# Select a contiguous 44-hour block (approx. 4 continuous daylight observation days)
# We take a window from the high-volatility autumn/winter section of the array
window_size = 44
start_idx = len(stn_indices) // 2  # Grab a dynamic mid-timeline sequence
holdout_indices = stn_indices[start_idx : start_idx + window_size]

# Create training set by completely deleting these 44 chronological hours!
train_mask = np.ones(len(y_full), dtype=bool)
train_mask[holdout_indices] = False

X_train_holdout = X_full[train_mask]
y_train_holdout = y_full[train_mask]

X_test_holdout = X_full[holdout_indices]
y_test_holdout = y_full[holdout_indices]

print(f"🛡️ Temporal Hold-Out established! Removed {window_size} consecutive hours from training.")

# ==============================================================================
# 3. TRAIN DIGITAL TWIN ON NON-LEAKED DATA
# ==============================================================================
print("⏳ Training XGBoost digital twin (strictly without hold-out dates)...")
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
best_xgb.fit(X_train_holdout, y_train_holdout)
print(f"✅ Model trained in {time.time() - start_train:.2f}s!")

# Predict continuously across the untouched 4-day chronological window
y_pred_holdout = best_xgb.predict(X_test_holdout)

r2_val = r2_score(y_test_holdout, y_pred_holdout)
rmse_val = np.sqrt(mean_squared_error(y_test_holdout, y_pred_holdout))
mae_val = mean_absolute_error(y_test_holdout, y_pred_holdout)
mean_bias = np.mean(y_pred_holdout - y_test_holdout)

print("\n📊 Continuous 4-Day Chronological Hold-Out Metrics:")
print(f"   * Hold-Out R² Score: {r2_val:.3f}")
print(f"   * Hold-Out RMSE:     {rmse_val:.2f} ppb")
print(f"   * Hold-Out MAE:      {mae_val:.2f} ppb")
print(f"   * Hold-Out Bias:     {mean_bias:+.2f} ppb")

# ==============================================================================
# 4. RENDER PUBLICATION-GRADE CONTINUOUS TIME SERIES
# ==============================================================================
print("🎨 Rendering publication-grade continuous time-series chart...")
fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

x_timeline = range(len(y_test_holdout))

# Plot Observed Ground Telemetry (True Smog)
ax.plot(
    x_timeline,
    y_test_holdout,
    marker="o",
    color="#991b1b",
    linewidth=3.0,
    markersize=7,
    label="Observed EPA Sensor (True $\\text{NO}_2$)",
    zorder=3,
)

# Plot XGBoost Digital Twin Prediction
ax.plot(
    x_timeline,
    y_pred_holdout,
    marker="s",
    color="#2563eb",
    linewidth=2.5,
    linestyle="--",
    markersize=6,
    label="20-Feature XGBoost Digital Twin (Unseen Dates)",
    zorder=4,
)

# Add visual shading for distinct diurnal cycles (approx. 11 active daylight hours per day)
for day in range(4):
  day_start = day * 11
  day_end = min((day + 1) * 11, len(y_test_holdout))
  if day % 2 == 0:
    ax.axvspan(
        day_start,
        day_end,
        color="#f1f5f9",
        alpha=0.7,
        zorder=1,
        label=(
            "Alternating 11-Hour Daylight Days (06:00–16:00 PDT)"
            if day == 0
            else ""
        ),
    )

# Formatting and aesthetics
ax.set_title(
    "Continuous Temporal Hold-Out Evaluation (Ontario Route 60 Freight Corridor)\n"
    "4-Day Unbroken Chronological Tracking of Diurnal Traffic Spikes and Photochemical Decay",
    fontsize=15,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel("Sequential Daylight Observation Hours (13:00 to 23:00 UTC / 06:00 to 16:00 PDT Daily)", fontsize=13, fontweight="bold")
ax.set_ylabel("Ground-Level $\\text{NO}_2$ Concentration (ppb)", fontsize=13, fontweight="bold")

ax.set_xlim(0, len(y_test_holdout) - 1)
ax.tick_params(axis="both", labelsize=11)
for label in ax.get_xticklabels() + ax.get_yticklabels():
  label.set_fontweight("bold")

ax.grid(True, linestyle="--", alpha=0.4, color="#94a3b8", zorder=0)

# Statistical Diagnostic Callout Box
stats_text = (
    f"Temporal Hold-Out Diagnostics\n"
    f"Unseen Window:  44 Contiguous Hours\n"
    f"Hold-Out $R^2$:   {r2_val:.3f}\n"
    f"Hold-Out RMSE:  {rmse_val:.2f} ppb\n"
    f"Hold-Out MAE:   {mae_val:.2f} ppb\n"
    f"Mean Bias:      {mean_bias:+.2f} ppb"
)

ax.text(
    0.03,
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
        alpha=0.94,
        linewidth=1.5,
    ),
    zorder=10,
)

ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95, fontsize=11.5, edgecolor="#4b5563")

out_file = OUTPUT_DIR / "tempo_continuous_timeseries.png"
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"✅ Saved Publication Time-Series Chart to: {out_file}")
print("🎯 TEMPORAL HOLD-OUT EXPERIMENT FINISHED!")