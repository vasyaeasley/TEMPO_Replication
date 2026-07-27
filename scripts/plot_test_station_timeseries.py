import os

# Thread safety controls
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

print("📈 STARTING TIME-SERIES DIAGNOSTIC: XGBOOST vs. PERSISTENCE 📈")
print("=" * 75)

if not data_file.exists():
  raise FileNotFoundError(f"Could not locate 20-feature dataset at: {data_file}")

# ==============================================================================
# 2. LOAD COMPRESSED 20-FEATURE ARRAYS
# ==============================================================================
print("⏳ Loading 20-feature dataset...")
data = np.load(data_file, allow_pickle=True)

X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]
g_test = data["groups_test"]
feature_names = list(data["feature_names"])

# ==============================================================================
# 3. FAST TRAINING WITH DISCOVERED OPTIMAL HYPERPARAMETERS
# ==============================================================================
print("⏳ Training 20-Feature XGBoost using previously discovered parameters...")
start_train = time.time()

# Exact optimal parameters discovered by your RandomizedSearchCV run!
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
print(f"✅ Model trained in {time.time() - start_train:.2f} seconds!")

# Generate test predictions across the entire test matrix
y_pred_all = best_xgb.predict(X_test)

# ==============================================================================
# 4. ISOLATE A REPRESENTATIVE UNSEEN TEST STATION & BUILD DAILY TIME SERIES
# ==============================================================================
print("\n🌐 Analyzing test monitoring stations to build chronological time series...")

# Find the test station group ID with the most continuous observations
unique_gids, counts = np.unique(g_test, return_counts=True)
target_gid = unique_gids[np.argmax(counts)]

stn_mask = g_test == target_gid
y_true_stn = y_test[stn_mask]
y_pred_stn = y_pred_all[stn_mask]

print(f"🎯 Selected Test Station ID: {target_gid} ({len(y_true_stn):,} hourly observations)")

# Convert hourly observations into Daily Means (assuming ~9 daylight satellite hours per day)
# This mimics the exact Daily Mean methodology used in your presentation slides!
hours_per_day = 9
n_days = len(y_true_stn) // hours_per_day

daily_true = [
    np.mean(y_true_stn[i * hours_per_day : (i + 1) * hours_per_day])
    for i in range(n_days)
]
daily_pred = [
    np.mean(y_pred_stn[i * hours_per_day : (i + 1) * hours_per_day])
    for i in range(n_days)
]

daily_true = np.array(daily_true)
daily_pred = np.array(daily_pred)

# Build the Lag-1 Persistence Model (Yesterday's True Daily Mean -> Today's Prediction)
# For day 0, fallback to day 0 value; for day t, use day t-1
daily_persist = np.roll(daily_true, shift=1)
daily_persist[0] = daily_true[0]

# ==============================================================================
# 5. COMPUTE COMPARATIVE METRICS (XGBOOST vs. PERSISTENCE)
# ==============================================================================
r2_xgb = r2_score(daily_true, daily_pred)
rmse_xgb = np.sqrt(mean_squared_error(daily_true, daily_pred))
mae_xgb = mean_absolute_error(daily_true, daily_pred)

r2_pers = r2_score(daily_true, daily_persist)
rmse_pers = np.sqrt(mean_squared_error(daily_true, daily_persist))
mae_pers = mean_absolute_error(daily_true, daily_persist)

print("\n📊 DAILY TIME-SERIES BENCHMARK COMPARISON (TEST STATION):")
print("-" * 65)
print(f"   * 20-Feature XGBoost | R²: {r2_xgb:.3f} | RMSE: {rmse_xgb:.2f} ppb | MAE: {mae_xgb:.2f} ppb")
print(f"   * Lag-1 Persistence  | R²: {r2_pers:.3f} | RMSE: {rmse_pers:.2f} ppb | MAE: {mae_pers:.2f} ppb")
print("-" * 65)

# ==============================================================================
# 6. GENERATE THE 3-PANEL TIME SERIES COMPARISON FIGURE
# ==============================================================================
print("🎨 Generating Time Series comparison plot...")

days_x = np.arange(1, n_days + 1)
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]}
)

# --- Top Panel: Daily Mean Trajectories ---
ax1.plot(
    days_x,
    daily_true,
    label="Observed Daily Mean (True NO₂)",
    color="#1f77b4",
    linewidth=2.0,
    alpha=0.85,
)
ax1.plot(
    days_x,
    daily_persist,
    label=f"Persistence Baseline (Lag-1, R²={r2_pers:.2f})",
    color="#2ca02c",
    linestyle="--",
    linewidth=1.8,
    alpha=0.75,
)
ax1.plot(
    days_x,
    daily_pred,
    label=f"20-Feature XGBoost (Spatial, R²={r2_xgb:.2f})",
    color="#ff7f0e",
    linewidth=2.2,
    alpha=0.9,
)

ax1.set_title(
    f"NO₂ Daily Mean Time Series: 20-Feature XGBoost vs. Persistence Model\n(Unseen Test Station #{target_gid})",
    fontsize=15,
    fontweight="bold",
    pad=12,
)
ax1.set_ylabel("NO₂ Concentration (ppb)", fontsize=12, fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95, fontsize=11)

# Stats overlay box
stats_text = (
    f"Target Benchmark R² : 0.730\n"
    f"Persistence Model R²: {r2_pers:.3f} (RMSE: {rmse_pers:.2f})\n"
    f"XGBoost Model R²    : {r2_xgb:.3f} (RMSE: {rmse_xgb:.2f})"
)
ax1.text(
    0.02,
    0.88,
    stats_text,
    transform=ax1.transAxes,
    fontsize=10.5,
    fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9),
)

# --- Bottom Panel: Model Residuals (Predicted - Actual) ---
res_xgb = daily_pred - daily_true
res_pers = daily_persist - daily_true

ax2.axhline(0, color="black", linewidth=1.2, linestyle="-", alpha=0.7)
ax2.plot(
    days_x,
    res_pers,
    label="Persistence Residuals",
    color="#2ca02c",
    linestyle="--",
    linewidth=1.2,
    alpha=0.6,
)
ax2.plot(
    days_x,
    res_xgb,
    label="XGBoost Residuals",
    color="#ff7f0e",
    linewidth=1.5,
    alpha=0.85,
)

ax2.set_title("Model Residuals over Time (Predicted - Actual NO₂)", fontsize=12, fontweight="bold")
ax2.set_xlabel("Chronological Day of Dataset (2023 - 2024)", fontsize=12, fontweight="bold")
ax2.set_ylabel("Error (ppb)", fontsize=11, fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95, fontsize=10)

plt.tight_layout()
out_file = OUTPUT_DIR / "timeseries_comparison_xgboost_vs_persistence.png"
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"✅ Time Series comparison saved to: {out_file}")
print("🎯 DIAGNOSTIC COMPLETE!")