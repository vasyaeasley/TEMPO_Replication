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

print("☀️ STARTING TEMPO DIURNAL (HOURLY) CYCLE EVALUATION ☀️")
print("=" * 75)

if not data_file.exists():
  raise FileNotFoundError(f"Could not locate 20-feature dataset at: {data_file}")

# ==============================================================================
# 2. LOAD DATA & RECREATE THE 0.852 R² DOMAIN SPLIT
# ==============================================================================
print("⏳ Loading 20-feature master dataset...")
data = np.load(data_file, allow_pickle=True)

X_full = np.vstack([data["X_train"], data["X_test"]])
y_full = np.concatenate([data["y_train"], data["y_test"]])
feature_names = list(data["feature_names"])

print("🎯 Recreating 80/20 Literature Domain Split...")
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.20, random_state=42
)

# ==============================================================================
# 3. TRAIN THE OPTIMAL XGBOOST DIGITAL TWIN
# ==============================================================================
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

# Generate predictions across all unseen test samples
print("⏳ Predicting across unseen test hours...")
y_pred = best_xgb.predict(X_test)

# ==============================================================================
# 4. DECODE TRIGONOMETRIC TIMESTAMPS INTO LOCAL HOURS
# ==============================================================================
print("🔍 Extracting local solar/daylight hours from trigonometric features...")
# In our 20-feature array: index 8 is hour_sin, index 9 is hour_cos
hour_sin = X_test[:, 8]
hour_cos = X_test[:, 9]

# Reverse arctan2 to recover 24-hour integer timestamps (0 to 23)
hours_raw = (
    np.round(np.arctan2(hour_sin, hour_cos) * 24.0 / (2 * np.pi)) % 24
).astype(int)

# Compile into a clean analytical DataFrame
df_eval = pd.DataFrame({
    "Hour": hours_raw,
    "True_NO2": y_test,
    "Pred_NO2": y_pred,
    "Error": y_pred - y_test,
    "Abs_Error": np.abs(y_pred - y_test),
    "Squared_Error": (y_pred - y_test) ** 2,
})

# Filter for active daylight hours where TEMPO operates (typically 7 AM to 6 PM local)
unique_hours = sorted(df_eval["Hour"].unique())
print(
    f"📋 Active TEMPO Observation Hours Discovered in Test Set: {unique_hours}"
)

# ==============================================================================
# 5. COMPUTE HOURLY DIURNAL METRICS TABLE
# ==============================================================================
hourly_stats = []

for h in unique_hours:
  sub = df_eval[df_eval["Hour"] == h]
  if len(sub) < 10:
    continue

  r2_val = r2_score(sub["True_NO2"], sub["Pred_NO2"])
  rmse_val = np.sqrt(mean_squared_error(sub["True_NO2"], sub["Pred_NO2"]))
  mae_val = mean_absolute_error(sub["True_NO2"], sub["Pred_NO2"])

  hourly_stats.append({
      "Local Hour": f"{h:02d}:00",
      "Test Samples": len(sub),
      "True Mean (ppb)": sub["True_NO2"].mean(),
      "Pred Mean (ppb)": sub["Pred_NO2"].mean(),
      "Bias (ppb)": sub["Error"].mean(),
      "RMSE (ppb)": rmse_val,
      "MAE (ppb)": mae_val,
      "R²": r2_val,
  })

df_hourly = pd.DataFrame(hourly_stats)

print("\n" + "=" * 80)
print("🏆 TEMPO DIURNAL CYCLE EVALUATION TABLE (YOUR PAPER'S TABLE II)")
print("=" * 80)
print(df_hourly.to_string(index=False))
print("=" * 80)

# ==============================================================================
# 6. GENERATE THE TWO-PANEL DIURNAL PUBLICATION FIGURE
# ==============================================================================
print("\n🎨 Generating Figure 5: Diurnal Cycle Tracking & Hourly RMSE...")

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
)

x_hours = df_hourly["Local Hour"]
true_means = df_hourly["True Mean (ppb)"]
pred_means = df_hourly["Pred Mean (ppb)"]
rmse_vals = df_hourly["RMSE (ppb)"]
r2_vals = df_hourly["R²"]

# --- Top Panel: True vs. Predicted Diurnal Trajectory ---
ax1.plot(
    x_hours,
    true_means,
    marker="o",
    markersize=8,
    linewidth=2.5,
    color="#1f77b4",
    label="Observed Ground Telemetry (True NO₂)",
)
ax1.plot(
    x_hours,
    pred_means,
    marker="s",
    markersize=7,
    linewidth=2.5,
    linestyle="--",
    color="#d62728",
    label="20-Feature XGBoost Digital Twin",
)

ax1.set_title(
    "TEMPO Diurnal NO₂ Tracking Across Active Daylight Observation Hours\n(State-Wide"
    " 80/20 Unseen Test Set)",
    fontsize=15,
    fontweight="bold",
    pad=15,
)
ax1.set_ylabel("Mean NO₂ Concentration (ppb)", fontsize=12, fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(
    loc="upper right", frameon=True, facecolor="white", framealpha=0.95, fontsize=11
)

# Highlight Rush Hour Spike vs Photochemical Decay
ax1.annotate(
    "Morning Traffic Peak\n(Rush Hour Accumulation)",
    xy=(0.15, 0.85),
    xycoords="axes fraction",
    fontsize=10.5,
    fontweight="bold",
    color="#333333",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff2cc", edgecolor="#d6b656"),
)
ax1.annotate(
    "Afternoon Photolysis\n(Solar Decay & Mixing)",
    xy=(0.65, 0.20),
    xycoords="axes fraction",
    fontsize=10.5,
    fontweight="bold",
    color="#333333",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#e1f5fe", edgecolor="#81d4fa"),
)

# --- Bottom Panel: Hourly RMSE Stability ---
bars = ax2.bar(
    x_hours,
    rmse_vals,
    color="#2ca02c",
    alpha=0.8,
    width=0.5,
    edgecolor="black",
    label="Hourly RMSE (ppb)",
)

# Overlay R² scores on top of bars
for bar, r2_score_val in zip(bars, r2_vals):
  height = bar.get_height()
  ax2.text(
      bar.get_x() + bar.get_width() / 2.0,
      height - 0.6,
      f"R²={r2_score_val:.2f}",
      ha="center",
      va="top",
      color="white",
      fontweight="bold",
      fontsize=9.5,
  )

ax2.set_title(
    "Model Error Stability Throughout Daylight Hours",
    fontsize=12,
    fontweight="bold",
)
ax2.set_xlabel(
    "California Local Daylight Time (Hours)", fontsize=12, fontweight="bold"
)
ax2.set_ylabel("RMSE (ppb)", fontsize=11, fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
ax2.set_ylim(0, max(rmse_vals) * 1.2)

plt.xticks(rotation=0, fontweight="bold")
plt.tight_layout()

out_file = OUTPUT_DIR / "tempo_diurnal_cycle_evaluation.png"
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"✅ Saved Diurnal Evaluation Table & Plot to: {out_file}")
print("🎯 DIURNAL EVALUATION FINISHED!")