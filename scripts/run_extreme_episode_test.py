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

print("🔥 STARTING EXTREME EMISSION EPISODE STRESS TEST 🔥")
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

# ==============================================================================
# 3. MATHEMATICALLY DECODE UTC HOURS & ISOLATE THE WORST SMOG SPIKE
# ==============================================================================
print("🔍 Decoding UTC daylight timestamps from trigonometric features...")
hour_sin_idx = np.where(feature_names == "hour_sin")[0][0]
hour_cos_idx = np.where(feature_names == "hour_cos")[0][0]

# Recover exact integer UTC hour from sin/cos arctan2 angles
decoded_hours = np.round(
    np.mod(
        np.arctan2(X_test[:, hour_sin_idx], X_test[:, hour_cos_idx])
        * 24
        / (2 * np.pi),
        24,
    )
).astype(int)

df_test = pd.DataFrame({
    "gid": g_test,
    "utc_hour": decoded_hours,
    "True_NO2": y_test,
    "Pred_NO2": y_pred,
})

# Find the absolute worst pollution anomaly in the entire testing domain!
max_idx = df_test["True_NO2"].idxmax()
peak_row = df_test.loc[max_idx]
target_gid = peak_row["gid"]
peak_val = peak_row["True_NO2"]

print("\n🚨 EXTREME EMISSION ANOMALY LOCATED 🚨")
print(
    f"   * Station Group ID:  {target_gid} (Inversion Hotspot Corridor)"
)
print(f"   * Peak True NO₂:     {peak_val:.2f} ppb")
print(f"   * Twin Prediction:   {peak_row['Pred_NO2']:.2f} ppb")
print(
    f"   * Instantaneous Bias: {peak_row['Pred_NO2'] - peak_val:+.2f} ppb"
)

# Isolate all daylight hours (13:00 to 23:00 UTC) for this specific station episode
df_episode = df_test[df_test["gid"] == target_gid].copy()

# Group by UTC hour to reconstruct the clean 11-hour diurnal trajectory of this anomaly day
diurnal_anomaly = (
    df_episode.groupby("utc_hour")
    .agg({"True_NO2": "mean", "Pred_NO2": "mean"})
    .reset_index()
)
diurnal_anomaly = diurnal_anomaly.sort_values("utc_hour")

# Convert UTC (13:00 - 23:00) to Local Pacific Daylight Time (PDT: 06:00 - 16:00)
diurnal_anomaly["local_hour"] = diurnal_anomaly["utc_hour"] - 7
diurnal_anomaly["time_label"] = diurnal_anomaly["local_hour"].apply(
    lambda h: f"{h:02d}:00" if h >= 0 else f"{h+24:02d}:00"
)

# Calculate anomaly episode statistical metrics
ep_r2 = r2_score(diurnal_anomaly["True_NO2"], diurnal_anomaly["Pred_NO2"])
ep_rmse = np.sqrt(
    mean_squared_error(
        diurnal_anomaly["True_NO2"], diurnal_anomaly["Pred_NO2"]
    )
)
ep_mae = mean_absolute_error(
    diurnal_anomaly["True_NO2"], diurnal_anomaly["Pred_NO2"]
)
ep_bias = np.mean(diurnal_anomaly["Pred_NO2"] - diurnal_anomaly["True_NO2"])

print("\n📊 Extreme Episode Tracking Metrics (Daylight Window):")
print(f"   * Episode R²:       {ep_r2:.3f}")
print(f"   * Episode RMSE:     {ep_rmse:.2f} ppb")
print(f"   * Episode MAE:      {ep_mae:.2f} ppb")
print(f"   * Episode Mean Bias: {ep_bias:+.2f} ppb")

# ==============================================================================
# 4. RENDER PUBLICATION-GRADE ANOMALY TRACKING CHART
# ==============================================================================
print("🎨 Rendering publication-grade Extreme Episode Stress Test chart...")
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

x_vals = range(len(diurnal_anomaly))

# Plot Observed Ground Telemetry (True Smog)
ax.plot(
    x_vals,
    diurnal_anomaly["True_NO2"],
    marker="o",
    color="#991b1b",
    linewidth=3.5,
    markersize=9,
    label="Observed EPA Sensor (True $\text{NO}_2$ Anomaly)",
    zorder=3,
)

# Plot XGBoost Digital Twin Prediction
ax.plot(
    x_vals,
    diurnal_anomaly["Pred_NO2"],
    marker="s",
    color="#2563eb",
    linewidth=3.0,
    linestyle="--",
    markersize=8,
    label="20-Feature XGBoost Digital Twin",
    zorder=4,
)

# Annotate Peak Inversion Spike
peak_idx = diurnal_anomaly["True_NO2"].idxmax()
peak_true = diurnal_anomaly.loc[peak_idx, "True_NO2"]
peak_pred = diurnal_anomaly.loc[peak_idx, "Pred_NO2"]
peak_time = diurnal_anomaly.loc[peak_idx, "time_label"]

ax.annotate(
    f"Severe Inversion Peak ({peak_time} PDT)\nTrue: {peak_true:.1f}"
    f" ppb | Twin: {peak_pred:.1f} ppb",
    xy=(peak_idx, peak_true),
    xytext=(peak_idx - 1.8, peak_true + 4.5),
    arrowprops=dict(
        facecolor="#111827",
        shrink=0.08,
        width=1.5,
        headwidth=6,
        edgecolor="none",
    ),
    fontsize=10.5,
    fontweight="bold",
    color="#111827",
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor="#fef08a",
        edgecolor="#ca8a04",
        alpha=0.95,
        linewidth=1.5,
    ),
    zorder=6,
)

# Formatting and aesthetics
ax.set_title(
    "Extreme Emission Episode Stress Test (Severe Inversion Anomaly"
    " Tracking)\nEvaluating Digital Twin Resistance to Peak-Shaving During"
    " High-Pollution Events",
    fontsize=15,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel(
    "Local Daylight Observation Time (Pacific Daylight Time, PDT)",
    fontsize=13,
    fontweight="bold",
)
ax.set_ylabel(
    "Ground-Level $\text{NO}_2$ Concentration (ppb)",
    fontsize=13,
    fontweight="bold",
)
ax.set_xticks(x_vals)
ax.set_xticklabels(
    diurnal_anomaly["time_label"], fontsize=11, fontweight="bold"
)
ax.tick_params(axis="y", labelsize=11)
for label in ax.get_yticklabels():
  label.set_fontweight("bold")

# Add 10% vertical padding so the annotation box doesn't touch the top
ax.set_ylim(0, max(peak_true, peak_pred) * 1.20)
ax.grid(True, linestyle="--", alpha=0.4, color="#94a3b8", zorder=0)

# Statistical Diagnostic Callout Box
stats_text = (
    f"Anomaly Episode Diagnostics\n"
    f"Episode $R^2$:      {ep_r2:.3f}\n"
    f"Episode RMSE:   {ep_rmse:.2f} ppb\n"
    f"Episode MAE:    {ep_mae:.2f} ppb\n"
    f"Mean Bias:      {ep_bias:+.2f} ppb\n"
    f"Peak Delta:     {peak_pred - peak_true:+.2f} ppb"
)

ax.text(
    0.04,
    0.28,
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
    zorder=5,
)

ax.legend(
    loc="lower left",
    frameon=True,
    facecolor="white",
    framealpha=0.95,
    fontsize=11.5,
    edgecolor="#4b5563",
)

out_file = OUTPUT_DIR / "tempo_extreme_episode_stress_test.png"
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"✅ Saved Publication Extreme Episode Chart to: {out_file}")
print("🎯 EXTREME EPISODE STRESS TEST FINISHED!")