import os

# Thread safety controls to prevent server CPU lockups
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================================================================
# 1. SETUP & PATH CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

data_file = PROCESSED_DIR / "epa_point_dataset_14months_20features.npz"
airnow_dir = BASE_DIR / "data" / "raw" / "epa" / "AirNow"
if not airnow_dir.exists():
  airnow_dir = BASE_DIR / "data" / "raw" / "AirNow"

print("🌴 STARTING ANAHEIM POSITIVE CONTROL CLIMATOLOGY ANALYSIS 🌴")
print("=" * 75)

if not data_file.exists():
  raise FileNotFoundError(f"Could not locate master dataset at: {data_file}")

# ==============================================================================
# 2. ALGORITHMIC SPATIAL AUTO-LOCATOR FOR ANAHEIM MONITOR
# ==============================================================================
print("⏳ Loading master dataset & auto-detecting Anaheim station ID...")
data = np.load(data_file, allow_pickle=True)

X_full = np.vstack([data["X_train"], data["X_test"]])
y_full = np.concatenate([data["y_train"], data["y_test"]])
g_full = np.concatenate([data["groups_train"], data["groups_test"]])
unique_gids = np.sort(np.unique(g_full))

# Locate coordinates from AirNow files to find the Anaheim monitor
airnow_files = sorted(list(airnow_dir.glob("*.nc")))
with nc.Dataset(airnow_files[0], "r") as a_ds:
  stn_lats_all = a_ds.variables["latitude"][:]
  stn_lons_all = a_ds.variables["longitude"][:]

# Anaheim EPA Monitor GPS Coordinates (~33.8306°N, -117.9383°W in Orange County)
anaheim_lat, anaheim_lon = 33.8306, -117.9383

# Find the Station Group ID with the shortest Euclidean spatial distance to Anaheim
best_gid = -1
min_dist = float("inf")

for gid in unique_gids:
  # Grab first occurrence index of this gid to check its coordinates
  idx = np.where(g_full == gid)[0][0]
  # Map back to raw station list (using proportional index scaling or coordinate lookup)
  # Here we use direct distance against the active station array
  if gid < len(stn_lats_all):
    dist = (stn_lats_all[gid] - anaheim_lat) ** 2 + (
        stn_lons_all[gid] - anaheim_lon
    ) ** 2
    if dist < min_dist:
      min_dist = dist
      best_gid = gid

target_gid = best_gid
print(f"🎯 Auto-Detected Anaheim Station Group ID: {target_gid}")
print(
    f"📍 Locked onto Orange County Coastal Basin coordinates (~{anaheim_lat}°N,"
    f" {anaheim_lon}°W)"
)

# Extract Anaheim Timeline
stn_mask = g_full == target_gid
X_stn = X_full[stn_mask]
y_stn = y_full[stn_mask]

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

# Decode Integer Month (1 to 12)
m_sin_idx = np.where(feature_names == "month_sin")[0][0]
m_cos_idx = np.where(feature_names == "month_cos")[0][0]
months_hourly = (
    np.round(
        np.mod(
            np.arctan2(X_stn[:, m_sin_idx], X_stn[:, m_cos_idx])
            * 12
            / (2 * np.pi),
            12,
        )
    ).astype(int)
    + 1
)
months_hourly[months_hourly == 13] = 1

# ==============================================================================
# 3. AGGREGATE INTO DAILY MEANS & SORT CHRONOLOGICALLY
# ==============================================================================
print("📅 Aggregating hourly telemetry into clean Daily Means...")
hours_per_day = 11
n_days = len(y_stn) // hours_per_day

daily_no2 = []
daily_month = []

for d in range(n_days):
  start_idx = d * hours_per_day
  end_idx = start_idx + hours_per_day
  daily_no2.append(np.mean(y_stn[start_idx:end_idx]))
  daily_month.append(int(np.round(np.mean(months_hourly[start_idx:end_idx]))))

df_daily = pd.DataFrame({
    "raw_idx": range(n_days),
    "month": daily_month,
    "Daily_Mean_NO2": daily_no2,
})

# Sort chronologically from Month 1 (Jan) to Month 12 (Dec)
df_daily.sort_values(by="month", inplace=True)
df_daily["day_idx"] = range(len(df_daily))

print(f"✅ Reconstructed {len(df_daily)} continuous daily means for Anaheim.")

climatology = df_daily.groupby("month")["Daily_Mean_NO2"].mean().reset_index()
climatology.rename(
    columns={"Daily_Mean_NO2": "Climatological_Mean"}, inplace=True
)

df_daily = df_daily.merge(climatology, on="month", how="left")
df_daily.sort_values(by="day_idx", inplace=True)


# ==============================================================================
# 4. FIT HARMONIC COSINE CURVE & GENERATE CONTINUOUS WAVE
# ==============================================================================
def cosine_model(m, amplitude, phase_shift, baseline_mean):
  return amplitude * np.cos(2 * np.pi * (m - phase_shift) / 12) + baseline_mean


print("🌊 Fitting harmonic cosine curve to Anaheim climatology...")
initial_guess = [7.0, 12.0, np.mean(df_daily["Daily_Mean_NO2"])]
params, _ = curve_fit(
    cosine_model,
    climatology["month"],
    climatology["Climatological_Mean"],
    p0=initial_guess,
)

amp_fit, phase_fit, base_fit = params
phase_clean = ((phase_fit - 1) % 12) + 1

df_daily["Cosine_Pred"] = cosine_model(df_daily["month"], *params)
df_daily["Residuals"] = df_daily["Cosine_Pred"] - df_daily["Daily_Mean_NO2"]

# Generate 1,000 continuous time steps for a silky-smooth wave
m_smooth = np.linspace(1, 12, 1000)
wave_smooth = cosine_model(m_smooth, *params)
day_smooth = np.linspace(0, len(df_daily) - 1, 1000)

r2_val = r2_score(df_daily["Daily_Mean_NO2"], df_daily["Cosine_Pred"])
rmse_val = np.sqrt(
    mean_squared_error(df_daily["Daily_Mean_NO2"], df_daily["Cosine_Pred"])
)
mae_val = mean_absolute_error(
    df_daily["Daily_Mean_NO2"], df_daily["Cosine_Pred"]
)
res_mean = np.mean(df_daily["Residuals"])
res_std = np.std(df_daily["Residuals"])

print("\n🌴 Anaheim Monitor Climatology & Cosine Parameters:")
print(f"   * Baseline Average (B): {base_fit:.2f} ppb (Benchmark ~10.58 ppb)")
print(f"   * Wave Amplitude (A):   {amp_fit:.2f} ppb (Benchmark ~7.30 ppb)")
print(f"   * Peak Phase Month (φ): {phase_clean:.2f} (Late Winter Trapping)")
print(f"   * Cosine Model R²:      {r2_val:.3f}")
print(f"   * Model RMSE:           {rmse_val:.2f} ppb")

# ==============================================================================
# 5. RENDER CHART 1: DAILY MEANS WITH CLIMATOLOGY STAIRCASE
# ==============================================================================
print("🎨 Rendering Chart 1: Anaheim Climatology Staircase...")
fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)

ax.scatter(
    df_daily["day_idx"],
    df_daily["Daily_Mean_NO2"],
    color="#3b82f6",
    alpha=0.6,
    s=30,
    label="Daily Mean $\\text{NO}_2$",
    zorder=3,
)
ax.plot(
    df_daily["day_idx"],
    df_daily["Climatological_Mean"],
    color="#dc2626",
    linewidth=3.0,
    label="Climatological Monthly Mean",
    zorder=4,
)

ax.set_title(
    "Anaheim $\\text{NO}_2$ Time Series with Climatological Monthly Means\n(Orange"
    " County Coastal Basin Daily Averages)",
    fontsize=15,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel(
    "Sequential Observation Days (Jan through Dec)", fontsize=12, fontweight="bold"
)
ax.set_ylabel("$\\text{NO}_2$ Daily Mean (ppb)", fontsize=12, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="upper right", frameon=True, facecolor="white", fontsize=11)

stats_text1 = (
    f"Statistics:\n"
    f"Overall daily mean: {np.mean(df_daily['Daily_Mean_NO2']):.2f} ppb\n"
    f"Daily std:          {np.std(df_daily['Daily_Mean_NO2']):.2f} ppb\n"
    f"Clim. monthly range: {climatology['Climatological_Mean'].max() - climatology['Climatological_Mean'].min():.2f} ppb\n"
    f"Highest Month:      {climatology['Climatological_Mean'].max():.2f} ppb\n"
    f"Lowest Month:       {climatology['Climatological_Mean'].min():.2f} ppb"
)
ax.text(
    0.02,
    0.95,
    stats_text1,
    transform=ax.transAxes,
    fontsize=11,
    family="monospace",
    verticalalignment="top",
    bbox=dict(
        boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", alpha=0.9
    ),
)

out1 = OUTPUT_DIR / "anaheim_01_climatology_staircase.jpg"
plt.savefig(out1, dpi=300, bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# 6. RENDER CHART 2: CONTINUOUS SILKY-SMOOTH COSINE FIT
# ==============================================================================
print("🎨 Rendering Chart 2: Silky-Smooth Cosine Wave Fit...")
fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)

ax.scatter(
    df_daily["day_idx"],
    df_daily["Daily_Mean_NO2"],
    color="#3b82f6",
    alpha=0.5,
    s=30,
    label="Daily Mean $\\text{NO}_2$",
    zorder=3,
)
ax.plot(
    df_daily["day_idx"],
    df_daily["Climatological_Mean"],
    color="#10b981",
    linewidth=4.0,
    linestyle="--",
    alpha=0.8,
    label="Climatological Monthly Mean",
    zorder=4,
)
ax.plot(
    day_smooth,
    wave_smooth,
    color="#dc2626",
    linewidth=3.5,
    label=f"Cosine Fit: y = {amp_fit:.2f}·cos(2π(m-{phase_clean:.2f})/12) + {base_fit:.2f} ($R^2$={r2_val:.3f})",
    zorder=5,
)

ax.set_title(
    "Anaheim $\\text{NO}_2$ Time Series with Cosine Fit to Climatological"
    " Means\n(Continuous Harmonic Wave Tracking Seasonal Orbital Cycles)",
    fontsize=15,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel(
    "Sequential Observation Days (Jan through Dec)", fontsize=12, fontweight="bold"
)
ax.set_ylabel("$\\text{NO}_2$ Daily Mean (ppb)", fontsize=12, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="upper right", frameon=True, facecolor="white", fontsize=11)

stats_text2 = (
    f"Cosine Fit:\n"
    f"A = {amp_fit:.2f} ppb\n"
    f"φ = {phase_clean:.2f} months\n"
    f"B = {base_fit:.2f} ppb\n"
    f"R² = {r2_val:.4f}\n\n"
    f"Data Stats:\n"
    f"Daily mean: {np.mean(df_daily['Daily_Mean_NO2']):.2f} ppb\n"
    f"Daily std:  {np.std(df_daily['Daily_Mean_NO2']):.2f} ppb"
)
ax.text(
    0.02,
    0.95,
    stats_text2,
    transform=ax.transAxes,
    fontsize=11,
    family="monospace",
    verticalalignment="top",
    bbox=dict(
        boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", alpha=0.9
    ),
)

out2 = OUTPUT_DIR / "anaheim_02_cosine_wave_fit.jpg"
plt.savefig(out2, dpi=300, bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# 7. RENDER CHART 3: ACTUAL VS PREDICTED SCATTER PLOT
# ==============================================================================
print("🎨 Rendering Chart 3: Actual vs. Predicted Parity Scatter...")
fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)

ax.scatter(
    df_daily["Cosine_Pred"],
    df_daily["Daily_Mean_NO2"],
    color="#3b82f6",
    alpha=0.6,
    s=45,
    edgecolor="none",
    label="Observed vs. Cosine Model Predictions",
)

max_lim = max(df_daily["Daily_Mean_NO2"].max(), df_daily["Cosine_Pred"].max()) * 1.1
ax.plot(
    [0, max_lim],
    [0, max_lim],
    "k--",
    linewidth=2.0,
    alpha=0.8,
    label="1:1 Line (Perfect Prediction)",
)

ax.set_title(
    "Actual vs Predicted $\\text{NO}_2$ - Cosine Fit Model\nAnaheim Site"
    " Evaluation",
    fontsize=15,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel(
    "Predicted $\\text{NO}_2$ (Cosine Model) [ppb]", fontsize=12, fontweight="bold"
)
ax.set_ylabel("Actual $\\text{NO}_2$ [ppb]", fontsize=12, fontweight="bold")
ax.set_xlim(0, max_lim)
ax.set_ylim(0, max_lim)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="upper left", frameon=True, facecolor="white", fontsize=11)

stats_text3 = (
    f"Cosine Model\n"
    f"y = {amp_fit:.2f}·cos(2π(m-{phase_clean:.2f})/12) + {base_fit:.2f}\n\n"
    f"Performance:\n"
    f"RMSE: {rmse_val:.2f} ppb\n"
    f"MAE:  {mae_val:.2f} ppb\n"
    f"R²:   {r2_val:.3f}\n\n"
    f"N = {len(df_daily)} days"
)
ax.text(
    0.65,
    0.15,
    stats_text3,
    transform=ax.transAxes,
    fontsize=11,
    family="monospace",
    verticalalignment="top",
    bbox=dict(
        boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", alpha=0.9
    ),
)

out3 = OUTPUT_DIR / "anaheim_03_cosine_scatter_parity.jpg"
plt.savefig(out3, dpi=300, bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# 8. RENDER CHART 4: RESIDUALS WITH ±1 STD DEV BAND
# ==============================================================================
print("🎨 Rendering Chart 4: Anaheim Residuals Band...")
fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)

ax.plot(
    df_daily["day_idx"],
    df_daily["Residuals"],
    color="#3b82f6",
    linewidth=1.5,
    label="Residuals (Predicted - Actual)",
)
ax.axhline(0, color="black", linewidth=1.5, label="Zero Line")

ax.axhline(
    res_std,
    color="#dc2626",
    linestyle="--",
    linewidth=1.2,
    label=f"±1 Std Dev (±{res_std:.2f} ppb)",
)
ax.axhline(-res_std, color="#dc2626", linestyle="--", linewidth=1.2)
ax.axhspan(-res_std, res_std, color="#fef2f2", alpha=0.8, zorder=0)

ax.set_title(
    "Cosine Model Residuals - Anaheim Site\n(Predicted - Actual $\\text{NO}_2$)",
    fontsize=15,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel(
    "Sequential Observation Days (Jan through Dec)", fontsize=12, fontweight="bold"
)
ax.set_ylabel(
    "Residual (Predicted - Actual) [ppb]", fontsize=12, fontweight="bold"
)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="upper right", frameon=True, facecolor="white", fontsize=11)

stats_text4 = (
    f"Cosine Model\n"
    f"y = {amp_fit:.2f}·cos(2π(m-{phase_clean:.2f})/12) + {base_fit:.2f}\n\n"
    f"Residual mean: {res_mean:+.4f} ppb\n"
    f"Residual std:  {res_std:.2f} ppb\n\n"
    f"Performance:\n"
    f"RMSE: {rmse_val:.2f} ppb\n"
    f"MAE:  {mae_val:.2f} ppb\n\n"
    f"Residual range:\n"
    f"Min: {df_daily['Residuals'].min():+.2f} ppb\n"
    f"Max: {df_daily['Residuals'].max():+.2f} ppb"
)
ax.text(
    0.02,
    0.95,
    stats_text4,
    transform=ax.transAxes,
    fontsize=11,
    family="monospace",
    verticalalignment="top",
    bbox=dict(
        boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", alpha=0.9
    ),
)

note_text = (
    "Note:\n"
    "Positive residuals = overprediction\n"
    "Negative residuals = underprediction\n"
    "Good models have residuals centered around zero\n"
    "with minimal systematic patterns."
)
ax.text(
    0.78,
    0.18,
    note_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="#f8fafc",
        edgecolor="#94a3b8",
        alpha=0.9,
    ),
)

out4 = OUTPUT_DIR / "anaheim_04_cosine_residuals_band.jpg"
plt.savefig(out4, dpi=300, bbox_inches="tight")
plt.close(fig)

print("=" * 75)
print(f"✅ All 4 Anaheim Verification Charts generated in: {OUTPUT_DIR}")
print("🎯 ANAHEIM POSITIVE CONTROL ANALYSIS FINISHED!")