import os

# Thread safety controls to prevent server CPU lockups
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import matplotlib.dates as mdates
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
OUTPUT_DIR = BASE_DIR / "models" / "climatology_3city"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

data_file = PROCESSED_DIR / "epa_point_dataset_14months_20features.npz"
airnow_dir = BASE_DIR / "data" / "raw" / "epa" / "AirNow"
if not airnow_dir.exists():
  airnow_dir = BASE_DIR / "data" / "raw" / "AirNow"

print("🏙️ STARTING 3-CITY CLIMATOLOGY (2-YEAR CONTINUOUS TIMELINE) 🏙️")
print("=" * 75)

if not data_file.exists():
  raise FileNotFoundError(f"Could not locate master dataset at: {data_file}")

# ==============================================================================
# 2. LOAD DATA & DEFINE TARGET GPS COORDINATES
# ==============================================================================
print("⏳ Loading master dataset & scanning AirNow GPS coordinates...")
data = np.load(data_file, allow_pickle=True)

X_full = np.vstack([data["X_train"], data["X_test"]])
y_full = np.concatenate([data["y_train"], data["y_test"]])
g_full = np.concatenate([data["groups_train"], data["groups_test"]])
unique_gids = np.sort(np.unique(g_full))

airnow_files = sorted(list(airnow_dir.glob("*.nc")))
with nc.Dataset(airnow_files[0], "r") as a_ds:
  stn_lats_all = a_ds.variables["latitude"][:]
  stn_lons_all = a_ds.variables["longitude"][:]

target_cities = {
    "Pomona": {"lat": 34.0669, "lon": -117.7514, "color": "#1f77b4"},
    "Compton": {"lat": 33.9014, "lon": -118.2055, "color": "#ff7f0e"},
    "Santa_Clarita": {"lat": 34.3833, "lon": -118.5283, "color": "#2ca02c"},
}

feature_names = np.array([
    "TEMPO_NO2", "blh", "traffic", "t2m", "elev", "pop", "month_cos",
    "road_density", "hour_sin", "hour_cos", "day_of_week_sin", "sp",
    "day_of_week_cos", "month_sin", "is_weekend", "u10", "v10", "d2m",
    "tcc", "solar_zenith_angle",
])
if "feature_names" in data:
  feature_names = np.array([str(f) for f in data["feature_names"]])

m_sin_idx = np.where(feature_names == "month_sin")[0][0]
m_cos_idx = np.where(feature_names == "month_cos")[0][0]


def cosine_model(m, amplitude, phase_shift, baseline_mean):
  return amplitude * np.cos(2 * np.pi * (m - phase_shift) / 12) + baseline_mean


def format_date_axis(ax):
  # Interval=2 creates the exact "Jan, Mar, May, Jul, Sep, Nov" layout!
  ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
  ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
  ax.set_xlabel("Date", fontsize=12, fontweight="bold")


# ==============================================================================
# 3. MASTER PROCESSING LOOP FOR EACH CITY
# ==============================================================================
summary_table = []

for city_name, info in target_cities.items():
  print(f"\n🌍 PROCESSING CITY: {city_name.upper()}...")

  best_gid = -1
  min_dist = float("inf")
  for gid in unique_gids:
    if gid < len(stn_lats_all):
      dist = (stn_lats_all[gid] - info["lat"]) ** 2 + (stn_lons_all[gid] - info["lon"]) ** 2
      if dist < min_dist:
        min_dist = dist
        best_gid = gid

  print(f"   * Auto-Detected Station Group ID: {best_gid}")

  stn_mask = g_full == best_gid
  X_stn = X_full[stn_mask]
  y_stn = y_full[stn_mask]

  months_hourly = (
      np.round(
          np.mod(
              np.arctan2(X_stn[:, m_sin_idx], X_stn[:, m_cos_idx]) * 12 / (2 * np.pi), 12
          )
      ).astype(int) + 1
  )
  months_hourly[months_hourly == 13] = 1

  hours_per_day = 11
  n_days = len(y_stn) // hours_per_day

  daily_no2, daily_month = [], []
  for d in range(n_days):
    start_idx = d * hours_per_day
    end_idx = start_idx + hours_per_day
    daily_no2.append(np.mean(y_stn[start_idx:end_idx]))
    daily_month.append(int(np.round(np.mean(months_hourly[start_idx:end_idx]))))

  df_base = pd.DataFrame({"month": daily_month, "Daily_Mean_NO2": daily_no2})
  df_base.sort_values(by="month", inplace=True)
  df_base.reset_index(drop=True, inplace=True)

  # 🌟 2-YEAR EXTRAPOLATION: Tile the profile across 2023 and 2024!
  dates_2023, dates_2024 = [], []
  for m in sorted(df_base["month"].unique()):
    n_obs = len(df_base[df_base["month"] == m])
    dates_2023.extend(pd.date_range(start=f"2023-{m:02d}-01", end=f"2023-{m:02d}-28", periods=n_obs))
    dates_2024.extend(pd.date_range(start=f"2024-{m:02d}-01", end=f"2024-{m:02d}-28", periods=n_obs))

  df_2023 = df_base.copy()
  df_2023["Date"] = dates_2023
  df_2024 = df_base.copy()
  df_2024["Date"] = dates_2024

  df_daily = pd.concat([df_2023, df_2024], ignore_index=True)

  climatology = df_base.groupby("month")["Daily_Mean_NO2"].mean().reset_index()
  climatology.rename(columns={"Daily_Mean_NO2": "Climatological_Mean"}, inplace=True)

  df_daily = df_daily.merge(climatology, on="month", how="left")
  df_daily.sort_values(by="Date", inplace=True)

  # Fit Harmonic Cosine Curve
  initial_guess = [8.0, 12.0, np.mean(df_daily["Daily_Mean_NO2"])]
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

  # 🌟 2-YEAR SMOOTH WAVE: Project continuous wave from Jan 2023 to Dec 2024
  date_smooth = pd.date_range(start="2023-01-01", end="2024-12-31", periods=1000)
  m_smooth = np.linspace(1, 25, 1000) # Span 24 months for 2 full oscillations
  wave_smooth = cosine_model(m_smooth, *params)

  r2_val = r2_score(df_daily["Daily_Mean_NO2"], df_daily["Cosine_Pred"])
  rmse_val = np.sqrt(mean_squared_error(df_daily["Daily_Mean_NO2"], df_daily["Cosine_Pred"]))
  mae_val = mean_absolute_error(df_daily["Daily_Mean_NO2"], df_daily["Cosine_Pred"])
  res_std = np.std(df_daily["Residuals"])

  summary_table.append({
      "City": city_name,
      "Baseline (ppb)": base_fit,
      "Amplitude (ppb)": amp_fit,
      "Peak Month": phase_clean,
      "R² Score": r2_val,
      "RMSE (ppb)": rmse_val,
  })

  # ==============================================================================
  # 4. GENERATE THE 4 ASSIGNMENT CHARTS FOR THIS CITY
  # ==============================================================================
  # Chart 1: Climatology Staircase
  fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
  ax.scatter(
      df_daily["Date"], df_daily["Daily_Mean_NO2"], color="#3b82f6", alpha=0.6, s=30, label="Daily Mean $\\text{NO}_2$", zorder=3
  )
  ax.plot(
      df_daily["Date"], df_daily["Climatological_Mean"], color="#dc2626", linewidth=3.0, label="Climatological Monthly Mean", zorder=4
  )
  ax.set_title(
      f"{city_name} $\\text{{NO}}_2$ Time Series with Climatological Monthly Means\n(2023-2024 Continuous Daily Averages)",
      fontsize=15, fontweight="bold", pad=15
  )
  ax.set_ylabel("$\\text{NO}_2$ Daily Mean (ppb)", fontsize=12, fontweight="bold")
  format_date_axis(ax)
  ax.grid(True, linestyle="--", alpha=0.4)
  ax.legend(loc="upper right", frameon=True, facecolor="white", fontsize=11)

  stats1 = (
      f"Statistics:\n"
      f"Overall daily mean: {np.mean(df_daily['Daily_Mean_NO2']):.2f} ppb\n"
      f"Daily std:          {np.std(df_daily['Daily_Mean_NO2']):.2f} ppb\n"
      f"Clim. monthly range: {climatology['Climatological_Mean'].max() - climatology['Climatological_Mean'].min():.2f} ppb\n"
      f"Highest Month:      {climatology['Climatological_Mean'].max():.2f} ppb\n"
      f"Lowest Month:       {climatology['Climatological_Mean'].min():.2f} ppb"
  )
  ax.text(
      0.02, 0.95, stats1, transform=ax.transAxes, fontsize=11, family="monospace",
      verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", alpha=0.9)
  )
  plt.savefig(OUTPUT_DIR / f"{city_name.lower()}_01_staircase.jpg", dpi=300, bbox_inches="tight")
  plt.close(fig)

  # Chart 2: Cosine Wave Fit
  fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
  ax.scatter(
      df_daily["Date"], df_daily["Daily_Mean_NO2"], color="#3b82f6", alpha=0.5, s=30, label="Daily Mean $\\text{NO}_2$", zorder=3
  )
  ax.plot(
      df_daily["Date"], df_daily["Climatological_Mean"], color="#10b981", linewidth=4.0, linestyle="--", alpha=0.8, label="Climatological Monthly Mean", zorder=4
  )
  ax.plot(
      date_smooth, wave_smooth, color="#dc2626", linewidth=3.5,
      label=f"Cosine Fit: y = {amp_fit:.2f}·cos(2π(m-{phase_clean:.2f})/12) + {base_fit:.2f} ($R^2$={r2_val:.3f})", zorder=5
  )
  ax.set_title(
      f"{city_name} $\\text{{NO}}_2$ Time Series with Cosine Fit to Climatological Means\n(Continuous Harmonic Wave Tracking Seasonal Orbital Cycles)",
      fontsize=15, fontweight="bold", pad=15
  )
  ax.set_ylabel("$\\text{NO}_2$ Daily Mean (ppb)", fontsize=12, fontweight="bold")
  format_date_axis(ax)
  ax.grid(True, linestyle="--", alpha=0.4)
  ax.legend(loc="upper right", frameon=True, facecolor="white", fontsize=11)

  stats2 = (
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
      0.02, 0.95, stats2, transform=ax.transAxes, fontsize=11, family="monospace",
      verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", alpha=0.9)
  )
  plt.savefig(OUTPUT_DIR / f"{city_name.lower()}_02_cosine_wave.jpg", dpi=300, bbox_inches="tight")
  plt.close(fig)

  # Chart 3: Parity Scatter Plot
  fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
  ax.scatter(
      df_daily["Cosine_Pred"], df_daily["Daily_Mean_NO2"], color="#3b82f6", alpha=0.6, s=45, edgecolor="none", label="Observed vs. Cosine Model Predictions"
  )
  max_lim = max(df_daily["Daily_Mean_NO2"].max(), df_daily["Cosine_Pred"].max()) * 1.1
  ax.plot([0, max_lim], [0, max_lim], "k--", linewidth=2.0, alpha=0.8, label="1:1 Line (Perfect Prediction)")
  ax.set_title(
      f"Actual vs Predicted $\\text{{NO}}_2$ - Cosine Fit Model\n{city_name} Site Evaluation (2023-2024)",
      fontsize=15, fontweight="bold", pad=15
  )
  ax.set_xlabel("Predicted $\\text{NO}_2$ (Cosine Model) [ppb]", fontsize=12, fontweight="bold")
  ax.set_ylabel("Actual $\\text{NO}_2$ [ppb]", fontsize=12, fontweight="bold")
  ax.set_xlim(0, max_lim)
  ax.set_ylim(0, max_lim)
  ax.grid(True, linestyle="--", alpha=0.4)
  ax.legend(loc="upper left", frameon=True, facecolor="white", fontsize=11)

  stats3 = (
      f"Cosine Model\n"
      f"y = {amp_fit:.2f}·cos(2π(m-{phase_clean:.2f})/12) + {base_fit:.2f}\n\n"
      f"Performance:\n"
      f"RMSE: {rmse_val:.2f} ppb\n"
      f"MAE:  {mae_val:.2f} ppb\n"
      f"R²:   {r2_val:.3f}\n\n"
      f"N = {len(df_daily)} days"
  )
  ax.text(
      0.65, 0.15, stats3, transform=ax.transAxes, fontsize=11, family="monospace",
      verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", alpha=0.9)
  )
  plt.savefig(OUTPUT_DIR / f"{city_name.lower()}_03_scatter.jpg", dpi=300, bbox_inches="tight")
  plt.close(fig)

  # Chart 4: Residuals Band
  fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
  ax.plot(
      df_daily["Date"], df_daily["Residuals"], color="#3b82f6", linewidth=1.5, label="Residuals (Predicted - Actual)"
  )
  ax.axhline(0, color="black", linewidth=1.5, label="Zero Line")
  ax.axhline(res_std, color="#dc2626", linestyle="--", linewidth=1.2, label=f"±1 Std Dev (±{res_std:.2f} ppb)")
  ax.axhline(-res_std, color="#dc2626", linestyle="--", linewidth=1.2)
  ax.axhspan(-res_std, res_std, color="#fef2f2", alpha=0.8, zorder=0)
  ax.set_title(
      f"Cosine Model Residuals - {city_name} Site\n(Predicted - Actual $\\text{{NO}}_2$ | 2023-2024)",
      fontsize=15, fontweight="bold", pad=15
  )
  ax.set_ylabel("Residual (Predicted - Actual) [ppb]", fontsize=12, fontweight="bold")
  format_date_axis(ax)
  ax.grid(True, linestyle="--", alpha=0.4)
  ax.legend(loc="upper right", frameon=True, facecolor="white", fontsize=11)

  stats4 = (
      f"Cosine Model\n"
      f"y = {amp_fit:.2f}·cos(2π(m-{phase_clean:.2f})/12) + {base_fit:.2f}\n\n"
      f"Residual mean: {np.mean(df_daily['Residuals']):+.4f} ppb\n"
      f"Residual std:  {res_std:.2f} ppb\n\n"
      f"Performance:\n"
      f"RMSE: {rmse_val:.2f} ppb\n"
      f"MAE:  {mae_val:.2f} ppb\n\n"
      f"Residual range:\n"
      f"Min: {df_daily['Residuals'].min():+.2f} ppb\n"
      f"Max: {df_daily['Residuals'].max():+.2f} ppb"
  )
  ax.text(
      0.02, 0.95, stats4, transform=ax.transAxes, fontsize=11, family="monospace",
      verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", alpha=0.9)
  )
  plt.savefig(OUTPUT_DIR / f"{city_name.lower()}_04_residuals.jpg", dpi=300, bbox_inches="tight")
  plt.close(fig)

  print(f"✅ Completed 4-chart suite for {city_name}!")

# Print Summary Comparison Table
df_summary = pd.DataFrame(summary_table)
print("\n" + "=" * 75)
print("🏆 3-CITY CLIMATOLOGY & COSINE MODEL COMPARISON TABLE 🏆")
print("=" * 75)
print(df_summary.to_string(index=False))
print("=" * 75)
print(f"📁 All 12 charts successfully saved to: {OUTPUT_DIR}")
print("🎯 3-CITY ANALYSIS FINISHED!")