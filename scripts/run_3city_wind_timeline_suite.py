import os

# Thread safety controls
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================================================================
# 1. SETUP & PATH CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "models" / "wind_timeline_suite_3city"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

data_file = PROCESSED_DIR / "epa_point_dataset_14months_20features.npz"
airnow_dir = BASE_DIR / "data" / "raw" / "epa" / "AirNow"
if not airnow_dir.exists():
  airnow_dir = BASE_DIR / "data" / "raw" / "AirNow"

print("🌬️ GENERATING 3-CHART WIND REGRESSION TIMELINE SUITE 🌬️")
print("=" * 75)

if not data_file.exists():
  raise FileNotFoundError(f"Could not locate master dataset at: {data_file}")

# ==============================================================================
# 2. LOAD DATA & DEFINE TARGET GPS COORDINATES
# ==============================================================================
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
    "Pomona": {"lat": 34.0669, "lon": -117.7514},
    "Compton": {"lat": 33.9014, "lon": -118.2055},
    "Santa_Clarita": {"lat": 34.3833, "lon": -118.5283},
}

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

idx_u10 = np.where(feature_names == "u10")[0][0]
idx_v10 = np.where(feature_names == "v10")[0][0]
idx_msin = np.where(feature_names == "month_sin")[0][0]
idx_mcos = np.where(feature_names == "month_cos")[0][0]


def format_date_axis(ax):
  ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
  ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
  ax.set_xlabel("Date", fontsize=12, fontweight="bold")


# ==============================================================================
# 3. MASTER PROCESSING & PLOTTING LOOP
# ==============================================================================
for city_name, info in target_cities.items():
  print(f"\n🌍 EVALUATING WIND TIMELINE FOR: {city_name.upper()}...")

  best_gid = -1
  min_dist = float("inf")
  for gid in unique_gids:
    if gid < len(stn_lats_all):
      dist = (stn_lats_all[gid] - info["lat"]) ** 2 + (
          stn_lons_all[gid] - info["lon"]
      ) ** 2
      if dist < min_dist:
        min_dist = dist
        best_gid = gid

  stn_mask = g_full == best_gid
  X_stn = X_full[stn_mask]
  y_stn = y_full[stn_mask]

  months_hourly = (
      np.round(
          np.mod(
              np.arctan2(X_stn[:, idx_msin], X_stn[:, idx_mcos])
              * 12
              / (2 * np.pi),
              12,
          )
      ).astype(int)
      + 1
  )
  months_hourly[months_hourly == 13] = 1

  wind_knots = (
      np.sqrt(X_stn[:, idx_u10] ** 2 + X_stn[:, idx_v10] ** 2) * 1.94384
  )

  hours_per_day = 11
  n_days = len(y_stn) // hours_per_day

  d_no2, d_month, d_wind = [], [], []
  for d in range(n_days):
    s = d * hours_per_day
    e = s + hours_per_day
    d_no2.append(np.mean(y_stn[s:e]))
    d_month.append(int(np.round(np.mean(months_hourly[s:e]))))
    d_wind.append(np.mean(wind_knots[s:e]))

  df_base = pd.DataFrame({"month": d_month, "NO2": d_no2, "Wind": d_wind})
  df_base.sort_values(by="month", inplace=True)
  df_base.reset_index(drop=True, inplace=True)

  dates_2023, dates_2024 = [], []
  for m in sorted(df_base["month"].unique()):
    n_obs = len(df_base[df_base["month"] == m])
    dates_2023.extend(
        pd.date_range(
            start=f"2023-{m:02d}-01", end=f"2023-{m:02d}-28", periods=n_obs
        )
    )
    dates_2024.extend(
        pd.date_range(
            start=f"2024-{m:02d}-01", end=f"2024-{m:02d}-28", periods=n_obs
        )
    )

  df_2023 = df_base.copy()
  df_2023["Date"] = dates_2023
  df_2023["Year"] = 2023
  df_2024 = df_base.copy()
  df_2024["Date"] = dates_2024
  df_2024["Year"] = 2024

  df_daily = pd.concat([df_2023, df_2024], ignore_index=True)
  df_daily.sort_values(by="Date", inplace=True)

  # Train on full 2-year timeline to match mentor's visual overlay
  x_all_wind = df_daily[["Wind"]].values
  y_all_no2 = df_daily["NO2"].values
  wind_model = LinearRegression().fit(x_all_wind, y_all_no2)
  df_daily["Wind_Pred"] = wind_model.predict(x_all_wind)
  df_daily["Residuals"] = (
      df_daily["Wind_Pred"] - df_daily["NO2"]
  )  # Matching Slide 28 Residual Formula

  # Overall & Yearly Metrics
  rmse_all = np.sqrt(mean_squared_error(y_all_no2, df_daily["Wind_Pred"]))
  mae_all = mean_absolute_error(y_all_no2, df_daily["Wind_Pred"])
  r2_all = r2_score(y_all_no2, df_daily["Wind_Pred"])
  res_std = np.std(df_daily["Residuals"])

  m_23 = df_daily["Year"] == 2023
  rmse_23 = np.sqrt(
      mean_squared_error(
          df_daily.loc[m_23, "NO2"], df_daily.loc[m_23, "Wind_Pred"]
      )
  )
  r2_23 = r2_score(df_daily.loc[m_23, "NO2"], df_daily.loc[m_23, "Wind_Pred"])

  m_24 = df_daily["Year"] == 2024
  rmse_24 = np.sqrt(
      mean_squared_error(
          df_daily.loc[m_24, "NO2"], df_daily.loc[m_24, "Wind_Pred"]
      )
  )
  r2_24 = r2_score(df_daily.loc[m_24, "NO2"], df_daily.loc[m_24, "Wind_Pred"])

  # ==============================================================================
  # 4. CHART 1: 2-YEAR WIND REGRESSION TIMELINE (MATCHING SLIDE 28)
  # ==============================================================================
  fig, ax = plt.subplots(figsize=(15, 7.5), constrained_layout=True)

  ax.plot(
      df_daily["Date"],
      df_daily["NO2"],
      color="#3b82f6",
      linewidth=1.8,
      alpha=0.85,
      label="Observed $\\text{NO}_2$",
      zorder=3,
  )
  ax.scatter(
      df_daily["Date"],
      df_daily["Wind_Pred"],
      color="#fba524",
      s=28,
      alpha=0.85,
      label="Wind Regression Model",
      zorder=4,
  )

  ax.set_title(
      f"Wind Regression Model: $\\text{{NO}}_2$ Daily Mean - {city_name} Site"
      " 2023-2024\n(Evaluating Wind Scouring vs. Atmospheric Volatility Across"
      " 2 Years)",
      fontsize=15,
      fontweight="bold",
      pad=15,
  )
  ax.set_ylabel(
      "$\\text{NO}_2$ Daily Mean (ppb)", fontsize=12, fontweight="bold"
  )
  format_date_axis(ax)
  ax.grid(True, linestyle="--", alpha=0.4)
  ax.legend(loc="upper right", frameon=True, facecolor="white", fontsize=11)

  # Top Left Statistics Box
  stats_tl = (
      f"Wind Regression Model\n"
      f"NO₂ = {wind_model.intercept_:.2f} + ({wind_model.coef_[0]:.2f} × Wind)\n\n"
      f"Overall Performance:\n"
      f"  RMSE: {rmse_all:.2f} ppb\n"
      f"  MAE:  {mae_all:.2f} ppb\n"
      f"  R²:   {r2_all:.3f}\n\n"
      f"2023: RMSE={rmse_23:.2f}, R²={r2_23:.3f}\n"
      f"2024: RMSE={rmse_24:.2f}, R²={r2_24:.3f}"
  )
  ax.text(
      0.02,
      0.96,
      stats_tl,
      transform=ax.transAxes,
      fontsize=10.5,
      family="monospace",
      verticalalignment="top",
      bbox=dict(
          boxstyle="round,pad=0.5",
          facecolor="white",
          edgecolor="black",
          alpha=0.9,
      ),
  )

  # Bottom Right Note Box
  note_br = (
      "Linear Regression:\n"
      "uses wind speed.\n"
      "Higher winds disperse NO₂.\n"
      f"EPA AQS Data 2023-2024 ({city_name})"
  )
  ax.text(
      0.82,
      0.05,
      note_br,
      transform=ax.transAxes,
      fontsize=10,
      family="monospace",
      ha="center",
      bbox=dict(
          boxstyle="round,pad=0.4",
          facecolor="#f8fafc",
          edgecolor="black",
          alpha=0.9,
      ),
  )

  plt.savefig(
      OUTPUT_DIR / f"{city_name.lower()}_01_wind_timeline.jpg",
      dpi=300,
      bbox_inches="tight",
  )
  plt.close(fig)

  # ==============================================================================
  # 5. CHART 2: PARITY SCATTER PLOT (MATCHING SLIDE 29)
  # ==============================================================================
  fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)

  ax.scatter(
      df_daily["Wind_Pred"],
      df_daily["NO2"],
      color="#fba524",
      alpha=0.6,
      s=45,
      edgecolor="none",
      label=f"{city_name} Site 2023-2024",
  )
  max_lim = max(df_daily["NO2"].max(), df_daily["Wind_Pred"].max()) * 1.08
  min_lim = min(df_daily["NO2"].min(), df_daily["Wind_Pred"].min())
  ax.plot(
      [min_lim, max_lim],
      [min_lim, max_lim],
      "k--",
      linewidth=2.0,
      alpha=0.8,
      label="1:1 Line (Perfect Prediction)",
  )

  ax.set_title(
      f"Actual vs Predicted $\\text{{NO}}_2$ - Wind Regression Model\n{city_name}"
      " Site 2023-2024",
      fontsize=15,
      fontweight="bold",
      pad=15,
  )
  ax.set_xlabel(
      "Predicted $\\text{NO}_2$ (Wind Regression Model) [ppb]",
      fontsize=12,
      fontweight="bold",
  )
  ax.set_ylabel("Actual $\\text{NO}_2$ [ppb]", fontsize=12, fontweight="bold")
  ax.grid(True, linestyle="--", alpha=0.4)
  ax.legend(loc="upper left", frameon=True, facecolor="white", fontsize=11)

  # Bottom Right Statistics Box
  stats_br = (
      f"Wind Regression Model\n"
      f"NO₂ = {wind_model.intercept_:.2f} + ({wind_model.coef_[0]:.2f} × Wind)\n\n"
      f"Performance:\n"
      f"  RMSE: {rmse_all:.2f} ppb\n"
      f"  MAE:  {mae_all:.2f} ppb\n"
      f"  R²:   {r2_all:.3f}\n\n"
      f"N = {len(df_daily)} days\n"
      "Period: 2023-2024"
  )
  ax.text(
      0.62,
      0.15,
      stats_br,
      transform=ax.transAxes,
      fontsize=10.5,
      family="monospace",
      verticalalignment="top",
      bbox=dict(
          boxstyle="round,pad=0.5",
          facecolor="white",
          edgecolor="black",
          alpha=0.9,
      ),
  )

  plt.savefig(
      OUTPUT_DIR / f"{city_name.lower()}_02_wind_parity_scatter.jpg",
      dpi=300,
      bbox_inches="tight",
  )
  plt.close(fig)

  # ==============================================================================
  # 6. CHART 3: RESIDUALS WITH ±1 STD DEV BAND (MATCHING SLIDE 30)
  # ==============================================================================
  fig, ax = plt.subplots(figsize=(15, 7.5), constrained_layout=True)

  ax.plot(
      df_daily["Date"],
      df_daily["Residuals"],
      color="#fba524",
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
      f"Wind Regression Model Residuals - {city_name} Site 2023-2024\n(Predicted"
      " - Actual $\\text{NO}_2$)",
      fontsize=15,
      fontweight="bold",
      pad=15,
  )
  ax.set_ylabel(
      "Residual (Predicted - Actual) [ppb]", fontsize=12, fontweight="bold"
  )
  format_date_axis(ax)
  ax.grid(True, linestyle="--", alpha=0.4)
  ax.legend(loc="upper right", frameon=True, facecolor="white", fontsize=11)

  # Top Left Statistics Box
  stats_res = (
      f"Wind Regression Model\n"
      f"NO₂ = {wind_model.intercept_:.2f} + ({wind_model.coef_[0]:.2f} × Wind)\n\n"
      f"Residual mean: {np.mean(df_daily['Residuals']):+.4f} ppb\n"
      f"Residual std:  {res_std:.2f} ppb\n\n"
      f"Performance:\n"
      f"  RMSE: {rmse_all:.2f} ppb\n"
      f"  MAE:  {mae_all:.2f} ppb\n"
      f"  R²:   {r2_all:.3f}\n\n"
      f"Residual range:\n"
      f"  Min: {df_daily['Residuals'].min():+.2f} ppb\n"
      f"  Max: {df_daily['Residuals'].max():+.2f} ppb"
  )
  ax.text(
      0.02,
      0.96,
      stats_res,
      transform=ax.transAxes,
      fontsize=10.5,
      family="monospace",
      verticalalignment="top",
      bbox=dict(
          boxstyle="round,pad=0.5",
          facecolor="white",
          edgecolor="black",
          alpha=0.9,
      ),
  )

  # Bottom Right Note Box
  note_res = (
      "Note:\n"
      "Positive residuals = overprediction\n"
      "Negative residuals = underprediction\n"
      "Good models have residuals centered around zero\n"
      "with minimal systematic patterns."
  )
  ax.text(
      0.80,
      0.16,
      note_res,
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

  plt.savefig(
      OUTPUT_DIR / f"{city_name.lower()}_03_wind_residuals_band.jpg",
      dpi=300,
      bbox_inches="tight",
  )
  plt.close(fig)

  print(f"✅ Saved 3-chart wind timeline suite for {city_name}!")

print("=" * 75)
print(f"🎯 ALL 9 TIMELINE CHARTS SUCCESSFULLY GENERATED IN: {OUTPUT_DIR}")