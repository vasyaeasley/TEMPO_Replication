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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================================================================
# 1. SETUP & PATH CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "models" / "harmonic_regression_3city"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

data_file = PROCESSED_DIR / "epa_point_dataset_14months_20features.npz"
airnow_dir = BASE_DIR / "data" / "raw" / "epa" / "AirNow"
if not airnow_dir.exists():
  airnow_dir = BASE_DIR / "data" / "raw" / "AirNow"

print(
    "🏙️ STARTING 3-CITY HARMONIC REGRESSION (2023 TRAIN / 2024 TEST SPLIT) 🏙️"
)
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

m_sin_idx = np.where(feature_names == "month_sin")[0][0]
m_cos_idx = np.where(feature_names == "month_cos")[0][0]


# Helper to engineer DOY Harmonic Features
def create_harmonic_features(doy_series):
  doy = doy_series.values
  f1 = np.sin(2 * np.pi * doy / 365.25)
  f2 = np.cos(2 * np.pi * doy / 365.25)
  f3 = np.sin(4 * np.pi * doy / 365.25)
  f4 = np.cos(4 * np.pi * doy / 365.25)
  return np.column_stack([f1, f2, f3, f4])


def format_date_axis(ax):
  ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
  ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
  ax.set_xlabel("Date", fontsize=12, fontweight="bold")


# ==============================================================================
# 3. MASTER PROCESSING & MODELING LOOP
# ==============================================================================
summary_table = []

for city_name, info in target_cities.items():
  print(f"\n🌍 BUILDING HARMONIC REGRESSION FOR: {city_name.upper()}...")

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

  print(f"   * Auto-Detected Station Group ID: {best_gid}")

  stn_mask = g_full == best_gid
  X_stn = X_full[stn_mask]
  y_stn = y_full[stn_mask]

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

  hours_per_day = 11
  n_days = len(y_stn) // hours_per_day

  daily_no2, daily_month = [], []
  for d in range(n_days):
    start_idx = d * hours_per_day
    end_idx = start_idx + hours_per_day
    daily_no2.append(np.mean(y_stn[start_idx:end_idx]))
    daily_month.append(
        int(np.round(np.mean(months_hourly[start_idx:end_idx])))
    )

  df_base = pd.DataFrame({"month": daily_month, "Daily_Mean_NO2": daily_no2})
  df_base.sort_values(by="month", inplace=True)
  df_base.reset_index(drop=True, inplace=True)

  # Project continuous 2-year timeline (2023 and 2024)
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
  df_daily["DOY"] = df_daily["Date"].dt.dayofyear

  # 🌟 TRAIN ON 2023, TEST ON 2024
  train_mask = df_daily["Year"] == 2023
  test_mask = df_daily["Year"] == 2024

  X_train_harm = create_harmonic_features(df_daily.loc[train_mask, "DOY"])
  y_train = df_daily.loc[train_mask, "Daily_Mean_NO2"].values

  X_test_harm = create_harmonic_features(df_daily.loc[test_mask, "DOY"])
  y_test = df_daily.loc[test_mask, "Daily_Mean_NO2"].values

  # 1. Fit 2-Harmonic Regression Model (DOY 2 Harmonics)
  harm_model = LinearRegression()
  harm_model.fit(X_train_harm, y_train)

  df_daily["Harmonic_Pred"] = 0.0
  df_daily.loc[train_mask, "Harmonic_Pred"] = harm_model.predict(X_train_harm)
  df_daily.loc[test_mask, "Harmonic_Pred"] = harm_model.predict(X_test_harm)
  df_daily["Residuals"] = (
      df_daily["Harmonic_Pred"] - df_daily["Daily_Mean_NO2"]
  )

  # 2. Fit 1-Harmonic Cosine Climatology Baseline (for comparison table)
  X_train_cos = X_train_harm[:, :2]  # Keep only annual sin/cos
  X_test_cos = X_test_harm[:, :2]
  cos_model = LinearRegression()
  cos_model.fit(X_train_cos, y_train)
  cos_preds_test = cos_model.predict(X_test_cos)

  # 3. Persistence Model Baseline (t-1 day)
  persist_preds_test = np.roll(y_test, 1)
  persist_preds_test[0] = y_test[0]

  # Evaluate Test Set (2024) Performance Metrics
  rmse_harm = np.sqrt(
      mean_squared_error(
          y_test, df_daily.loc[test_mask, "Harmonic_Pred"].values
      )
  )
  mae_harm = mean_absolute_error(
      y_test, df_daily.loc[test_mask, "Harmonic_Pred"].values
  )
  r2_harm = r2_score(y_test, df_daily.loc[test_mask, "Harmonic_Pred"].values)

  rmse_cos = np.sqrt(mean_squared_error(y_test, cos_preds_test))
  mae_cos = mean_absolute_error(y_test, cos_preds_test)
  r2_cos = r2_score(y_test, cos_preds_test)

  rmse_pers = np.sqrt(mean_squared_error(y_test, persist_preds_test))
  mae_pers = mean_absolute_error(y_test, persist_preds_test)
  r2_pers = r2_score(y_test, persist_preds_test)

  res_std = np.std(df_daily.loc[test_mask, "Residuals"])

  print(f"   * 2024 Test RMSE (Harmonic): {rmse_harm:.2f} ppb")
  print(f"   * 2024 Test R² (Harmonic):   {r2_harm:.3f}")

  # Store for Comparison Table
  summary_table.extend([
      {
          "City": city_name,
          "Model": "1. Persistence (t-1)",
          "Prediction Type": "Previous Day",
          "RMSE (ppb)": rmse_pers,
          "MAE (ppb)": mae_pers,
          "R² Score": r2_pers,
      },
      {
          "City": city_name,
          "Model": "2. Harmonic Regression",
          "Prediction Type": "DOY (2 Harmonics)",
          "RMSE (ppb)": rmse_harm,
          "MAE (ppb)": mae_harm,
          "R² Score": r2_harm,
      },
      {
          "City": city_name,
          "Model": "3. Cosine Climatology",
          "Prediction Type": "DOY (1 Harmonic)",
          "RMSE (ppb)": rmse_cos,
          "MAE (ppb)": mae_cos,
          "R² Score": r2_cos,
      },
  ])

  # ==============================================================================
  # 4. GENERATE THE SLIDE CHARTS
  # ==============================================================================
  # Chart 1: Two-Panel Time Series & DOY Function Fit (Matching Slide 10)
  fig, (ax_top, ax_bot) = plt.subplots(
      2, 1, figsize=(14, 12), constrained_layout=True
  )

  # Top Panel: Time Series 2023-2024
  ax_top.scatter(
      df_daily.loc[train_mask, "Date"],
      df_daily.loc[train_mask, "Daily_Mean_NO2"],
      color="#1f77b4",
      alpha=0.5,
      s=25,
      label="2023 Observed (Train)",
      zorder=3,
  )
  ax_top.scatter(
      df_daily.loc[test_mask, "Date"],
      df_daily.loc[test_mask, "Daily_Mean_NO2"],
      color="#ff7f0e",
      alpha=0.5,
      s=25,
      label="2024 Observed (Test)",
      zorder=3,
  )
  ax_top.plot(
      df_daily.loc[train_mask, "Date"],
      df_daily.loc[train_mask, "Harmonic_Pred"],
      color="#1f77b4",
      linewidth=3.0,
      label="2023 Model Fit",
      zorder=4,
  )
  ax_top.plot(
      df_daily.loc[test_mask, "Date"],
      df_daily.loc[test_mask, "Harmonic_Pred"],
      color="#ff7f0e",
      linewidth=3.0,
      label="2024 Model Prediction",
      zorder=4,
  )

  ax_top.set_title(
      f"$\\text{{NO}}_2$ Model (DOY-based Harmonic Regression) - {city_name}"
      " Site\nTrain (2023) vs. Test (2024) Time Series Evaluation",
      fontsize=14,
      fontweight="bold",
  )
  ax_top.set_ylabel(
      "$\\text{NO}_2$ Daily Mean (ppb)", fontsize=12, fontweight="bold"
  )
  format_date_axis(ax_top)
  ax_top.grid(True, linestyle="--", alpha=0.4)
  ax_top.legend(loc="upper right", frameon=True, facecolor="white", ncol=2)

  # Bottom Panel: NO2 as Function of Day of Year (DOY 1-365)
  doy_grid = np.linspace(1, 365, 1000)
  doy_harm_grid = create_harmonic_features(pd.Series(doy_grid))
  doy_pred_grid = harm_model.predict(doy_harm_grid)

  ax_bot.scatter(
      df_daily.loc[train_mask, "DOY"],
      df_daily.loc[train_mask, "Daily_Mean_NO2"],
      color="#1f77b4",
      alpha=0.4,
      s=20,
      label="2023 Observed",
  )
  ax_bot.scatter(
      df_daily.loc[test_mask, "DOY"],
      df_daily.loc[test_mask, "Daily_Mean_NO2"],
      color="#ff7f0e",
      alpha=0.4,
      s=20,
      label="2024 Observed",
  )
  ax_bot.plot(
      doy_grid,
      doy_pred_grid,
      color="#2ca02c",
      linewidth=3.5,
      label="Model (2-Harmonic DOY Function)",
      zorder=5,
  )

  ax_bot.set_title(
      f"$\\text{{NO}}_2$ as Function of Day of Year (DOY) - {city_name}"
      " Seasonal Profile",
      fontsize=14,
      fontweight="bold",
  )
  ax_bot.set_xlabel("Day of Year (1 to 365)", fontsize=12, fontweight="bold")
  ax_bot.set_ylabel(
      "$\\text{NO}_2$ Daily Mean (ppb)", fontsize=12, fontweight="bold"
  )
  ax_bot.set_xlim(1, 365)
  ax_bot.grid(True, linestyle="--", alpha=0.4)
  ax_bot.legend(loc="upper right", frameon=True, facecolor="white")

  plt.savefig(
      OUTPUT_DIR / f"{city_name.lower()}_01_timeseries_and_doy_fit.jpg",
      dpi=300,
      bbox_inches="tight",
  )
  plt.close(fig)

  # Chart 2: Parity Scatter Plot (Matching Slide 13)
  fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
  ax.scatter(
      df_daily.loc[test_mask, "Harmonic_Pred"],
      df_daily.loc[test_mask, "Daily_Mean_NO2"],
      color="#1f77b4",
      alpha=0.6,
      s=45,
      edgecolor="none",
      label="2024 Out-of-Sample Predictions",
  )
  max_lim = (
      max(df_daily["Daily_Mean_NO2"].max(), df_daily["Harmonic_Pred"].max())
      * 1.1
  )
  ax.plot(
      [0, max_lim],
      [0, max_lim],
      "k--",
      linewidth=2.0,
      alpha=0.8,
      label="1:1 Line (Perfect Prediction)",
  )

  ax.set_title(
      f"Actual vs Predicted $\\text{{NO}}_2$ - Harmonic Regression Model\n{city_name} Site (2024 Test Set)",
      fontsize=15,
      fontweight="bold",
      pad=15,
  )
  ax.set_xlabel(
      "Predicted $\\text{NO}_2$ (Harmonic Model) [ppb]",
      fontsize=12,
      fontweight="bold",
  )
  ax.set_ylabel("Actual $\\text{NO}_2$ [ppb]", fontsize=12, fontweight="bold")
  ax.set_xlim(0, max_lim)
  ax.set_ylim(0, max_lim)
  ax.grid(True, linestyle="--", alpha=0.4)
  ax.legend(loc="upper left", frameon=True, facecolor="white", fontsize=11)

  stats_scatter = (
      f"Harmonic Model (2 Harmonics)\n"
      f"DOY Regression Architecture\n\n"
      f"2024 Test Performance:\n"
      f"RMSE: {rmse_harm:.2f} ppb\n"
      f"MAE:  {mae_harm:.2f} ppb\n"
      f"R²:   {r2_harm:.3f}\n\n"
      f"N = {len(df_daily[test_mask])} test days"
  )
  ax.text(
      0.62,
      0.15,
      stats_scatter,
      transform=ax.transAxes,
      fontsize=11,
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
      OUTPUT_DIR / f"{city_name.lower()}_02_scatter_parity.jpg",
      dpi=300,
      bbox_inches="tight",
  )
  plt.close(fig)

  # Chart 3: Residuals Band (Matching Slide 14)
  fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
  ax.plot(
      df_daily["Date"],
      df_daily["Residuals"],
      color="#1f77b4",
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
      f"Harmonic Model Residuals - {city_name} Site\n(Predicted - Actual"
      " $\\text{NO}_2$ | 2023-2024)",
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

  stats_res = (
      f"Harmonic Model (2 Harmonics)\n\n"
      f"Residual mean: {np.mean(df_daily['Residuals']):+.4f} ppb\n"
      f"Residual std:  {res_std:.2f} ppb\n\n"
      f"2024 Performance:\n"
      f"RMSE: {rmse_harm:.2f} ppb\n"
      f"MAE:  {mae_harm:.2f} ppb\n"
      f"R²:   {r2_harm:.3f}"
  )
  ax.text(
      0.02,
      0.95,
      stats_res,
      transform=ax.transAxes,
      fontsize=11,
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
      OUTPUT_DIR / f"{city_name.lower()}_03_residuals_band.jpg",
      dpi=300,
      bbox_inches="tight",
  )
  plt.close(fig)

  print(f"✅ Completed Harmonic Regression suite for {city_name}!")

# Print Summary Comparison Table (Matching Slide 15)
df_summary = pd.DataFrame(summary_table)
print("\n" + "=" * 80)
print(
    "🏆 MODEL COMPARISON: PERSISTENCE, COSINE, AND HARMONIC (2024 TEST SET) 🏆"
)
print("=" * 80)
print(df_summary.to_string(index=False))
print("=" * 80)
print(f"📁 All charts successfully saved to: {OUTPUT_DIR}")
print("🎯 3-CITY HARMONIC ANALYSIS FINISHED!")