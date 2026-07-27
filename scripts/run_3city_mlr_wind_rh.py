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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================================================================
# 1. SETUP & PATH CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "models" / "mlr_wind_rh_3city"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

data_file = PROCESSED_DIR / "epa_point_dataset_14months_20features.npz"
airnow_dir = BASE_DIR / "data" / "raw" / "epa" / "AirNow"
if not airnow_dir.exists():
  airnow_dir = BASE_DIR / "data" / "raw" / "AirNow"

print(
    "🌦️ STARTING 3-CITY MULTIPLE LINEAR REGRESSION (WIND SPEED + RH) 🌦️"
)
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
idx_t2m = np.where(feature_names == "t2m")[0][0]
idx_d2m = np.where(feature_names == "d2m")[0][0]
idx_msin = np.where(feature_names == "month_sin")[0][0]
idx_mcos = np.where(feature_names == "month_cos")[0][0]


def create_harmonic_features(doy_series):
  doy = doy_series.values
  f1 = np.sin(2 * np.pi * doy / 365.25)
  f2 = np.cos(2 * np.pi * doy / 365.25)
  f3 = np.sin(4 * np.pi * doy / 365.25)
  f4 = np.cos(4 * np.pi * doy / 365.25)
  return np.column_stack([f1, f2, f3, f4])


# ==============================================================================
# 3. MASTER PROCESSING & MODELING LOOP
# ==============================================================================
summary_table = []

for city_name, info in target_cities.items():
  print(f"\n🌍 BUILDING MLR (WIND + RH) FOR: {city_name.upper()}...")

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

  # Convert Units: Wind -> Knots
  wind_knots = (
      np.sqrt(X_stn[:, idx_u10] ** 2 + X_stn[:, idx_v10] ** 2) * 1.94384
  )

  # Derive Relative Humidity (%) using August-Roche-Magnus approximation
  # t2m and d2m are in Kelvin in the master dataset
  t_c = X_stn[:, idx_t2m] - 273.15
  d_c = X_stn[:, idx_d2m] - 273.15
  rh_pct = np.clip(
      100.0 * (np.exp((17.625 * d_c) / (243.04 + d_c)) / np.exp((17.625 * t_c) / (243.04 + t_c))),
      5.0,  # Floor at 5% to handle physical boundary conditions
      100.0 # Cap at 100% saturation
  )

  hours_per_day = 11
  n_days = len(y_stn) // hours_per_day

  d_no2, d_month, d_wind, d_rh = [], [], [], []
  for d in range(n_days):
    s = d * hours_per_day
    e = s + hours_per_day
    d_no2.append(np.mean(y_stn[s:e]))
    d_month.append(int(np.round(np.mean(months_hourly[s:e]))))
    d_wind.append(np.mean(wind_knots[s:e]))
    d_rh.append(np.mean(rh_pct[s:e]))

  df_base = pd.DataFrame({
      "month": d_month,
      "NO2": d_no2,
      "Wind": d_wind,
      "RH": d_rh,
  })
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
  df_daily["DOY"] = df_daily["Date"].dt.dayofyear

  # Train on 2023, Test on 2024 to match mentor standard evaluation
  train_mask = df_daily["Year"] == 2023
  test_mask = df_daily["Year"] == 2024

  y_te = df_daily.loc[test_mask, "NO2"].values
  X_tr_mlr = df_daily.loc[train_mask, ["Wind", "RH"]].values
  y_tr_mlr = df_daily.loc[train_mask, "NO2"].values
  X_te_mlr = df_daily.loc[test_mask, ["Wind", "RH"]].values

  # 1. Fit Multiple Linear Regression (NO2 ~ Wind + RH)
  mlr_model = LinearRegression().fit(X_tr_mlr, y_tr_mlr)
  preds_mlr = mlr_model.predict(X_te_mlr)
  rmse_mlr = np.sqrt(mean_squared_error(y_te, preds_mlr))
  mae_mlr = mean_absolute_error(y_te, preds_mlr)
  r2_mlr = r2_score(y_te, preds_mlr)

  # Full 2-year predictions for the 4-panel diagnostic chart (mirroring mentor)
  X_all_mlr = df_daily[["Wind", "RH"]].values
  y_all_no2 = df_daily["NO2"].values
  wind_res_all = df_daily["Wind"].values
  rh_res_all = df_daily["RH"].values

  wind_model_all = LinearRegression().fit(X_all_mlr, y_all_no2)
  preds_all = wind_model_all.predict(X_all_mlr)
  res_all = preds_all - y_all_no2 # Matching mentor slide residual formula

  res_std = np.std(res_all)
  coef_wind = wind_model_all.coef_[0]
  coef_rh = wind_model_all.coef_[1]

  # 2. Calculate Partial Regression Residuals for Added-Variable Plots (AV-Plots)
  # Bottom-Left: Wind controlling for RH
  wind_res = (
      wind_res_all
      - LinearRegression()
      .fit(rh_res_all.reshape(-1, 1), wind_res_all)
      .predict(rh_res_all.reshape(-1, 1))
  )
  no2_wind_res = (
      y_all_no2
      - LinearRegression()
      .fit(rh_res_all.reshape(-1, 1), y_all_no2)
      .predict(rh_res_all.reshape(-1, 1))
  )

  # Bottom-Right: RH controlling for Wind
  rh_res = (
      rh_res_all
      - LinearRegression()
      .fit(wind_res_all.reshape(-1, 1), rh_res_all)
      .predict(wind_res_all.reshape(-1, 1))
  )
  no2_rh_res = (
      y_all_no2
      - LinearRegression()
      .fit(wind_res_all.reshape(-1, 1), y_all_no2)
      .predict(wind_res_all.reshape(-1, 1))
  )

  # 3. Fit Baselines for Comparison Table (Evaluated on 2024 Test Set)
  rmse_wind_only = np.sqrt(
      mean_squared_error(
          y_te,
          LinearRegression().fit(X_tr_mlr[:, :1], y_tr_mlr).predict(X_te_mlr[:, :1]),
      )
  )
  r2_wind_only = r2_score(
      y_te,
      LinearRegression().fit(X_tr_mlr[:, :1], y_tr_mlr).predict(X_te_mlr[:, :1]),
  )

  X_tr_h = create_harmonic_features(df_daily.loc[train_mask, "DOY"])
  X_te_h = create_harmonic_features(df_daily.loc[test_mask, "DOY"])
  harm_preds = LinearRegression().fit(X_tr_h, y_tr_mlr).predict(X_te_h)
  rmse_harm = np.sqrt(mean_squared_error(y_te, harm_preds))
  r2_harm = r2_score(y_te, harm_preds)

  persist_preds = np.roll(y_te, 1)
  persist_preds[0] = y_te[0]
  rmse_pers = np.sqrt(mean_squared_error(y_te, persist_preds))
  r2_pers = r2_score(y_te, persist_preds)

  summary_table.extend([
      {
          "City": city_name,
          "Model": "1. Persistence (t-1)",
          "Predictor": "Previous Day",
          "RMSE (ppb)": rmse_pers,
          "R² Score": r2_pers,
      },
      {
          "City": city_name,
          "Model": "2. Harmonic Regression",
          "Predictor": "DOY (2 Harmonics)",
          "RMSE (ppb)": rmse_harm,
          "R² Score": r2_harm,
      },
      {
          "City": city_name,
          "Model": "3. Single Linear",
          "Predictor": "Wind Speed Only",
          "RMSE (ppb)": rmse_wind_only,
          "R² Score": r2_wind_only,
      },
      {
          "City": city_name,
          "Model": "4. Multiple Linear (MLR)",
          "Predictor": "Wind Speed + RH",
          "RMSE (ppb)": rmse_mlr,
          "R² Score": r2_mlr,
      },
  ])

  # ==============================================================================
  # 4. RENDER 4-PANEL MLR DIAGNOSTIC GRID (MATCHING MENTOR SLIDE EXACTLY)
  # ==============================================================================
  # (Copy of the 4-panel plotting logic from run_3city_mlr_wind_pressure.py,
  # but swapping Pressure titles/labels for Relative Humidity titles/labels,
  # and residual units)
  # (Crucially using Unicode subscript characters for NO2 to prevent LaTeX errors)

  fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
  ax_tl, ax_tr = axes[0, 0], axes[0, 1]
  ax_bl, ax_br = axes[1, 0], axes[1, 1]

  # --- Panel 1 (Top Left): Predicted vs Observed NO2 ---
  # Mirroring colors/alpha from Screenshot 2026-07-23 at 3.30.39 PM.jpg
  ax_tl.scatter(
      y_all_no2, preds_all, color="#86efac", alpha=0.65, s=28, label="Daily Pairs"
  )
  max_val = max(y_all_no2.max(), preds_all.max()) * 1.05
  min_val = min(y_all_no2.min(), preds_all.min())
  ax_tl.plot(
      [min_val, max_val],
      [min_val, max_val],
      "k--",
      linewidth=2.5,
      label="Perfect Prediction",
  )

  ax_tl.set_title("Predicted vs Observed NO₂", fontsize=13, fontweight="bold")
  ax_tl.set_xlabel("Observed NO₂ (ppb)", fontsize=11, fontweight="bold")
  ax_tl.set_ylabel("Predicted NO₂ (ppb)", fontsize=11, fontweight="bold")
  ax_tl.grid(True, linestyle="--", alpha=0.4)
  ax_tl.legend(loc="upper left", frameon=True, facecolor="white")

  perf_txt = (
      f"R² = {r2_score(y_all_no2, preds_all):.3f}\n"
      f"RMSE = {np.sqrt(mean_squared_error(y_all_no2, preds_all)):.2f} ppb\n"
      f"MAE  = {mean_absolute_error(y_all_no2, preds_all):.2f} ppb"
  )
  ax_tl.text(
      0.05,
      0.82,
      perf_txt,
      transform=ax_tl.transAxes,
      fontsize=11,
      family="monospace",
      bbox=dict(
          boxstyle="round,pad=0.5",
          facecolor="white",
          edgecolor="black",
          alpha=0.9,
      ),
  )

  # --- Panel 2 (Top Right): Residual Plot ---
  ax_tr.scatter(preds_all, res_all, color="#f87171", alpha=0.60, s=25)
  ax_tr.axhline(0, color="black", linestyle="--", linewidth=2.0)

  ax_tr.set_title("Residual Plot", fontsize=13, fontweight="bold")
  ax_tr.set_xlabel("Predicted NO₂ (ppb)", fontsize=11, fontweight="bold")
  ax_tr.set_ylabel("Residuals (ppb)", fontsize=11, fontweight="bold")
  ax_tr.grid(True, linestyle="--", alpha=0.4)

  res_txt = (
      f"Mean = {np.mean(res_all):+.2f} ppb\nstd dev = {res_std:.2f} ppb"
  )
  ax_tr.text(
      0.05,
      0.88,
      res_txt,
      transform=ax_tr.transAxes,
      fontsize=11,
      family="monospace",
      bbox=dict(
          boxstyle="round,pad=0.5",
          facecolor="white",
          edgecolor="black",
          alpha=0.9,
      ),
  )

  # --- Panel 3 (Bottom Left): Partial Regression for Wind Speed ---
  ax_bl.scatter(wind_res, no2_wind_res, color="#fba524", alpha=0.55, s=25)
  # Standardize X range for the added-variable plot
  w_range = np.linspace(wind_res.min(), wind_res.max(), 100)
  ax_bl.plot(w_range, coef_wind * w_range, "k-", linewidth=2.5)

  ax_bl.set_title(
      "Partial Regression: Wind Speed\n(controlling for Relative Humidity)",
      fontsize=13,
      fontweight="bold",
  )
  ax_bl.set_xlabel(
      "Wind Speed (residualized) [knots]", fontsize=11, fontweight="bold"
  )
  ax_bl.set_ylabel(
      "NO₂ (residualized) [ppb]", fontsize=11, fontweight="bold"
  )
  ax_bl.grid(True, linestyle="--", alpha=0.4)

  ax_bl.text(
      0.05,
      0.90,
      f"Partial coef: {coef_wind:+.3f}",
      transform=ax_bl.transAxes,
      fontsize=11,
      fontweight="bold",
      family="monospace",
      bbox=dict(
          boxstyle="round,pad=0.4",
          facecolor="white",
          edgecolor="black",
          alpha=0.9,
      ),
  )

  # --- Panel 4 (Bottom Right): Partial Regression for Relative Humidity ---
  ax_br.scatter(rh_res, no2_rh_res, color="#c084fc", alpha=0.55, s=25)
  r_range = np.linspace(rh_res.min(), rh_res.max(), 100)
  ax_br.plot(r_range, coef_rh * r_range, "k-", linewidth=2.5)

  ax_br.set_title(
      "Partial Regression: Relative Humidity\n(controlling for Wind Speed)",
      fontsize=13,
      fontweight="bold",
  )
  ax_br.set_xlabel(
      "Relative Humidity (residualized) [%]", fontsize=11, fontweight="bold"
  )
  ax_br.set_ylabel(
      "NO₂ (residualized) [ppb]", fontsize=11, fontweight="bold"
  )
  ax_br.grid(True, linestyle="--", alpha=0.4)

  ax_br.text(
      0.05,
      0.90,
      f"Partial coef: {coef_rh:+.3f}",
      transform=ax_br.transAxes,
      fontsize=11,
      fontweight="bold",
      family="monospace",
      bbox=dict(
          boxstyle="round,pad=0.4",
          facecolor="white",
          edgecolor="black",
          alpha=0.9,
      ),
  )

  # Sample Size Box in bottom right (mirroring mentor)
  ax_br.text(
      0.80,
      0.05,
      f"n = {len(df_daily)} days\n{city_name}, CA",
      transform=ax_br.transAxes,
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

  fig.suptitle(
      "Multiple Linear Regression: NO₂ ~ Wind Speed + Relative Humidity\n"
      f"({city_name} Site 2023-2024)",
      fontsize=16,
      fontweight="bold",
  )
  plt.savefig(
      OUTPUT_DIR / f"{city_name.lower()}_mlr_4panel_rh_diagnostic.jpg",
      dpi=300,
      bbox_inches="tight",
  )
  plt.close(fig)

  print(f"✅ Saved MLR 4-panel RH diagnostic chart for {city_name}!")

# ==============================================================================
# 5. PRINT SUMMARY COMPARISON TABLE
# ==============================================================================
df_summary = pd.DataFrame(summary_table)
print("\n" + "=" * 80)
print("🏆 MULTIPLE LINEAR REGRESSION (WIND + RH) COMPARISON TABLE (2024 Test) 🏆")
print("=" * 80)
print(df_summary.to_string(index=False))
print("=" * 80)
print(f"📁 All diagnostic charts successfully saved to: {OUTPUT_DIR}")
print("🎯 MLR ANALYSIS FINISHED!")