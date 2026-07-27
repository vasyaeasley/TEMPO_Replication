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
import seaborn as sns
from sklearn.linear_model import LinearRegression

# ==============================================================================
# 1. SETUP & PATH CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "models" / "meteorology_suite_3city"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

data_file = PROCESSED_DIR / "epa_point_dataset_14months_20features.npz"
airnow_dir = BASE_DIR / "data" / "raw" / "epa" / "AirNow"
if not airnow_dir.exists():
  airnow_dir = BASE_DIR / "data" / "raw" / "AirNow"

print("🌦️ STARTING 3-CITY METEOROLOGICAL TIME-SERIES & CORRELATION SUITE 🌦️")
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

# Identify column indices for weather variables
idx_t2m = np.where(feature_names == "t2m")[0][0]
idx_sp = np.where(feature_names == "sp")[0][0]
idx_u10 = np.where(feature_names == "u10")[0][0]
idx_v10 = np.where(feature_names == "v10")[0][0]
idx_d2m = np.where(feature_names == "d2m")[0][0]
idx_msin = np.where(feature_names == "month_sin")[0][0]
idx_mcos = np.where(feature_names == "month_cos")[0][0]


def format_date_axis(ax):
  ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
  ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
  ax.set_xlabel("Date", fontsize=12, fontweight="bold")


# ==============================================================================
# 3. MASTER PROCESSING & PLOTTING LOOP FOR EACH CITY
# ==============================================================================
for city_name, info in target_cities.items():
  print(f"\n🌍 EVALUATING METEOROLOGY SUITE FOR: {city_name.upper()}...")

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

  # Decode Integer Months (1 to 12)
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

  # Convert raw meteorological variables into slide-standard units
  t_kelvin = X_stn[:, idx_t2m]
  temp_f = (t_kelvin - 273.15) * 9 / 5 + 32.0  # Kelvin -> Fahrenheit
  press_mb = X_stn[:, idx_sp] / 100.0  # Pascals -> Millibars / hPa
  wind_knots = (
      np.sqrt(X_stn[:, idx_u10] ** 2 + X_stn[:, idx_v10] ** 2) * 1.94384
  )  # m/s -> Knots

  # August-Roche-Magnus approximation for Relative Humidity (%)
  t_c = t_kelvin - 273.15
  d_c = X_stn[:, idx_d2m] - 273.15
  rh_pct = np.clip(
      100.0
      * (
          np.exp((17.625 * d_c) / (243.04 + d_c))
          / np.exp((17.625 * t_c) / (243.04 + t_c))
      ),
      5.0,
      100.0,
  )

  # Aggregate into Daily Means (~11 Daylight Hours = 1 Day)
  hours_per_day = 11
  n_days = len(y_stn) // hours_per_day

  d_no2, d_month, d_temp, d_press, d_wind, d_rh = [], [], [], [], [], []
  for d in range(n_days):
    s = d * hours_per_day
    e = s + hours_per_day
    d_no2.append(np.mean(y_stn[s:e]))
    d_month.append(int(np.round(np.mean(months_hourly[s:e]))))
    d_temp.append(np.mean(temp_f[s:e]))
    d_press.append(np.mean(press_mb[s:e]))
    d_wind.append(np.mean(wind_knots[s:e]))
    d_rh.append(np.mean(rh_pct[s:e]))

  df_base = pd.DataFrame({
      "month": d_month,
      "NO2": d_no2,
      "Temp": d_temp,
      "Press": d_press,
      "Wind": d_wind,
      "RH": d_rh,
  })
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
  df_2024 = df_base.copy()
  df_2024["Date"] = dates_2024
  df_daily = pd.concat([df_2023, df_2024], ignore_index=True)
  df_daily.sort_values(by="Date", inplace=True)

  # ==============================================================================
  # 4. RENDER CHARTS 1-4: DUAL-AXIS TIME SERIES PLOTS
  # ==============================================================================
  weather_configs = [
      (
          "Temp",
          "Temperature (°F)",
          "#dc2626",
          f"{city_name.lower()}_01_time_series_temp.jpg",
          "Temperature",
      ),
      (
          "Press",
          "Pressure (millibars)",
          "#16a34a",
          f"{city_name.lower()}_02_time_series_press.jpg",
          "Pressure",
      ),
      (
          "RH",
          "Relative Humidity (%)",
          "#9333ea",
          f"{city_name.lower()}_03_time_series_rh.jpg",
          "Relative Humidity",
      ),
      (
          "Wind",
          "Wind Speed (knots)",
          "#ea580c",
          f"{city_name.lower()}_04_time_series_wind.jpg",
          "Wind Speed",
      ),
  ]

  for col_key, y_label, color_hex, fname, title_word in weather_configs:
    fig, ax1 = plt.subplots(figsize=(14, 7), constrained_layout=True)

    # Left Y-Axis: Weather Variable
    ax1.plot(
        df_daily["Date"],
        df_daily[col_key],
        color=color_hex,
        linewidth=2.0,
        label=title_word,
    )
    ax1.set_ylabel(y_label, color=color_hex, fontsize=13, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_hex)
    format_date_axis(ax1)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Right Y-Axis: NO2 Concentration
    ax2 = ax1.twinx()
    ax2.plot(
        df_daily["Date"],
        df_daily["NO2"],
        color="#2563eb",
        linewidth=1.8,
        alpha=0.85,
        label="$\\text{NO}_2$",
    )
    ax2.set_ylabel(
        "$\\text{NO}_2$ Daily Mean (ppb)",
        color="#2563eb",
        fontsize=13,
        fontweight="bold",
    )
    ax2.tick_params(axis="y", labelcolor="#2563eb")

    ax1.set_title(
        f"{title_word} & $\\text{{NO}}_2$ Daily Mean - {city_name} Site"
        " 2023-2024\n(Evaluating Temporal Evolution Across 2-Year Timeline)",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )

    # Combined Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right",
        frameon=True,
        facecolor="white",
        fontsize=11,
    )

    # Statistics Text Box
    stats_txt = (
        f"{title_word}:\n"
        f"Mean: {df_daily[col_key].mean():.1f}\n"
        f"Range: {df_daily[col_key].min():.1f}-{df_daily[col_key].max():.1f}\n\n"
        f"NO2:\n"
        f"Mean: {df_daily['NO2'].mean():.2f} ppb\n"
        f"Range: {df_daily['NO2'].min():.1f}-{df_daily['NO2'].max():.1f} ppb\n"
        f"Records: {len(df_daily)}"
    )
    ax1.text(
        0.02,
        0.95,
        stats_txt,
        transform=ax1.transAxes,
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

    plt.savefig(OUTPUT_DIR / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)

  # ==============================================================================
  # 5. RENDER CHART 5: CORRELATION MATRIX HEATMAP
  # ==============================================================================
  df_corr = df_daily[["NO2", "Temp", "Press", "RH", "Wind"]].rename(columns={
      "NO2": "NO₂\n(ppb)",
      "Temp": "Temperature\n(°F)",
      "Press": "Pressure\n(mb)",
      "RH": "Relative Humidity\n(%)",
      "Wind": "Wind Speed\n(knots)",
  })
  corr_matrix = df_corr.corr()

  fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
  sns.heatmap(
      corr_matrix,
      annot=True,
      fmt=".3f",
      cmap="RdBu_r",
      vmin=-1.0,
      vmax=1.0,
      center=0.0,
      square=True,
      linewidths=1.5,
      cbar_kws={"label": "Correlation Coefficient"},
      annot_kws={"size": 12, "weight": "bold"},
      ax=ax,
  )

  ax.set_title(
      f"Correlation Matrix: $\\text{{NO}}_2$ and Meteorological"
      f" Variables\n{city_name} Site 2023-2024 (n = {len(df_daily)} days)",
      fontsize=15,
      fontweight="bold",
      pad=15,
  )
  plt.savefig(
      OUTPUT_DIR / f"{city_name.lower()}_05_correlation_matrix.jpg",
      dpi=300,
      bbox_inches="tight",
  )
  plt.close(fig)

  # ==============================================================================
  # 6. RENDER CHART 6: 4-PANEL SCATTER PLOT GRID WITH OLS TREND LINES
  # ==============================================================================
  fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
  axes_flat = axes.flatten()

  scatter_configs = [
      ("Temp", "Temperature (°F)", "#dc2626", axes_flat[0]),
      ("Press", "Pressure (millibars)", "#16a34a", axes_flat[1]),
      ("RH", "Relative Humidity (%)", "#9333ea", axes_flat[2]),
      ("Wind", "Wind Speed (knots)", "#ea580c", axes_flat[3]),
  ]

  for col_key, x_label, color_hex, ax_sub in scatter_configs:
    x_vals = df_daily[col_key].values.reshape(-1, 1)
    y_vals = df_daily["NO2"].values

    # Fit OLS Linear Trend Line
    reg = LinearRegression().fit(x_vals, y_vals)
    x_range = np.linspace(x_vals.min(), x_vals.max(), 100).reshape(-1, 1)
    y_pred_range = reg.predict(x_range)
    r_val = np.corrcoef(df_daily[col_key], df_daily["NO2"])[0, 1]

    ax_sub.scatter(
        x_vals,
        y_vals,
        color=color_hex,
        alpha=0.45,
        s=25,
        edgecolor="none",
        label="Daily Observations",
    )
    ax_sub.plot(
        x_range,
        y_pred_range,
        "k--",
        linewidth=2.0,
        label=f"y = {reg.coef_[0]:.3f}x + {reg.intercept_:.2f}",
    )

    ax_sub.set_title(
        f"$\\text{{NO}}_2$ vs {x_label.split(' ')[0]}",
        fontsize=13,
        fontweight="bold",
    )
    ax_sub.set_xlabel(x_label, fontsize=11, fontweight="bold")
    ax_sub.set_ylabel("$\\text{NO}_2$ (ppb)", fontsize=11, fontweight="bold")
    ax_sub.grid(True, linestyle="--", alpha=0.4)

    # Boxed Correlation Callout (Top Left)
    ax_sub.text(
        0.05,
        0.92,
        f"r = {r_val:+.3f}",
        transform=ax_sub.transAxes,
        fontsize=11,
        fontweight="bold",
        color=color_hex,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor=color_hex,
            linewidth=1.5,
        ),
    )

    # Boxed Equation Callout (Top Right)
    ax_sub.legend(loc="upper right", frameon=True, facecolor="white", fontsize=10)

  fig.suptitle(
      f"$\\text{{NO}}_2$ vs Meteorological Variables - {city_name} Site"
      " 2023-2024\n(Evaluating Linear Sensitivity & Physical Coupling)",
      fontsize=16,
      fontweight="bold",
  )
  plt.savefig(
      OUTPUT_DIR / f"{city_name.lower()}_06_scatter_plots_grid.jpg",
      dpi=300,
      bbox_inches="tight",
  )
  plt.close(fig)

  print(f"✅ Completed 6-slide meteorological suite for {city_name}!")

print("=" * 75)
print(
    "🎯 ALL 18 METEOROLOGICAL SLIDES SUCCESSFULLY GENERATED IN:"
    f" {OUTPUT_DIR}"
)