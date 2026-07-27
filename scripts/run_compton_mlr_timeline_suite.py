import os

# Thread safety controls to prevent server CPU lockups
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================================================================
# 1. SETUP & PATH CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DAILY_DIR = BASE_DIR / "data" / "raw" / "epa_from_internet_daily"
if not DAILY_DIR.exists():
  DAILY_DIR = BASE_DIR / "epa_from_internet_daily"

OUTPUT_DIR = BASE_DIR / "models" / "compton_mlr_timeline_suite"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("🌴 GENERATING COMPTON 24-HOUR [WIND + RH] 3-CHART TIMELINE SUITE 🌴")
print("=" * 75)

if not DAILY_DIR.exists():
  raise FileNotFoundError(
      f"Could not locate daily CSV directory at: {DAILY_DIR}"
  )


# ==============================================================================
# 2. ROBUST DATA LOADING FOR COMPTON
# ==============================================================================
def load_and_filter_csv(file_patterns, city_keyword, param_keyword=None):
  dfs = []
  for pat in file_patterns:
    for f in sorted(DAILY_DIR.glob(pat)):
      print(f"   --> Scanning {f.name} for {city_keyword}...")
      df = pd.read_csv(f, low_memory=False)

      mask = (
          df.get("City Name", pd.Series(dtype=str)).str.contains(
              city_keyword, case=False, na=False
          )
          | df.get("Local Site Name", pd.Series(dtype=str)).str.contains(
              city_keyword, case=False, na=False
          )
          | df.get("Address", pd.Series(dtype=str)).str.contains(
              city_keyword, case=False, na=False
          )
      )
      df_city = df[mask].copy()

      if param_keyword and "Parameter Name" in df_city.columns:
        df_city = df_city[
            df_city["Parameter Name"].str.contains(
                param_keyword, case=False, na=False
            )
        ]

      if not df_city.empty:
        df_city["Date"] = pd.to_datetime(df_city["Date Local"])
        daily = (
            df_city.groupby("Date")["Arithmetic Mean"].mean().reset_index()
        )
        dfs.append(daily)

  if not dfs:
    return pd.DataFrame()
  res = (
      pd.concat(dfs, ignore_index=True)
      .sort_values("Date")
      .drop_duplicates("Date")
  )
  return res


search_kw = "Compton"
print("\n📡 Extracting Compton 24-Hour NO₂ Data...")
df_no2 = load_and_filter_csv(
    ["daily_42602_2023.csv", "daily_42602_2024.csv"], search_kw
)
if df_no2.empty:
  raise ValueError("Failed to load NO2 data for Compton!")
df_no2.rename(columns={"Arithmetic Mean": "NO2"}, inplace=True)

print("📡 Extracting Compton 24-Hour Wind Speed Data...")
df_wind = load_and_filter_csv(
    ["daily_WIND_2023.csv", "daily_WIND_2024.csv"], search_kw, "Speed"
)
if df_wind.empty:
  raise ValueError("Failed to load Wind Speed data for Compton!")
df_wind.rename(columns={"Arithmetic Mean": "Wind"}, inplace=True)

print("📡 Extracting Compton 24-Hour Relative Humidity Data...")
df_rh = load_and_filter_csv(
    ["daily_RH_DP_2023.csv", "daily_RH_DP_2024.csv"],
    search_kw,
    "Relative Humidity|Humid",
)
if df_rh.empty:
  raise ValueError("Failed to load Relative Humidity data for Compton!")
df_rh.rename(columns={"Arithmetic Mean": "RH"}, inplace=True)

# Merge onto unified calendar date index
df_daily = df_no2.merge(df_wind, on="Date", how="inner").merge(
    df_rh, on="Date", how="inner"
)
df_daily.sort_values("Date", inplace=True)
df_daily["Year"] = df_daily["Date"].dt.year

if df_daily.empty:
  raise ValueError("Merged Compton dataframe is empty! Check date alignment.")

print(
    f"\n✅ Successfully constructed {len(df_daily)}-day continuous 24-hour"
    " Compton timeline!"
)

# ==============================================================================
# 3. TRAIN MLR MODEL & CALCULATE METRICS
# ==============================================================================
X_all = df_daily[["Wind", "RH"]].values
y_all = df_daily["NO2"].values
mlr_model = LinearRegression().fit(X_all, y_all)

df_daily["MLR_Pred"] = mlr_model.predict(X_all)
df_daily["Residuals"] = (
    df_daily["MLR_Pred"] - df_daily["NO2"]
)  # Matching mentor slide residual formula

rmse_all = np.sqrt(mean_squared_error(y_all, df_daily["MLR_Pred"]))
mae_all = mean_absolute_error(y_all, df_daily["MLR_Pred"])
r2_all = r2_score(y_all, df_daily["MLR_Pred"])
res_std = np.std(df_daily["Residuals"])

m_23 = df_daily["Year"] == 2023
rmse_23 = np.sqrt(
    mean_squared_error(
        df_daily.loc[m_23, "NO2"], df_daily.loc[m_23, "MLR_Pred"]
    )
)
r2_23 = r2_score(df_daily.loc[m_23, "NO2"], df_daily.loc[m_23, "MLR_Pred"])

m_24 = df_daily["Year"] == 2024
rmse_24 = np.sqrt(
    mean_squared_error(
        df_daily.loc[m_24, "NO2"], df_daily.loc[m_24, "MLR_Pred"]
    )
)
r2_24 = r2_score(df_daily.loc[m_24, "NO2"], df_daily.loc[m_24, "MLR_Pred"])

coef_wind = mlr_model.coef_[0]
coef_rh = mlr_model.coef_[1]
intercept = mlr_model.intercept_


def format_date_axis(ax):
  ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
  ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
  ax.set_xlabel("Date", fontsize=12, fontweight="bold")


# ==============================================================================
# 4. CHART 1: 2-YEAR MLR TIMELINE (MATCHING SLIDE 34)
# ==============================================================================
print("\n🎨 Rendering Chart 1: 2-Year MLR Timeline...")
fig, ax = plt.subplots(figsize=(15, 7.5), constrained_layout=True)

ax.plot(
    df_daily["Date"],
    df_daily["NO2"],
    color="#3b82f6",
    linewidth=1.8,
    alpha=0.85,
    label="Observed NO₂",
    zorder=3,
)
ax.scatter(
    df_daily["Date"],
    df_daily["MLR_Pred"],
    color="#86efac",
    s=28,
    alpha=0.90,
    label="Multiple Regression Model",
    zorder=4,
)

ax.set_title(
    "Multiple Regression Model vs Observed NO₂ - Compton Site"
    " 2023-2024\n(NO₂ = f(Wind Speed, Relative Humidity))",
    fontsize=15,
    fontweight="bold",
    pad=15,
)
ax.set_ylabel("NO₂ Daily Mean (ppb)", fontsize=12, fontweight="bold")
format_date_axis(ax)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="upper right", frameon=True, facecolor="white", fontsize=11)

stats_tl = (
    f"Multiple Regression Model\n"
    f"NO₂ = {intercept:.2f} + ({coef_wind:.2f}×Wind) + ({coef_rh:.2f}×RH)\n\n"
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

note_br = (
    "Multi-Variable Model:\n"
    "Wind speed + Humidity\n"
    "Wind disperses, RH removes\n"
    "EPA AQS Data 2023-2024 (Compton)"
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

chart1_path = OUTPUT_DIR / "compton_01_mlr_timeline.jpg"
plt.savefig(chart1_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"   --> Saved: {chart1_path}")

# ==============================================================================
# 5. CHART 2: PARITY SCATTER PLOT (MATCHING SLIDE 35)
# ==============================================================================
print("🎨 Rendering Chart 2: Parity Scatter Plot...")
fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)

ax.scatter(
    df_daily["MLR_Pred"],
    df_daily["NO2"],
    color="#86efac",
    alpha=0.65,
    s=45,
    edgecolor="none",
    label="Compton Site 2023-2024",
)
max_lim = max(df_daily["NO2"].max(), df_daily["MLR_Pred"].max()) * 1.08
min_lim = min(df_daily["NO2"].min(), df_daily["MLR_Pred"].min())
ax.plot(
    [min_lim, max_lim],
    [min_lim, max_lim],
    "k--",
    linewidth=2.0,
    alpha=0.8,
    label="1:1 Line (Perfect Prediction)",
)

ax.set_title(
    "Actual vs Predicted NO₂ - Multiple Regression Model\nCompton Site"
    " 2023-2024",
    fontsize=15,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel(
    "Predicted NO₂ (Multiple Regression Model) [ppb]",
    fontsize=12,
    fontweight="bold",
)
ax.set_ylabel("Actual NO₂ [ppb]", fontsize=12, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="upper left", frameon=True, facecolor="white", fontsize=11)

stats_br = (
    f"Multiple Regression Model\n"
    f"NO₂ = {intercept:.2f} + ({coef_wind:.2f}×Wind)\n"
    f"       + ({coef_rh:.2f}×RH)\n\n"
    f"Performance:\n"
    f"  RMSE: {rmse_all:.2f} ppb\n"
    f"  MAE:  {mae_all:.2f} ppb\n"
    f"  R²:   {r2_all:.3f}\n\n"
    f"N = {len(df_daily)} days\n"
    "Period: 2023-2024"
)
ax.text(
    0.60,
    0.18,
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

chart2_path = OUTPUT_DIR / "compton_02_mlr_parity_scatter.jpg"
plt.savefig(chart2_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"   --> Saved: {chart2_path}")

# ==============================================================================
# 6. CHART 3: RESIDUALS WITH ±1 STD DEV BAND (MATCHING SLIDE 36)
# ==============================================================================
print("🎨 Rendering Chart 3: Residuals Band...")
fig, ax = plt.subplots(figsize=(15, 7.5), constrained_layout=True)

ax.plot(
    df_daily["Date"],
    df_daily["Residuals"],
    color="#86efac",
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
    "Multiple Regression Model Residuals - Compton Site 2023-2024\n(Predicted"
    " - Actual NO₂)",
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
    f"Multiple Regression Model\n"
    f"NO₂ = {intercept:.2f} + ({coef_wind:.2f}×Wind)\n"
    f"       + ({coef_rh:.2f}×RH)\n\n"
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

chart3_path = OUTPUT_DIR / "compton_03_mlr_residuals_band.jpg"
plt.savefig(chart3_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"   --> Saved: {chart3_path}")

print("\n" + "=" * 75)
print(
    "🎯 ALL 3 COMPTON CHARTS SUCCESSFULLY GENERATED & SAVED TO:"
    f" {OUTPUT_DIR}"
)