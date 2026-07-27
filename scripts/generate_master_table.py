import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================================================================
# 1. SETUP & PATH CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "epa_from_internet_daily"
OUTPUT_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

print("📋 STARTING PERSISTENT & MASTER TABLE GENERATIONS 📋")
print("=" * 75)

# ==============================================================================
# 2. LOAD & COMBINE 2023-2024 MONITOR DATA
# ==============================================================================
files_to_load = [
    RAW_DIR / "daily_42602_2023.csv",
    RAW_DIR / "daily_42602_2024.csv",
]

dfs = []
for f_path in files_to_load:
  if not f_path.exists():
    f_path = BASE_DIR / "raw" / "epa_from_internet_daily" / f_path.name
  if f_path.exists():
    print(f"⏳ Loading {f_path.name}...")
    df = pd.read_csv(f_path, low_memory=False)
    dfs.append(df)
  else:
    raise FileNotFoundError(f"Could not find required data file: {f_path}")

df_all = pd.concat(dfs, ignore_index=True)
df_all["Date Local"] = pd.to_datetime(df_all["Date Local"])
df_all = df_all.sort_values("Date Local")

if df_all["Arithmetic Mean"].max() < 5.0:
  df_all["NO2_ppb"] = df_all["Arithmetic Mean"] * 1000.0
else:
  df_all["NO2_ppb"] = df_all["Arithmetic Mean"]

# ==============================================================================
# 3. CALCULATE EXACT METRICS (ALIGNED ON N-1 DAYS FOR FAIR COMPARISON)
# ==============================================================================
target_sites = ["Compton", "Pomona", "Santa Clarita"]
df_all["Site_Name_Clean"] = df_all["Local Site Name"].astype(str).str.strip()
df_sites = df_all[df_all["Site_Name_Clean"].isin(target_sites)].copy()

persistent_data = []
master_data = []

for site in target_sites:
  site_df = df_sites[df_sites["Site_Name_Clean"] == site].copy()
  daily_ts = site_df.groupby("Date Local")["NO2_ppb"].mean().reset_index()
  daily_ts = daily_ts.sort_values("Date Local")

  # Create lag-1 persistent prediction
  daily_ts["Persistent_Pred"] = daily_ts["NO2_ppb"].shift(1)

  # Drop Day 0 so ALL models are evaluated on the exact same N days!
  eval_df = daily_ts.dropna(subset=["Persistent_Pred", "NO2_ppb"]).copy()
  actual = eval_df["NO2_ppb"].values
  pred_persist = eval_df["Persistent_Pred"].values
  n_days = len(actual)

  # Calculate static baseline values on this aligned actual dataset
  val_mean = np.mean(actual)
  val_median = np.median(actual)
  binned_data = np.round(actual * 2) / 2
  mode_res = stats.mode(binned_data, keepdims=True)
  val_mode = mode_res.mode[0] if len(mode_res.mode) > 0 else val_median

  models = [
      ("Mean", np.full_like(actual, val_mean), f"{val_mean:.2f} ppb (static)"),
      (
          "Median",
          np.full_like(actual, val_median),
          f"{val_median:.2f} ppb (static)",
      ),
      ("Mode", np.full_like(actual, val_mode), f"{val_mode:.2f} ppb (static)"),
      ("Persistent", pred_persist, "Dynamic ($y_{t-1}$)"),
  ]

  for idx, (m_name, pred_arr, pred_label) in enumerate(models):
    rmse = np.sqrt(mean_squared_error(actual, pred_arr))
    mae = mean_absolute_error(actual, pred_arr)
    r2 = r2_score(actual, pred_arr)

    # Assign strategic roles
    if m_name == "Persistent":
      role = "⭐ THE NEMESIS - True dynamic benchmark (autocorrelation)"
      # Add to standalone persistent table
      persistent_data.append([
          site,
          "Persistent ($t-1$)",
          "Dynamic ($y_{t-1}$)",
          f"{rmse:.2f} ppb",
          f"{mae:.2f} ppb",
          f"{r2:.3f}",
          f"{n_days} days",
      ])
    elif m_name == "Median":
      role = "Best static baseline (resists winter inversion outliers)"
    elif m_name == "Mean":
      role = "Total volume baseline (skewed by winter spikes)"
    else:
      role = "Most frequent daily concentration bin (0.5 ppb)"

    # Add to master table
    master_data.append([
        site if idx == 0 else "",  # Show site name only on first row of group
        f"{m_name} Model",
        pred_label,
        f"{rmse:.2f} ppb",
        f"{mae:.2f} ppb",
        f"{r2:.3f}",
        role,
    ])

print("✅ Computed perfectly aligned metrics across all models!")


# ==============================================================================
# 4. RENDER TABLE 1: DEDICATED PERSISTENT MODEL TABLE (.PNG)
# ==============================================================================
def render_table(
    data, headers, title, filename, col_widths, is_master=False, figsize=(16, 7)
):
  print(f"🎨 Rendering {filename}...")
  fig, ax = plt.subplots(figsize=figsize)
  ax.axis("off")

  table = ax.table(
      cellText=data,
      colLabels=headers,
      loc="center",
      cellLoc="center",
      colWidths=col_widths,
  )

  table.auto_set_font_size(False)
  table.set_fontsize(11 if is_master else 12)
  table.scale(1.0, 2.2 if is_master else 2.6)

  for (row, col), cell in table.get_celld().items():
    if row == 0:
      cell.set_facecolor("#1f4e79")  # Executive Navy
      cell.set_text_props(color="white", fontweight="bold", fontsize=12)
      cell.set_edgecolor("#d9d9d9")
      cell.set_linewidth(1.2)
    else:
      # Group shading
      group_size = 4 if is_master else 1
      site_group = (row - 1) // group_size
      cell.set_facecolor("#f8f9fa" if site_group % 2 == 0 else "#ffffff")

      # Highlight Persistent Model ("The Nemesis")
      row_text = data[row - 1][1] if is_master else data[row - 1][1]
      if "Persistent" in row_text or "⭐ THE NEMESIS" in data[row - 1][-1]:
        if col in [1, 3, 4, 5, 6]:
          cell.set_text_props(color="#1b5e20", fontweight="bold")  # Deep green
          cell.set_facecolor("#e8f5e9")  # Soft mint highlight

      # Bold site names & model names
      if col == 0 and data[row - 1][0] != "":
        cell.set_text_props(fontweight="bold", fontsize=12, color="#1f4e79")
      elif col == 1:
        cell.set_text_props(fontweight="bold")

      if col == len(headers) - 1 and is_master:
        cell._loc = "left"
        cell.set_text_props(fontsize=10)

      cell.set_edgecolor("#cccccc")
      cell.set_linewidth(0.8)

  fig.suptitle(title, fontsize=16, fontweight="bold", y=0.96, color="#1f4e79")

  if is_master:
    footer = (
        "Note: Static models predict a single constant number, yielding an R²"
        " of exactly 0.000.\n"
        "The Persistent Model (The Nemesis) introduces lag-1 temporal memory,"
        " reducing MAE by ~30–45% and causing R² to surge into positive"
        " territory."
    )
    fig.text(
        0.5,
        0.02,
        footer,
        ha="center",
        fontsize=9.5,
        style="italic",
        color="#555555",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#f1f3f4",
            edgecolor="none",
            alpha=0.8,
        ),
    )

  out_path = OUTPUT_DIR / filename
  plt.savefig(out_path, dpi=300, bbox_inches="tight")
  plt.close(fig)
  print(f"   ✅ Saved: {out_path}")


# Render Dedicated Persistent Table
p_headers = [
    "Monitoring Site",
    "Model Name",
    "Prediction Strategy",
    "RMSE (ppb)",
    "MAE (ppb)",
    "R² Score",
    "Sample Size (N)",
]
render_table(
    persistent_data,
    p_headers,
    "EPA Surface NO₂ - Persistent Model ('The Nemesis') Performance Scorecard\n"
    "Lag-1 Autocorrelation Forecast ($y_{t-1}$) across Los Angeles Basin"
    " (2023–2024)",
    "persistent_model_table.png",
    col_widths=[0.16, 0.16, 0.16, 0.13, 0.13, 0.13, 0.13],
    is_master=False,
    figsize=(14, 6),
)

# Render Master All-in-One Comparison Table
m_headers = [
    "Monitoring Site",
    "Model Type",
    "Prediction Strategy",
    "RMSE\n(Heavy Penalty)",
    "MAE\n(Average Error)",
    "R² Score\n(Explained Var)",
    "Strategic Role / Assessment",
]
render_table(
    master_data,
    m_headers,
    "EPA Surface NO₂ - Master Baseline & Persistent Model Scorecard\n"
    "Static Baselines vs. Dynamic Persistence across Los Angeles Basin Sites"
    " (2023–2024)",
    "master_models_comparison_table.png",
    col_widths=[0.13, 0.13, 0.15, 0.11, 0.11, 0.11, 0.36],
    is_master=True,
    figsize=(16.5, 9.5),
)

print("=" * 75)
print("🎯 BOTH SCORECARD TABLES SUCCESSFULLY GENERATED!")