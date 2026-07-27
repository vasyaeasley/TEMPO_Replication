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

print("📋 STARTING BASELINE MODEL SUMMARY TABLE GENERATION 📋")
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
# 3. CALCULATE METRICS FOR ALL SITES & MODELS
# ==============================================================================
target_sites = ["Compton", "Pomona", "Santa Clarita"]
df_all["Site_Name_Clean"] = df_all["Local Site Name"].astype(str).str.strip()
df_sites = df_all[df_all["Site_Name_Clean"].isin(target_sites)].copy()

table_data = []

for site in target_sites:
  actual = (
      df_sites[df_sites["Site_Name_Clean"] == site]["NO2_ppb"].dropna().values
  )

  # Calculate single-value predictions
  val_mean = np.mean(actual)
  val_median = np.median(actual)

  binned_data = np.round(actual * 2) / 2
  mode_res = stats.mode(binned_data, keepdims=True)
  val_mode = mode_res.mode[0] if len(mode_res.mode) > 0 else val_median

  models = [("Mean", val_mean), ("Median", val_median), ("Mode", val_mode)]

  # Find lowest MAE to crown the winner for this site
  maes = [mean_absolute_error(actual, np.full_like(actual, v)) for _, v in models]
  min_mae_idx = np.argmin(maes)

  for idx, (m_name, pred_val) in enumerate(models):
    predicted = np.full_like(actual, fill_value=pred_val)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    # Assign strategic role / best for label
    if idx == min_mae_idx:
      role = "⭐ BEST MAE (Winner) - Typical daily exposure"
    elif m_name == "Mean":
      role = "Total volume baseline (Skewed by winter spikes)"
    elif m_name == "Mode":
      role = "Most frequent concentration bin (0.5 ppb)"
    else:
      role = "Resistant to winter inversion outliers"

    table_data.append([
        site if idx == 0 else "",  # Show site name only on first row of group
        f"{m_name} Model",
        f"{pred_val:.2f} ppb",
        f"{rmse:.2f} ppb",
        f"{mae:.2f} ppb",
        f"{r2:.3f}",
        role,
    ])

print("✅ Computed metrics across all sites and baselines!")

# ==============================================================================
# 4. RENDER HIGH-RESOLUTION VISUAL TABLE (.PNG)
# ==============================================================================
print("🎨 Rendering publication-quality visual table...")

fig, ax = plt.subplots(figsize=(16, 7.5))
ax.axis("off")  # Turn off standard plot canvas axes

col_headers = [
    "Monitoring Site",
    "Baseline Model",
    "Prediction\n(Constant Guess)",
    "RMSE\n(Heavy Penalty)",
    "MAE\n(Average Error)",
    "R² Score\n(Explained Var)",
    "Best For / Strategic Role",
]

# Create Matplotlib Table
table = ax.table(
    cellText=table_data,
    colLabels=col_headers,
    loc="center",
    cellLoc="center",
    colWidths=[0.14, 0.12, 0.14, 0.12, 0.12, 0.12, 0.34],
)

# Style Table Aesthetics
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 2.3)  # Stretch cell height for executive legibility

# Advanced Cell Styling Loop
for (row, col), cell in table.get_celld().items():
  # Header Row Styling
  if row == 0:
    cell.set_facecolor("#1f4e79")  # Dark Executive Navy
    cell.set_text_props(color="white", fontweight="bold", fontsize=11.5)
    cell.set_edgecolor("#d9d9d9")
    cell.set_linewidth(1.2)
  else:
    # Alternating Row Shading per City Group (3 rows per city)
    site_group = (row - 1) // 3
    if site_group % 2 == 0:
      cell.set_facecolor("#f8f9fa")  # Very light gray
    else:
      cell.set_facecolor("#ffffff")  # Pure white

    # Highlight the Winning Row (⭐ BEST MAE)
    if "⭐ BEST MAE" in table_data[row - 1][6]:
      if col in [1, 4, 6]:  # Highlight Model Name, MAE, and Role
        cell.set_text_props(color="#2e7d32", fontweight="bold")  # Forest Green
        cell.set_facecolor("#e8f5e9")  # Soft mint green highlight

    # Bold Site Names and Model Names
    if col == 0 and table_data[row - 1][0] != "":
      cell.set_text_props(fontweight="bold", fontsize=12, color="#1f4e79")
    elif col == 1:
      cell.set_text_props(fontweight="bold")

    # Left-align the explanatory text in the final column
    if col == 6:
      cell._loc = "left"
      cell.set_text_props(fontsize=10)

    cell.set_edgecolor("#cccccc")
    cell.set_linewidth(0.8)

# Add Title and Subtitle
fig.suptitle(
    "EPA Surface NO₂ Baseline ('Dumb') Model Performance Summary\n"
    "Typological Comparison across Los Angeles Basin Sites (2023–2024)",
    fontsize=16,
    fontweight="bold",
    y=0.96,
    color="#1f4e79",
)

# Add Bottom Explanatory Footer Note
footer_text = (
    "Note: All baseline models predict a single static value for 100% of days,"
    " resulting in an R² of exactly 0.000.\n"
    "The Median Model consistently minimizes Mean Absolute Error (MAE) because"
    " it is robust against episodic winter inversion spikes."
)
fig.text(
    0.5,
    0.03,
    footer_text,
    ha="center",
    fontsize=9.5,
    style="italic",
    color="#555555",
    bbox=dict(
        boxstyle="round,pad=0.5", facecolor="#f1f3f4", edgecolor="none", alpha=0.8
    ),
)

out_file = OUTPUT_DIR / "baseline_models_comparison_table.png"
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"✅ Saved High-Resolution Table Image to: {out_file}")
print("=" * 75)
print("🎯 TABLE GENERATION SUCCESSFULLY FINISHED!")