import os

# Thread safety controls
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import matplotlib.pyplot as plt
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

print("🌍 STARTING NO₂ CLIMATOLOGY & BASLINE MODEL COMPARISON 🌍")
print("=" * 75)

if not data_file.exists():
  raise FileNotFoundError(f"Could not locate master dataset at: {data_file}")

# ==============================================================================
# 2. LOAD DATA & EXTRACT STATION TIMELINE
# ==============================================================================
print("⏳ Loading master dataset...")
data = np.load(data_file, allow_pickle=True)

# Reconstruct full historical arrays
X_full = np.vstack([data["X_train"], data["X_test"]])
y_full = np.concatenate([data["y_train"], data["y_test"]])
g_full = np.concatenate([data["groups_train"], data["groups_test"]])

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

# Isolate Station 41 (Ontario Route 60 Near-Road Freight Corridor)
target_gid = 41
stn_mask = g_full == target_gid
X_stn = X_full[stn_mask]
y_stn = y_full[stn_mask]

# Decode Integer Month (1 to 12) from Trigonometric Encodings
m_sin_idx = np.where(feature_names == "month_sin")[0][0]
m_cos_idx = np.where(feature_names == "month_cos")[0][0]
months = (
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
months[months == 13] = 1  # Handle boundary wrap

df = pd.DataFrame({"month": months, "True_NO2": y_stn})

print(
    f"🎯 Isolated {len(df):,} historical daylight observations for Station"
    f" {target_gid}."
)

# ==============================================================================
# 3. COMPUTE CLIMATOLOGY & FIT HARMONIC COSINE CURVE
# ==============================================================================
print("⏳ Computing monthly mean climatological values...")
climatology = df.groupby("month")["True_NO2"].mean().reset_index()
climatology.rename(
    columns={"True_NO2": "Climatological_Mean"}, inplace=True
)

# Map climatological means back to every individual hour in the time series
df = df.merge(climatology, on="month", how="left")


# Define the simplest Time Series Harmonic Cosine Model
def cosine_model(m, amplitude, phase_shift, baseline_mean):
  return amplitude * np.cos(2 * np.pi * (m - phase_shift) / 12) + baseline_mean


print("🌊 Fitting harmonic cosine curve to monthly climatological values...")
# Initial guess: Amplitude=8 ppb, Phase=1 (January peak), Baseline=mean(y)
initial_guess = [8.0, 1.0, np.mean(y_stn)]
params, _ = curve_fit(
    cosine_model,
    climatology["month"],
    climatology["Climatological_Mean"],
    p0=initial_guess,
)

amp_fit, phase_fit, base_fit = params
print("   * Cosine Fit Parameters:")
print(f"     - Amplitude (A):     {amp_fit:.2f} ppb")
print(f"     - Phase Shift (φ):   Month {phase_fit:.2f} (Peak Winter Trapping)")
print(f"     - Annual Baseline:   {base_fit:.2f} ppb")

# Generate Cosine predictions for every hour
df["Cosine_Model"] = cosine_model(df["month"], *params)
df["Residuals"] = df["True_NO2"] - df["Cosine_Model"]

# ==============================================================================
# 4. COMPUTE ALL BASELINE MODELS FOR COMPARISON TABLE
# ==============================================================================
print("⏳ Computing classical statistical & persistence baselines...")
# 1. Mean Baseline
mean_pred = np.full_like(y_stn, np.mean(y_stn))

# 2. Median Baseline
median_pred = np.full_like(y_stn, np.median(y_stn))

# 3. Mode Baseline (using continuous histogram bin peak as proxy for regression mode)
counts, bin_edges = np.histogram(y_stn, bins=50)
mode_val = (bin_edges[np.argmax(counts)] + bin_edges[np.argmax(counts) + 1]) / 2
mode_pred = np.full_like(y_stn, mode_val)

# 4. Persistence Model (y_hat(t) = y(t-1), shifting array by 1 hour)
persist_pred = np.roll(y_stn, 1)
persist_pred[0] = y_stn[0]  # Handle boundary

# Evaluate Metrics
models_dict = {
    "1. Mean Baseline": mean_pred,
    "2. Median Baseline": median_pred,
    "3. Mode Baseline": mode_pred,
    "4. Persistence Model (t-1)": persist_pred,
    "5. Cosine Climatology": df["Cosine_Model"].values,
}

results_table = []
for name, preds in models_dict.items():
  r2 = r2_score(y_stn, preds)
  rmse = np.sqrt(mean_squared_error(y_stn, preds))
  mae = mean_absolute_error(y_stn, preds)
  results_table.append(
      {"Model Architecture": name, "R² Score": r2, "RMSE (ppb)": rmse, "MAE (ppb)": mae}
  )

df_results = pd.DataFrame(results_table)

print("\n" + "=" * 75)
print("🏆 NO₂ TIME SERIES & BASELINE MODEL COMPARISON TABLE 🏆")
print("=" * 75)
print(df_results.to_string(index=False))
print("=" * 75)

# ==============================================================================
# 5. RENDER 3-PANEL ASSIGNMENT DIAGNOSTIC CHART
# ==============================================================================
print("🎨 Rendering publication-grade assignment charts...")
fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(13, 16), constrained_layout=True
)

# Panel 1: Time Series Climatology vs Cosine Fit
# Plotting first 500 consecutive hours so visual oscillations are legible
subset_len = min(500, len(df))
x_idx = range(subset_len)

ax1.plot(
    x_idx,
    df["True_NO2"].iloc[:subset_len],
    color="#94a3b8",
    alpha=0.6,
    linewidth=1.5,
    label="Raw Observed Telemetry (True NO₂)",
    zorder=2,
)
ax1.plot(
    x_idx,
    df["Climatological_Mean"].iloc[:subset_len],
    color="#dc2626",
    linewidth=2.5,
    linestyle="--",
    label="Monthly Mean Climatology (Step Function)",
    zorder=3,
)
ax1.plot(
    x_idx,
    df["Cosine_Model"].iloc[:subset_len],
    color="#2563eb",
    linewidth=3.0,
    label="Fitted Harmonic Cosine Model",
    zorder=4,
)

ax1.set_title(
    "1. Time Series Climatology & Harmonic Cosine Curve Overlay (Station 41)",
    fontsize=14,
    fontweight="bold",
)
ax1.set_ylabel("NO₂ Concentration (ppb)", fontsize=12, fontweight="bold")
ax1.legend(loc="upper right", frameon=True, facecolor="white", fontsize=10.5)
ax1.grid(True, linestyle="--", alpha=0.5)

# Panel 2: Scatter Plot (True vs. Modeled)
ax2.scatter(
    df["True_NO2"],
    df["Cosine_Model"],
    color="#0891b2",
    alpha=0.15,
    s=25,
    edgecolor="none",
    label="Observed vs. Cosine Model Predictions",
)
# Add 1:1 Parity Line for reference
max_val = max(df["True_NO2"].max(), df["Cosine_Model"].max())
ax2.plot(
    [0, max_val],
    [0, max_val],
    "k--",
    linewidth=1.5,
    alpha=0.7,
    label="1:1 Perfect Parity Reference Line",
)

ax2.set_title(
    "2. Scatter Plot: True vs. Modeled Cosine Values (Visualizing Horizontal"
    " Climatology Banding)",
    fontsize=14,
    fontweight="bold",
)
ax2.set_xlabel("True Observed NO₂ Concentration (ppb)", fontsize=12, fontweight="bold")
ax2.set_ylabel("Modeled Cosine Prediction (ppb)", fontsize=12, fontweight="bold")
ax2.legend(loc="upper left", frameon=True, facecolor="white", fontsize=10.5)
ax2.grid(True, linestyle="--", alpha=0.5)

# Panel 3: Residuals Time Series (True - Model)
ax3.plot(
    x_idx,
    df["Residuals"].iloc[:subset_len],
    color="#7c3aed",
    linewidth=1.5,
    label="Model Error Residuals (True - Modeled)",
)
ax3.axhline(
    0, color="black", linestyle="--", linewidth=1.5, label="Zero Error Line"
)

# Highlight high-frequency diurnal rush-hour features
ax3.set_title(
    "3. Residuals Time Series (True - Model): Highlighting Unresolved"
    " Rush-Hour Spikes & Weather Volatility",
    fontsize=14,
    fontweight="bold",
)
ax3.set_xlabel(
    "Sequential Daylight Observation Hours (Subset Timeline)",
    fontsize=12,
    fontweight="bold",
)
ax3.set_ylabel("Residual Error (ppb)", fontsize=12, fontweight="bold")
ax3.legend(loc="upper right", frameon=True, facecolor="white", fontsize=10.5)
ax3.grid(True, linestyle="--", alpha=0.5)

out_chart = OUTPUT_DIR / "no2_climatology_cosine_assignment.png"
plt.savefig(out_chart, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"✅ Saved Assignment Diagnostic Chart to: {out_chart}")
print("🎯 CLIMATOLOGY & BASELINE ANALYSIS FINISHED!")