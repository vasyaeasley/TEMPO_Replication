from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ==============================================================================
# 1. SETUP & PATHS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
processed_dir = BASE_DIR / "data" / "processed"
input_file = processed_dir / "la_basin_point_dataset.npz"

# Fallback to statewide file if LA Basin file isn't found
if not input_file.exists():
    print(f"⚠️ {input_file.name} not found! Falling back to statewide dataset...")
    input_file = processed_dir / "epa_point_dataset_14months.npz"

print("=" * 65)
print(f"🌍 GENERATING LA BASIN CORRELATION MATRIX & DENSITY PLOTS 🌍")
print(f"📂 Loading dataset: {input_file.name}")
print("=" * 65)

# ==============================================================================
# 2. LOAD & UN-SCALE DATA TO PHYSICAL UNITS
# ==============================================================================
data = np.load(input_file)
X_all = np.vstack([data["X_train"], data["X_test"]])
y_all = np.concatenate([data["y_train"], data["y_test"]])

# Extract raw normalized columns based on extract_point_dataset.py indices
tempo_no2 = X_all[:, 0]               # Already in 10^15 molec/cm^2
t2m_c     = (X_all[:, 1] * 15.0) + 11.85  # Unscale back to Celsius (°C)
u10_ms    = X_all[:, 2] * 10.0            # Unscale back to m/s
v10_ms    = X_all[:, 3] * 10.0            # Unscale back to m/s
sp_hpa    = ((X_all[:, 4] * 5000.0) + 95000.0) / 100.0  # Unscale to hPa
blh_m     = X_all[:, 5] * 1000.0          # Unscale back to meters (m)

# Compile into a clean 2D matrix for correlation analysis
covariates = np.column_stack([t2m_c, u10_ms, v10_ms, sp_hpa, blh_m, tempo_no2])
labels = [
    "2m Temp (°C)",
    "10m U-Wind",
    "10m V-Wind",
    "Pressure (hPa)",
    "BL Height (m)",
    "TEMPO NO₂",
]

print(f"✅ Extracted {len(y_all):,} localized station-hour observations.")

# ==============================================================================
# 3. GENERATE THE CORRELATION MATRIX (Figure 1)
# ==============================================================================
print("\n⏳ Computing Pearson Correlation Matrix...")
corr_matrix = np.corrcoef(covariates, rowvar=False)

plt.figure(figsize=(9, 8))
sns.set_theme(style="white")
heatmap = sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1.0,
    vmax=1.0,
    xticklabels=labels,
    yticklabels=labels,
    cbar_kws={"label": "Pearson Correlation (r)"},
    square=True,
    annot_kws={"size": 11, "weight": "bold"},
)

plt.title(
    "Atmospheric Covariate Correlation Matrix (LA Basin Digital Twin)",
    fontsize=14,
    pad=15,
    weight="bold",
)
plt.xticks(rotation=45, ha="right", fontsize=11, weight="bold")
plt.yticks(rotation=0, fontsize=11, weight="bold")
plt.tight_layout()

corr_out = processed_dir / "la_basin_correlation_matrix.png"
plt.savefig(corr_out, dpi=300, bbox_inches="tight")
plt.close()
print(f"📈 Correlation Matrix saved to:\n   {corr_out}")

# ==============================================================================
# 4. GENERATE PAIRWISE DENSITY SCATTERPLOTS (Figure 2)
# ==============================================================================
print("\n⏳ Generating Hexbin Density Distributions vs. TEMPO NO₂...")
plot_vars = [
    (t2m_c, "2m Temperature (°C)", -5, 45),
    (u10_ms, "10m U-Wind (m/s)", -8, 12),
    (v10_ms, "10m V-Wind (m/s)", -6, 10),
    (sp_hpa, "Surface Pressure (hPa)", 970, 1025),
    (blh_m, "Boundary Layer Height (m)", 0, 3500),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, (var_data, var_name, x_min, x_max) in enumerate(plot_vars):
    ax = axes[i]
    r_val = np.corrcoef(var_data, tempo_no2)[0, 1]
    
    # Hexbin log-density plot
    hb = ax.hexbin(
        var_data,
        tempo_no2,
        gridsize=50,
        cmap="YlOrRd",
        bins="log",
        mincnt=1,
        extent=[x_min, x_max, 0, np.percentile(tempo_no2, 99.5)],
    )
    
    ax.set_title(
        f"{var_name} vs NO₂ (r = {r_val:.2f})", fontsize=12, weight="bold"
    )
    ax.set_xlabel(var_name, fontsize=10, weight="bold")
    ax.set_ylabel("TEMPO NO₂ (10¹⁵ molec/cm²)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Log(Density Count)", fontsize=9)

# Hide the unused 6th subplot frame
axes[5].axis("off")

plt.suptitle(
    f"Pairwise Density Distributions: LA Basin Weather Covariates vs. TEMPO NO₂\n"
    f"(Sampled across {len(y_all):,} localized urban observations)",
    fontsize=15,
    weight="bold",
    y=0.98,
)
plt.tight_layout(rect=[0, 0, 1, 0.95])

scatter_out = processed_dir / "la_basin_density_distributions.png"
plt.savefig(scatter_out, dpi=300, bbox_inches="tight")
plt.close()
print(f"📊 Density Distributions saved to:\n   {scatter_out}")
print("=" * 65)
print("🎉 VISUALIZATION PIPELINE COMPLETE!")
print("=" * 65)