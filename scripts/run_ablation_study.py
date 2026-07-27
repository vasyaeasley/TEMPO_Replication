import os

# Thread safety controls to prevent server CPU lockups
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# ==============================================================================
# 1. SETUP & PATH CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

data_file = PROCESSED_DIR / "epa_point_dataset_14months_20features.npz"

print("🧪 STARTING HIERARCHICAL FEATURE ABLATION STUDY 🧪")
print("=" * 75)

if not data_file.exists():
  raise FileNotFoundError(f"Could not locate 20-feature dataset at: {data_file}")

# ==============================================================================
# 2. LOAD DATA & DEFINE FEATURE INVENTORY
# ==============================================================================
print("⏳ Loading 20-feature master dataset...")
data = np.load(data_file, allow_pickle=True)

X_full = np.vstack([data["X_train"], data["X_test"]])
y_full = np.concatenate([data["y_train"], data["y_test"]])
g_full = np.concatenate([data["groups_train"], data["groups_test"]])

print("🎯 Recreating 80/20 Literature Domain Split...")
X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
    X_full, y_full, g_full, test_size=0.20, random_state=42
)

# Canonical list of all 20 features in our data fusion pipeline
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

# If the .npz file has embedded feature names, use them as absolute ground truth
if "feature_names" in data:
  feature_names = np.array([str(f) for f in data["feature_names"]])

print(f"📋 Verified {len(feature_names)} total covariates in dataset.")

# ==============================================================================
# 3. TRAIN BASELINE 20-FEATURE MODEL & EXTRACT RANKINGS
# ==============================================================================
print("⏳ Training Full 20-Feature Baseline XGBoost Model...")
start_time = time.time()
base_xgb = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=673,
    learning_rate=0.0327,
    max_depth=7,
    subsample=0.7100,
    colsample_bytree=0.6610,
    gamma=0.1324,
    reg_alpha=32.5183,
    reg_lambda=7.2963,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)
base_xgb.fit(X_train, y_train)
print(f"✅ Baseline model trained in {time.time() - start_time:.2f}s!")

# Rank features dynamically by their native XGBoost importance
importances = base_xgb.feature_importances_
ranked_indices = np.argsort(importances)[::-1]
ranked_names = feature_names[ranked_indices]

print("\n🏆 Top 10 Most Influential Variables:")
for i in range(10):
  print(f"   {i + 1:2d}. {ranked_names[i]:<20} ({importances[ranked_indices[i]]:.4f})")

# ==============================================================================
# 4. EXECUTE HIERARCHICAL ABLATION TIERS
# ==============================================================================
ablation_tiers = [3, 5, 8, 12, 16, len(feature_names)]
results = []

print("\n⚙️ Systematically evaluating feature subsets...")
print("-" * 75)
print(f"{'Tier':<10} {'Features Included':<30} {'R²':<8} {'RMSE (ppb)':<12} {'MAE (ppb)':<10}")
print("-" * 75)

for k in ablation_tiers:
  # Slice training and testing matrices to strictly include only the Top K features
  top_k_idx = ranked_indices[:k]
  X_tr_slice = X_train[:, top_k_idx]
  X_te_slice = X_test[:, top_k_idx]

  # Train optimized XGBoost on subset
  model_k = XGBRegressor(
      objective="reg:squarederror",
      n_estimators=673,
      learning_rate=0.0327,
      max_depth=7,
      subsample=0.7100,
      colsample_bytree=0.6610,
      gamma=0.1324,
      reg_alpha=32.5183,
      reg_lambda=7.2963,
      random_state=42,
      n_jobs=-1,
      tree_method="hist",
  )
  model_k.fit(X_tr_slice, y_train)

  # Predict on unseen test hours
  preds_k = model_k.predict(X_te_slice)

  r2_k = r2_score(y_test, preds_k)
  rmse_k = np.sqrt(mean_squared_error(y_test, preds_k))
  mae_k = mean_absolute_error(y_test, preds_k)

  tier_label = f"Top {k}" if k < len(feature_names) else "Full 20"
  feat_summary = ", ".join(ranked_names[: min(3, k)]) + (
      "..." if k > 3 else ""
  )

  results.append({
      "Tier": tier_label,
      "Count": k,
      "R2": r2_k,
      "RMSE": rmse_k,
      "MAE": mae_k,
      "Top_Features": ", ".join(ranked_names[:k]),
  })

  print(f"{tier_label:<10} {feat_summary:<30} {r2_k:.3f}   {rmse_k:<12.2f} {mae_k:<10.2f}")

# ==============================================================================
# 5. GENERATE PUBLICATION-GRADE ABLATION CHART
# ==============================================================================
print("\n🎨 Rendering publication-grade ablation comparison chart...")
df_res = pd.DataFrame(results)

fig, ax1 = plt.subplots(figsize=(10, 6))

color_r2 = "#2563eb"  # Deep royal blue
color_rmse = "#dc2626"  # Crimson red

# Plot R² on primary y-axis
bars = ax1.bar(
    df_res["Tier"],
    df_res["R2"],
    color=color_r2,
    alpha=0.85,
    width=0.45,
    edgecolor="black",
    linewidth=1.2,
    label="Test R² Score",
)
ax1.set_xlabel("Ablation Feature Tier", fontsize=13, fontweight="bold")
ax1.set_ylabel("Unseen Test Set Accuracy ($R^2$)", fontsize=13, fontweight="bold", color=color_r2)
ax1.set_ylim(0.60, 0.90)
ax1.tick_params(axis="y", labelcolor=color_r2, labelsize=11)
ax1.tick_params(axis="x", labelsize=12)
ax1.grid(True, linestyle="--", alpha=0.3, zorder=0)

# Annotate R² values directly above bars
for bar in bars:
  yval = bar.get_height()
  ax1.text(
      bar.get_x() + bar.get_width() / 2.0,
      yval + 0.005,
      f"{yval:.3f}",
      ha="center",
      va="bottom",
      fontsize=11,
      fontweight="bold",
  )

# Plot RMSE on secondary y-axis
ax2 = ax1.twinx()
line = ax2.plot(
    df_res["Tier"],
    df_res["RMSE"],
    color=color_rmse,
    marker="o",
    linewidth=3,
    markersize=8,
    label="Test RMSE (ppb)",
)
ax2.set_ylabel("Root Mean Squared Error (ppb)", fontsize=13, fontweight="bold", color=color_rmse)
ax2.set_ylim(2.5, 4.5)
ax2.tick_params(axis="y", labelcolor=color_rmse, labelsize=11)

# Combined Legend
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(
    lines_1 + lines_2,
    labels_1 + labels_2,
    loc="lower right",
    frameon=True,
    facecolor="white",
    framealpha=0.95,
    fontsize=11,
    edgecolor="#4b5563",
)

plt.title(
    "XGBoost Digital Twin Feature Ablation Analysis\nEvaluating Occam's Razor"
    " in Geostationary Downscaling",
    fontsize=15,
    fontweight="bold",
    pad=15,
)
plt.tight_layout()

out_chart = OUTPUT_DIR / "tempo_ablation_study_chart.png"
plt.savefig(out_chart, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"✅ Saved Ablation Chart to: {out_chart}")

# ==============================================================================
# 6. PRINT PRE-FORMATTED LATEX TABLE
# ==============================================================================
print("\n📑 DROP-IN LATEX TABLE FOR SECTION 4:")
print("=" * 75)
print("\\begin{table}[htbp]")
print("    \\centering")
print("    \\caption{Hierarchical Feature Ablation \& Sensitivity Analysis}")
print("    \\label{tab:ablation}")
print("    \\resizebox{\\linewidth}{!}{%")
print("    \\begin{tabular}{lrrr}")
print("    \\toprule")
print("    \\textbf{Ablation Tier} & \\textbf{Test $R^2$} & \\textbf{RMSE (ppb)} & \\textbf{MAE (ppb)} \\\\")
print("    \\midrule")
for r in results:
  print(
      f"    {r['Tier']:<12} & {r['R2']:.3f} & {r['RMSE']:.2f} & {r['MAE']:.2f} \\\\"
  )
print("    \\bottomrule")
print("    \\end{tabular}%")
print("    }")
print("\\end{table}")
print("=" * 75)
print("🎯 ABLATION STUDY FINISHED!")