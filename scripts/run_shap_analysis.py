import os

# Thread safety controls to prevent server CPU lockups
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import builtins
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
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

print("🧠 STARTING GAME-THEORETIC SHAP FEATURE IMPORTANCE ANALYSIS 🧠")
print("=" * 75)

if not data_file.exists():
  raise FileNotFoundError(f"Could not locate 20-feature dataset at: {data_file}")

# ==============================================================================
# 2. LOAD DATA & RECREATE THE 0.852 R² DOMAIN SPLIT
# ==============================================================================
print("⏳ Loading 20-feature master dataset...")
data = np.load(data_file, allow_pickle=True)

X_full = np.vstack([data["X_train"], data["X_test"]])
y_full = np.concatenate([data["y_train"], data["y_test"]])
feature_names = list(data["feature_names"])

print("🎯 Recreating 80/20 Literature Domain Split...")
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.20, random_state=42
)

# ==============================================================================
# 3. TRAIN THE OPTIMAL XGBOOST DIGITAL TWIN
# ==============================================================================
print("⏳ Training optimal 20-feature XGBoost model...")
start_train = time.time()

best_xgb = XGBRegressor(
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

best_xgb.fit(X_train, y_train)
print(f"✅ Model trained in {time.time() - start_train:.2f}s!")

# ==============================================================================
# 4. EXECUTE SHAP TREE EXPLAINER (WITH BUILT-IN FLOAT INTERCEPTOR)
# ==============================================================================
print(
    "\n⚙️ Initializing SHAP TreeExplainer (applying built-in float"
    " interceptor)..."
)

# Save Python's original built-in float function
orig_float = builtins.float


def patched_float(val):
  """Intercepts bracketed strings like '[7.72E0]' from XGBoost 2.0+ and strips brackets!"""
  if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
    val = val.strip("[]")
  return orig_float(val)


# Temporarily wrap float during SHAP initialization
builtins.float = patched_float
try:
  explainer = shap.TreeExplainer(best_xgb)
finally:
  # Immediately restore normal Python float behavior!
  builtins.float = orig_float

# Subsample 10,000 test points for rapid, statistically converged SHAP values
sample_size = min(10000, len(X_test))
np.random.seed(42)
subsample_idx = np.random.choice(len(X_test), size=sample_size, replace=False)
X_sub = X_test[subsample_idx]
X_sub_df = pd.DataFrame(X_sub, columns=feature_names)

print(f"⏳ Calculating Shapley values across {sample_size:,} test samples...")
start_shap = time.time()
shap_values = explainer(X_sub_df)
print(f"✅ SHAP calculation complete in {time.time() - start_shap:.2f}s!")

# ==============================================================================
# 5. PRINT CONSOLE FEATURE RANKING TABLE
# ==============================================================================
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
ranking_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean Absolute SHAP (ppb impact)": mean_abs_shap,
}).sort_values(by="Mean Absolute SHAP (ppb impact)", ascending=False)

ranking_df["Relative Importance (%)"] = (
    ranking_df["Mean Absolute SHAP (ppb impact)"]
    / ranking_df["Mean Absolute SHAP (ppb impact)"].sum()
    * 100.0
)

print("\n" + "=" * 75)
print("🏆 OFFICIAL SHAP FEATURE IMPORTANCE RANKING (YOUR PAPER'S TABLE I)")
print("=" * 75)
print(ranking_df.to_string(index=False))
print("=" * 75)

# ==============================================================================
# 6. GENERATE PUBLICATION FIGURES
# ==============================================================================
print("\n🎨 Generating Figure 4a: SHAP Summary Bee-Swarm Plot...")
plt.figure(figsize=(11, 8))
shap.summary_plot(shap_values, X_sub_df, plot_type="dot", show=False, max_display=15)
plt.title(
    "SHAP Feature Impact on Ground-Level NO₂ Predictions\n(Top 15 Covariates)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
plt.xlabel(
    "SHAP Value (Impact on Model Output in ppb)", fontsize=11, fontweight="bold"
)
plt.tight_layout()

out_beeswarm = OUTPUT_DIR / "shap_summary_beeswarm_plot.png"
plt.savefig(out_beeswarm, dpi=300, bbox_inches="tight")
plt.close()

print("🎨 Generating Figure 4b: SHAP Absolute Feature Importance Bar Chart...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_sub_df, plot_type="bar", show=False, max_display=15)
plt.title(
    "Mean Absolute Feature Importance (Global Impact in ppb)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
plt.xlabel("mean(|SHAP value|) (Average impact on model output magnitude in ppb)", fontsize=11, fontweight="bold")
plt.tight_layout()

out_bar = OUTPUT_DIR / "shap_feature_importance_bar.png"
plt.savefig(out_bar, dpi=300, bbox_inches="tight")
plt.close()

print(f"✅ Saved Bee-Swarm Plot to: {out_beeswarm}")
print(f"✅ Saved Bar Chart to:     {out_bar}")
print("🎯 SHAP ANALYSIS FINISHED!")