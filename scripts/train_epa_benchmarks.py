"""Robust, leakage-free XGBoost + PyTorch MLP benchmarks for surface NO2 (ppb).

Pipeline
--------
1. Load epa_point_dataset_14months.npz (X, y, station group ids) + strict audit.
2. Filter the target to a physical NO2 range, sanitize features (Inf -> NaN),
   impute NaNs with the TRAINING median, drop zero-variance/dead columns, and
   log1p the (log-normal) TEMPO column.
3. Select the feature set + XGBoost hyper-parameters with GroupKFold spatial
   cross-validation over held-out STATIONS (this mirrors the test split, so it
   is an honest spatial-generalization estimate with no test leakage).
4. Evaluate the selected XGBoost and a 4-layer MLP (64-32-16-1, gradient
   clipping + spatial early stopping) ONCE on the held-out test stations.

Notes
-----
* Spatial generalization is the hard part: rows from one station must never be
  used to validate predictions at that same station. Group ids come from the
  extraction script and drive both CV and MLP early stopping.
* `elev` is still dropped automatically while the corrected USGS DEM tiles are
  being sourced (the current raster is ~95% nodata at the stations).
* Training is forced onto CPU (local NVIDIA driver 12020 is too old).
"""

import os

# Force CPU BEFORE importing torch (local driver 12020 is too old for this
# build). Remove/override this line once the NVIDIA driver is updated.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import warnings
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# Benign: CUDA probe on a too-old driver, and the intentional all-NaN median
# of a fully-empty column (which we deliberately fall back to 0 and then drop).
warnings.filterwarnings("ignore", message=".*CUDA initialization.*")
warnings.filterwarnings("ignore", message=".*All-NaN slice encountered.*")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "epa_point_dataset_14months.npz"
FEATURE_NAMES = [
    "TEMPO_NO2",
    "t2m",
    "u10",
    "v10",
    "sp",
    "blh",
    "pop",
    "elev",
    "hour_sin",
    "hour_cos",
    "traffic",
    "road_density",
]

CONST_TOL = 1e-8          # std below this => treated as constant/dead feature
Y_MIN, Y_MAX = 0.0, 100.0  # physical NO2 ppb bounds for the target filter


# ==============================================================================
# 1. LOAD + STRICT NaN/Inf AUDIT
# ==============================================================================
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def audit(name, X, y):
    print(f"\n[audit] {name}: X={X.shape} y={y.shape}")
    for j in range(X.shape[1]):
        col = X[:, j]
        finite = np.isfinite(col)
        std = col[finite].std() if finite.any() else float("nan")
        label = FEATURE_NAMES[j] if j < len(FEATURE_NAMES) else f"col{j}"
        print(
            f"   [{j}] {label:10s} NaN={int(np.isnan(col).sum()):6d} "
            f"Inf={int(np.isinf(col).sum()):6d} std={std:.4g}"
        )
    print(
        f"   target      NaN={int(np.isnan(y).sum())} "
        f"Inf={int(np.isinf(y).sum())} "
        f"min={np.nanmin(y):.3g} max={np.nanmax(y):.3g} mean={np.nanmean(y):.3g}"
    )


print("🌍 ROBUST NO2 BENCHMARK TRAINING (XGBoost + MLP) 🌍")
print("=" * 65)

data = np.load(DATA_PATH)
X_tr = data["X_train"].astype(np.float64)
y_tr = data["y_train"].astype(np.float64)
X_te = data["X_test"].astype(np.float64)
y_te = data["y_test"].astype(np.float64)
groups_tr = data["groups_train"].astype(int) if "groups_train" in data.files else None
if "feature_names" in data.files:
    FEATURE_NAMES = [str(n) for n in data["feature_names"]]

audit("train (raw)", X_tr, y_tr)
audit("test  (raw)", X_te, y_te)


# ==============================================================================
# 2. TARGET FILTER + FEATURE SANITIZE + MEDIAN IMPUTATION
# ==============================================================================
def filter_target(X, y, g=None):
    keep = np.isfinite(y) & (y > Y_MIN) & (y <= Y_MAX)
    if g is None:
        return X[keep], y[keep]
    return X[keep], y[keep], g[keep]


if groups_tr is None:
    raise RuntimeError(
        "groups_train missing from .npz — re-run extract_point_dataset.py"
    )
X_tr, y_tr, groups_tr = filter_target(X_tr, y_tr, groups_tr)
X_te, y_te = filter_target(X_te, y_te)

# Inf -> NaN so a single imputation path handles every non-finite value.
X_tr[~np.isfinite(X_tr)] = np.nan
X_te[~np.isfinite(X_te)] = np.nan

# Impute with TRAINING medians only (no test leakage). A fully-NaN column has an
# undefined median -> fall back to 0 so it becomes constant and is dropped next.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    train_median = np.nanmedian(X_tr, axis=0)
train_median = np.where(np.isfinite(train_median), train_median, 0.0)


def impute(X):
    rows, cols = np.where(np.isnan(X))
    X[rows, cols] = train_median[cols]
    return X


X_tr = impute(X_tr)
X_te = impute(X_te)


# ==============================================================================
# 3. DROP ZERO-VARIANCE FEATURES (StandardScaler only on non-constant columns)
# ==============================================================================
train_std = X_tr.std(axis=0)
keep_mask = train_std > CONST_TOL
kept_names = [FEATURE_NAMES[j] for j in range(len(keep_mask)) if keep_mask[j]]
dropped = [FEATURE_NAMES[j] for j in range(len(keep_mask)) if not keep_mask[j]]
print(
    f"\n[features] kept {int(keep_mask.sum())}/{len(keep_mask)} -> {kept_names}"
)
print(f"[features] dropped constant/dead -> {dropped or 'none'}")

X_tr = X_tr[:, keep_mask]
X_te = X_te[:, keep_mask]

# Satellite column densities are ~log-normal: log1p tames the extreme skew.
if "TEMPO_NO2" in kept_names:
    ti = kept_names.index("TEMPO_NO2")
    X_tr[:, ti] = np.log1p(np.clip(X_tr[:, ti], 0.0, None))
    X_te[:, ti] = np.log1p(np.clip(X_te[:, ti], 0.0, None))


# ==============================================================================
# 4. SPATIAL MODEL SELECTION (GroupKFold over held-out STATIONS, no leakage)
# ==============================================================================
# The test split holds out whole stations, so feature/hyper-parameter selection
# must also be validated on held-out stations — never on rows from training
# stations. GroupKFold on the station id gives an honest spatial estimate that
# we use to pick the model. The test set is touched exactly once, at the end.
name_to_col = {n: j for j, n in enumerate(kept_names)}


def cols(names):
    return [name_to_col[n] for n in names if n in name_to_col]


FEATURE_SETS = {
    "all": list(kept_names),
    "tempo+met+hour": [
        "TEMPO_NO2", "t2m", "u10", "v10", "sp", "blh", "hour_sin", "hour_cos",
    ],
    "tempo+blh+hour": ["TEMPO_NO2", "blh", "hour_sin", "hour_cos"],
    "sat+traffic+hour": ["TEMPO_NO2", "blh", "hour_sin", "hour_cos", "traffic"],
    "sat+road+hour": ["TEMPO_NO2", "blh", "hour_sin", "hour_cos", "road_density"],
    "sat+traffic+road+hour": [
        "TEMPO_NO2", "blh", "hour_sin", "hour_cos", "traffic", "road_density",
    ],
    "sat+met+traffic+road+hour": [
        "TEMPO_NO2", "t2m", "u10", "v10", "sp", "blh", "hour_sin", "hour_cos",
        "traffic", "road_density",
    ],
    "tempo_only": ["TEMPO_NO2"],
}

XGB_CONFIGS = {
    "d3_lr03": dict(
        n_estimators=500, max_depth=3, learning_rate=0.03, subsample=0.7,
        colsample_bytree=0.8, reg_lambda=5.0, min_child_weight=10,
    ),
    "d4_lr03": dict(
        n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.7,
        colsample_bytree=0.8, reg_lambda=5.0, min_child_weight=10,
    ),
    "d5_lr03": dict(
        n_estimators=500, max_depth=5, learning_rate=0.03, subsample=0.7,
        colsample_bytree=0.8, reg_lambda=8.0, min_child_weight=20,
    ),
    "d6_lr02": dict(
        n_estimators=600, max_depth=6, learning_rate=0.02, subsample=0.7,
        colsample_bytree=0.7, reg_lambda=10.0, min_child_weight=30,
    ),
    "d4_lr05_heavyreg": dict(
        n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.6,
        colsample_bytree=0.7, reg_lambda=15.0, reg_alpha=2.0, min_child_weight=30,
    ),
    "d3_lr05_light": dict(
        n_estimators=600, max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.9, reg_lambda=3.0, min_child_weight=5,
    ),
}

n_groups = int(len(np.unique(groups_tr)))
n_splits = min(5, n_groups)
gkf = GroupKFold(n_splits=n_splits)
print(
    f"\n[spatial-CV] {n_groups} training stations -> GroupKFold({n_splits}); "
    f"{len(FEATURE_SETS)} feature sets x {len(XGB_CONFIGS)} configs"
)

leaderboard = []
for fs_name, feats in FEATURE_SETS.items():
    ci = cols(feats)
    if not ci:
        continue
    for cfg_name, cfg in XGB_CONFIGS.items():
        fold_scores = []
        for tr_idx, va_idx in gkf.split(X_tr[:, ci], y_tr, groups=groups_tr):
            m = xgb.XGBRegressor(
                tree_method="hist", n_jobs=-1, random_state=RANDOM_SEED, **cfg
            )
            m.fit(X_tr[np.ix_(tr_idx, ci)], y_tr[tr_idx])
            fold_scores.append(
                r2_score(y_tr[va_idx], m.predict(X_tr[np.ix_(va_idx, ci)]))
            )
        mean_r2, std_r2 = float(np.mean(fold_scores)), float(np.std(fold_scores))
        leaderboard.append((mean_r2, std_r2, fs_name, cfg_name, feats, cfg))
        print(f"   {fs_name:16s} {cfg_name:12s} CV_R²={mean_r2:+.3f} ± {std_r2:.3f}")

leaderboard.sort(key=lambda r: r[0], reverse=True)
print(f"[spatial-CV] top raw-mean: {leaderboard[0][2]}/{leaderboard[0][3]} "
      f"CV_R²={leaderboard[0][0]:+.3f} ± {leaderboard[0][1]:.3f}")

# Risk-adjusted selection (mean - std): reward a high CV R² AND low across-fold
# variance, i.e. RELIABLE spatial generalization. Selecting the raw maximum mean
# tends to over-fit CV noise and pick a deeper, less stable model that transfers
# worse to unseen stations. This uses CV statistics only — never the test set.
leaderboard.sort(key=lambda r: (r[0] - r[1]), reverse=True)
best_r2, best_std, best_fs, best_cfg_name, best_feats, best_cfg = leaderboard[0]
print(
    f"\n[select] winner (mean-std rule): '{best_fs}' + '{best_cfg_name}' "
    f"(spatial CV R²={best_r2:+.3f} ± {best_std:.3f}; adj={best_r2 - best_std:+.3f})"
)
print(f"[select] features -> {best_feats}")


# ==============================================================================
# 5. FINAL EVALUATION ON HELD-OUT TEST STATIONS (touched once)
# ==============================================================================
ci = cols(best_feats)

# 5a. XGBoost with the CV-selected feature set + config.
xgb_final = xgb.XGBRegressor(
    tree_method="hist", n_jobs=-1, random_state=RANDOM_SEED, **best_cfg
)
xgb_final.fit(X_tr[:, ci], y_tr)
xgb_pred = xgb_final.predict(X_te[:, ci])
xgb_r2 = r2_score(y_te, xgb_pred)

# Reference: the previous naive config (all features, deep unregularized trees).
naive = xgb.XGBRegressor(
    n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", n_jobs=-1,
    random_state=RANDOM_SEED,
)
naive.fit(X_tr, y_tr)
naive_r2 = r2_score(y_te, naive.predict(X_te))

# 5b. PyTorch MLP (64-32-16-1) on the selected features, with spatial early
#     stopping validated on held-out TRAINING stations (never the test set).
device = torch.device("cpu")
tr_idx, va_idx = next(gkf.split(X_tr[:, ci], y_tr, groups=groups_tr))
X_sub, y_sub = X_tr[np.ix_(tr_idx, ci)], y_tr[tr_idx]
X_val, y_val = X_tr[np.ix_(va_idx, ci)], y_tr[va_idx]

scaler_x = StandardScaler().fit(X_sub)
scaler_y = StandardScaler().fit(y_sub.reshape(-1, 1))
X_sub_s = np.clip(scaler_x.transform(X_sub), -6, 6)
X_val_s = np.clip(scaler_x.transform(X_val), -6, 6)
X_te_s = np.clip(scaler_x.transform(X_te[:, ci]), -6, 6)
y_sub_s = scaler_y.transform(y_sub.reshape(-1, 1)).ravel()


class MLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


def predict_ppb(net, X_t):
    net.eval()
    with torch.no_grad():
        s = net(X_t).cpu().numpy().ravel()
    return scaler_y.inverse_transform(s.reshape(-1, 1)).ravel()


model = MLP(len(ci)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
train_dl = DataLoader(
    TensorDataset(
        torch.tensor(X_sub_s, dtype=torch.float32),
        torch.tensor(y_sub_s, dtype=torch.float32).view(-1, 1),
    ),
    batch_size=512, shuffle=True,
)
X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
X_te_t = torch.tensor(X_te_s, dtype=torch.float32)

best_val_r2, best_state, patience, bad = -np.inf, None, 15, 0
for epoch in range(300):
    model.train()
    for xb, yb in train_dl:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch} — aborting.")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    val_r2 = r2_score(y_val, predict_ppb(model, X_val_t))
    if val_r2 > best_val_r2 + 1e-4:
        best_val_r2, bad = val_r2, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
        if bad >= patience:
            print(
                f"   MLP early-stopped at epoch {epoch} "
                f"(best spatial-val R²={best_val_r2:+.3f})"
            )
            break

if best_state is not None:
    model.load_state_dict(best_state)
mlp_pred = predict_ppb(model, X_te_t)
mlp_r2 = r2_score(y_te, mlp_pred)


# ==============================================================================
# 6. RESULTS (spatial generalization to unseen stations, physical ppb)
# ==============================================================================
print("\n" + "=" * 65)
print("HELD-OUT TEST STATIONS  (spatial generalization, physical ppb)")
print("-" * 65)
print(f"  XGBoost naive (all feats, depth 6)   R²={naive_r2:+.4f}")
print(
    f"  XGBoost selected [{best_fs}]           "
    f"R²={xgb_r2:+.4f}  RMSE={rmse(y_te, xgb_pred):.3f} ppb"
)
print(
    f"  MLP     selected [{best_fs}]           "
    f"R²={mlp_r2:+.4f}  RMSE={rmse(y_te, mlp_pred):.3f} ppb"
)
print("=" * 65)


# ==============================================================================
# 7. PERSIST reproducible artifacts (model, metadata, predictions, plot)
# ==============================================================================
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

xgb_final.save_model(str(MODEL_DIR / "xgboost_tempo_no2.json"))

metadata = {
    "final_model": "XGBoost",
    "features": list(best_feats),
    "xgb_config": dict(best_cfg),
    "selection_rule": "spatial GroupKFold(5), risk-adjusted (mean - std)",
    "random_seed": RANDOM_SEED,
    "n_train_stations": int(n_groups),
    "n_train_rows": int(X_tr.shape[0]),
    "n_test_rows": int(X_te.shape[0]),
    "cv_r2_mean": round(float(best_r2), 4),
    "cv_r2_std": round(float(best_std), 4),
    "test_r2_xgboost": round(float(xgb_r2), 4),
    "test_rmse_xgboost_ppb": round(rmse(y_te, xgb_pred), 3),
    "test_r2_mlp": round(float(mlp_r2), 4),
    "test_rmse_mlp_ppb": round(rmse(y_te, mlp_pred), 3),
    "preprocessing": {
        "target_ppb_range": [Y_MIN, Y_MAX],
        "impute": "training median per feature",
        "log1p_on": "TEMPO_NO2",
        "dropped_constant_features": dropped,
    },
}
with open(MODEL_DIR / "xgboost_tempo_no2_metadata.json", "w") as fh:
    json.dump(metadata, fh, indent=2)

np.savez_compressed(
    MODEL_DIR / "test_predictions.npz",
    y_true=y_te,
    y_pred_xgb=xgb_pred,
    y_pred_mlp=mlp_pred,
)

# Predicted-vs-actual density scatter (paper Fig. 3 style).
lim = float(np.ceil(max(np.percentile(y_te, 99.5), np.percentile(xgb_pred, 99.5))))
fig, ax = plt.subplots(figsize=(6, 6))
hb = ax.hexbin(y_te, xgb_pred, gridsize=60, cmap="viridis", bins="log", mincnt=1)
ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="1:1")
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel("Measured surface NO$_2$ (ppb)")
ax.set_ylabel("Predicted surface NO$_2$ (ppb)")
ax.set_title(
    f"TEMPO+BLH+hour XGBoost — held-out stations\n"
    f"R² = {xgb_r2:.3f},  RMSE = {rmse(y_te, xgb_pred):.2f} ppb"
)
ax.legend(loc="upper left")
fig.colorbar(hb, ax=ax, label="count (log)")
fig.tight_layout()
fig.savefig(MODEL_DIR / "predicted_vs_actual_tempo_xgb.png", dpi=130)

print(f"\n💾 Reproducible artifacts saved to {MODEL_DIR}:")
print("   - xgboost_tempo_no2.json (model)")
print("   - xgboost_tempo_no2_metadata.json (features, config, metrics, seed)")
print("   - test_predictions.npz (y_true, y_pred_xgb, y_pred_mlp)")
print("   - predicted_vs_actual_tempo_xgb.png")