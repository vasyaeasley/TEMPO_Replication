# TEMPO Replication

Replication and benchmarking workflows for downscaling **NASA TEMPO** geostationary satellite NO₂ observations into **ground-level (surface) NO₂** estimates over California, with a focus on the **LA Basin**. The project fuses satellite NO₂, ERA5 meteorology, and static geospatial covariates (traffic, roads, elevation, population) to train machine-learning "digital twins" of air quality and evaluate them against EPA/AirNow ground stations.

## What the project does

Based on the scripts in [`scripts/`](scripts/), the pipeline covers the full research workflow:

- **Data acquisition & preprocessing** — download ERA5 reanalysis (`download_era5.py`, `download_copernicus_era5_missing.py`), inspect and regrid TEMPO and ERA5 to a common 1 km grid (`regrid_tempo_pipeline.py`, `regrid_era5_pipeline.py`), process static rasters and traffic/road layers (`process_static_rasters.py`, `process_traffic_road.py`), and extract point datasets at station locations (`extract_point_dataset.py`).
- **Modeling** — an **XGBoost** "digital twin" using ~20 fused covariates (`train_xgboost_20features.py`, `train_baseline_xgboost.py`) and a **ConvLSTM** spatiotemporal model for hourly, 1 km LA Basin forecasting (`train_convlstm_digital_twin.py`, `convlstm_architecture.py`). A U-Net twin (`train_unet_digital_twin.py`) is also included.
- **Baselines & benchmarks** — persistence, climatology/cosine, harmonic regression, and multiple-linear-regression baselines across cities (`run_baseline_models.py`, `run_climatology_cosine_baselines.py`, `run_3city_*`, `run_pomona_*`, `run_persistent_model.py`).
- **Evaluation & interpretability** — temporal holdout and diurnal-cycle evaluation (`run_temporal_holdout.py`, `run_diurnal_cycle_evaluation.py`), extreme-episode stress tests (`run_extreme_episode_test.py`), feature ablation (`run_ablation_study.py`), and game-theoretic **SHAP** feature importance (`run_shap_analysis.py`).
- **Visualization** — spatial residual maps, density scatterplots, cluster maps, and diurnal animations (`generate_*` and `plot_*` scripts).


## Repository layout

- `scripts/` — data pipeline, training, baselines, evaluation, and plotting scripts
- `models/` — trained model weights and generated analysis charts
- `graphs_for_paper/` — publication-ready figures (see below)
- `data/raw/`, `data/processed/` — local data (excluded from version control)

## Figures (`graphs_for_paper/`)

The `graphs_for_paper/` directory contains the publication figures produced by the scripts above:

### Model performance & evaluation
![True vs. predicted density scatterplot](graphs_for_paper/tempo_true_vs_pred_density_scatterplot.png)
*True vs. predicted ground-level NO₂ density scatterplot.*

![Continuous time series](graphs_for_paper/tempo_continuous_timeseries.png)
*Continuous predicted vs. observed NO₂ time series.*

![Diurnal cycle evaluation](graphs_for_paper/tempo_diurnal_cycle_evaluation.png)
*Diurnal (24-hour) cycle evaluation capturing rush-hour peaks.*

![Extreme episode stress test](graphs_for_paper/tempo_extreme_episode_stress_test.png)
*Model behavior during extreme pollution episodes.*

![Spatial residual map](graphs_for_paper/tempo_spatial_residual_map.png)
*Spatial map of model residuals.*

### Interpretability (SHAP) & ablation
![SHAP summary beeswarm plot](graphs_for_paper/shap_summary_beeswarm_plot.png)
*SHAP bee-swarm plot of feature impact on NO₂ predictions.*

![SHAP feature importance bar](graphs_for_paper/shap_feature_importance_bar.png)
*Mean absolute SHAP feature importance.*

![Ablation study chart](graphs_for_paper/tempo_ablation_study_chart.png)
*Hierarchical feature ablation (R² and RMSE vs. number of features).*

### Baselines & maps
![Pomona XGBoost vs. classical baselines](graphs_for_paper/pomona_xgboost_vs_classical_baselines.png)
*XGBoost digital twin vs. classical baselines at Pomona.*

![California standard science map](graphs_for_paper/california_standard_science_map.png)
*Standard science map of California NO₂.*

![California gridded cluster map (80/20)](graphs_for_paper/california_gridded_cluster_map_80_20.jpg)
*Gridded station-cluster map for the 80/20 domain split.*

### Animations
![LA Basin diurnal twin animation](graphs_for_paper/la_basin_diurnal_twin_animation.gif)
*LA Basin diurnal digital-twin animation.*

![California diurnal twin animation](graphs_for_paper/california_diurnal_twin_animation.gif)
*California diurnal digital-twin animation.*

## Data

Large raw and processed datasets are intentionally excluded from version control. Expected local layout:

- `data/raw/`
- `data/processed/` — e.g. `epa_point_dataset_14months_20features.npz`, `era5_california_1x1km_master.nc`, `tempo_monthly/`

Most scripts resolve paths relative to the repository root, so the project can run on other machines without hardcoded local paths.

## Quick start

1. Create a Python environment (Python 3.10+ recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place required datasets under `data/raw/` and/or `data/processed/`.
4. Reproduce paper analyses, e.g.:

```bash
python scripts/run_ablation_study.py      # feature ablation chart
python scripts/run_shap_analysis.py       # SHAP importance figures
python scripts/train_convlstm_digital_twin.py   # ConvLSTM LA Basin twin
```

## Reproducibility notes

- This repo expects preprocessed NetCDF/NPZ inputs under `data/processed/`.
- If you publish this repository publicly, share dataset download/preprocessing instructions and file manifests separately.
- Files larger than 100 MB should be hosted externally or tracked with Git LFS.

## Suggested public release checklist

- Add a data manifest (filenames, sizes, source URLs, licenses).
- Add preprocessing commands in this README.
- Add a small sample dataset for quick CI/smoke tests.
- Verify at least one end-to-end script from a clean environment.
