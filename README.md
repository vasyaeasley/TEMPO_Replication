# TEMPO Replication

Replication and benchmarking workflows for TEMPO NO2 and surface NO2 modeling experiments.

## What is included
- Training and evaluation scripts in `scripts/`
- Quick smoke-test entry points:
  - `quick_test_train.py`
  - `ultra_fast_test.py`
- Plotting and map-generation scripts for analysis outputs

## What is not included by default
Large raw and processed datasets are intentionally excluded from version control.

Expected local data layout:
- `data/raw/`
- `data/processed/`

Most scripts resolve paths relative to the repository root, so this project can run on other machines without hardcoded local paths.

## Quick start
1. Create a Python environment (Python 3.10+ recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place required datasets under `data/raw/` and/or `data/processed/`.
4. Run a smoke test:

```bash
python quick_test_train.py
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
