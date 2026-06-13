# FP-ML — Reasonix workspace guide

## Stack
- **Language:** Python 3.12 (PyTorch venv) / Python 3.14 (XGBoost venv)
- **ML:** PyTorch 2.x + ROCm (AMD GPU), XGBoost 3.x (CPU hist), scikit-learn
- **Data:** pandas, numpy, matplotlib, seaborn, mplfinance, scipy, statsmodels
- **GPU:** AMD Radeon RX 9060 XT (ROCm 7.2.4, RDNA 4 `gfx1201`)
- **Notebook:** Jupyter (`forex_forecasting.ipynb` is the main deliverable)

## Layout
- `forex_forecasting.ipynb` — **main deliverable**, self-contained full pipeline
- `src/` — active v2 model scripts: `preprocess.py`, `train_mlp.py`, `train_knn.py`, `train_xgboost.py`, `compare.py`
- `scripts/` — data scraping utilities (`scrape_usdchf.py`, `aggregate_usdchf.py`)
- `data/processed/` — raw CSV (gitignored, ~119MB), `data/processed/USDCHF_1min_2020_2026.csv`
- `outputs/` — models, plots, metrics, preprocessed tensors (gitignored)
- `docs/HANDOFF.md` — exhaustive 343-line codebase reference
- `archive/v1/` + `archive/prototypes/` — failed experiments, kept for reference
- `PLAN.md` — 853-line notebook implementation spec

## Commands
- **Preprocess:** `bash -c "source .venv-torch/bin/activate && python3 -u src/preprocess.py"`
- **MLP:** `bash -c "source .venv-torch/bin/activate && python3 -u src/train_mlp.py"`
- **KNN:** `bash -c "source .venv-torch/bin/activate && python3 -u src/train_knn.py"`
- **XGBoost:** `bash -c "source .venv/bin/activate && python3 -u src/train_xgboost.py"`
- **Notebook:** `jupyter notebook forex_forecasting.ipynb` (Run All = full pipeline)

## Conventions
- **Two venvs:** `.venv` (Python 3.14, XGBoost/sklearn) and `.venv-torch` (Python 3.12, PyTorch+ROCm) — shell is fish, so venv activation must use `bash -c "source …/bin/activate && …"`
- **Shared preprocess:** `src/preprocess.py` → `outputs/preprocessed/data.pt` (301MB) consumed by all three models — run once, never re-run unless features change
- **Target:** log-return `ln(close[t+1] / close[t])` (stationary), **not** absolute price
- **Split:** chronological only (no shuffle) — Train < 2025, Val 2025H1, Test ≥ 2025-09
- **Seed:** `torch.manual_seed(42)` throughout
- **Shebang:** all `src/` scripts use `#!/usr/bin/env python3`

## Watch out for
- `src/config.py` and `src/evaluate.py` are listed in `README.md` but **don't exist on disk** — the README is stale; constants are inlined in each script
- `outputs/` and `data/` are in `.gitignore` and **large** (119MB CSV, 301MB tensor) — don't `read_file` / `search_content` them unless explicitly asked
- `PLAN.md` is a notebook implementation spec (853 lines), not an architecture doc — it describes what each notebook cell should do, not what the code currently does
