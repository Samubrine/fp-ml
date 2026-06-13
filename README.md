# FP-ML: USD/CHF Forex Forecasting

> **Final Project — Machine Learning for Forex Prediction**

Predict next-minute USD/CHF price direction using three GPU-accelerated regression models trained on 2.3M rows of 1-minute OHLC data (2020–2026).

## Quick Start

```bash
# Open the notebook (self-contained, Run All = full pipeline)
jupyter notebook forex_forecasting.ipynb
```

The notebook is the **main deliverable** — everything from data loading to model evaluation runs inline. Training takes ~30 minutes total (mostly XGBoost grid search).

## Models (v2)

| Model | Framework | GPU | RMSE | R² | DirAcc |
|-------|-----------|-----|------|-----|--------|
| MLP | PyTorch + ROCm | ✅ | 0.000151 | -0.331 | 45.8% |
| KNN | PyTorch cdist (batched) | ✅ | 0.000132 | -0.015 | 46.2% |
| XGBoost | Native (CPU hist) | ❌ | 0.000130 | +0.006 | 46.6% |

Target: log return `ln(close[t+1] / close[t])` — stationary across multi-year data.

## Environment

- **GPU:** AMD Radeon RX 9060 XT (ROCm 7.2.4)
- **Two venvs:** `.venv` (Python 3.14, XGBoost/sklearn) + `.venv-torch` (Python 3.12, PyTorch+ROCm)
- **Shell:** fish (use `bash -c` for venv activation)

## Repository Structure

```
fp-ml/
├── forex_forecasting.ipynb      # ★ Main deliverable — full pipeline
├── README.md
│
├── docs/
│   └── HANDOFF.md               # Detailed codebase reference
│
├── src/                         # Active v2 code (tested before notebook)
│   ├── config.py                # Central constants
│   ├── preprocess.py            # Data pipeline: CSV → data.pt
│   ├── train_mlp.py             # PyTorch MLP (1024→512→256→128)
│   ├── train_knn.py             # GPU batched KNN (k=50)
│   ├── train_xgboost.py         # XGBoost grid search (216 combos)
│   └── evaluate.py              # Metrics + comparison
│
├── archive/                     # v1 code (kept for reference)
│   ├── v1/                      # Failed v1 (absolute price target)
│   └── prototypes/              # Failed experiments
│
├── scripts/                     # Data scraping utilities
│
├── data/                        # Raw CSV (gitignored, 119MB)
└── outputs/                     # Models, plots, results (gitignored)
```

## Running Individual Models

```bash
# MLP (needs .venv-torch)
bash -c "source .venv-torch/bin/activate && python3 -u src/train_mlp.py"

# KNN (needs .venv-torch)
bash -c "source .venv-torch/bin/activate && python3 -u src/train_knn.py"

# XGBoost (needs .venv)
bash -c "source .venv/bin/activate && python3 -u src/train_xgboost.py"
```

## Key Technical Decisions

1. **Log-return target** — absolute price fails on non-stationary multi-year data (v1 R² = -3.26 → v2 = +0.006)
2. **Chronological split** — no shuffle (time series)
3. **Preprocess once** — `data.pt` (301MB) shared by all models
4. **CPU XGBoost** — GPU build failed, but `hist` is plenty fast

## Dataset

- **Source:** [histdata.com](https://www.histdata.com) — USD/CHF 1-minute
- **Period:** 2020-01-01 → 2026-05-29
- **Features:** 34 (lags, rolling stats, indicators)
- **Split:** Train <2025 (~1.8M) / Val 2025H1 (~247K) / Test ≥2025-09 (~276K)

## References

- Full codebase reference: [`docs/HANDOFF.md`](docs/HANDOFF.md)
- GitHub: [Samubrine/fp-ml](https://github.com/Samubrine/fp-ml)
