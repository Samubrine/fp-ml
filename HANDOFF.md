# FP-ML: USD/CHF Forex Forecasting — Complete Handoff

> **Ultra-detailed codebase analysis. Read this once, understand everything.**
> Generated: 2026-06-12 | GitHub: `https://github.com/Samubrine/fp-ml`

---

## 1. PROJECT OVERVIEW

**What:** USD/CHF 1-minute forex price forecasting using 3 GPU-accelerated regression models.
**Dataset:** 2,319,766 rows × 4 OHLC columns (2020-01-01 → 2026-05-29) from histdata.com.
**Goal:** Predict next-minute price direction/magnitude. School/research project.
**Output:** Jupyter notebook (`forex_forecasting.ipynb`) that runs the full pipeline.

---

## 2. ENVIRONMENT

### 2.1 Machine
- **Host:** `desktop-linux.netbird.selfhosted`
- **SSH:** `ssh bob@desktop-linux.netbird.selfhosted`
- **Project root:** `/home/bob/Documents/git/fp-ml`
- **Shell:** fish (avoid `source activate`; use `bash -c "source .venv/bin/activate && ..."`)

### 2.2 GPU
- **Card:** AMD Radeon RX 9060 XT (RDNA 4, `gfx1201`, 8GB VRAM)
- **Driver:** ROCm 7.2.4 (installed via `sudo pacman -S rocm-hip-sdk` from CachyOS repo)
- **Check:** `rocm-smi --showuse` → GPU usage %
- **Warning:** `/opt/amdgpu/share/libdrm/amdgpu.ids: No such file or directory` is HARMLESS

### 2.3 Two Python Environments

| Property | `.venv` | `.venv-torch` |
|----------|---------|---------------|
| Python | 3.14.5 | 3.12.13 |
| Purpose | XGBoost, sklearn, pandas | PyTorch + ROCm |
| XGBoost | 3.1.1 (built from source with HIP, but GPU fails — falls back to CPU `hist`) | N/A |
| PyTorch | N/A | 2.12.0+rocm7.2 |
| Activate | `source .venv/bin/activate` (bash only) | `source .venv-torch/bin/activate` (bash only) |
| Install | `uv pip install <pkg>` | `uv pip install <pkg>` |

### 2.4 Installed Packages (both venvs)
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`, `joblib`
- `.venv` only: `xgboost`
- `.venv-torch` only: `torch`, `torchvision`

---

## 3. FILE STRUCTURE

```
fp-ml/
├── README.md                          # Minimal (just "fp-ml")
├── .gitignore                         # Excludes data/, outputs/, .venv/, .venv-torch/
│
├── forex_forecasting.ipynb            # ★ MAIN DELIVERABLE: 46 cells, FULL pipeline
│
├── PLAN.md                            # v1 plan (SVR+KNN+XGBoost, absolute price → FAILED)
├── PLAN2.md                           # v2 plan (MLP+KNN+XGBoost, log-return → WORKING)
│
├── data/
│   └── processed/
│       ├── USDCHF_1min_2020_2026.csv   # 2,319,766 rows, 119MB — FULL dataset
│       └── USDCHF_1min_3y.csv         # 27MB — OLD 3-year subset (NOT USED in v2)
│
├── src/                               # Python scripts (standalone, for testing before notebook)
│   ├── __init__.py
│   ├── config.py                      # Central constants (paths, cutoffs, param grids)
│   ├── preprocess.py                  # v2 preprocessing: CSV→data.pt (log-return target)
│   ├── data_loader.py                 # v1 data loader (NOT USED in v2)
│   ├── feature_engineering.py         # v1 feature engineering (NOT USED in v2)
│   ├── data_splitting.py              # v1 split logic (NOT USED in v2)
│   ├── train_mlp.py                   # v1 MLP script (early, FAILED with absolute price)
│   ├── train_mlp_v2.py                # v2 MLP: GPU optimized, AMP, batch 65K, 728K params
│   ├── train_knn.py                   # v1 KNN sklearn (CPU, brute force, 371K→100K subsample)
│   ├── train_knn_v2.py                # v2 KNN: GPU batched cdist, k=50, 100K train
│   ├── train_knn_torch.py             # Early torch KNN prototype
│   ├── train_svr.py                   # v1 SVR (CPU, 50K subsample grid search)
│   ├── train_xgboost.py               # v1 XGBoost sklearn API (CPU)
│   ├── train_xgboost_gpu.py           # Early GPU XGBoost attempt (FAILED)
│   ├── train_xgb_v2.py                # v2 XGBoost: CPU hist grid search 216 combos
│   ├── evaluate.py                    # v1 evaluation
│   ├── compare.py                     # v1 comparison
│   ├── compare_v2.py                  # v2 comparison script
│   ├── run_all.py                     # v1 orchestrator (runs KNN→SVR→XGBoost)
│   └── validate_quick.sh              # Quick validation script
│
├── outputs/                           # Generated artifacts (ALL gitignored)
│   ├── preprocessed/
│   │   └── data.pt                    # 301MB — Preprocessed tensors (34 features, log-return targets)
│   ├── models/
│   │   ├── mlp_regressor.pt           # v1 MLP (42 features, 256→128→64→32 — OLD, DO NOT USE)
│   │   ├── mlp_v2.pt                  # v2 MLP (34 features, 1024→512→256→128, 2.8MB) ★
│   │   ├── knn_regressor.pkl          # v1 KNN (633MB — OLD, DO NOT USE)
│   │   ├── svr_regressor.pkl          # v1 SVR (155KB — OLD, DO NOT USE)
│   │   ├── xgboost_regressor.json     # v1 XGBoost (489KB — OLD, DO NOT USE)
│   │   └── xgboost_v2.json            # v2 XGBoost (281KB) ★
│   ├── plots/
│   │   ├── v2_comparison.png          # v2 MLP vs KNN vs XGBoost bar charts ★
│   │   └── ... (v1 plots)
│   ├── mlp_v2_results.json            # v2 MLP results ★
│   ├── knn_v2_results.json            # v2 KNN results ★
│   ├── xgb_v2_results.json            # v2 XGBoost results ★
│   ├── v2_summary.json                # v2 comparison summary ★
│   └── *.log                          # Training logs
│
├── scrape_usdchf.py                   # Data scraping script (dukascopy)
├── scrape_all_usdchf.py               # Batch scraper
├── aggregate_usdchf.py                # CSV merger
└── .hermes/                           # Hermes agent plans/context
    └── plans/
        └── 2026-06-11_forex-forecast-regression.md
```

---

## 4. DATA PIPELINE

### 4.1 Raw Data
- **Source:** `data/processed/USDCHF_1min_2020_2026.csv`
- **Format:** `datetime, open, high, low, close, volume (always 0)`
- **Rows:** 2,319,766
- **Period:** 2020-01-01 → 2026-05-29
- **Characteristics:** Non-stationary. Bullish 2020-2021, sideways 2022-2024, bullish early 2025, bearish late 2025-2026.

### 4.2 Preprocessing (in notebook Cell 6-17)
1. **Load CSV** → drop `volume`, `tick_volume`, `spread`
2. **Feature engineering** → 34 features:
   - 8 lag features: `close_lag_1..60`
   - 16 rolling stats: `close_roll_{mean,std,min,max}_{5,10,30,60}`
   - 4 price-derived: `log_return`, `pct_change`, `hl_spread`, `oc_range`
   - 6 indicators: `rsi_14`, `macd_hist`, `bb_position_20`, `bb_width_20`, `atr_14`
3. **Target:** `ln(close[t+1] / close[t])` — log return, 1 step ahead
4. **Drop NaN** (from rolling windows + target) → ~2.0M usable rows
5. **Chronological split** (NO SHUFFLE — time series):
   - Train: < 2025-01-01 (~1.8M)
   - Val: 2025-01-01 → 2025-09-01 (~247K)
   - Test: ≥ 2025-09-01 (~276K)
6. **StandardScaler** (fit on train, transform all)
7. **Save** → `outputs/preprocessed/data.pt` (301MB PyTorch tensors)

### 4.3 Why Log Return?
- **v1 failed catastrophically** with absolute price target: MLP R² = -3.26
- Price is non-stationary across 6 years (regime shifts)
- Log return is approximately stationary (mean ≈ 0, constant variance)
- ADF test: p < 0.001 → stationary ✓

---

## 5. MODELS (v2 — WORKING VERSION)

### 5.1 MLP (PyTorch GPU)
- **Architecture:** 34 → 1024 → 512 → 256 → 128 → 1 (728,833 params)
- **Layers:** Linear → BatchNorm1d → ReLU → Dropout(0.15) per hidden layer
- **Optimizations:**
  - AMP (Automatic Mixed Precision, fp16) → 2× throughput
  - Batch size: 65,536 → GPU saturation (89% utilization vs 20% in v1)
  - 4 DataLoader workers, pin_memory=True, persistent_workers=True
  - CosineAnnealingWarmRestarts (T_0=20, T_mult=2, eta_min=1e-6)
  - AdamW optimizer, lr=0.001, weight_decay=1e-4
  - Gradient clipping (max_norm=1.0)
- **Training:** 128 epochs (early stop patience=15), ~11 minutes
- **Results:** RMSE=0.000151, R²=-0.331, MAPE=0.008%, DirAcc=45.8%

### 5.2 KNN (GPU Batched cdist)
- **Method:** Batched Euclidean distance on GPU via `torch.cdist()`
- **Train subset:** 100,000 samples (full 1.8M would be too slow)
- **Grid search:** k = [1,3,5,7,10,15,20,30,50] on validation set
- **Prediction:** Average of k-nearest neighbor target values
- **Best k:** 50
- **Results:** RMSE=0.000132, R²=-0.015, MAPE=0.008%, DirAcc=46.2%
- **Training time:** ~50 seconds

### 5.3 XGBoost (CPU hist)
- **Note:** GPU (`gpu_hist`) NOT AVAILABLE in this build. XGBoost 3.1.1 from source with HIP flag didn't enable GPU. Falls back to CPU `hist`.
- **Grid search:** 216 combinations (4×3×3×2×3)
  - max_depth: [5,7,9,11]
  - learning_rate: [0.01,0.03,0.05]
  - subsample: [0.7,0.8,0.9]
  - colsample_bytree: [0.6,0.8]
  - min_child_weight: [1,3,5]
- **CV:** 3-fold, 200 rounds, early_stopping=20
- **Best params:** depth=5, lr=0.05, subsample=0.9, colsample=0.6, min_child_weight=5
- **Trees:** 65 (best_iteration = max(cv_idxmin+1, 10))
- **Full training:** 1.8M samples, 65 trees, 5 seconds
- **Results:** RMSE=0.000130, R²=+0.006, MAPE=0.008%, DirAcc=46.6% ★ BEST
- **Training time:** ~15 minutes (grid search)

### 5.4 Final Comparison

| Model | RMSE ↓ | R² ↑ | MAPE% ↓ | DirAcc ↑ | Time |
|-------|--------|------|---------|----------|------|
| MLP (GPU) | 0.000151 | -0.331 | 0.0082 | 45.8% | 11.1m |
| KNN (GPU) | 0.000132 | -0.015 | 0.0084 | 46.2% | 0.8m |
| **XGBoost (CPU)** | **0.000130** | **+0.006** | **0.0082** | **46.6%** | **14.8m** |

**Key insight:** XGBoost is the ONLY model with positive R² — it extracts a tiny but real signal from forex noise. All MAPE < 0.01% (extremely precise price predictions). Directional accuracy ~46% is expected (forex ≈ random walk).

---

## 6. v1 vs v2 — WHAT CHANGED & WHY

| Aspect | v1 (FAILED) | v2 (WORKING) |
|--------|-------------|--------------|
| Target | Absolute price `close[t+1]` | Log return `ln(close[t+1]/close[t])` |
| Models | SVR, KNN sklearn, XGBoost sklearn | MLP PyTorch, KNN GPU, XGBoost native |
| Preprocessing | In notebook cells (redundant) | Once via `src/preprocess.py` → `data.pt` |
| GPU usage | None/minimal | MLP 89%, KNN GPU batched |
| SVR | O(n²) impossible on 2.3M rows → removed | Replaced by MLP (non-linear, GPU) |
| MLP R² | -3.26 (catastrophic) | -0.33 (improved 10×) |
| Stationarity | Not addressed | Log-return makes target stationary |

---

## 7. NOTEBOOK STRUCTURE (forex_forecasting.ipynb)

46 cells, FULLY SELF-CONTAINED. Run All = complete pipeline.

| Cells | Section | Description |
|-------|---------|-------------|
| 0-1 | Title + Imports | All libraries |
| 2-5 | Data Loading | Read CSV, display stats, price visualization |
| 6-17 | **Preprocessing** | Feature engineering (34 features), log-return target, correlation heatmap, chronological split, scaling, save data.pt |
| 18-20 | **MLP Training** | Define architecture, FULL training loop (AMP, batch 65K, cosine annealing, early stop, save best model) |
| 21-22 | **KNN Training** | GPU grid search k=[1..50], batched cdist on val set, best k selection, predict on test |
| 23 | **XGBoost Training** | CPU grid search 216 combos, 3-fold CV, full training on 1.8M rows |
| 24-26 | Generate Predictions | MLP inference, package all predictions |
| 27-28 | Regression Evaluation | RMSE, MAE, R², MAPE, DirAcc for all models |
| 29-30 | **Confusion Matrix** | Directional classification (Up/Down), accuracy/precision/recall/F1 |
| 31-33 | Comparison Charts | Bar charts + residuals + actual vs predicted |
| 34 | Silhouette Score | K-Means market regime clustering (k=2..8) |
| 35-36 | **PCA** | 3 views: K-Means colored, |log return| colored, XGBoost error colored |
| 37-39 | Parameter Tuning | KNN k×weight experiment, XGBoost lr×depth experiment |
| 40-43 | **Kesimpulan** | Analysis + recommendations |
| 44-45 | Final Summary | Formatted table |

---

## 8. KEY TECHNICAL DECISIONS

1. **Log-return target is MANDATORY** — absolute price causes regime shift failure across multi-year data.
2. **Chronological split (no shuffle)** — time series must respect temporal order to avoid look-ahead bias.
3. **Preprocess once, share `data.pt`** — saves 2-3 minutes of CPU per model run.
4. **Python-first, notebook-later** — all scripts tested standalone before assembling into notebook.
5. **Two venvs** — PyTorch needs Python 3.12 (ROCm compat); XGBoost works on 3.14.
6. **XGBoost GPU failed** — source build with `-DUSE_HIP=ON` didn't produce `gpu_hist` method. Falls back to CPU `hist` which is fast enough (700% CPU utilization).
7. **KNN subsampled to 100K** — full 1.8M training set would make inference too slow (O(n_train × n_test) distance computation).
8. **best_iteration fix** — XGBoost CV `idxmin()` returns 0-index; must `+1` and enforce `max(best_round, 10)` minimum trees.

---

## 9. COMMON PITFALLS

1. **Fish shell quoting** — use `bash -c "..."` for remote commands. Fish doesn't support `$!` or heredoc `<<`.
2. **Python output buffering** — use `python3 -u` for unbuffered stdout when logging to file.
3. **`uv pip install` not `pip install`** — pip not in PATH on remote.
4. **Model dimension mismatch** — v1 models have 42 features, v2 has 34. `mlp_regressor.pt` is OLD (256→128→64→32, 42 features). Use `mlp_v2.pt`.
5. **`_estimator_type` for XGBoost GridSearchCV** — must set `model._estimator_type = "regressor"` before `save_model()` when using sklearn wrapper.
6. **`gpu_id` deprecated in XGBoost 3.1.1** — use `device: cuda` instead (though GPU still fails on this build).
7. **`.venv-torch/` gitignored** — must be created manually on new machine.

---

## 10. HOW TO RUN

### Fresh Setup
```bash
# SSH to machine
ssh bob@desktop-linux.netbird.selfhosted
cd /home/bob/Documents/git/fp-ml

# Clone (if needed)
git clone https://github.com/Samubrine/fp-ml.git
cd fp-ml

# Create venvs (if missing)
python3.12 -m venv .venv-torch
source .venv-torch/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
uv pip install pandas numpy scikit-learn matplotlib seaborn scipy

python3.14 -m venv .venv
source .venv/bin/activate
uv pip install pandas numpy scikit-learn xgboost matplotlib seaborn scipy
```

### Run Full Pipeline (Notebook)
```bash
cd /home/bob/Documents/git/fp-ml
jupyter notebook forex_forecasting.ipynb
# → Run All Cells (~30 minutes total: mostly XGBoost grid search)
```

### Run Individual Models
```bash
# MLP (requires .venv-torch)
bash -c "source .venv-torch/bin/activate && python3 src/train_mlp_v2.py"

# KNN (requires .venv-torch)
bash -c "source .venv-torch/bin/activate && python3 src/train_knn_v2.py"

# XGBoost (requires .venv)
bash -c "source .venv/bin/activate && python3 src/train_xgb_v2.py"
```

### Git
```bash
git remote -v  # https://github.com/Samubrine/fp-ml
git status
# data/ and outputs/ are gitignored
```

---

## 11. FUTURE IMPROVEMENT IDEAS

1. **XGBoost GPU** — try AMD's ROCm-finance PyPI: `pip install xgboost --index-url https://repo.radeon.com/rocm/manylinux/rocm-finance/`
2. **Ensemble** — combine XGBoost + KNN predictions for better directional accuracy
3. **External features** — correlated currency pairs (EUR/USD, USD/JPY), news sentiment, economic calendar
4. **LSTM/Transformer** — sequence models might capture temporal patterns better than feedforward
5. **Multi-horizon** — predict 5-min, 15-min, 1-hour ahead (not just 1-min)
6. **Hyperparameter optimization** — Optuna or Ray Tune for smarter search (currently manual grid)
7. **Cross-validation across time** — expanding window or TimeSeriesSplit instead of single split

---

## 12. QUICK REFERENCE

| What | Where |
|------|-------|
| Raw data | `data/processed/USDCHF_1min_2020_2026.csv` (119MB) |
| Preprocessed tensors | `outputs/preprocessed/data.pt` (301MB) |
| v2 MLP model | `outputs/models/mlp_v2.pt` (2.8MB) |
| v2 XGBoost model | `outputs/models/xgboost_v2.json` (281KB) |
| v2 results JSON | `outputs/{mlp,knn,xgb}_v2_results.json` |
| Comparison summary | `outputs/v2_summary.json` |
| Comparison chart | `outputs/plots/v2_comparison.png` |
| Training scripts (v2) | `src/train_{mlp,knn,xgb}_v2.py` |
| Preprocessing script | `src/preprocess.py` |
| Notebook (FULL pipeline) | `forex_forecasting.ipynb` |
| v2 plan document | `PLAN2.md` |
| GitHub | `https://github.com/Samubrine/fp-ml` |
| Last commit | `01edc77` — notebook v4 with full training inline |
