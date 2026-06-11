# USD/CHF Forex Forecasting — 3 Model Regression Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Implement & compare 3 regression models (KNN Regressor, SVR, XGBoost Regressor) untuk forecasting harga USD/CHF close price, menggunakan dataset 1-minute OHLCV dari histdata.com.

**Architecture:** Modular pipeline: Data Load → Feature Engineering → Train/Val/Test Split → 3 Model Training + Tuning → Evaluation → Comparison Report.

**Tech Stack:** Python 3.14, pandas 3.0, numpy 2.4, scikit-learn, xgboost, matplotlib, seaborn, uv (package manager)

**Dataset:** `data/processed/USDCHF_1min_3y.csv` — 522,862 rows, columns: datetime, open, high, low, close, volume (all 0 → dropped). Range: 2025-01-01 to 2026-05-29.

---

## Environment

- **Project root:** `/home/bob/Documents/git/fp-ml`
- **Venv:** `.venv/` (uv-managed, Python 3.14)
- **Remote access:** `ssh bob@desktop-linux.netbird.selfhosted`
- **Shell:** fish (use `.venv/bin/python3` directly, NOT `source .venv/bin/activate`)

---

## File Map (Final Structure)

```
fp-ml/
├── data/
│   └── processed/
│       └── USDCHF_1min_3y.csv          # Input dataset
├── src/
│   ├── __init__.py
│   ├── config.py                       # Constants: paths, hyperparam ranges
│   ├── data_loader.py                  # Load & validate CSV
│   ├── feature_engineering.py          # Lag, rolling, indicators, time features
│   ├── data_splitting.py               # Chronological train/val/test split
│   ├── train_knn.py                    # KNN Regressor training + tuning
│   ├── train_svr.py                    # SVR training + tuning (with sampling)
│   ├── train_xgboost.py                # XGBoost training + tuning
│   ├── evaluate.py                     # Shared evaluation: RMSE, MAE, MAPE, plots
│   ├── compare.py                      # Model comparison table + charts
│   └── run_all.py                      # Master pipeline runner
├── outputs/
│   ├── models/                         # Saved model pickle files
│   ├── plots/                          # Prediction vs actual, residuals
│   └── metrics/                        # JSON/CSV metric tables
└── .hermes/
    └── plans/
        └── 2026-06-11_forex-forecast-regression.md  # THIS FILE
```

---

# PHASE 0: ENVIRONMENT & PROJECT SETUP

---

## Task 0.1: Install ML dependencies via uv

**Action:**
```bash
cd /home/bob/Documents/git/fp-ml && .venv/bin/uv pip install scikit-learn xgboost matplotlib seaborn
```

**Verify:**
```bash
.venv/bin/python3 -c "import sklearn; import xgboost; import matplotlib; import seaborn; print('ALL OK')"
```

**Expected:** `ALL OK`

---

## Task 0.2: Create project directories

**Action:**
```bash
mkdir -p /home/bob/Documents/git/fp-ml/src
mkdir -p /home/bob/Documents/git/fp-ml/outputs/models
mkdir -p /home/bob/Documents/git/fp-ml/outputs/plots
mkdir -p /home/bob/Documents/git/fp-ml/outputs/metrics
```

**Verify:**
```bash
ls -d /home/bob/Documents/git/fp-ml/src /home/bob/Documents/git/fp-ml/outputs/*
```

---

## Task 0.3: Create `src/__init__.py`

**Create:** `src/__init__.py`
```python
"""USD/CHF Forex Forecasting — ML Regression Models."""
```

**Verify:**
```bash
cat /home/bob/Documents/git/fp-ml/src/__init__.py
```

---

# PHASE 1: CONFIG FILE

---

## Task 1.1: Create `src/config.py`

**Create:** `src/config.py` — Centralised constants and paths.

```python
"""Centralised constants and paths for the USD/CHF forecasting project."""
import os

# ── Paths ──────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
METRIC_DIR = os.path.join(OUTPUT_DIR, "metrics")

INPUT_CSV = os.path.join(DATA_DIR, "USDCHF_1min_3y.csv")

# ── Data ───────────────────────────────────────────────────────────────
TRAIN_CUTOFF = "2026-01-01"           # ~78% train
VAL_CUTOFF = "2026-03-15"             # ~11% val  (rest=test)
TARGET_COL = "close"                  # What we're predicting
DROP_COLS = ["volume"]                # All-zero in our dataset

# ── Feature Engineering ────────────────────────────────────────────────
LAG_PERIODS = [1, 2, 3, 5, 10, 15, 30, 60]  # Minutes back
ROLLING_WINDOWS = [5, 10, 30, 60]   # Minutes for rolling stats
ROLLING_STATS = ["mean", "std", "min", "max"]
LOOKAHEAD = 1  # Predict close at t+LOOKAHEAD (1-minute ahead)

# ── KNN ────────────────────────────────────────────────────────────────
KNN_PARAM_GRID = {
    "n_neighbors": [3, 5, 10, 20, 50, 100],
    "weights": ["uniform", "distance"],
    "p": [1, 2],  # 1=manhattan, 2=euclidean
}

# ── SVR ────────────────────────────────────────────────────────────────
SVR_SAMPLE_SIZE = 50_000              # SVR O(n²) — must subsample
SVR_PARAM_GRID = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    "epsilon": [0.001, 0.01, 0.1],
}

# ── XGBoost ────────────────────────────────────────────────────────────
XGB_PARAM_GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.1, 0.3],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}

# ── General ────────────────────────────────────────────────────────────
RANDOM_SEED = 42
CV_FOLDS = 3  # TimeSeriesSplit — keep small for speed
```

**Verify:**
```bash
cd /home/bob/Documents/git/fp-ml && .venv/bin/python3 -c "from src.config import *; print('Config OK'); print('INPUT_CSV:', INPUT_CSV)"
```

**Expected:** `Config OK` with correct path.

---

# PHASE 2: DATA LOADER

---

## Task 2.1: Create `src/data_loader.py`

**Create:** `src/data_loader.py`

```python
"""Load, validate, and return the USD/CHF 1-minute dataset."""
import pandas as pd
from src.config import INPUT_CSV, DROP_COLS, TARGET_COL


def load_data(csv_path: str = INPUT_CSV) -> pd.DataFrame:
    """Load CSV, parse datetime index, drop useless columns, validate.

    Returns:
        DataFrame with DatetimeIndex, sorted ascending.
    """
    # 1. Read CSV
    df = pd.read_csv(csv_path)

    # 2. Parse datetime & set as index
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    # 3. Drop columns that add no value
    df.drop(columns=DROP_COLS, inplace=True, errors="ignore")

    # 4. Validate — no NaNs in critical columns
    assert df.isnull().sum().sum() == 0, "NaN values found in dataset"
    assert len(df) > 100_000, "Dataset too small: {n} rows".format(n=len(df))

    # 5. Ensure numeric types
    for c in [TARGET_COL, "open", "high", "low"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(inplace=True)

    return df


if __name__ == "__main__":
    df = load_data()
    print("Loaded: {n:,} rows".format(n=len(df)))
    print("Range:  {start}  →  {end}".format(start=df.index.min(), end=df.index.max()))
    print("Nulls:  {n}".format(n=df.isnull().sum().sum()))
    print(df.head())
```

**Verify:**
```bash
cd /home/bob/Documents/git/fp-ml && .venv/bin/python3 -m src.data_loader
```

**Expected:**
```
Loaded: 522,862 rows
Range:  2025-01-01 17:04:00  →  2026-05-29 16:58:00
Nulls:  0
```

---

# PHASE 3: FEATURE ENGINEERING

---

## Task 3.1: Create `src/feature_engineering.py` — Core Structure

**Create:** `src/feature_engineering.py` — all feature functions in one module.

```python
"""Feature engineering pipeline for USD/CHF 1-min data.

Builds: lag features, rolling statistics, technical indicators (RSI, MACD, BB),
price-derived features (returns, spreads), time-based cyclical features,
and the target variable (close at t+LOOKAHEAD).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config import LAG_PERIODS, ROLLING_WINDOWS, ROLLING_STATS, TARGET_COL, LOOKAHEAD


# ── 3a. Lag Features ──────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lagged close price features.

    Example: close_lag_1 = close at t-1, close_lag_5 = close at t-5
    """
    for lag in LAG_PERIODS:
        df["close_lag_{lag}".format(lag=lag)] = df[TARGET_COL].shift(lag)
    return df


# ── 3b. Rolling Window Features ───────────────────────────────────────

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling statistics of close price.

    For each window in ROLLING_WINDOWS, compute each stat in ROLLING_STATS.
    """
    for window in ROLLING_WINDOWS:
        roll = df[TARGET_COL].rolling(window=window)
        for stat in ROLLING_STATS:
            col_name = "close_roll_{stat}_{window}".format(stat=stat, window=window)
            df[col_name] = getattr(roll, stat)()
    return df


# ── 3c. Price-Derived Features ────────────────────────────────────────

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived price features: returns, spreads, ratios."""
    # Log return (close vs prev close)
    df["log_return"] = np.log(df[TARGET_COL] / df[TARGET_COL].shift(1))

    # Simple return (fractional change)
    df["pct_change"] = df[TARGET_COL].pct_change()

    # High-Low spread
    df["hl_spread"] = df["high"] - df["low"]

    # Open-Close range
    df["oc_range"] = df[TARGET_COL] - df["open"]

    # OHLC mean (central tendency)
    df["ohlc_mean"] = (df["open"] + df["high"] + df["low"] + df[TARGET_COL]) / 4

    return df


# ── 3d. Technical Indicators ──────────────────────────────────────────

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Relative Strength Index (RSI) — momentum oscillator.

    RSI = 100 - (100 / (1 + RS)), where RS = avg_gain / avg_loss
    """
    delta = df[TARGET_COL].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands: middle=MA, upper/lower = MA ± num_std * std."""
    ma = df[TARGET_COL].rolling(window=period).mean()
    std = df[TARGET_COL].rolling(window=period).std()
    df["bb_mid_20"] = ma
    df["bb_upper_20"] = ma + num_std * std
    df["bb_lower_20"] = ma - num_std * std
    df["bb_width_20"] = df["bb_upper_20"] - df["bb_lower_20"]
    df["bb_position_20"] = (df[TARGET_COL] - df["bb_lower_20"]) / df["bb_width_20"]
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """MACD: MACD line = EMA12 - EMA26, Signal = EMA9(MACD), Hist = MACD - Signal."""
    ema_12 = df[TARGET_COL].ewm(span=12, adjust=False).mean()
    ema_26 = df[TARGET_COL].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


# ── 3e. Time-Based Cyclical Features ──────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract cyclical time features from DatetimeIndex."""
    df["hour"] = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dayofweek"] = df.index.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    return df


# ── 3f. Master Pipeline ───────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Complete feature engineering pipeline.

    Builds ALL features, then drops NaN rows.

    Returns:
        DataFrame with all features + target column (close at t+LOOKAHEAD).
    """
    # 1. Target: future close price
    df["target"] = df[TARGET_COL].shift(-LOOKAHEAD)

    # 2. Feature groups (order matters — later groups may use earlier cols)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_price_features(df)
    df = add_rsi(df)
    df = add_bollinger_bands(df)
    df = add_macd(df)
    df = add_time_features(df)

    # 3. Drop rows with NaN (first ~60 rows + last LOOKAHEAD row)
    df.dropna(inplace=True)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of feature column names (exclude target + raw OHLC)."""
    exclude = {"open", "high", "low", TARGET_COL, "target",
               "hour", "dayofweek"}  # raw time cols dropped (use sin/cos)
    return [c for c in df.columns if c not in exclude]


# ── 3g. Feature Scaling ───────────────────────────────────────────────

def scale_features(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> tuple:
    """Fit StandardScaler on train, transform all splits.

    Returns:
        X_train_scaled, X_val_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
```

**Verify (each function individually):**

```bash
cd /home/bob/Documents/git/fp-ml && .venv/bin/python3 -c "
from src.data_loader import load_data
from src.feature_engineering import add_lag_features, add_rolling_features, add_price_features
from src.feature_engineering import add_rsi, add_bollinger_bands, add_macd, add_time_features
from src.feature_engineering import build_features, get_feature_columns

df = load_data()
df = build_features(df)
feats = get_feature_columns(df)

print('Rows after build:', len(df))
print('Feature count:', len(feats))
print('Nulls:', df.isnull().sum().sum())
print('Target nulls:', df['target'].isnull().sum())
print('Features:', feats[:5], '...', feats[-3:])
print()
print('Sample:\n', df[['close', 'target'] + feats[:3]].head())
"
```

**Expected:**
```
Rows after build: ~522,700
Feature count: 48-50
Nulls: 0
Target nulls: 0
```

---

# PHASE 4: DATA SPLITTING

---

## Task 4.1: Create `src/data_splitting.py`

**Create:** `src/data_splitting.py`

```python
"""Chronological train/validation/test split for time-series data."""
import numpy as np
from src.config import TRAIN_CUTOFF, VAL_CUTOFF


def split_data(df):
    """Split into train/val/test preserving temporal order.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names
    """
    from src.feature_engineering import get_feature_columns

    feature_cols = get_feature_columns(df)

    # Chronological split by date
    train_mask = df.index < TRAIN_CUTOFF
    val_mask = (df.index >= TRAIN_CUTOFF) & (df.index < VAL_CUTOFF)
    test_mask = df.index >= VAL_CUTOFF

    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, "target"].values
    X_val = df.loc[val_mask, feature_cols].values
    y_val = df.loc[val_mask, "target"].values
    X_test = df.loc[test_mask, feature_cols].values
    y_test = df.loc[test_mask, "target"].values

    print("Train: {n:,} rows ({pct:.0f}%)".format(
        n=len(X_train), pct=train_mask.sum() / len(df) * 100))
    print("Val:   {n:,} rows ({pct:.0f}%)".format(
        n=len(X_val), pct=val_mask.sum() / len(df) * 100))
    print("Test:  {n:,} rows ({pct:.0f}%)".format(
        n=len(X_test), pct=test_mask.sum() / len(df) * 100))

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


if __name__ == "__main__":
    from src.data_loader import load_data
    from src.feature_engineering import build_features

    df = load_data()
    df = build_features(df)
    X_tr, X_v, X_te, y_tr, y_v, y_te, feats = split_data(df)
    print("Feature count:", len(feats))
    print("y_train range: {min:.5f} – {max:.5f}".format(min=y_tr.min(), max=y_tr.max()))
    print("y_test  range: {min:.5f} – {max:.5f}".format(min=y_te.min(), max=y_te.max()))
```

**Verify:**
```bash
cd /home/bob/Documents/git/fp-ml && .venv/bin/python3 -m src.data_splitting
```

**Expected:**
```
Train: ~410,000 rows (78%)
Val:   ~56,000 rows (11%)
Test:  ~56,000 rows (11%)
y_train range: 0.76xxx – 0.91xxx
```

---

# PHASE 5: KNN REGRESSOR

---

## Task 5.1: Create `src/train_knn.py`

**Create:** `src/train_knn.py`

```python
"""K-Nearest Neighbors Regressor for USD/CHF close price forecasting."""
import pickle
import time
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from src.config import KNN_PARAM_GRID, RANDOM_SEED, MODEL_DIR, CV_FOLDS


def train_knn(X_train_scaled: np.ndarray, y_train: np.ndarray) -> KNeighborsRegressor:
    """Train KNN Regressor with GridSearchCV.

    Args:
        X_train_scaled: Scaled feature matrix. SCALING REQUIRED.
        y_train: Target values (close price at t+1).

    Returns:
        Best KNeighborsRegressor from grid search.
    """
    print("=" * 60)
    print("KNN REGRESSOR — GridSearchCV")
    print("=" * 60)

    base_model = KNeighborsRegressor(n_jobs=-1)
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

    grid = GridSearchCV(
        base_model,
        KNN_PARAM_GRID,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=2,
    )

    t0 = time.time()
    grid.fit(X_train_scaled, y_train)
    elapsed = time.time() - t0

    print("\nTraining complete in {elapsed:.1f}s".format(elapsed=elapsed))
    print("Best params:", grid.best_params_)
    print("Best RMSE (CV): {score:.6f}".format(score=-grid.best_score_))

    # Save model
    model_path = MODEL_DIR + "/knn_regressor.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(grid.best_estimator_, f)
    print("Saved:", model_path)

    return grid.best_estimator_


if __name__ == "__main__":
    from src.data_loader import load_data
    from src.feature_engineering import build_features, scale_features
    from src.data_splitting import split_data

    df = load_data()
    df = build_features(df)
    X_tr, X_v, X_te, y_tr, y_v, y_te, feats = split_data(df)
    X_tr_s, X_v_s, X_te_s, _ = scale_features(X_tr, X_v, X_te)

    model = train_knn(X_tr_s, y_tr)
```

**Verify:**
```bash
cd /home/bob/Documents/git/fp-ml && .venv/bin/python3 -m src.train_knn
```

**Expected:** Grid search runs (2-5 min on 410K rows), prints best n_neighbors, weights, p. Saves `outputs/models/knn_regressor.pkl`.

---

# PHASE 6: SVR (Support Vector Regression)

---

## Task 6.1: Create `src/train_svr.py`

**Create:** `src/train_svr.py`

```python
"""Support Vector Regression (SVR) for USD/CHF close price forecasting.

IMPORTANT: SVR has O(n²)~O(n³) complexity — we MUST subsample to ~50K rows.
"""
import pickle
import time
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from src.config import SVR_PARAM_GRID, SVR_SAMPLE_SIZE, RANDOM_SEED, MODEL_DIR, CV_FOLDS


def subsample(X: np.ndarray, y: np.ndarray, n: int = SVR_SAMPLE_SIZE,
              random_seed: int = RANDOM_SEED) -> tuple:
    """Randomly subsample to n rows, preserving price distribution via quantile bins."""
    n = min(n, len(y))
    if n >= len(y):
        return X, y

    # Bin prices into deciles
    bins = np.quantile(y, np.linspace(0, 1, 11))
    bin_indices = np.digitize(y, bins)

    rng = np.random.RandomState(random_seed)
    sample_idx = []
    rows_per_bin = n // 10

    for bin_id in range(1, 12):
        idx = np.where(bin_indices == bin_id)[0]
        if len(idx) == 0:
            continue
        take = min(rows_per_bin, len(idx))
        sample_idx.extend(rng.choice(idx, size=take, replace=False))

    # Pad to exact n if needed
    if len(sample_idx) < n:
        remaining = np.setdiff1d(np.arange(len(y)), sample_idx)
        sample_idx.extend(rng.choice(remaining, size=n - len(sample_idx), replace=False))

    sample_idx = np.array(sample_idx)
    print("Subsampled: {orig:,} → {new:,} rows (stratified by target quantiles)".format(
        orig=len(X), new=n))
    return X[sample_idx], y[sample_idx]


def train_svr(X_train_scaled: np.ndarray, y_train: np.ndarray) -> SVR:
    """Train SVR with GridSearchCV on subsampled data."""
    print("=" * 60)
    print("SVR — GridSearchCV (with subsampling)")
    print("=" * 60)

    # Subsample — O(n²) scaling makes full dataset impractical
    X_sub, y_sub = subsample(X_train_scaled, y_train)

    base_model = SVR(kernel="rbf")
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

    grid = GridSearchCV(
        base_model,
        SVR_PARAM_GRID,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=2,
    )

    t0 = time.time()
    grid.fit(X_sub, y_sub)
    elapsed = time.time() - t0

    print("\nTraining complete in {elapsed:.1f}s".format(elapsed=elapsed))
    print("Best params:", grid.best_params_)
    print("Best RMSE (CV): {score:.6f}".format(score=-grid.best_score_))

    # Save model
    model_path = MODEL_DIR + "/svr_regressor.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(grid.best_estimator_, f)
    print("Saved:", model_path)

    return grid.best_estimator_


if __name__ == "__main__":
    from src.data_loader import load_data
    from src.feature_engineering import build_features, scale_features
    from src.data_splitting import split_data

    df = load_data()
    df = build_features(df)
    X_tr, X_v, X_te, y_tr, y_v, y_te, feats = split_data(df)
    X_tr_s, X_v_s, X_te_s, _ = scale_features(X_tr, X_v, X_te)

    model = train_svr(X_tr_s, y_tr)
```

**Verify:**
```bash
cd /home/bob/Documents/git/fp-ml && .venv/bin/python3 -m src.train_svr
```

**Expected:** Subsampled to 50K, grid search runs (5-15 min), prints best C, gamma, epsilon. Saves `outputs/models/svr_regressor.pkl`.

---

# PHASE 7: XGBOOST REGRESSOR

---

## Task 7.1: Create `src/train_xgboost.py`

**Create:** `src/train_xgboost.py`

```python
"""XGBoost Regressor for USD/CHF close price forecasting."""
import time
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from src.config import XGB_PARAM_GRID, RANDOM_SEED, MODEL_DIR, CV_FOLDS


def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray = None, y_val: np.ndarray = None,
) -> xgb.XGBRegressor:
    """Train XGBoost Regressor with GridSearchCV + optional early stopping.

    Args:
        X_train: Feature matrix (scaling optional — tree-based models
                 are scale-invariant, but we pass scaled for consistency).
        y_train: Target values.
        X_val, y_val: Optional validation set for early stopping.

    Returns:
        Best XGBRegressor from grid search.
    """
    print("=" * 60)
    print("XGBOOST REGRESSOR — GridSearchCV")
    print("=" * 60)

    base_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
    )
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

    grid = GridSearchCV(
        base_model,
        XGB_PARAM_GRID,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=2,
    )

    t0 = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - t0

    print("\nTraining complete in {elapsed:.1f}s".format(elapsed=elapsed))
    print("Best params:", grid.best_params_)
    print("Best RMSE (CV): {score:.6f}".format(score=-grid.best_score_))

    # Optional: retrain best model with early stopping on val set
    if X_val is not None and y_val is not None:
        print("\nRetraining with early stopping on validation set...")
        best_params = dict(grid.best_params_)
        best_params.pop("early_stopping_rounds", None)
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=1,
            **best_params,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        model = grid.best_estimator_

    # Save model
    model_path = MODEL_DIR + "/xgboost_regressor.json"
    model.save_model(model_path)
    print("Saved:", model_path)

    # Feature importance
    importance = model.feature_importances_
    top_indices = np.argsort(importance)[-10:][::-1]
    print("\nTop 10 feature importances:")
    for i in top_indices:
        print("  feat_{idx}: {val:.4f}".format(idx=i, val=importance[i]))

    return model


if __name__ == "__main__":
    from src.data_loader import load_data
    from src.feature_engineering import build_features, scale_features
    from src.data_splitting import split_data

    df = load_data()
    df = build_features(df)
    X_tr, X_v, X_te, y_tr, y_v, y_te, feats = split_data(df)
    X_tr_s, X_v_s, X_te_s, _ = scale_features(X_tr, X_v, X_te)

    model = train_xgboost(X_tr_s, y_tr, X_v_s, y_v)
```

**Verify:**
```bash
cd /home/bob/Documents/git/fp-ml && .venv/bin/python3 -m src.train_xgboost
```

**Expected:** Grid search runs (10-30 min depending on combinations), prints best params, top 10 feature importances. Saves `outputs/models/xgboost_regressor.json`.

---

# PHASE 8: EVALUATION MODULE

---

## Task 8.1: Create `src/evaluate.py`

**Create:** `src/evaluate.py`

```python
"""Evaluation metrics and plotting for regression models."""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for SSH
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import PLOT_DIR, METRIC_DIR


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "model") -> dict:
    """Compute standard regression metrics.

    Returns:
        dict with: rmse, mae, mape, r2, directional_accuracy_pct, n_samples
    """
    # Filter out NaN predictions
    mask = ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE — avoid division by zero
    nonzero_mask = y_true != 0
    mape = np.mean(np.abs(
        (y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask]
    )) * 100

    # Directional accuracy (% of times model correctly predicts up/down)
    actual_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    dir_acc = np.mean(actual_dir == pred_dir) * 100

    metrics = {
        "model": model_name,
        "rmse": round(rmse, 8),
        "mae": round(mae, 8),
        "mape_pct": round(mape, 4),
        "r2": round(r2, 6),
        "directional_accuracy_pct": round(dir_acc, 2),
        "n_samples": len(y_true),
    }

    print("\n" + "=" * 50)
    print("  {name} — Evaluation".format(name=model_name))
    print("=" * 50)
    for k, v in metrics.items():
        print("  {key}: {val}".format(key=k, val=v))
    return metrics


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model",
    n_points: int = 500,
) -> str:
    """Plot actual vs predicted prices (time-series overlay + residuals).

    Returns:
        Path to saved PNG.
    """
    # Plot last n_points
    y_true = y_true[-n_points:]
    y_pred = y_pred[-n_points:]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(
        "{name} — Actual vs Predicted (Last {n} Points)".format(name=model_name, n=n_points),
        fontsize=14,
    )

    # Top: Overlay plot
    ax = axes[0]
    ax.plot(y_true, label="Actual", color="blue", linewidth=0.8, alpha=0.8)
    ax.plot(y_pred, label="Predicted", color="red", linewidth=0.8, alpha=0.8)
    ax.set_ylabel("Close Price")
    ax.set_title("Price Overlay")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Bottom: Residuals
    ax = axes[1]
    residuals = y_true - y_pred
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.plot(residuals, color="purple", linewidth=0.5, alpha=0.7)
    ax.set_ylabel("Residual (Actual - Pred)")
    ax.set_xlabel("Test Sample Index")
    ax.set_title(
        "Residuals (μ={mean:.6f}, σ={std:.6f})".format(
            mean=residuals.mean(), std=residuals.std()
        )
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = "{plot_dir}/{name}_predictions.png".format(
        plot_dir=PLOT_DIR, name=model_name.lower().replace(" ", "_")
    )
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved plot:", path)
    return path


def plot_residual_hist(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "model"
) -> str:
    """Plot histogram of residuals (prediction errors).

    Returns:
        Path to saved PNG.
    """
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=80, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(x=0, color="red", linewidth=1, linestyle="--")
    ax.set_xlabel("Prediction Error (Actual — Predicted)")
    ax.set_ylabel("Frequency")
    ax.set_title(
        "{name} — Residual Distribution\n(μ={mean:.6f}, σ={std:.6f})".format(
            name=model_name, mean=residuals.mean(), std=residuals.std()
        )
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = "{plot_dir}/{name}_residual_hist.png".format(
        plot_dir=PLOT_DIR, name=model_name.lower().replace(" ", "_")
    )
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved plot:", path)
    return path
```

**Verify:**
```bash
cd /home/bob/Documents/git/fp-ml && .venv/bin/python3 -c "from src.evaluate import compute_metrics, plot_predictions; print('OK')"
```

**Expected:** `OK`

---

# PHASE 9: MODEL COMPARISON

---

## Task 9.1: Create `src/compare.py`

**Create:** `src/compare.py`

```python
"""Compare all 3 models: metrics table, bar charts, CSV export."""
import json
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import METRIC_DIR, PLOT_DIR


def plot_metrics_bar(all_metrics: list) -> str:
    """Side-by-side bar chart comparing RMSE, MAE, R², DirAcc across models."""
    models = [m["model"] for m in all_metrics]
    rmse_vals = [m["rmse"] for m in all_metrics]
    mae_vals = [m["mae"] for m in all_metrics]
    r2_vals = [m["r2"] for m in all_metrics]
    dir_vals = [m["directional_accuracy_pct"] for m in all_metrics]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Comparison — USD/CHF Forecasting", fontsize=16, fontweight="bold")

    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    # RMSE
    axes[0, 0].bar(models, rmse_vals, color=colors)
    axes[0, 0].set_title("RMSE (lower is better)")
    axes[0, 0].set_ylabel("RMSE")
    for bar, val in zip(axes[0, 0].patches, rmse_vals):
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            "{:.6f}".format(val), ha="center", va="bottom", fontsize=9,
        )

    # MAE
    axes[0, 1].bar(models, mae_vals, color=colors)
    axes[0, 1].set_title("MAE (lower is better)")
    axes[0, 1].set_ylabel("MAE")
    for bar, val in zip(axes[0, 1].patches, mae_vals):
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            "{:.6f}".format(val), ha="center", va="bottom", fontsize=9,
        )

    # R²
    axes[1, 0].bar(models, r2_vals, color=colors)
    axes[1, 0].set_title("R² Score (higher is better)")
    axes[1, 0].set_ylabel("R²")
    for bar, val in zip(axes[1, 0].patches, r2_vals):
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            "{:.4f}".format(val), ha="center", va="bottom", fontsize=9,
        )

    # Directional Accuracy
    axes[1, 1].bar(models, dir_vals, color=colors)
    axes[1, 1].set_title("Directional Accuracy % (higher is better)")
    axes[1, 1].set_ylabel("%")
    axes[1, 1].axhline(y=50, color="gray", linewidth=0.5, linestyle="--", label="Random")
    axes[1, 1].legend()
    for bar, val in zip(axes[1, 1].patches, dir_vals):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            "{:.1f}%".format(val), ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()
    path = PLOT_DIR + "/model_comparison_metrics.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved:", path)
    return path


def save_metrics(all_metrics: list) -> str:
    """Save all metrics as JSON and CSV."""
    # JSON
    json_path = METRIC_DIR + "/all_metrics.json"
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print("Saved:", json_path)

    # CSV
    csv_path = METRIC_DIR + "/all_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
        writer.writeheader()
        writer.writerows(all_metrics)
    print("Saved:", csv_path)
    return json_path


if __name__ == "__main__":
    # Test with dummy data
    test_metrics = [
        {
            "model": "KNN Regressor", "rmse": 0.0005, "mae": 0.0003,
            "mape_pct": 0.05, "r2": 0.95, "directional_accuracy_pct": 51.2,
            "n_samples": 56000,
        },
        {
            "model": "SVR", "rmse": 0.0008, "mae": 0.0005,
            "mape_pct": 0.08, "r2": 0.88, "directional_accuracy_pct": 50.5,
            "n_samples": 56000,
        },
        {
            "model": "XGBoost Regressor", "rmse": 0.0004, "mae": 0.0002,
            "mape_pct": 0.04, "r2": 0.97, "directional_accuracy_pct": 53.1,
            "n_samples": 56000,
        },
    ]
    plot_metrics_bar(test_metrics)
    save_metrics(test_metrics)
    print("Compare module OK")
```

**Verify:**
```bash
cd /home/bob/Documents/git/fp-ml && .venv/bin/python3 -m src.compare
```

**Expected:** Creates PNG, JSON, CSV. Prints "Compare module OK".

---

# PHASE 10: MASTER PIPELINE RUNNER

---

## Task 10.1: Create `src/run_all.py`

**Create:** `src/run_all.py`

```python
#!/usr/bin/env python3
"""Master pipeline: load → features → split → scale → train 3 → evaluate → compare.

Usage:
    .venv/bin/python3 -m src.run_all
"""
from src.data_loader import load_data
from src.feature_engineering import build_features, scale_features
from src.data_splitting import split_data
from src.train_knn import train_knn
from src.train_svr import train_svr
from src.train_xgboost import train_xgboost
from src.evaluate import compute_metrics, plot_predictions, plot_residual_hist
from src.compare import plot_metrics_bar, save_metrics


def main():
    print("=" * 70)
    print("  USD/CHF FOREX FORECASTING — 3 MODEL COMPARISON")
    print("  KNN Regressor  |  SVR  |  XGBoost Regressor")
    print("=" * 70)

    # ── 1. Load & Build Features ──
    print("\n[1/7] Loading data...")
    df = load_data()
    print("       Loaded: {n:,} rows".format(n=len(df)))

    print("\n[2/7] Building features...")
    df = build_features(df)
    print("       Feature-built: {n:,} rows".format(n=len(df)))

    # ── 2. Split ──
    print("\n[3/7] Splitting data (chronological)...")
    X_tr, X_v, X_te, y_tr, y_v, y_te, feats = split_data(df)
    print("       {n} features".format(n=len(feats)))

    # ── 3. Scale ──
    print("\n[4/7] Scaling features (StandardScaler)...")
    X_tr_s, X_v_s, X_te_s, _ = scale_features(X_tr, X_v, X_te)

    # ── 4. Train 3 Models ──
    all_metrics = []

    print("\n[5a/7] Training KNN Regressor...")
    knn_model = train_knn(X_tr_s, y_tr)
    knn_pred = knn_model.predict(X_te_s)
    knn_metrics = compute_metrics(y_te, knn_pred, "KNN Regressor")
    plot_predictions(y_te, knn_pred, "KNN Regressor")
    plot_residual_hist(y_te, knn_pred, "KNN Regressor")
    all_metrics.append(knn_metrics)

    print("\n[5b/7] Training SVR...")
    svr_model = train_svr(X_tr_s, y_tr)
    svr_pred = svr_model.predict(X_te_s)
    svr_metrics = compute_metrics(y_te, svr_pred, "SVR")
    plot_predictions(y_te, svr_pred, "SVR")
    plot_residual_hist(y_te, svr_pred, "SVR")
    all_metrics.append(svr_metrics)

    print("\n[5c/7] Training XGBoost Regressor...")
    xgb_model = train_xgboost(X_tr_s, y_tr, X_v_s, y_v)
    xgb_pred = xgb_model.predict(X_te_s)
    xgb_metrics = compute_metrics(y_te, xgb_pred, "XGBoost Regressor")
    plot_predictions(y_te, xgb_pred, "XGBoost Regressor")
    plot_residual_hist(y_te, xgb_pred, "XGBoost Regressor")
    all_metrics.append(xgb_metrics)

    # ── 5. Compare ──
    print("\n[6/7] Generating comparison report...")
    save_metrics(all_metrics)
    plot_metrics_bar(all_metrics)

    # ── 6. Summary ──
    print("\n[7/7] ====== FINAL COMPARISON ======")
    header = "{:<20} {:>10} {:>10} {:>8} {:>8} {:>8}".format(
        "Model", "RMSE", "MAE", "MAPE%", "R²", "DirAcc%"
    )
    print(header)
    print("-" * 70)
    for m in all_metrics:
        print("{:<20} {:>10.6f} {:>10.6f} {:>8.4f} {:>8.4f} {:>8.2f}".format(
            m["model"], m["rmse"], m["mae"], m["mape_pct"], m["r2"],
            m["directional_accuracy_pct"],
        ))

    print("\n✅ Pipeline complete!")
    print("   Models:  outputs/models/")
    print("   Plots:   outputs/plots/")
    print("   Metrics: outputs/metrics/")


if __name__ == "__main__":
    main()
```

**Verify (dry run — load + features only):**
```bash
cd /home/bob/Documents/git/fp-ml && .venv/bin/python3 -c "
from src.data_loader import load_data
from src.feature_engineering import build_features, scale_features
from src.data_splitting import split_data
df = load_data()
df = build_features(df)
X_tr, X_v, X_te, y_tr, y_v, y_te, feats = split_data(df)
X_tr_s, X_v_s, X_te_s, _ = scale_features(X_tr, X_v, X_te)
print('Pipeline OK. Train:', X_tr_s.shape, 'Test:', X_te_s.shape)
"
```

**Expected:** `Pipeline OK. Train: (~410000, 48) Test: (~56000, 48)`

---

# PHASE 11: QUICK VALIDATION SCRIPT

---

## Task 11.1: Create `src/validate_quick.sh`

**Create:** `src/validate_quick.sh`

```bash
#!/usr/bin/env bash
# Quick validation: train all 3 models on a 20K sample
# to verify the pipeline works before full training.

set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python3

echo "=== QUICK VALIDATION (20K subsample) ==="

$PY -c "
from src.data_loader import load_data
from src.feature_engineering import build_features, scale_features
from src.data_splitting import split_data
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import numpy as np

# Load last 20K rows for recency
df = load_data().iloc[-20000:]
df = build_features(df)
X_tr, X_v, X_te, y_tr, y_v, y_te, feats = split_data(df)
X_tr_s, X_v_s, X_te_s, _ = scale_features(X_tr, X_v, X_te)

# Train mini models
for name, model in [
    ('KNN', KNeighborsRegressor(n_neighbors=5)),
    ('SVR', SVR(C=1.0, gamma='scale', epsilon=0.01)),
    ('XGBoost', XGBRegressor(n_estimators=50, max_depth=3, verbosity=0)),
]:
    model.fit(X_tr_s, y_tr)
    pred = model.predict(X_te_s)
    rmse = np.sqrt(((y_te - pred)**2).mean())
    mae = np.abs(y_te - pred).mean()
    print('{:.<10} RMSE={:.6f}  MAE={:.6f}'.format(name, rmse, mae))

print('✅ Quick validation PASSED — pipeline works!')
"
```

Make executable:
```bash
chmod +x src/validate_quick.sh
```

**Verify:**
```bash
cd /home/bob/Documents/git/fp-ml && bash src/validate_quick.sh
```

**Expected:** 3 lines with RMSE/MAE. Takes <60 seconds.

---

# MASTER EXECUTION CHECKLIST

| Step | Action | Command | Est. Time |
|------|--------|---------|-----------|
| 1 | Install deps | `.venv/bin/uv pip install scikit-learn xgboost matplotlib seaborn` | 2 min |
| 2 | Create dirs | `mkdir -p src outputs/models outputs/plots outputs/metrics` | instant |
| 3 | Write src/__init__.py | — | instant |
| 4 | Write src/config.py | — | instant |
| 5 | Write src/data_loader.py | — | instant |
| 6 | Test loader | `.venv/bin/python3 -m src.data_loader` | 5 sec |
| 7 | Write src/feature_engineering.py | — | instant |
| 8 | Test features | `.venv/bin/python3 -c "..." ` (see verify above) | 30 sec |
| 9 | Write src/data_splitting.py | — | instant |
| 10 | Test split | `.venv/bin/python3 -m src/data_splitting` | 5 sec |
| 11 | Write src/train_knn.py | — | instant |
| 12 | Write src/train_svr.py | — | instant |
| 13 | Write src/train_xgboost.py | — | instant |
| 14 | Write src/evaluate.py | — | instant |
| 15 | Write src/compare.py | — | instant |
| 16 | Write src/run_all.py | — | instant |
| 17 | Quick validate | `bash src/validate_quick.sh` | 1 min |
| 18 | **FULL RUN** | `.venv/bin/python3 -m src.run_all` | 30-90 min |

---

# RISKS & MITIGATIONS

| Risk | Mitigation |
|------|-----------|
| SVR O(n²) — training hangs | Subsampling to 50K rows via `subsample()` |
| All models predict naive (close_t ≈ close_t+1) | Directional accuracy catches this. DirAcc ≈ 50% = no edge |
| KNN predict slow on large test set | 56K test rows × ~10K train neighbors = fast |
| XGBoost grid: 3×4×3×3×3 = 324 combos | Only 3 CV folds via TimeSeriesSplit |
| Fish shell can't `source activate` | All commands use `.venv/bin/python3` directly |
| Volume column all zeros | Dropped in `data_loader.py` |
| Price data non-stationary | Lags + returns + rolling stats handle this |
| matplotlib needs display | `matplotlib.use("Agg")` in all plotting code |

---

# DELIVERABLES

1. **`src/`** — 10 Python modules
2. **`outputs/models/`** — 3 trained model files
3. **`outputs/plots/`** — 7 PNG plots
4. **`outputs/metrics/`** — `all_metrics.json` + `all_metrics.csv`

---

# GIT COMMIT

```bash
cd /home/bob/Documents/git/fp-ml
git add src/ outputs/ .hermes/
git commit -m "feat: KNN, SVR, XGBoost forex forecasting pipeline — 3 model comparison"
```
