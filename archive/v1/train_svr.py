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
    print(f"Subsampled: {len(X):,} → {n:,} rows (stratified by target quantiles)")
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

    print(f"\nTraining complete in {elapsed:.1f}s")
    print("Best params:", grid.best_params_)
    print(f"Best RMSE (CV): {-grid.best_score_:.6f}")

    # Save model
    model_path = f"{MODEL_DIR}/svr_regressor.pkl"
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
