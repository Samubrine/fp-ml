"""K-Nearest Neighbors Regressor for USD/CHF close price forecasting.

KNN grid search on full 371K rows is too slow (even with KDTree).
Strategy: subsample to 100K for grid search, then fit best model on full data.
"""
import pickle
import time
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from src.config import KNN_PARAM_GRID, RANDOM_SEED, MODEL_DIR, CV_FOLDS


def train_knn(X_train_scaled: np.ndarray, y_train: np.ndarray) -> KNeighborsRegressor:
    """Train KNN Regressor with GridSearchCV on subsample."""
    print("=" * 60)
    print("KNN REGRESSOR — GridSearchCV (subsampled)")
    print("=" * 60)

    rng = np.random.RandomState(RANDOM_SEED)
    n_grid = min(100_000, len(X_train_scaled))
    idx = rng.choice(len(X_train_scaled), size=n_grid, replace=False)
    X_grid = X_train_scaled[idx]
    y_grid = y_train[idx]
    print(f"Grid search on {n_grid:,} rows (subsampled from {len(X_train_scaled):,})")

    base_model = KNeighborsRegressor(n_jobs=-1, algorithm="kd_tree", leaf_size=30)
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    grid = GridSearchCV(base_model, KNN_PARAM_GRID, cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=2)

    t0 = time.time()
    grid.fit(X_grid, y_grid)
    elapsed = time.time() - t0
    print(f"\nGrid search complete in {elapsed:.1f}s")
    print(f"Best params: {grid.best_params_}")
    print(f"Best RMSE (CV): {-grid.best_score_:.6f}")

    print(f"\nFitting best model on FULL training data ({len(X_train_scaled):,} rows)...")
    best_params = dict(grid.best_params_)
    best_model = KNeighborsRegressor(n_jobs=-1, algorithm="kd_tree", leaf_size=30, **best_params)
    t0 = time.time()
    best_model.fit(X_train_scaled, y_train)
    elapsed = time.time() - t0
    print(f"Full fit complete in {elapsed:.1f}s")

    model_path = f"{MODEL_DIR}/knn_regressor.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Saved: {model_path}")
    return best_model


if __name__ == "__main__":
    from src.data_loader import load_data
    from src.feature_engineering import build_features, scale_features
    from src.data_splitting import split_data
    df = load_data()
    df = build_features(df)
    X_tr, X_v, X_te, y_tr, y_v, y_te, feats = split_data(df)
    X_tr_s, X_v_s, X_te_s, _ = scale_features(X_tr, X_v, X_te)
    model = train_knn(X_tr_s, y_tr)
