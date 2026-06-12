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

    print(f"\nTraining complete in {elapsed:.1f}s")
    print("Best params:", grid.best_params_)
    print(f"Best RMSE (CV): {-grid.best_score_:.6f}")

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
    model_path = f"{MODEL_DIR}/xgboost_regressor.json"
    model.save_model(model_path)
    print("Saved:", model_path)

    # Feature importance
    importance = model.feature_importances_
    top_indices = np.argsort(importance)[-10:][::-1]
    print("\nTop 10 feature importances:")
    for i in top_indices:
        print(f"  feat_{i}: {importance[i]:.4f}")

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
