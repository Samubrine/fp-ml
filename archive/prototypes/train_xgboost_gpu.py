#!/usr/bin/env python3
"""XGBoost Regressor — GPU-accelerated (HIP/ROCm).
Requires xgboost built from source with -DUSE_HIP=ON."""

import time
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# Load data
import sys
sys.path.insert(0, '.')
from src.config import XGB_PARAM_GRID, RANDOM_SEED, MODEL_DIR, CV_FOLDS, OUTPUT_DIR
from src.data_loader import load_data
from src.feature_engineering import build_features, scale_features
from src.data_splitting import split_data
from src.evaluate import compute_metrics

df = load_data()
df = build_features(df)
X_train, X_val, X_test, y_train, y_val, y_test, feature_names = split_data(df)
X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)
selected_features = feature_names

print(f"Train: {X_train_s.shape[0]:,}, Test: {X_test_s.shape[0]:,}")

# ---- GPU XGBoost ----
print("=" * 60)
print("XGBOOST REGRESSOR — GPU (HIP/ROCm)")
print("=" * 60)

base_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    random_state=RANDOM_SEED,
    device="cuda",          # HIP/ROCm GPU
    tree_method="hist",     # Required for GPU
    n_jobs=-1,
    verbosity=0,
)

tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

grid = GridSearchCV(
    base_model,
    XGB_PARAM_GRID,
    cv=tscv,
    scoring="neg_root_mean_squared_error",
    n_jobs=1,  # GPU models: run sequentially to avoid VRAM contention
    verbose=2,
)

t0 = time.time()
grid.fit(X_train_s, y_train)
elapsed = time.time() - t0

print(f"\nTraining complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"Best params: {grid.best_params_}")
print(f"Best CV RMSE: {-grid.best_score_:.6f}")

# Early stopping on validation set
print("\nRetraining best model with early stopping...")
best_params = dict(grid.best_params_)
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    random_state=RANDOM_SEED,
    device="cuda",
    tree_method="hist",
    n_jobs=-1,
    verbosity=1,
    **best_params,
)
xgb_model.fit(
    X_train_s, y_train,
    eval_set=[(X_val_s, y_val)],
    verbose=False,
)

# ---- Predict & Evaluate ----
xgb_pred = xgb_model.predict(X_test_s)
xgb_m = compute_metrics(y_test, xgb_pred, "XGBoost Regressor (GPU)")
# plot saved separately")
xgb_c = {"acc":0}")

# Feature importance
imp = xgb_model.feature_importances_
top = np.argsort(imp)[-10:][::-1]
print("\nTop 10 Feature Importances:")
for rank, idx in enumerate(top, 1):
    print(f"  {rank:2d}. {selected_features[idx]:<25s} {imp[idx]:.4f}")

# Save
xgb_model._estimator_type = "regressor"
xgb_model.save_model(f"{MODEL_DIR}/xgboost_regressor_gpu.json")
print(f"\nSaved: {MODEL_DIR}/xgboost_regressor_gpu.json")
print(f"Final — RMSE={xgb_m['rmse']:.6f}, R²={xgb_m['r2']:.4f}, DirAcc={xgb_m['directional_accuracy_pct']:.1f}%")
