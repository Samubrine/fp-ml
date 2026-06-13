#!/usr/bin/env python3
"""XGBoost Regressor v2 — GPU (HIP) + log returns. 
Uses source-built XGBoost with HIP/ROCm on AMD GPU."""
import time, json, os
import numpy as np
import xgboost as xgb

print(f"XGBoost version: {xgb.__version__}")

# Load data
print("Loading data...")
import torch
data = torch.load("/home/bob/Documents/git/fp-ml/outputs/preprocessed/data.pt", weights_only=False)
X_train, y_train = data["X_train"].numpy(), data["y_train"].numpy().ravel()
X_val, y_val = data["X_val"].numpy(), data["y_val"].numpy().ravel()
X_test, y_test = data["X_test"].numpy(), data["y_test"].numpy().ravel()

print(f"Train: {X_train.shape[0]:,}, Val: {X_val.shape[0]:,}, Test: {X_test.shape[0]:,}")
print(f"y_train mean={y_train.mean():.8f} std={y_train.std():.6f}")

# Subsample for grid search speed
GRID_SAMPLES = 200000
if len(X_train) > GRID_SAMPLES:
    idx = np.random.RandomState(42).choice(len(X_train), GRID_SAMPLES, replace=False)
    X_grid, y_grid = X_train[idx], y_train[idx]
    print(f"Grid search on {GRID_SAMPLES:,} samples")
else:
    X_grid, y_grid = X_train, y_train

# ---- GPU test ----
print("\nTesting GPU...")
dtrain_test = xgb.DMatrix(X_grid[:1000], label=y_grid[:1000])
try:
    params_test = {"tree_method": "gpu_hist", "device": "cuda", "max_depth": 3, "verbosity": 0}
    bst = xgb.train(params_test, dtrain_test, num_boost_round=2)
    print("GPU (gpu_hist) works!")
    TREE_METHOD = "gpu_hist"
    GPU_OK = True
except Exception as e:
    print(f"gpu_hist failed: {e}. Using 'hist' (CPU).")
    TREE_METHOD = "hist"
    GPU_OK = False

# ---- Grid search ----
t0 = time.time()
param_grid = {
    "max_depth": [5, 7, 9, 11],
    "learning_rate": [0.01, 0.03, 0.05],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.6, 0.8],
    "min_child_weight": [1, 3, 5],
}

best_params, best_rmse = None, float("inf")
n_trials = 0
print(f"\nGrid search ({len(param_grid['max_depth'])*len(param_grid['learning_rate'])*len(param_grid['subsample'])*len(param_grid['colsample_bytree'])*len(param_grid['min_child_weight'])} combos, on {GRID_SAMPLES:,} samples)...")

for md in param_grid["max_depth"]:
    for lr in param_grid["learning_rate"]:
        for ss in param_grid["subsample"]:
            for cs in param_grid["colsample_bytree"]:
                for mcw in param_grid["min_child_weight"]:
                    params = {
                        "tree_method": TREE_METHOD,
                        "objective": "reg:squarederror",
                        "eval_metric": "rmse",
                        "max_depth": md, "learning_rate": lr,
                        "subsample": ss, "colsample_bytree": cs,
                        "min_child_weight": mcw,
                        "verbosity": 0, "n_jobs": 0,
                    }
                    if GPU_OK:
                        params["device"] = "cuda"
                    dtrain = xgb.DMatrix(X_grid, label=y_grid)
                    cv = xgb.cv(params, dtrain, num_boost_round=200,
                                nfold=3, early_stopping_rounds=20,
                                verbose_eval=False, seed=42)
                    rmse = cv["test-rmse-mean"].min()
                    best_round = int(cv["test-rmse-mean"].idxmin()) + 1  # 1-indexed boosting round
                    n_trials += 1
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_params = params.copy()
                        best_params["best_iteration"] = max(best_round, 10)  # min 10 trees
                    if n_trials % 20 == 0:
                        print(f"  [{n_trials}] best RMSE={best_rmse:.8f} (md={best_params['max_depth']}, lr={best_params['learning_rate']})")

print(f"\nGrid search done in {time.time()-t0:.0f}s. Best CV RMSE={best_rmse:.8f}")
print(f"Best params: {best_params}")

# ---- Train on full data ----
print(f"\nTraining on full {len(X_train):,} samples...")
full_params = {k: v for k, v in best_params.items() if k not in ("best_iteration",)}
dtrain_full = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
evals = [(dtrain_full, "train"), (dval, "val")]

t1 = time.time()
model = xgb.train(full_params, dtrain_full,
                  num_boost_round=best_params["best_iteration"],
                  evals=evals, verbose_eval=50)
train_time = time.time() - t1
print(f"Training done in {train_time:.0f}s")

# ---- Evaluate ----
dtest = xgb.DMatrix(X_test)
y_pred = model.predict(dtest)

mse = np.mean((y_pred - y_test) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_pred - y_test))
ss_res = np.sum((y_pred - y_test) ** 2)
ss_tot = np.sum((y_test - y_test.mean()) ** 2)
r2 = 1 - ss_res / ss_tot
dir_acc = (np.sign(y_pred) == np.sign(y_test)).mean() * 100
ratio_true = np.exp(y_test)
ratio_pred = np.exp(y_pred)
mape = np.mean(np.abs(ratio_true - ratio_pred) / ratio_true) * 100

total_time = time.time() - t0

print(f"\n{'='*50}")
print(f"  XGBoost v2 Regressor ({TREE_METHOD}) — Log Returns")
print(f"{'='*50}")
print(f"  MSE:   {mse:.10f}")
print(f"  RMSE:  {rmse:.8f}")
print(f"  MAE:   {mae:.8f}")
print(f"  R²:    {r2:.6f}")
print(f"  MAPE:  {mape:.4f}%")
print(f"  DirAcc:{dir_acc:.1f}%")
print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")

# Save model
model.save_model("/home/bob/Documents/git/fp-ml/outputs/models/xgboost_v2.json")

results = {
    "model": f"XGBoost v2 ({TREE_METHOD})",
    "version": xgb.__version__,
    "tree_method": TREE_METHOD,
    "best_params": best_params,
    "mse": float(mse), "rmse": float(rmse),
    "mae": float(mae), "r2": float(r2),
    "mape": float(mape), "directional_accuracy": float(dir_acc),
    "total_time_s": total_time, "training_time_s": train_time,
    "test_samples": len(y_test), "y_true_std": float(np.std(y_test)),
}
with open("/home/bob/Documents/git/fp-ml/outputs/xgb_v2_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: outputs/models/xgboost_v2.json + results JSON")
