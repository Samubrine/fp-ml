#!/usr/bin/env bash
# Quick validation: train all 3 models on a 20K sample
# to verify the pipeline works before full training.

set -e
cd "$(dirname \"$0\")"/..
PY=.venv/bin/python3

echo "=== QUICK VALIDATION (20K subsample) ==="

$PY -c "
from src.data_loader import load_data
from src.feature_engineering import build_features, scale_features
from src.data_splitting import split_data
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBoostRegressor
import numpy as np

df = load_data().iloc[-20000:]
df = build_features(df)
X_tr, X_v, X_te, y_tr, y_v, y_te, feats = split_data(df)
X_tr_s, X_v_s, X_te_s, _ = scale_features(X_tr, X_v, X_te)

for name, model in [
    (\"KNN\", KNeighborsRegressor(n_neighbors=5)),
    (\"SVR\", SVR(C=1.0, gamma=\"scale\", epsilon=0.01)),
    (\"XGBoost\", XGBoostRegressor(n_estimators=50, max_depth=3, verbosity=0)),
]:
    model.fit(X_tr_s, y_tr)
    pred = model.predict(X_te_s)
    rmse = np.sqrt(((y_te - pred)**2).mean())
    mae = np.abs(y_te - pred).mean()
    print(f\{name:<10} RMSE={rmse:.6f}  MAE={mae:.6f}\)

print(\"Quick validation PASSED - pipeline works!\")
