#!/usr/bin/env python3
"""
PyTorch GPU-Accelerated KNN Regressor (ROCm/HIP)
- Grid search on CPU with subsample (fast)
- Final fit on GPU with all training data
- Batched distance computation for 8GB VRAM
"""

import numpy as np
import time
import pickle
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch KNN using: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

# Load data
import sys
sys.path.insert(0, '.')
from src.config import *
from src.data_loader import load_data
from src.feature_engineering import build_features, scale_features
from src.data_splitting import split_data

df = load_data()
df = build_features(df)
X_train, X_val, X_test, y_train, y_val, y_test, feature_names = split_data(df)
X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)
selected_features = feature_names

# Convert to torch tensors
X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test_s, dtype=torch.float32)

print(f"Train: {X_train_t.shape[0]:,} x {X_train_t.shape[1]}, Test: {X_test_t.shape[0]:,}")

# ---- Grid Search (CPU, subsampled) ----
print("\n=== Grid Search (CPU, 80K subsample) ===")
rng = np.random.RandomState(42)
idx = rng.choice(len(X_train_s), min(80000, len(X_train_s)), replace=False)
tscv = TimeSeriesSplit(n_splits=3)

param_grid = {
    'n_neighbors': [5, 10, 20, 50],
    'weights': ['uniform', 'distance'],
}

best_score = float('inf')
best_params = None

for k in param_grid['n_neighbors']:
    for w in param_grid['weights']:
        scores = []
        for train_i, val_i in tscv.split(idx):
            X_tr = X_train_s[idx][train_i]
            y_tr = y_train[idx][train_i]
            X_va = X_train_s[idx][val_i]
            y_va = y_train[idx][val_i]
            
            # Use sklearn KNN for grid search (fast enough on 80K)
            from sklearn.neighbors import KNeighborsRegressor
            knn = KNeighborsRegressor(n_neighbors=k, weights=w, algorithm='kd_tree', n_jobs=-1)
            knn.fit(X_tr, y_tr)
            pred = knn.predict(X_va)
            scores.append(np.sqrt(mean_squared_error(y_va, pred)))
        
        avg_rmse = np.mean(scores)
        print(f"  k={k:3d}  weight={w:<10s}  CV_RMSE={avg_rmse:.6f}")
        if avg_rmse < best_score:
            best_score = avg_rmse
            best_params = {'n_neighbors': k, 'weights': w}

print(f"\nBest: {best_params}, CV_RMSE={best_score:.6f}")
k = best_params['n_neighbors']
weights = best_params['weights']

# ---- GPU KNN Final Fit ----
print(f"\n=== GPU KNN (k={k}, {weights}) ===")
t0 = time.time()

# Move training data to GPU
X_train_gpu = X_train_t.to(device)
y_train_gpu = y_train_t.to(device)
n_train = X_train_gpu.shape[0]

# Process test set in batches
batch_size = 500
train_chunk = 400_000  # ~1.3GB per chunk (400K × 500 × 4)
n_test = X_test_t.shape[0]
predictions = torch.zeros(n_test, device='cpu')

print(f"Training samples: {n_train:,}, Test samples: {n_test:,}")
print(f"Batch size: {batch_size}, Train chunk: {train_chunk:,}")

for start in range(0, n_test, batch_size):
    end = min(start + batch_size, n_test)
    X_batch = X_test_t[start:end].to(device)  # (B, F)
    B = X_batch.shape[0]
    
    # Accumulate k-nearest distances and indices across chunks
    all_dists = []
    all_indices = []
    
    for chunk_start in range(0, n_train, train_chunk):
        chunk_end = min(chunk_start + train_chunk, n_train)
        X_chunk = X_train_gpu[chunk_start:chunk_end]  # (C, F)
        
        # Compute pairwise distances: (B, C)
        dists = torch.cdist(X_batch, X_chunk, p=2)  # L2 distance
        
        # Get top-k for this chunk
        chunk_dists, chunk_idx = torch.topk(dists, k=min(k, chunk_end - chunk_start), 
                                            dim=1, largest=False)
        # Offset indices
        chunk_idx += chunk_start
        
        all_dists.append(chunk_dists.cpu())
        all_indices.append(chunk_idx.cpu())
    
    # Merge chunks: take top-k across all chunks
    merged_dists = torch.cat(all_dists, dim=1)  # (B, total_k)
    merged_idx = torch.cat(all_indices, dim=1)
    
    final_dists, final_order = torch.topk(merged_dists, k=k, dim=1, largest=False)
    final_idx = merged_idx.gather(1, final_order)  # (B, k)
    
    # Weighted prediction
    if weights == 'uniform':
        pred = y_train_gpu[final_idx].mean(dim=1)
    else:
        # Inverse distance weighting
        inv_dists = 1.0 / (final_dists + 1e-10)
        weights_sum = inv_dists.sum(dim=1, keepdim=True)
        pred = (y_train_gpu[final_idx] * inv_dists).sum(dim=1) / weights_sum.squeeze()
    
    predictions[start:end] = pred.cpu()
    
    if (end % 5000) == 0 or end == n_test:
        elapsed = time.time() - t0
        rate = end / elapsed
        eta = (n_test - end) / rate
        print(f"  {end:,}/{n_test:,} ({100*end/n_test:.1f}%) — {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

total_time = time.time() - t0
print(f"GPU KNN done in {total_time:.0f}s ({total_time/60:.1f} min)")

# ---- Evaluate ----
from src.evaluate import compute_metrics

knn_pred = predictions.numpy()
knn_m = compute_metrics(y_test, knn_pred, "KNN Regressor (GPU)")
# plot saved separately")
knn_c = {"acc":0}")

# Save
pickle.dump({'k': k, 'weights': weights, 'X_train': X_train_s, 'y_train': y_train}, 
            open("outputs/models/knn_regressor_gpu.pkl", "wb"))
print("KNN GPU model saved")
print(f"\nFinal — RMSE={knn_m['rmse']:.6f}, R²={knn_m['r2']:.4f}, DirAcc={knn_m['directional_accuracy_pct']:.1f}%")
