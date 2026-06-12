#!/usr/bin/env python3
"""KNN Regressor v2 — GPU-batched distance computation. Predicts log returns.
Uses PyTorch for batched cdist on GPU to avoid O(n²) memory blowup."""
import time, json
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"KNN using: {device}")
torch.manual_seed(42)

# Load data
print("Loading data...")
data = torch.load("/home/bob/Documents/git/fp-ml/outputs/preprocessed/data.pt", weights_only=False)
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]

print(f"Train: {X_train.shape[0]:,} x {X_train.shape[1]}, Val: {X_val.shape[0]:,}, Test: {X_test.shape[0]:,}")

# Subsample train for speed (KNN inference is O(n_train * n_test))
# 100K train samples gives good coverage; full 1.8M would be ~3h
SUBSAMPLE = 100000
if X_train.shape[0] > SUBSAMPLE:
    idx = torch.randperm(X_train.shape[0])[:SUBSAMPLE]
    X_train_sub, y_train_sub = X_train[idx], y_train[idx]
    print(f"Subsampled train to {SUBSAMPLE:,} for inference speed")
else:
    X_train_sub, y_train_sub = X_train, y_train

# Move to GPU in float32
X_train_gpu = X_train_sub.float().to(device)
y_train_gpu = y_train_sub.float().to(device)
X_val_gpu = X_val.float().to(device)
y_val_gpu = y_val.float().to(device)
X_test_gpu = X_test.float().to(device)
y_test_gpu = y_test.float().to(device)

print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ---- Grid search for optimal k ----
# Use val set to pick k
BATCH_SIZE = 5000  # test samples per batch to keep GPU memory bounded
k_values = [1, 3, 5, 7, 10, 15, 20, 30, 50]

print("\nGrid search for k (on val set)...")
best_k, best_rmse = None, float("inf")
t0 = time.time()

for k in k_values:
    all_preds = []
    for i in range(0, X_val_gpu.shape[0], BATCH_SIZE):
        Xb = X_val_gpu[i:i+BATCH_SIZE]
        # Batched cdist: compute all-vs-all distances within batch range
        # (B, F) vs (N, F) → (B, N) using torch.cdist
        dists = torch.cdist(Xb, X_train_gpu)  # (B, N_train)
        _, indices = torch.topk(dists, k, largest=False)  # (B, k)
        # Gather predictions
        preds = y_train_gpu[indices].mean(dim=1)  # (B,)
        all_preds.append(preds.cpu())

    val_preds = torch.cat(all_preds)
    rmse = torch.sqrt(torch.mean((val_preds - y_val) ** 2))
    print(f"  k={k:2d}  RMSE={rmse.item():.8f}  {time.time()-t0:.0f}s")

    if rmse < best_rmse:
        best_rmse = rmse
        best_k = k

print(f"\nBest k={best_k}, val RMSE={best_rmse.item():.8f}")

# ---- Evaluate on test ----
print(f"\nEvaluating on test set with k={best_k}...")
all_preds = []
t1 = time.time()
for i in range(0, X_test_gpu.shape[0], BATCH_SIZE):
    Xb = X_test_gpu[i:i+BATCH_SIZE]
    dists = torch.cdist(Xb, X_train_gpu)
    _, indices = torch.topk(dists, best_k, largest=False)
    preds = y_train_gpu[indices].mean(dim=1)
    all_preds.append(preds.cpu())

test_preds = torch.cat(all_preds)
total_time = time.time() - t0
eval_time = time.time() - t1

# Metrics
y_true = y_test
mse = torch.mean((test_preds - y_true) ** 2).item()
rmse = np.sqrt(mse)
mae = torch.mean(torch.abs(test_preds - y_true)).item()
ss_res = torch.sum((test_preds - y_true) ** 2).item()
ss_tot = torch.sum((y_true - y_true.mean()) ** 2).item()
r2 = 1 - ss_res / ss_tot

y_true_np = y_true.numpy()
preds_np = test_preds.numpy()
dir_acc = (np.sign(preds_np) == np.sign(y_true_np)).mean() * 100
ratio_true = np.exp(y_true_np)
ratio_pred = np.exp(preds_np)
mape = np.mean(np.abs(ratio_true - ratio_pred) / ratio_true) * 100

print(f"\n{'='*50}")
print(f"  KNN v2 Regressor (GPU) — Log Returns")
print(f"  k={best_k}, train samples={SUBSAMPLE:,}")
print(f"{'='*50}")
print(f"  MSE:   {mse:.10f}")
print(f"  RMSE:  {rmse:.8f}")
print(f"  MAE:   {mae:.8f}")
print(f"  R²:    {r2:.6f}")
print(f"  MAPE:  {mape:.4f}%")
print(f"  DirAcc:{dir_acc:.1f}%")
print(f"  Grid time: {total_time:.0f}s ({total_time/60:.1f} min)")

# Save
results = {
    "model": "KNN v2 (GPU batched)",
    "k": int(best_k),
    "train_samples": SUBSAMPLE,
    "device": device,
    "mse": float(mse), "rmse": float(rmse),
    "mae": float(mae), "r2": float(r2),
    "mape": float(mape), "directional_accuracy": float(dir_acc),
    "total_time_s": total_time,
    "test_samples": len(y_true_np),
    "y_true_std": float(np.std(y_true_np)),
}
with open("/home/bob/Documents/git/fp-ml/outputs/knn_v2_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: outputs/knn_v2_results.json")
