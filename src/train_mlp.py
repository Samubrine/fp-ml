#!/usr/bin/env python3
"""MLP Regressor v2 — GPU-optimized (AMP + large batch + wide model). Predicts log returns."""
import time, os, json
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"MLP using: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
torch.manual_seed(42)

# ---- Load preprocessed data ----
print("Loading data...")
data = torch.load("outputs/preprocessed/data.pt", weights_only=False)
X_train, y_train = data["X_train"], data["y_train"].reshape(-1, 1)
X_val, y_val = data["X_val"], data["y_val"].reshape(-1, 1)
X_test, y_test = data["X_test"], data["y_test"].reshape(-1, 1)

print(f"Train: {X_train.shape[0]:,}, Val: {X_val.shape[0]:,}, Test: {X_test.shape[0]:,}")
print(f"Features: {X_train.shape[1]}")
print(f"y_train mean={y_train.mean().item():.8f} std={y_train.std().item():.6f}")

# ---- Model (optimized for GPU throughput) ----
class MLP(nn.Module):
    def __init__(self, input_dim, hidden=[1024, 512, 256, 128], dropout=0.15):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

model = MLP(X_train.shape[1]).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

BATCH_SIZE = 65536
EPOCHS = 150
LR = 0.001
PATIENCE = 15
USE_AMP = True

train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE,
                      shuffle=True, pin_memory=True, num_workers=4,
                      persistent_workers=True)
val_dl = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE * 2,
                    pin_memory=True, num_workers=2)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
scaler = torch.amp.GradScaler("cuda") if USE_AMP else None

amp_dtype = torch.float16
print(f"\nTraining (batch={BATCH_SIZE}, lr={LR}, AMP={USE_AMP}, workers=4)")
print(f"Batches/epoch: {len(train_dl)}")
t0 = time.time()
best_val = float("inf")
patience_cnt = 0
best_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0
    for Xb, yb in train_dl:
        Xb, yb = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad()
        if USE_AMP:
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                out = model(Xb)
                loss = criterion(out, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        train_loss += loss.item()

    scheduler.step()
    train_loss /= len(train_dl)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for Xb, yb in val_dl:
            Xb, yb = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            if USE_AMP:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    out = model(Xb)
                    val_loss += criterion(out, yb).item()
            else:
                out = model(Xb)
                val_loss += criterion(out, yb).item()
    val_loss /= len(val_dl)

    dt = time.time() - t0
    marker = ""
    if val_loss < best_val:
        best_val = val_loss
        patience_cnt = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        marker = " *"
    else:
        patience_cnt += 1
    if patience_cnt >= PATIENCE:
        print(f"\nEarly stop at epoch {epoch}")
        break

    if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.8f} | val_loss={val_loss:.8f} | {dt:.0f}s{marker}")

total_time = time.time() - t0
model.load_state_dict(best_state)
print(f"Done in {total_time:.0f}s ({total_time/60:.1f} min). Best val_loss={best_val:.8f}")

# ---- Evaluate on test ----
model.eval()
with torch.no_grad():
    if USE_AMP:
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            preds = model(X_test.to(device)).cpu().flatten()
    else:
        preds = model(X_test.to(device)).cpu().flatten()

y_true = y_test.flatten()
mse = torch.mean((preds - y_true) ** 2).item()
rmse = np.sqrt(mse)
mae = torch.mean(torch.abs(preds - y_true)).item()
ss_res = torch.sum((preds - y_true) ** 2).item()
ss_tot = torch.sum((y_true - y_true.mean()) ** 2).item()
r2 = 1 - ss_res / ss_tot
# Fixed: directional accuracy = sign(pred) == sign(actual)
dir_acc = (torch.sign(preds) == torch.sign(y_true)).float().mean().item() * 100

y_true_np = y_true.numpy()
preds_np = preds.numpy()
ratio_true = np.exp(y_true_np)
ratio_pred = np.exp(preds_np)
mape = np.mean(np.abs(ratio_true - ratio_pred) / ratio_true) * 100

print(f"\n{'='*50}")
print(f"  MLP v2 Regressor (GPU) — Log Returns")
print(f"  Architecture: 1024→512→256→128")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"{'='*50}")
print(f"  MSE:   {mse:.10f}")
print(f"  RMSE:  {rmse:.8f}")
print(f"  MAE:   {mae:.8f}")
print(f"  R²:    {r2:.6f}")
print(f"  MAPE:  {mape:.4f}%")
print(f"  DirAcc:{dir_acc:.1f}%")
print(f"  GPU time: {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"  y std: {np.std(y_true_np):.8f}")

# Save model
torch.save(best_state, "outputs/models/mlp_v2.pt")
results = {
    "model": "MLP v2 (PyTorch GPU)",
    "architecture": "1024→512→256→128",
    "params": sum(p.numel() for p in model.parameters()),
    "device": device,
    "mse": float(mse), "rmse": float(rmse), "mae": float(mae), "r2": float(r2),
    "mape": float(mape), "directional_accuracy": float(dir_acc),
    "y_true_std": float(np.std(y_true_np)),
    "test_samples": len(y_true_np),
    "training_time_s": total_time,
    "best_val_loss": float(best_val),
    "epochs": epoch,
}
with open("outputs/mlp_v2_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: outputs/models/mlp_v2.pt + results JSON")
