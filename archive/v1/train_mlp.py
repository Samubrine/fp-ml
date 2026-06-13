#!/usr/bin/env python3
"""
PyTorch MLP Regressor — GPU-accelerated replacement for SVR
Architecture: [256, 128, 64, 32] with BatchNorm, Dropout, EarlyStopping
"""

import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch MLP using: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

# Seed
torch.manual_seed(42)
np.random.seed(42)

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

# Convert to tensors
X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
X_test_t = torch.tensor(X_test_s, dtype=torch.float32)

print(f"Train: {X_train_t.shape[0]:,}, Val: {X_val_t.shape[0]:,}, Test: {X_test_t.shape[0]:,}")
print(f"Features: {X_train_t.shape[1]}")

# ---- Model Definition ----
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

model = MLPRegressor(X_train_t.shape[1]).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ---- Training Setup ----
BATCH_SIZE = 8192
EPOCHS = 200
LR = 0.001
PATIENCE = 15

train_ds = TensorDataset(X_train_t, y_train_t)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_ds = TensorDataset(X_val_t, y_val_t)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, pin_memory=True)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# ---- Training Loop ----
print(f"\nTraining (batch={BATCH_SIZE}, lr={LR}, epochs={EPOCHS}, patience={PATIENCE})")
t0 = time.time()
best_val_loss = float('inf')
patience_counter = 0
best_state = None

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    train_loss = 0
    for Xb, yb in train_dl:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * Xb.shape[0]
    train_loss /= len(train_ds)
    
    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for Xb, yb in val_dl:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = criterion(model(Xb), yb)
            val_loss += loss.item() * Xb.shape[0]
    val_loss /= len(val_ds)
    
    scheduler.step(val_loss)
    
    if epoch % 10 == 0 or epoch == 1:
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{EPOCHS} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={optimizer.param_groups[0]['lr']:.6f} | {elapsed:.0f}s")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

model.load_state_dict(best_state)
total_time = time.time() - t0
print(f"Training done in {total_time:.0f}s ({total_time/60:.1f} min), best_val_loss={best_val_loss:.6f}")

# ---- Predict ----
model.eval()
with torch.no_grad():
    mlp_pred = model(X_test_t.to(device)).cpu().numpy().flatten()

# ---- Evaluate ----
from src.evaluate import compute_metrics
from sklearn.metrics import mean_squared_error, r2_score

mlp_m = compute_metrics(y_test, mlp_pred, "MLP Regressor (GPU)")

# Save model
torch.save(best_state, "outputs/models/mlp_regressor.pt")
print("MLP model saved")

rmse = np.sqrt(mean_squared_error(y_test, mlp_pred))
r2 = r2_score(y_test, mlp_pred)
print(f"\nFinal — RMSE={rmse:.6f}, R²={r2:.4f}")
