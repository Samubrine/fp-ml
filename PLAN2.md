# GPU-Accelerated Forex Forecasting — Complete Rewrite

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Rebuild forex forecasting pipeline with GPU-accelerated models (PyTorch MLP, PyTorch KNN, XGBoost HIP) using log-return targets and proper preprocessing for 2.3M rows (2020–2026).

**Architecture:** Preprocess once → save to `.pt` tensors → train 3 models independently from tensors. Target is `log_return = ln(close[t+1]/close[t])` (stationary). All models read shared `data.pt`.

**Strategy:** 🔬 **Python scripts first, notebook later.** Tasks 1–5 test everything via standalone `.py` files. Only after all models produce good results and all edge cases are resolved, Task 7 assembles the working code into a clean `.ipynb` notebook.

**Tech Stack:** PyTorch 2.12 + ROCm 7.2, XGBoost 3.x HIP, scikit-learn 1.9, pandas 3.x, Python 3.12 (`.venv-torch`)

---

## Environment

- **Host:** `ssh bob@desktop-linux.netbird.selfhosted`
- **Project root:** `/home/bob/Documents/git/fp-ml`
- **Shell:** fish (use `.venv-torch/bin/python3` directly, no `source activate`)
- **PyTorch venv:** `.venv-torch/bin/python3` (Python 3.12.13, PyTorch 2.12+rocm7.2)
- **XGBoost venv:** `.venv/bin/python3` (Python 3.14, XGBoost built from source with HIP)
- **GPU:** AMD Radeon RX 9060 XT, 8GB VRAM, `gfx1201`
- **Data:** `data/processed/USDCHF_1min_2020_2026.csv` (2,319,766 rows, 119MB)
- **Cutoffs:** TRAIN_CUTOFF="2025-01-01", VAL_CUTOFF="2025-09-01"

---

## Task 1: Create Preprocessing Pipeline

**Objective:** Load raw CSV, engineer features, split, scale, save as `.pt` tensors with log-return target.

**Files:**
- Create: `src/preprocess.py`

**Step 1: Write `src/preprocess.py`**

```python
#!/usr/bin/env python3
"""Preprocess 2.3M-row forex data into PyTorch tensors. Run once, all models share output."""
import sys, os, time
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# ---- Config (hardcoded to avoid circular imports) ----
DATA_PATH = "data/processed/USDCHF_1min_2020_2026.csv"
OUT_DIR = "outputs/preprocessed"
TRAIN_CUTOFF = "2025-01-01"
VAL_CUTOFF = "2025-09-01"

os.makedirs(OUT_DIR, exist_ok=True)

# ---- 1. Load ----
print("Loading data...", flush=True)
t0 = time.time()
df = pd.read_csv(DATA_PATH)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.set_index("datetime").sort_index()
print(f"  Loaded {len(df):,} rows in {time.time()-t0:.0f}s", flush=True)

# ---- 2. Feature Engineering ----
print("Engineering features...", flush=True)
t0 = time.time()

# Lag features
for lag in [1, 2, 3, 5, 10, 15, 30, 60]:
    df[f"close_lag_{lag}"] = df["close"].shift(lag)

# Rolling features
for w in [5, 10, 20, 50]:
    df[f"rolling_mean_{w}"] = df["close"].rolling(w).mean()
    df[f"rolling_std_{w}"] = df["close"].rolling(w).std()
    df[f"rolling_max_{w}"] = df["close"].rolling(w).max()
    df[f"rolling_min_{w}"] = df["close"].rolling(w).min()

# Price-derived features
df["hl_spread"] = df["high"] - df["low"]
df["ohlc_mean"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
df["log_return"] = np.log(df["close"] / df["close"].shift(1))
df["pct_change"] = df["close"].pct_change()

# Technical indicators
delta = df["close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
df["rsi_14"] = 100 - 100 / (1 + gain / loss)
ema12 = df["close"].ewm(span=12).mean()
ema26 = df["close"].ewm(span=26).mean()
df["macd_hist"] = ema12 - ema26 - (ema12 - ema26).ewm(span=9).mean()
bb_mid = df["close"].rolling(20).mean()
bb_std = df["close"].rolling(20).std()
df["bb_position_20"] = (df["close"] - bb_mid) / bb_std
df["bb_upper_20"] = bb_mid + 2 * bb_std
df["bb_lower_20"] = bb_mid - 2 * bb_std
df["bb_width_20"] = df["bb_upper_20"] - df["bb_lower_20"]

# Target: log return (stationary, regime-invariant)
df["target"] = np.log(df["close"].shift(-1) / df["close"])

# Drop NaN from rolling/lag windows
df = df.dropna()
print(f"  Engineered {len(df):,} rows, {len(df.columns)} cols in {time.time()-t0:.0f}s", flush=True)

# ---- 3. Feature Selection ----
exclude = ["target", "open", "high", "low", "close", "tick_volume", "volume", "spread"]
feature_cols = [c for c in df.columns if c not in exclude]
print(f"  Features: {len(feature_cols)}", flush=True)

# ---- 4. Chronological Split ----
train_mask = df.index < TRAIN_CUTOFF
val_mask = (df.index >= TRAIN_CUTOFF) & (df.index < VAL_CUTOFF)
test_mask = df.index >= VAL_CUTOFF

X_train = df.loc[train_mask, feature_cols].values.astype(np.float32)
y_train = df.loc[train_mask, "target"].values.astype(np.float32)
X_val = df.loc[val_mask, feature_cols].values.astype(np.float32)
y_val = df.loc[val_mask, "target"].values.astype(np.float32)
X_test = df.loc[test_mask, feature_cols].values.astype(np.float32)
y_test = df.loc[test_mask, "target"].values.astype(np.float32)

print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}", flush=True)

# ---- 5. Scale Features ----
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train).astype(np.float32)
X_val_s = scaler.transform(X_val).astype(np.float32)
X_test_s = scaler.transform(X_test).astype(np.float32)

# ---- 6. Save as PyTorch tensors ----
torch.save({
    "X_train": torch.from_numpy(X_train_s),
    "y_train": torch.from_numpy(y_train),
    "X_val": torch.from_numpy(X_val_s),
    "y_val": torch.from_numpy(y_val),
    "X_test": torch.from_numpy(X_test_s),
    "y_test": torch.from_numpy(y_test),
    "feature_names": feature_cols,
    "scaler_mean": scaler.mean_,
    "scaler_scale": scaler.scale_,
}, f"{OUT_DIR}/data.pt")

print(f"\nSaved to {OUT_DIR}/data.pt", flush=True)
print(f"  Train: {X_train_s.shape}, Val: {X_val_s.shape}, Test: {X_test_s.shape}")
print(f"  y_train mean={y_train.mean():.8f} std={y_train.std():.6f}")
print(f"  y_test  mean={y_test.mean():.8f} std={y_test.std():.6f}")
```

**Step 2: Run preprocessing**

```bash
ssh bob@desktop-linux.netbird.selfhosted 'cd /home/bob/Documents/git/fp-ml && .venv-torch/bin/python3 -u src/preprocess.py'
```

**Expected:** `outputs/preprocessed/data.pt` created with X_train shape ~(1.8M, 42), y_train ~(1.8M,)

**Step 3: Verify data integrity**

```bash
ssh bob@desktop-linux.netbird.selfhosted 'cd /home/bob/Documents/git/fp-ml && .venv-torch/bin/python3 -c "
import torch
d = torch.load(\"outputs/preprocessed/data.pt\", weights_only=True)
for k, v in d.items():
    if hasattr(v, \"shape\"):
        print(f\"{k}: {list(v.shape)}, dtype={v.dtype}\")
    else:
        print(f\"{k}: {type(v).__name__}\")
"'
```

**Step 4: Commit**

```bash
git add src/preprocess.py && git commit -m "feat: add preprocessing pipeline with log-return target"
```

---

## Task 2: MLP Regressor with Log-Return Target

**Objective:** Train a PyTorch MLP on log returns with GPU. Target: RMSE < std(y_test).

**Files:**
- Create: `src/train_mlp_v2.py`

**Step 1: Write `src/train_mlp_v2.py`**

```python
#!/usr/bin/env python3
"""MLP Regressor — GPU-accelerated (PyTorch + ROCm 7.2). Predicts log returns."""
import time, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"MLP using: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
torch.manual_seed(42)

# ---- Load preprocessed data ----
print("Loading data...")
data = torch.load("outputs/preprocessed/data.pt", weights_only=True)
X_train, y_train = data["X_train"], data["y_train"].reshape(-1, 1)
X_val, y_val = data["X_val"], data["y_val"].reshape(-1, 1)
X_test, y_test = data["X_test"], data["y_test"].reshape(-1, 1)

print(f"Train: {X_train.shape[0]:,}, Val: {X_val.shape[0]:,}, Test: {X_test.shape[0]:,}")
print(f"Features: {X_train.shape[1]}")
print(f"y_train mean={y_train.mean().item():.8f} std={y_train.std().item():.6f}")

# ---- Model ----
class MLP(nn.Module):
    def __init__(self, input_dim, hidden=[512, 256, 128, 64], dropout=0.15):
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

# ---- Training ----
BATCH_SIZE = 16384
EPOCHS = 150
LR = 0.0005
PATIENCE = 15

train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_dl = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE * 2, pin_memory=True)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)

print(f"\nTraining (batch={BATCH_SIZE}, lr={LR}, epochs={EPOCHS})")
t0 = time.time()
best_val = float("inf")
patience_cnt = 0
best_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0
    for Xb, yb in train_dl:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item() * Xb.shape[0]
    train_loss /= len(train_dl.dataset)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for Xb, yb in val_dl:
            Xb, yb = Xb.to(device), yb.to(device)
            val_loss += criterion(model(Xb), yb).item() * Xb.shape[0]
    val_loss /= len(val_dl.dataset)

    scheduler.step(epoch)

    if epoch % 10 == 0 or epoch == 1:
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d} | train={train_loss:.8f} | val={val_loss:.8f} | lr={optimizer.param_groups[0]['lr']:.6f} | {elapsed:.0f}s", flush=True)

    if val_loss < best_val:
        best_val = val_loss
        patience_cnt = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}", flush=True)
            break

model.load_state_dict(best_state)
print(f"Done in {time.time()-t0:.0f}s. Best val_loss={best_val:.8f}")

# ---- Evaluate ----
model.eval()
with torch.no_grad():
    preds = model(X_test.to(device)).cpu().flatten()

y_true = y_test.flatten()
rmse = torch.sqrt(torch.mean((preds - y_true) ** 2)).item()
ss_res = torch.sum((preds - y_true) ** 2).item()
ss_tot = torch.sum((y_true - y_true.mean()) ** 2).item()
r2 = 1 - ss_res / ss_tot
# Directional accuracy
dir_acc = (torch.sign(preds[1:] - preds[:-1]) == torch.sign(y_true[1:] - y_true[:-1])).float().mean().item() * 100

print(f"\n{'='*50}")
print(f"  MLP Regressor (GPU) — Log Returns")
print(f"{'='*50}")
print(f"  RMSE (return): {rmse:.8f}")
print(f"  R²  (return): {r2:.6f}")
print(f"  DirAcc:       {dir_acc:.1f}%")

# Save
torch.save(best_state, "outputs/models/mlp_v2.pt")
print(f"\nSaved: outputs/models/mlp_v2.pt")
```

**Step 2: Train MLP (~8-12 min)**

```bash
ssh bob@desktop-linux.netbird.selfhosted 'cd /home/bob/Documents/git/fp-ml && .venv-torch/bin/python3 -u src/train_mlp_v2.py'
```

**Step 3: Verify results exist** — `outputs/models/mlp_v2.pt` should be ~500KB

**Step 4: Commit**

```bash
git add src/train_mlp_v2.py && git commit -m "feat: MLP v2 with log-return target, deeper architecture, cosine annealing"
```

---

## Task 3: KNN Regressor GPU with Log Returns

**Objective:** Train GPU-accelerated KNN using batched `torch.cdist`. Grid search on CPU with 100K subsample.

**Files:**
- Create: `src/train_knn_v2.py`

**Step 1: Write `src/train_knn_v2.py`**

```python
#!/usr/bin/env python3
"""KNN Regressor — GPU-accelerated (PyTorch). Batched cdist for 8GB VRAM."""
import time, pickle, os
import numpy as np
import torch
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import TimeSeriesSplit

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"KNN using: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

# ---- Load data ----
print("Loading data...")
data = torch.load("outputs/preprocessed/data.pt", weights_only=True)
X_train_np = data["X_train"].numpy()
y_train_np = data["y_train"].numpy()
X_test_np = data["X_test"].numpy()
X_test_t = data["X_test"]
y_test_np = data["y_test"].numpy()

print(f"Train: {X_train_np.shape[0]:,}, Test: {X_test_np.shape[0]:,}, Feats: {X_train_np.shape[1]}")

# ---- Grid Search (CPU, subsampled) ----
print("\nGrid search (CPU, 100K subsample)...")
rng = np.random.RandomState(42)
idx = rng.choice(len(X_train_np), min(100_000, len(X_train_np)), replace=False)
tscv = TimeSeriesSplit(n_splits=3)

best_score = float("inf")
best_params = None

for k in [5, 10, 20, 50, 100]:
    for w in ["uniform", "distance"]:
        scores = []
        for tr_i, va_i in tscv.split(idx):
            knn = KNeighborsRegressor(n_neighbors=k, weights=w, algorithm="kd_tree", n_jobs=-1)
            knn.fit(X_train_np[idx][tr_i], y_train_np[idx][tr_i])
            p = knn.predict(X_train_np[idx][va_i])
            scores.append(np.sqrt(np.mean((p - y_train_np[idx][va_i]) ** 2)))
        avg = np.mean(scores)
        print(f"  k={k:3d}  {w:<10s}  CV_RMSE={avg:.8f}", flush=True)
        if avg < best_score:
            best_score = avg
            best_params = {"n_neighbors": k, "weights": w}

k = best_params["n_neighbors"]
weights = best_params["weights"]
print(f"Best: k={k}, {weights}, CV_RMSE={best_score:.8f}")

# ---- GPU KNN ----
print(f"\nGPU KNN (k={k}, {weights}) — this takes 15-25 min...")
X_tr = data["X_train"].to(device)
y_tr = data["y_train"].to(device)
n_train = X_tr.shape[0]
n_test = X_test_t.shape[0]

BATCH = 400
CHUNK = 500_000
preds = torch.zeros(n_test)

t0 = time.time()
for start in range(0, n_test, BATCH):
    end = min(start + BATCH, n_test)
    Xb = X_test_t[start:end].to(device)
    all_dists, all_idx = [], []

    for cs in range(0, n_train, CHUNK):
        ce = min(cs + CHUNK, n_train)
        Xc = X_tr[cs:ce]
        d = torch.cdist(Xb, Xc)
        chunk_d, chunk_i = torch.topk(d, min(k, ce - cs), dim=1, largest=False)
        all_dists.append(chunk_d.cpu())
        all_idx.append((chunk_i + cs).cpu())

    merged_d = torch.cat(all_dists, dim=1)
    merged_i = torch.cat(all_idx, dim=1)
    final_d, order = torch.topk(merged_d, k, dim=1, largest=False)
    final_i = merged_i.gather(1, order)

    if weights == "uniform":
        preds[start:end] = y_tr[final_i].mean(dim=1).cpu()
    else:
        inv_d = 1.0 / (final_d + 1e-10)
        w_sum = inv_d.sum(dim=1, keepdim=True)
        preds[start:end] = (y_tr[final_i] * inv_d).sum(dim=1).cpu() / w_sum.squeeze()

    if end % 10_000 == 0 or end == n_test:
        elapsed = time.time() - t0
        print(f"  {end:,}/{n_test:,} | {elapsed:.0f}s elapsed", flush=True)

total = time.time() - t0
print(f"GPU KNN done in {total:.0f}s ({total/60:.1f} min)")

# ---- Evaluate ----
y_true = data["y_test"]
rmse = torch.sqrt(torch.mean((preds - y_true) ** 2)).item()
ss_res = torch.sum((preds - y_true) ** 2).item()
ss_tot = torch.sum((y_true - y_true.mean()) ** 2).item()
r2 = 1 - ss_res / ss_tot
dir_acc = (torch.sign(preds[1:] - preds[:-1]) == torch.sign(y_true[1:] - y_true[:-1])).float().mean().item() * 100

print(f"\n{'='*50}")
print(f"  KNN Regressor (GPU) — Log Returns")
print(f"{'='*50}")
print(f"  RMSE (return): {rmse:.8f}")
print(f"  R²  (return): {r2:.6f}")
print(f"  DirAcc:       {dir_acc:.1f}%")

# Save metadata
pickle.dump({"k": k, "weights": weights, "rmse": rmse, "r2": r2},
            open("outputs/models/knn_v2.pkl", "wb"))
print(f"Saved: outputs/models/knn_v2.pkl")
```

**Step 2: Train KNN (~15-25 min)**

```bash
ssh bob@desktop-linux.netbird.selfhosted 'cd /home/bob/Documents/git/fp-ml && .venv-torch/bin/python3 -u src/train_knn_v2.py'
```

**Step 3: Verify** — check `outputs/models/knn_v2.pkl` exists

**Step 4: Commit**

```bash
git add src/train_knn_v2.py && git commit -m "feat: KNN v2 GPU with log-return target, batched cdist"
```

---

## Task 4: XGBoost HIP with Log Returns

**Objective:** Train XGBoost on GPU (HIP) with log-return target.

**Files:**
- Create: `src/train_xgb_v2.py`

**Note:** XGBoost HIP was built from source in `.venv/` (Python 3.14). Use that venv.

**Step 1: Write `src/train_xgb_v2.py`**

```python
#!/usr/bin/env python3
"""XGBoost Regressor — GPU (HIP/ROCm). Predicts log returns."""
import time, os, pickle
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# ---- Load data ----
print("Loading data...")
import torch
data = torch.load("outputs/preprocessed/data.pt", weights_only=True)
X_train = data["X_train"].numpy()
y_train = data["y_train"].numpy()
X_val = data["X_val"].numpy()
y_val = data["y_val"].numpy()
X_test = data["X_test"].numpy()
y_test = data["y_test"].numpy()

print(f"Train: {X_train.shape[0]:,}, Val: {X_val.shape[0]:,}, Test: {X_test.shape[0]:,}")

# ---- GPU XGBoost ----
print(f"\n{'='*50}")
print("XGBoost GPU (HIP/ROCm) — Log Returns")
print(f"{'='*50}")

base = xgb.XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    device="cuda",
    tree_method="hist",
    verbosity=0,
)

tscv = TimeSeriesSplit(n_splits=3)
param_grid = {
    "n_estimators": [100, 200, 400],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
}

grid = GridSearchCV(base, param_grid, cv=tscv, scoring="neg_root_mean_squared_error",
                    n_jobs=1, verbose=2)

t0 = time.time()
grid.fit(X_train, y_train)
print(f"Grid done in {time.time()-t0:.0f}s. Best: {grid.best_params_}, CV_RMSE={-grid.best_score_:.8f}")

# Retrain with early stopping
best = dict(grid.best_params_)
model = xgb.XGBRegressor(
    objective="reg:squarederror", random_state=42,
    device="cuda", tree_method="hist", **best
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# Predict
preds = model.predict(X_test)

# Evaluate
rmse = np.sqrt(np.mean((preds - y_test) ** 2))
ss_res = np.sum((preds - y_test) ** 2)
ss_tot = np.sum((y_test - y_test.mean()) ** 2)
r2 = 1 - ss_res / ss_tot
dir_acc = np.mean(np.sign(np.diff(preds)) == np.sign(np.diff(y_test))) * 100

print(f"\n{'='*50}")
print(f"  XGBoost Regressor (GPU) — Log Returns")
print(f"{'='*50}")
print(f"  RMSE (return): {rmse:.8f}")
print(f"  R²  (return): {r2:.6f}")
print(f"  DirAcc:       {dir_acc:.1f}%")

# Save
model._estimator_type = "regressor"
model.save_model("outputs/models/xgb_v2.json")
print(f"Saved: outputs/models/xgb_v2.json")
```

**Step 2: Train XGBoost (~5-10 min) — use OLD venv for HIP support**

```bash
ssh bob@desktop-linux.netbird.selfhosted 'cd /home/bob/Documents/git/fp-ml && .venv/bin/python3 -u src/train_xgb_v2.py'
```

**Step 3: Verify** — `outputs/models/xgb_v2.json` should exist

**Step 4: Commit**

```bash
git add src/train_xgb_v2.py && git commit -m "feat: XGBoost v2 GPU HIP with log-return target"
```

---

## Task 5: Comparison & Visualization Script

**Objective:** Single script that loads all 3 model outputs + preprocessed data, prints comparison table, generates plots.

**Files:**
- Create: `src/compare_v2.py`

**Step 1: Write `src/compare_v2.py`**

```python
#!/usr/bin/env python3
"""Compare all 3 GPU models on log-return prediction."""
import torch, pickle, numpy as np, xgboost as xgb, os, sys

sys.path.insert(0, ".")
from src.evaluate import compute_metrics, plot_predictions

# Load data
data = torch.load("outputs/preprocessed/data.pt", weights_only=True)
y_test = data["y_test"].numpy()
X_test = data["X_test"].numpy()

# ---- MLP ----
mlp_state = torch.load("outputs/models/mlp_v2.pt", weights_only=True)
from train_mlp_v2 import MLP
mlp = MLP(data["X_train"].shape[1])
mlp.load_state_dict(mlp_state)
mlp.eval()
with torch.no_grad():
    mlp_pred = mlp(data["X_test"]).flatten().numpy()

# ---- KNN (loaded as pickle metadata, predictions need re-run or saved) ----
# For KNN we re-run with saved params
knn_meta = pickle.load(open("outputs/models/knn_v2.pkl", "rb"))

# ---- XGBoost ----
xgb_m = xgb.XGBRegressor()
xgb_m.load_model("outputs/models/xgb_v2.json")
xgb_pred = xgb_m.predict(X_test)

# ---- Evaluate ----
mlp_m = compute_metrics(y_test, mlp_pred, "MLP (GPU)")
xgb_m = compute_metrics(y_test, xgb_pred, "XGBoost (GPU)")

# KNN metrics from saved file
knn_m = {"model": "KNN (GPU)", "rmse": knn_meta["rmse"], "r2": knn_meta["r2"]}

print(f"\n{'='*70}")
print(f"{'Model':<20} {'RMSE':>10} {'R²':>10}  Notes")
print(f"{'-'*70}")
for m in [mlp_m, knn_m, xgb_m]:
    print(f"{m['model']:<20} {m['rmse']:>10.8f} {m['r2']:>10.6f}")

# Convert log returns back to prices for visualization
# close[t+1] = close[t] * exp(return)
close_test = np.exp(np.cumsum(y_test)) * 1.0  # relative prices
mlp_close = np.exp(np.cumsum(mlp_pred))
xgb_close = np.exp(np.cumsum(xgb_pred))

import matplotlib.pyplot as plt
plt.figure(figsize=(14, 6))
plt.plot(close_test[:500], "k-", label="Actual", linewidth=1.5)
plt.plot(mlp_close[:500], alpha=0.7, label="MLP")
plt.plot(xgb_close[:500], alpha=0.7, label="XGBoost")
plt.legend()
plt.title("Cumulative Log-Return Comparison (first 500 test points)")
plt.tight_layout()
plt.savefig("outputs/plots/compare_v2.png", dpi=150)
print("Plot saved: outputs/plots/compare_v2.png")
```

**Step 2: Run comparison**

```bash
ssh bob@desktop-linux.netbird.selfhosted 'cd /home/bob/Documents/git/fp-ml && .venv-torch/bin/python3 -u src/compare_v2.py'
```

**Step 3: Verify plot** — `outputs/plots/compare_v2.png` should exist

**Step 4: Commit**

```bash
git add src/compare_v2.py && git commit -m "feat: model comparison script v2 for log-return predictions"
```

---

## Task 6: Consolidate Results

**Objective:** After all 3 models complete, verify their outputs and compile a summary JSON for the notebook.

**This is a checkpoint task — no new code. Just verification.**

**Step 1: Verify all model outputs exist**

```bash
ssh bob@desktop-linux.netbird.selfhosted 'ls -lh outputs/models/mlp_v2.pt outputs/models/knn_v2.pkl outputs/models/xgb_v2.json outputs/plots/compare_v2.png'
```

**Step 2: Run compare script to get final numbers**

```bash
ssh bob@desktop-linux.netbird.selfhosted 'cd /home/bob/Documents/git/fp-ml && .venv-torch/bin/python3 -u src/compare_v2.py 2>&1 | tee /tmp/compare_v2_output.txt'
```

**Step 3: Save comparison output for notebook use**

```bash
ssh bob@desktop-linux.netbird.selfhosted 'cat /tmp/compare_v2_output.txt'
```

**Step 4: Commit all results**

```bash
git add outputs/models/mlp_v2.pt outputs/models/knn_v2.pkl outputs/models/xgb_v2.json outputs/plots/compare_v2.png && git commit -m "results: GPU v2 models trained on log returns (2020-2026)"
```

---

## 🔬 Task 7: Assemble Working Code into Notebook

> **⚠️ CRITICAL: Do NOT execute this task until Tasks 1–6 are all verified working.**
> All Python scripts must produce correct results before assembly.

**Objective:** Port the working `.py` scripts into a clean `.ipynb` notebook. The notebook should tell a coherent story: preprocessing → training 3 models → evaluation → comparison → conclusion.

**Files:**
- Modify: `forex_forecasting.ipynb` (overwrite cells with v2 pipeline code)

**Step 1: Write notebook structure (markdown cells + code cells)**

Create a new notebook with these sections:

| Section | Cells | Source |
|---------|-------|--------|
| **Title + Spec** | 2 markdown | Project title, data description (2020-2026, 2.3M rows, log-return target) |
| **1. Imports** | 1 code | All imports: torch, xgboost, sklearn, pandas, matplotlib, seaborn |
| **2. Preprocessing** | 1 code | Copy from `src/preprocess.py` (load → engineer → split → scale → save data.pt) |
| **3. Helper Functions** | 1 code | `compute_metrics`, `plot_predictions`, plot helper functions |
| **4.1 MLP Training** | 1 code | Copy from `src/train_mlp_v2.py` (load data.pt → train → evaluate → save) |
| **4.2 KNN Training** | 1 code | Copy from `src/train_knn_v2.py` (grid search CPU → GPU inference → evaluate → save) |
| **4.3 XGBoost Training** | 1 code | Copy from `src/train_xgb_v2.py` (load data.pt → GridSearchCV → evaluate → save) |
| **5. Comparison** | 2 code | Model comparison table + cumulative return plot |
| **6. Analysis** | 2 code | Feature importance (XGBoost), correlation heatmap, silhouette, PCA |
| **7. Parameter Tuning** | 2 code | KNN k-values sweep, XGBoost lr vs depth heatmap |
| **8. Conclusion** | 1 markdown | Final results table, analysis, caveats (log returns, regime shift, random walk) |

**Step 2: Adapt code for notebook format**

Key adaptations from `.py` to `.ipynb`:
- Remove `#!/usr/bin/env python3` shebangs
- Replace `print(..., flush=True)` with `print(...)` (Jupyter auto-flushes)
- Replace `if __name__ == "__main__"` blocks — code runs directly in cell
- Use relative imports OR inline the code (recommended: inline to keep notebook self-contained)
- For XGBoost cell: use `!pip install xgboost` if needed, or document venv requirement

**Step 3: Verify notebook runs end-to-end**

```bash
ssh bob@desktop-linux.netbird.selfhosted 'cd /home/bob/Documents/git/fp-ml && .venv-torch/bin/python3 -m jupyter nbconvert --to notebook --execute --inplace forex_forecasting.ipynb 2>&1 | tail -20'
```

If `nbconvert` not installed:
```bash
ssh bob@desktop-linux.netbird.selfhosted 'cd /home/bob/Documents/git/fp-ml && .venv-torch/bin/pip install jupyter nbconvert'
```

**Step 4: Verify all outputs in the executed notebook**

Check that the executed notebook has:
- [ ] All cells have execution numbers (no execution errors)
- [ ] `outputs/preprocessed/data.pt` loaded
- [ ] MLP, KNN, XGBoost training completed with metrics printed
- [ ] Comparison table and plot exist in outputs
- [ ] Conclusion has real numbers (not placeholders)

**Step 5: Commit**

```bash
git add forex_forecasting.ipynb && git commit -m "notebook: v2 GPU log-return pipeline (MLP+KNN+XGBoost)"
git push
```

---

## Dependencies Between Tasks

```
Task 1 (preprocess) ──┬── Task 2 (MLP)
                      ├── Task 3 (KNN)
                      └── Task 4 (XGBoost)
                              │
                              ▼
                       Task 5 (compare)
                              │
                              ▼
                       Task 6 (consolidate)
                              │
                              ▼
                    🔬 Task 7 (notebook assembly)
                         [GATE: all .py verified]
```

Tasks 2, 3, 4 can run in parallel after Task 1 completes.
Task 7 is **GATED** — do NOT start until Tasks 1–6 all produce correct outputs.

---

## 🔬 Testing Philosophy

| Phase | Scope | Fail Fast? |
|-------|-------|-----------|
| Tasks 1–5 | Standalone `.py` files | ✅ Yes — fix bugs in scripts, not in notebook |
| Task 6 | Verify all outputs exist | ✅ Yes — gate before notebook assembly |
| Task 7 | Assemble into `.ipynb` | Only after all scripts verified |

**Rationale:** Debugging in Jupyter is painful — hidden state, cells run out of order, stale variables. Debug in `.py` files where errors are explicit and reproducible. Only port to notebook once everything works.

---

## Verification Checklist

- [ ] `outputs/preprocessed/data.pt` exists (~1.2GB)
- [ ] `outputs/models/mlp_v2.pt` exists (~500KB)
- [ ] `outputs/models/knn_v2.pkl` exists
- [ ] `outputs/models/xgb_v2.json` exists
- [ ] `outputs/plots/compare_v2.png` exists
- [ ] All 3 models have R² > 0 (positive, beating mean baseline)
- [ ] `forex_forecasting.ipynb` updated with v2 pipeline and executes end-to-end without errors
- [ ] All notebook cells produce output (no empty cells)
- [ ] Conclusion section has real numbers from Task 6 output
- [ ] Pushed to GitHub

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Log returns near-zero → R² numerically unstable | Use raw MSE/RMSE as primary metric, R² as secondary |
| KNN GPU OOM with CHUNK=500K | Reduce to 300K, accept slower runtime |
| XGBoost HIP not available in .venv | Fall back to CPU `tree_method='hist'`, adds ~50% time |
| data.pt too large for 8GB | Load tensors with `.to(device)` only when needed, keep on CPU otherwise |
| Jupyter kernel dies during nbconvert | Run cells individually first, identify memory-heavy cells |
| XGBoost needs different venv than PyTorch | Document that XGBoost cell uses `.venv/` Python or install xgboost in `.venv-torch` |
