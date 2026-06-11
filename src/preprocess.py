#!/usr/bin/env python3
"""Preprocess 2.3M-row forex data into PyTorch tensors. Run once, all models share output."""
import sys, os, time
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

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

for lag in [1, 2, 3, 5, 10, 15, 30, 60]:
    df[f"close_lag_{lag}"] = df["close"].shift(lag)

for w in [5, 10, 20, 50]:
    df[f"rolling_mean_{w}"] = df["close"].rolling(w).mean()
    df[f"rolling_std_{w}"] = df["close"].rolling(w).std()
    df[f"rolling_max_{w}"] = df["close"].rolling(w).max()
    df[f"rolling_min_{w}"] = df["close"].rolling(w).min()

df["hl_spread"] = df["high"] - df["low"]
df["ohlc_mean"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
df["log_return"] = np.log(df["close"] / df["close"].shift(1))
df["pct_change"] = df["close"].pct_change()

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
