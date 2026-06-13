#!/usr/bin/env python3
"""Compare v2 models (MLP, KNN, XGBoost) — load results + generate comparison chart."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

model_files = {
    "MLP (GPU)": "outputs/mlp_v2_results.json",
    "KNN (GPU)": "outputs/knn_v2_results.json",
    "XGBoost (GPU)": "outputs/xgb_v2_results.json",
}

results = {}
for name, path in model_files.items():
    if os.path.exists(path):
        with open(path) as f:
            results[name] = json.load(f)
        print(f"Loaded {name}: R²={results[name]['r2']:.4f}, RMSE={results[name]['rmse']:.6f}")
    else:
        print(f"SKIP {name}: {path} not found (model not trained yet?)")

if len(results) < 3:
    print(f"\nOnly {len(results)}/3 models available — skipping comparison chart.")
    sys.exit(0)

# ---- Comparison bar chart ----
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("MLP vs KNN vs XGBoost — Log-Return Forecasting (2020-2026 USD/CHF)", fontsize=14, fontweight="bold")

metrics = ["rmse", "mae", "r2", "mape", "directional_accuracy"]
titles = ["RMSE (lower=better)", "MAE (lower=better)", "R² (higher=better)", "MAPE % (lower=better)", "Directional Acc % (higher=better)"]
colors = ["#2196F3", "#FF9800", "#4CAF50"]

for ax, metric, title in zip(axes.flat[:5], metrics, titles):
    names = list(results.keys())
    values = [results[n][metric] for n in names]
    bars = ax.bar(names, values, color=colors[:len(names)], edgecolor="white", linewidth=1.2)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(metric.upper())
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(values)*0.02),
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Training time
ax = axes[1, 2]
names = list(results.keys())
times = [results[n].get("total_time_s", results[n].get("training_time_s", 0)) / 60 for n in names]
bars = ax.bar(names, times, color=colors[:len(names)], edgecolor="white", linewidth=1.2)
ax.set_title("Training Time (minutes)", fontsize=11, fontweight="bold")
ax.set_ylabel("Minutes")
for bar, val in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{val:.1f}m", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
os.makedirs("outputs/plots", exist_ok=True)
plt.savefig("outputs/plots/v2_comparison.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: outputs/plots/v2_comparison.png")

# ---- Summary table ----
print(f"\n{'='*80}")
print(f"{'Model':<18} {'R²':>8} {'RMSE':>10} {'MAE':>10} {'MAPE%':>8} {'DirAcc%':>9} {'Time':>8}")
print(f"{'-'*80}")
for name in ["MLP (GPU)", "KNN (GPU)", "XGBoost (GPU)"]:
    if name not in results: continue
    r = results[name]
    t = r.get("total_time_s", r.get("training_time_s", 0))
    print(f"{name:<18} {r['r2']:>8.4f} {r['rmse']:>10.6f} {r['mae']:>10.6f} {r['mape']:>8.2f} {r['directional_accuracy']:>9.1f} {t/60:>7.1f}m")

# ---- Best model ----
best = max(results.items(), key=lambda x: x[1]["r2"])
print(f"\n★ Best model: {best[0]} (R²={best[1]['r2']:.4f})")
print("="*80)

# Save summary JSON
summary = {
    "models": {n: {k: v for k, v in r.items() if k in ["r2", "rmse", "mae", "mape", "directional_accuracy"]}
               for n, r in results.items()},
    "best_model": best[0],
    "best_r2": best[1]["r2"],
}
with open("outputs/v2_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Saved: outputs/v2_summary.json")
