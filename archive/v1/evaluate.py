"""Evaluation metrics and plotting for regression models."""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for SSH
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import PLOT_DIR, METRIC_DIR


def compute_metrics(y_true, y_pred, model_name="model"):
    """Compute standard regression metrics.

    Returns:
        dict with: rmse, mae, mape, r2, directional_accuracy_pct, n_samples
    """
    # Filter out NaN predictions
    mask = ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE - avoid division by zero
    nonzero_mask = y_true != 0
    mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100

    # Directional accuracy
    actual_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    dir_acc = np.mean(actual_dir == pred_dir) * 100

    metrics = {
        "model": model_name,
        "rmse": round(rmse, 8),
        "mae": round(mae, 8),
        "mape_pct": round(mape, 4),
        "r2": round(r2, 6),
        "directional_accuracy_pct": round(dir_acc, 2),
        "n_samples": len(y_true),
    }

    print("\n" + "=" * 50)
    print(f"  {model_name} - Evaluation")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics


def plot_predictions(
    y_true, y_pred,
    model_name="model",
    n_points=500,
):
    """Plot actual vs predicted prices (time-series overlay + residuals)."""
    y_true = y_true[-n_points:]
    y_pred = y_pred[-n_points:]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(
        f"{model_name} - Actual vs Predicted (Last {n_points} Points)",
        fontsize=14,
    )

    ax = axes[0]
    ax.plot(y_true, label="Actual", color="blue", linewidth=0.8, alpha=0.8)
    ax.plot(y_pred, label="Predicted", color="red", linewidth=0.8, alpha=0.8)
    ax.set_ylabel("Close Price")
    ax.set_title("Price Overlay")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    residuals = y_true - y_pred
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.plot(residuals, color="purple", linewidth=0.5, alpha=0.7)
    ax.set_ylabel("Residual (Actual - Pred)")
    ax.set_xlabel("Test Sample Index")
    ax.set_title(f"Residuals (mean={residuals.mean():.6f}, std={residuals.std():.6f})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{PLOT_DIR}/{model_name.lower().replace(' ', '_')}_predictions.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved plot:", path)
    return path


def plot_residual_hist(y_true, y_pred, model_name="model"):
    """Plot histogram of residuals (prediction errors)."""
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=80, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(x=0, color="red", linewidth=1, linestyle="--")
    ax.set_xlabel("Prediction Error (Actual - Predicted)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{model_name} - Residual Distribution\n(mean={residuals.mean():.6f}, std={residuals.std():.6f})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{PLOT_DIR}/{model_name.lower().replace(' ', '_')}_residual_hist.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved plot:", path)
    return path
