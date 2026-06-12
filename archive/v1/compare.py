"""Compare all 3 models: metrics table, bar charts, CSV export."""
import json
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import METRIC_DIR, PLOT_DIR


def plot_metrics_bar(all_metrics: list) -> str:
    """Side-by-side bar chart comparing RMSE, MAE, R2, DirAcc across models."""
    models = [m["model"] for m in all_metrics]
    rmse_vals = [m["rmse"] for m in all_metrics]
    mae_vals = [m["mae"] for m in all_metrics]
    r2_vals = [m["r2"] for m in all_metrics]
    dir_vals = [m["directional_accuracy_pct"] for m in all_metrics]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Comparison - USD/CHF Forecasting", fontsize=16, fontweight="bold")

    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    # RMSE
    axes[0, 0].bar(models, rmse_vals, color=colors)
    axes[0, 0].set_title("RMSE (lower is better)")
    axes[0, 0].set_ylabel("RMSE")
    for bar, val in zip(axes[0, 0].patches, rmse_vals):
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.6f}", ha="center", va="bottom", fontsize=9,
        )

    # MAE
    axes[0, 1].bar(models, mae_vals, color=colors)
    axes[0, 1].set_title("MAE (lower is better)")
    axes[0, 1].set_ylabel("MAE")
    for bar, val in zip(axes[0, 1].patches, mae_vals):
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.6f}", ha="center", va="bottom", fontsize=9,
        )

    # R2
    axes[1, 0].bar(models, r2_vals, color=colors)
    axes[1, 0].set_title("R2 Score (higher is better)")
    axes[1, 0].set_ylabel("R2")
    for bar, val in zip(axes[1, 0].patches, r2_vals):
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.4f}", ha="center", va="bottom", fontsize=9,
        )

    # Directional Accuracy
    axes[1, 1].bar(models, dir_vals, color=colors)
    axes[1, 1].set_title("Directional Accuracy % (higher is better)")
    axes[1, 1].set_ylabel("%")
    axes[1, 1].axhline(y=50, color="gray", linewidth=0.5, linestyle="--", label="Random")
    axes[1, 1].legend()
    for bar, val in zip(axes[1, 1].patches, dir_vals):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()
    path = PLOT_DIR + "/model_comparison_metrics.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved:", path)
    return path


def save_metrics(all_metrics: list) -> str:
    """Save all metrics as JSON and CSV."""
    # JSON
    json_path = METRIC_DIR + "/all_metrics.json"
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print("Saved:", json_path)

    # CSV
    csv_path = METRIC_DIR + "/all_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(all_metrics)
    print("Saved:", csv_path)
    return json_path


if __name__ == "__main__":
    # Test with dummy data
    test_metrics = [
        {
            "model": "KNN Regressor", "rmse": 0.0005, "mae": 0.0003,
            "mape_pct": 0.05, "r2": 0.95, "directional_accuracy_pct": 51.2,
            "n_samples": 56000,
        },
        {
            "model": "SVR", "rmse": 0.0008, "mae": 0.0005,
            "mape_pct": 0.08, "r2": 0.88, "directional_accuracy_pct": 50.5,
            "n_samples": 56000,
        },
        {
            "model": "XGBoost Regressor", "rmse": 0.0004, "mae": 0.0002,
            "mape_pct": 0.04, "r2": 0.97, "directional_accuracy_pct": 53.1,
            "n_samples": 56000,
        },
    ]
    plot_metrics_bar(test_metrics)
    save_metrics(test_metrics)
    print("Compare module OK")

