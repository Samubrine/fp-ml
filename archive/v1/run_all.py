#!/usr/bin/env python3
"""Master pipeline: load -> features -> split -> scale -> train 3 -> evaluate -> compare.

Usage:
    .venv/bin/python3 -m src.run_all
"""
from src.data_loader import load_data
from src.feature_engineering import build_features, scale_features
from src.data_splitting import split_data
from src.train_knn import train_knn
from src.train_svr import train_svr
from src.train_xgboost import train_xgboost
from src.evaluate import compute_metrics, plot_predictions, plot_residual_hist
from src.compare import plot_metrics_bar, save_metrics


def main():
    print("=" * 70)
    print("  USD/CHF FOREX FORECASTING - 3 MODEL COMPARISON")
    print("  KNN Regressor  |  SVR  |  XGBoost Regressor")
    print("=" * 70)

    # 1. Load & Build Features
    print("\n[1/7] Loading data...")
    df = load_data()
    print(f"       Loaded: {len(df):,} rows")

    print("\n[2/7] Building features...")
    df = build_features(df)
    print(f"       Feature-built: {len(df):,} rows")

    # 2. Split
    print("\n[3/7] Splitting data (chronological)...")
    X_tr, X_v, X_te, y_tr, y_v, y_te, feats = split_data(df)
    print(f"       {len(feats)} features")

    # 3. Scale
    print("\n[4/7] Scaling features (StandardScaler)...")
    X_tr_s, X_v_s, X_te_s, _ = scale_features(X_tr, X_v, X_te)

    # 4. Train 3 Models
    all_metrics = []

    print("\n[5a/7] Training KNN Regressor...")
    knn_model = train_knn(X_tr_s, y_tr)
    knn_pred = knn_model.predict(X_te_s)
    knn_metrics = compute_metrics(y_te, knn_pred, "KNN Regressor")
    plot_predictions(y_te, knn_pred, "KNN Regressor")
    plot_residual_hist(y_te, knn_pred, "KNN Regressor")
    all_metrics.append(knn_metrics)

    print("\n[5b/7] Training SVR...")
    svr_model = train_svr(X_tr_s, y_tr)
    svr_pred = svr_model.predict(X_te_s)
    svr_metrics = compute_metrics(y_te, svr_pred, "SVR")
    plot_predictions(y_te, svr_pred, "SVR")
    plot_residual_hist(y_te, svr_pred, "SVR")
    all_metrics.append(svr_metrics)

    print("\n[5c/7] Training XGBoost Regressor...")
    xgb_model = train_xgboost(X_tr_s, y_tr, X_v_s, y_v)
    xgb_pred = xgb_model.predict(X_te_s)
    xgb_metrics = compute_metrics(y_te, xgb_pred, "XGBoost Regressor")
    plot_predictions(y_te, xgb_pred, "XGBoost Regressor")
    plot_residual_hist(y_te, xgb_pred, "XGBoost Regressor")
    all_metrics.append(xgb_metrics)

    # 5. Compare
    print("\n[6/7] Generating comparison report...")
    save_metrics(all_metrics)
    plot_metrics_bar(all_metrics)

    # 6. Summary
    print("\n[7/7] ====== FINAL COMPARISON ======")
    header = "{:20s} {:>10s} {:>10s} {:>8s} {:>8s} {:>8s}".format("Model", "RMSE", "MAE", "MAPE%", "R2", "DirAcc%")
    print(header)
    print("-" * 70)
    for m in all_metrics:
        print(f"{m["model"]:<20} {m["rmse"]:>10.6f} {m["mae"]:>10.6f} {m["mape_pct"]:>8.4f} {m["r2"]:>8.4f} {m["directional_accuracy_pct"]:>8.2f}")

    print("\nPipeline complete!")
    print("   Models:  outputs/models/")
    print("   Plots:   outputs/plots/")
    print("   Metrics: outputs/metrics/")


if __name__ == "__main__":
    main()

