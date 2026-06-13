"""Centralised constants and paths for the USD/CHF forecasting project."""
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
METRIC_DIR = os.path.join(OUTPUT_DIR, "metrics")

INPUT_CSV = os.path.join(DATA_DIR, "USDCHF_1min_2020_2026.csv")

TRAIN_CUTOFF = "2025-01-01"
VAL_CUTOFF = "2025-09-01"
TARGET_COL = "close"
DROP_COLS = ["volume"]

LAG_PERIODS = [1, 2, 3, 5, 10, 15, 30, 60]
ROLLING_WINDOWS = [5, 10, 30, 60]
ROLLING_STATS = ["mean", "std", "min", "max"]
LOOKAHEAD = 1

KNN_PARAM_GRID = {
    "n_neighbors": [5, 10, 20, 50],
    "weights": ["uniform", "distance"],
    "p": [2],
}

SVR_SAMPLE_SIZE = 50_000
SVR_PARAM_GRID = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    "epsilon": [0.001, 0.01, 0.1],
}

XGB_PARAM_GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.1, 0.3],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}

RANDOM_SEED = 42
CV_FOLDS = 3
