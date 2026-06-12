"""Chronological train/validation/test split for time-series data."""
import numpy as np
from src.config import TRAIN_CUTOFF, VAL_CUTOFF


def split_data(df):
    """Split into train/val/test preserving temporal order.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names
    """
    from src.feature_engineering import get_feature_columns

    feature_cols = get_feature_columns(df)

    # Chronological split by date
    train_mask = df.index < TRAIN_CUTOFF
    val_mask = (df.index >= TRAIN_CUTOFF) & (df.index < VAL_CUTOFF)
    test_mask = df.index >= VAL_CUTOFF

    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, "target"].values
    X_val = df.loc[val_mask, feature_cols].values
    y_val = df.loc[val_mask, "target"].values
    X_test = df.loc[test_mask, feature_cols].values
    y_test = df.loc[test_mask, "target"].values

    total = len(df)
    pct_train = train_mask.sum() / total * 100
    pct_val = val_mask.sum() / total * 100
    pct_test = test_mask.sum() / total * 100
    print(f"Train: {len(X_train):,} rows ({pct_train:.0f}%)")
    print(f"Val:   {len(X_val):,} rows ({pct_val:.0f}%)")
    print(f"Test:  {len(X_test):,} rows ({pct_test:.0f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


if __name__ == "__main__":
    from src.data_loader import load_data
    from src.feature_engineering import build_features

    df = load_data()
    df = build_features(df)
    X_tr, X_v, X_te, y_tr, y_v, y_te, feats = split_data(df)
    print(f"Feature count: {len(feats)}")
    print(f"y_train range: {y_tr.min():.5f} - {y_tr.max():.5f}")
    print(f"y_test  range: {y_te.min():.5f} - {y_te.max():.5f}")

