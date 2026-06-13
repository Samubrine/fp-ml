"""Feature engineering pipeline for USD/CHF 1-min data.

Builds: lag features, rolling statistics, technical indicators (RSI, MACD, BB),
price-derived features (returns, spreads), time-based cyclical features,
and the target variable (close at t+LOOKAHEAD).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config import LAG_PERIODS, ROLLING_WINDOWS, ROLLING_STATS, TARGET_COL, LOOKAHEAD


# --- 3a. Lag Features -------------------------------------------------

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lagged close price features.

    Example: close_lag_1 = close at t-1, close_lag_5 = close at t-5
    """
    for lag in LAG_PERIODS:
        df[f"close_lag_{lag}"] = df[TARGET_COL].shift(lag)
    return df


# --- 3b. Rolling Window Features --------------------------------------

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling statistics of close price.

    For each window in ROLLING_WINDOWS, compute each stat in ROLLING_STATS.
    """
    for window in ROLLING_WINDOWS:
        roll = df[TARGET_COL].rolling(window=window)
        for stat in ROLLING_STATS:
            col_name = f"close_roll_{stat}_{window}"
            df[col_name] = getattr(roll, stat)()
    return df


# --- 3c. Price-Derived Features ---------------------------------------

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived price features: returns, spreads, ratios."""
    # Log return (close vs prev close)
    df["log_return"] = np.log(df[TARGET_COL] / df[TARGET_COL].shift(1))

    # Simple return (fractional change)
    df["pct_change"] = df[TARGET_COL].pct_change()

    # High-Low spread
    df["hl_spread"] = df["high"] - df["low"]

    # Open-Close range
    df["oc_range"] = df[TARGET_COL] - df["open"]

    # OHLC mean (central tendency)
    df["ohlc_mean"] = (df["open"] + df["high"] + df["low"] + df[TARGET_COL]) / 4

    return df


# --- 3d. Technical Indicators -----------------------------------------

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Relative Strength Index (RSI) — momentum oscillator.

    RSI = 100 - (100 / (1 + RS)), where RS = avg_gain / avg_loss
    """
    delta = df[TARGET_COL].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands: middle=MA, upper/lower = MA ± num_std * std."""
    ma = df[TARGET_COL].rolling(window=period).mean()
    std = df[TARGET_COL].rolling(window=period).std()
    df["bb_mid_20"] = ma
    df["bb_upper_20"] = ma + num_std * std
    df["bb_lower_20"] = ma - num_std * std
    df["bb_width_20"] = df["bb_upper_20"] - df["bb_lower_20"]
    df["bb_position_20"] = (df[TARGET_COL] - df["bb_lower_20"]) / df["bb_width_20"]
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """MACD: MACD line = EMA12 - EMA26, Signal = EMA9(MACD), Hist = MACD - Signal."""
    ema_12 = df[TARGET_COL].ewm(span=12, adjust=False).mean()
    ema_26 = df[TARGET_COL].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


# --- 3e. Time-Based Cyclical Features ---------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract cyclical time features from DatetimeIndex."""
    df["hour"] = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dayofweek"] = df.index.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    return df


# --- 3f. Master Pipeline ----------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Complete feature engineering pipeline.

    Builds ALL features, then drops NaN rows.

    Returns:
        DataFrame with all features + target column (close at t+LOOKAHEAD).
    """
    # 1. Target: future close price
    df["target"] = df[TARGET_COL].shift(-LOOKAHEAD)

    # 2. Feature groups (order matters — later groups may use earlier cols)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_price_features(df)
    df = add_rsi(df)
    df = add_bollinger_bands(df)
    df = add_macd(df)
    df = add_time_features(df)

    # 3. Drop rows with NaN (first ~60 rows + last LOOKAHEAD row)
    df.dropna(inplace=True)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of feature column names (exclude target + raw OHLC)."""
    exclude = {"open", "high", "low", TARGET_COL, "target",
               "hour", "dayofweek"}  # raw time cols dropped (use sin/cos)
    return [c for c in df.columns if c not in exclude]


# --- 3g. Feature Scaling ----------------------------------------------

def scale_features(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> tuple:
    """Fit StandardScaler on train, transform all splits.

    Returns:
        X_train_scaled, X_val_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
