"""Load, validate, and return the USD/CHF 1-minute dataset."""
import pandas as pd
from src.config import INPUT_CSV, DROP_COLS, TARGET_COL


def load_data(csv_path: str = INPUT_CSV) -> pd.DataFrame:
    """Load CSV, parse datetime index, drop useless columns, validate."""
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    df.drop(columns=DROP_COLS, inplace=True, errors="ignore")
    assert df.isnull().sum().sum() == 0, "NaN values found in dataset"
    assert len(df) > 100_000, f"Dataset too small: {len(df)} rows"
    for c in [TARGET_COL, "open", "high", "low"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(inplace=True)
    return df


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded: {len(df):,} rows")
    print(f"Range:  {df.index.min()}  ->  {df.index.max()}")
    print(f"Nulls:  {df.isnull().sum().sum()}")
    print(df.head())
