import os
import zipfile
from io import StringIO
import pandas as pd

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
OUT_FILE = os.path.join(OUT_DIR, "USDCHF_1min_3y.csv")
os.makedirs(OUT_DIR, exist_ok=True)

COL_NAMES = ["date", "time", "open", "high", "low", "close", "volume"]


def _parse_csv(text):
    df = pd.read_csv(
        StringIO(text),
        sep=",",
        header=None,
        names=COL_NAMES,
    )
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M")
    df.drop(columns=["date", "time"], inplace=True)
    df.set_index("datetime", inplace=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def aggregate():
    zips = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(".zip")])
    print(f"Found {len(zips)} zip files")

    frames = []
    for fname in zips:
        fpath = os.path.join(RAW_DIR, fname)
        print(f"  {fname}")
        with zipfile.ZipFile(fpath) as z:
            csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
            text = z.read(csv_name).decode("utf-8")
            df = _parse_csv(text)
            frames.append(df)

    full = pd.concat(frames)
    full.sort_index(inplace=True)
    full = full[~full.index.duplicated(keep="first")]

    full.to_csv(OUT_FILE)
    print(f"\nWritten: {OUT_FILE}")
    print(f"Rows:    {len(full):,}")
    print(f"Range:   {full.index.min()} → {full.index.max()}")


if __name__ == "__main__":
    aggregate()
