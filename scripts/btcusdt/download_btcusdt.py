#!/usr/bin/env python3
"""Download BTCUSDT 1-min kline CSVs from data.binance.vision.

Usage:
    python scripts/btcusdt/download_btcusdt.py

Extends through 2026-05. Auto-handles Binance's 2025+ microsecond timestamp change.
"""
import os
import time
import zipfile
from io import BytesIO

import pandas as pd
import requests

OUT_DIR = "data/btcusdt"
OUT_FILE = os.path.join(OUT_DIR, "BTCUSDT_1min.csv")
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/"

START_YEAR, START_MONTH = 2020, 1
END_YEAR, END_MONTH = 2026, 5

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "n_trades",
    "taker_buy_base_vol", "taker_buy_quote_vol", "ignore",
]

os.makedirs(OUT_DIR, exist_ok=True)


def iter_months(start_year, start_month, end_year, end_month):
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def download_month(year: int, month: int) -> pd.DataFrame | None:
    fname = f"BTCUSDT-1m-{year}-{month:02d}.zip"
    fpath = os.path.join(OUT_DIR, fname)

    if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
        print(f"  SKIP {year}-{month:02d} (cached)", flush=True)
    else:
        url = BASE_URL + fname
        print(f"  GET  {url} ... ", end="", flush=True)
        r = requests.get(url, timeout=120)
        if r.status_code != 200:
            print(f"HTTP {r.status_code} — skipping")
            return None
        with open(fpath, "wb") as f:
            f.write(r.content)
        print(f"OK ({len(r.content) // 1024:,} KB)")
        time.sleep(0.3)

    try:
        with zipfile.ZipFile(fpath) as z:
            csv_name = next(n for n in z.namelist() if n.endswith(".csv"))
            raw = z.read(csv_name)
    except Exception as e:
        print(f"  ERROR reading {fpath}: {e}")
        return None

    df = pd.read_csv(BytesIO(raw), header=None, names=KLINE_COLS)
    # Auto-detect timestamp unit: Binance changed from ms to us in 2025+
    sample_ts = df["open_time"].iloc[0]
    unit = "us" if sample_ts > 1e15 else "ms"
    df["datetime"] = pd.to_datetime(df["open_time"], unit=unit, utc=True)
    df = df.set_index("datetime").sort_index()
    keep = ["open", "high", "low", "close", "volume"]
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[keep]


def main():
    frames = []
    for year, month in iter_months(START_YEAR, START_MONTH, END_YEAR, END_MONTH):
        df = download_month(year, month)
        if df is not None:
            frames.append(df)

    if not frames:
        print("No data downloaded!")
        return

    full = pd.concat(frames)
    full.sort_index(inplace=True)
    full = full[~full.index.duplicated(keep="first")]
    full.dropna(inplace=True)

    full.to_csv(OUT_FILE)
    for f in os.listdir(OUT_DIR):
        if f.endswith(".zip"):
            os.remove(os.path.join(OUT_DIR, f))

    print(f"\n=== DONE ===")
    print(f"Written : {OUT_FILE}")
    print(f"Rows    : {len(full):,}")
    print(f"Range   : {full.index.min()} -> {full.index.max()}")
    print(f"Size    : {os.path.getsize(OUT_FILE) // 1024:,} KB")


if __name__ == "__main__":
    main()
