#!/usr/bin/env python3
"""Download USDCHF 1-min ASCII yearly zips from histdata.com (2020-2026)."""
import os
import time
import zipfile
from io import StringIO
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup
import pandas as pd

OUT_DIR = "data/usdchf"
OUT_FILE = os.path.join(OUT_DIR, "USDCHF_1min_2020_2026.csv")
os.makedirs(OUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}
BASE = "https://www.histdata.com"
YEARS = list(range(2020, 2026))
MONTHS_2026 = range(1, 6)


def download_yearly_ascii(year):
    fname = f"HISTDATA_COM_ASCII_USDCHF_M1_{year}.zip"
    fpath = os.path.join(OUT_DIR, fname)
    if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
        print(f"  SKIP {year} (exists)")
        return fpath

    print(f"  Download {year} yearly ...", end=" ", flush=True)
    s = requests.Session()
    s.headers.update(HEADERS)

    url = f"{BASE}/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/USDCHF/{year}"
    r = s.get(url, timeout=30)
    if r.status_code != 200:
        print(f"HTTP {r.status_code}")
        return None

    soup = BeautifulSoup(r.text, "html.parser")
    form = soup.find("form", id="file_down")
    if not form:
        print("NO FORM")
        return None

    data = {i.get("name"): i.get("value", "") for i in form.find_all("input") if i.get("name")}
    if not data.get("tk"):
        print("NO TOKEN")
        return None

    headers = dict(HEADERS)
    headers["Referer"] = url
    headers["Origin"] = BASE

    r2 = s.post(BASE + "/get.php", data=data, headers=headers, timeout=120, allow_redirects=True)
    if r2.status_code != 200 or len(r2.content) < 100:
        print(f"FAIL ({len(r2.content)} bytes)")
        return None

    with open(fpath, "wb") as f:
        f.write(r2.content)
    print(f"OK ({len(r2.content)//1024:,} KB)")
    time.sleep(1)
    return fpath


def download_month_ascii(year, month):
    fname = f"USDCHF_{year}_{month:02d}.zip"
    fpath = os.path.join(OUT_DIR, fname)
    if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
        print(f"  SKIP {year}-{month:02d} (exists)")
        return fpath

    print(f"  Download {year}-{month:02d} ...", end=" ", flush=True)
    s = requests.Session()
    s.headers.update(HEADERS)

    url = f"{BASE}/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/USDCHF/{year}/{month}"
    r = s.get(url, timeout=30)
    if r.status_code != 200:
        print(f"HTTP {r.status_code}")
        return None

    soup = BeautifulSoup(r.text, "html.parser")
    form = soup.find("form", id="file_down")
    if not form:
        print("NO FORM")
        return None

    data = {i.get("name"): i.get("value", "") for i in form.find_all("input") if i.get("name")}
    if not data.get("tk"):
        print("NO DATA")
        return None

    headers = dict(HEADERS)
    headers["Referer"] = url
    headers["Origin"] = BASE

    r2 = s.post(BASE + "/get.php", data=data, headers=headers, timeout=120, allow_redirects=True)
    if r2.status_code != 200 or len(r2.content) < 100:
        print(f"FAIL ({len(r2.content)} bytes)")
        return None

    with open(fpath, "wb") as f:
        f.write(r2.content)
    print(f"OK ({len(r2.content)//1024:,} KB)")
    time.sleep(1)
    return fpath


def parse_ascii_csv(text):
    df = pd.read_csv(
        StringIO(text),
        sep=";",
        header=None,
        names=["datetime_str", "open", "high", "low", "close", "volume"],
    )
    df["datetime"] = pd.to_datetime(df["datetime_str"], format="%Y%m%d %H%M%S")
    df.drop(columns=["datetime_str"], inplace=True)
    df.set_index("datetime", inplace=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def process_zip(fpath):
    with zipfile.ZipFile(fpath) as z:
        csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
        text = z.read(csv_name).decode("utf-8")
        return parse_ascii_csv(text)


def main():
    frames = []
    for year in YEARS:
        fpath = download_yearly_ascii(year)
        if fpath:
            print(f"    Parsing ...", end=" ", flush=True)
            df = process_zip(fpath)
            frames.append(df)
            print(f"{len(df):,} rows")

    for month in MONTHS_2026:
        fpath = download_month_ascii(2026, month)
        if fpath:
            print(f"    Parsing ...", end=" ", flush=True)
            df = process_zip(fpath)
            frames.append(df)
            print(f"{len(df):,} rows")

    if not frames:
        print("No data downloaded!")
        return

    full = pd.concat(frames)
    full.sort_index(inplace=True)
    full = full[~full.index.duplicated(keep="first")]

    full.to_csv(OUT_FILE)
    # Clean up zips after successful merge
    for f in os.listdir(OUT_DIR):
        if f.endswith(".zip"):
            os.remove(os.path.join(OUT_DIR, f))

    print(f"\n=== DONE ===")
    print(f"Written: {OUT_FILE}")
    print(f"Rows:    {len(full):,}")
    print(f"Range:   {full.index.min()} -> {full.index.max()}")
    print(f"Size:    {os.path.getsize(OUT_FILE)//1024:,} KB")


if __name__ == "__main__":
    main()
