"""
UCMS - Step 01: Data Inspection
================================
Purpose : Quick structural audit of the Chicago Crimes CSV before any heavy
          processing.  Run this FIRST to confirm column names, dtypes, size,
          and missing-value profile so every downstream script can rely on a
          known schema.

Usage   : python 01_data_inspection.py
Output  : outputs/01_inspection_report.txt  (saved + printed)
"""

import os, sys, time
import pandas as pd
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
OUT_DIR    = "outputs"
LOG_DIR    = "logs"

# ▶ Primary Indian Crime Dataset
CSV_NAME   = "indian_crimes_2010_2025.csv"
CSV_PATH   = os.path.join(DATA_DIR, CSV_NAME)

SAMPLE_N   = 100_000   # rows read for the quick inspection pass
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def log(msg, f=None):
    print(msg)
    if f:
        f.write(msg + "\n")


def inspect(csv_path: str, sample_n: int):
    report_path = os.path.join(OUT_DIR, "01_inspection_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:

        log("=" * 70, f)
        log("UCMS  -  Indian Urban Crimes Dataset (Delhi NCT / NCRB) - Audit Report", f)
        log("=" * 70, f)

        # ── 1. File size ──────────────────────────────────────────────────────
        fsize_mb = os.path.getsize(csv_path) / (1024 ** 2)
        log(f"\nFile      : {csv_path}", f)
        log(f"    Size      : {fsize_mb:,.1f} MB", f)

        # ── 2. Row count (fast wc-style count) ───────────────────────────────
        log("\nCounting total rows (streaming) ...", f)
        t0 = time.time()
        total_rows = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1  # minus header
        log(f"    Total rows: {total_rows:,}  ({time.time()-t0:.1f}s)", f)

        # ── 3. Sample read ────────────────────────────────────────────────────
        log(f"\nReading sample of {sample_n:,} rows ...", f)
        df = pd.read_csv(csv_path, nrows=sample_n, low_memory=False)

        log(f"\n{'-'*50}", f)
        log(f"  Columns ({len(df.columns)}):", f)
        log(f"{'-'*50}", f)
        for i, col in enumerate(df.columns, 1):
            dtype   = str(df[col].dtype)
            nullpct = df[col].isna().mean() * 100
            sample_val = df[col].dropna().iloc[0] if df[col].notna().any() else "ALL NULL"
            log(f"  {i:2d}. {col:<30} | {dtype:<10} | null={nullpct:5.1f}% | e.g. {sample_val}", f)

        # ── 4. Date column analysis ───────────────────────────────────────────
        log(f"\n{'-'*50}", f)
        log("  DATE column analysis:", f)
        log(f"{'-'*50}", f)
        if "Date" in df.columns:
            df["_date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
            log(f"  Min date  : {df['_date'].min()}", f)
            log(f"  Max date  : {df['_date'].max()}", f)
            log(f"  Parse fail: {df['_date'].isna().sum():,} rows", f)
        else:
            log(" 'Date' column not found - check column names above!", f)

        # ── 5. Year column ────────────────────────────────────────────────────
        if "Year" in df.columns:
            yr = df["Year"].value_counts().sort_index()
            log(f"\n  Year distribution (sample):\n{yr.to_string()}", f)

        # ── 6. Primary Type ───────────────────────────────────────────────────
        if "Primary Type" in df.columns:
            pt = df["Primary Type"].value_counts().head(20)
            log(f"\n  Top-20 Primary Types (sample):\n{pt.to_string()}", f)

        # ── 7. Spatial coverage ───────────────────────────────────────────────
        log(f"\n{'-'*50}", f)
        log("  Spatial fields:", f)
        log(f"{'-'*50}", f)
        for col in ["Beat", "District", "Ward", "Community Area", "Latitude", "Longitude"]:
            if col in df.columns:
                nnull = df[col].isna().sum()
                uniq  = df[col].nunique()
                log(f"  {col:<20}: {uniq:>5} unique vals | {nnull:>6} nulls in sample", f)

        # ── 8. Missing value heatmap summary ─────────────────────────────────
        log(f"\n{'-'*50}", f)
        log("  Missing value summary:", f)
        log(f"{'-'*50}", f)
        miss = df.isna().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        log(miss.to_string(), f)

        log(f"\n\nReport saved - {report_path}", f)
        log("=" * 70, f)

    return df.columns.tolist()


if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print(f"\nCSV not found at: {CSV_PATH}")
        print(f"    Place the file inside the  '{DATA_DIR}/'  folder and retry.\n")
        sys.exit(1)

    cols = inspect(CSV_PATH, SAMPLE_N)
    print(f"\nColumns detected: {cols}")
    print("\nSTEP 01 COMPLETE -  run  02_preprocessing.py  next.\n")