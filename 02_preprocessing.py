"""
UCMS - Step 02: Preprocessing & Feature Engineering
=====================================================
Purpose : Load the full Chicago Crimes CSV (chunked for RAM safety), clean it,
          filter to 2010-2025, engineer temporal + spatial features, assign
          severity weights, and persist a cleaned Parquet file for all
          downstream analysis.

Usage   : python 02_preprocessing.py
Output  : outputs/chicago_crimes_clean.parquet   (~150-400 MB depending on years)
          outputs/02_preprocessing_report.txt
"""

import os, sys, time, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR  = "data"
OUT_DIR   = "outputs"
LOG_DIR   = "logs"

# Use Indian crime dataset if present, else fallback
INDIAN_CSV = "indian_crimes_2010_2025.csv"
CHICAGO_CSV = "Crimes_-_2001_to_Present_20250603.csv"
CSV_NAME  = INDIAN_CSV if os.path.exists(os.path.join(DATA_DIR, INDIAN_CSV)) else CHICAGO_CSV
CSV_PATH  = os.path.join(DATA_DIR, CSV_NAME)

OUT_PARQ  = os.path.join(OUT_DIR, "chicago_crimes_clean.parquet")
OUT_PARQ_IND = os.path.join(OUT_DIR, "indian_crimes_clean.parquet")

YEAR_MIN  = 2010
YEAR_MAX  = 2025

CHUNK_SZ  = 200_000   # rows per chunk — safe for 8 GB RAM laptops
# ──────────────────────────────────────────────────────────────────────────────

# ── Crime Severity Weights ─────────────────────────────────────────────────────
# Calibrated for Indian IPC / BNS offenses + FBI / Chicago mappings
SEVERITY_MAP = {
    # Indian IPC / BNS Offenses
    "MURDER":                           1.00,
    "CULPABLE HOMICIDE":                0.95,
    "RAPE / SEXUAL ASSAULT":            0.95,
    "DACOITY":                          0.90,
    "ROBBERY":                          0.85,
    "KIDNAPPING & ABDUCTION":           0.85,
    "DOMESTIC VIOLENCE / 498A":         0.80,
    "ARSON":                            0.80,
    "GRIEVOUS HURT / ASSAULT":          0.75,
    "CRIME AGAINST WOMEN":              0.70,
    "ARMS ACT VIOLATION":               0.65,
    "BURGLARY / HOUSE BREAKING":        0.65,
    "EXTORTION":                        0.65,
    "RIOTING / UNLAWFUL ASSEMBLY":      0.60,
    "RIOTS / UNLAWFUL ASSEMBLY":        0.60,
    "MOTOR VEHICLE THEFT":              0.55,
    "SNATCHING / STREET THEFT":         0.50,
    "NARCOTICS / NDPS ACT":             0.50,
    "CYBER CRIME":                      0.45,
    "CHEATING & FORGERY":               0.40,

    # Legacy mappings (for full compatibility)
    "HOMICIDE":                         1.00,
    "CRIMINAL SEXUAL ASSAULT":          0.90,
    "KIDNAPPING":                       0.85,
    "HUMAN TRAFFICKING":                0.80,
    "ASSAULT":                          0.70,
    "BATTERY":                          0.65,
    "BURGLARY":                         0.60,
    "CRIM SEXUAL ASSAULT":              0.90,
    "WEAPONS VIOLATION":                0.75,
    "SEX OFFENSE":                      0.70,
    "STALKING":                         0.60,
    "INTIMIDATION":                     0.55,
    "THEFT":                            0.50,
    "CRIMINAL DAMAGE":                  0.45,
    "OFFENSE INVOLVING CHILDREN":       0.80,
    "PROSTITUTION":                     0.35,
    "GAMBLING":                         0.20,
    "NARCOTICS":                        0.40,
    "OTHER NARCOTIC VIOLATION":         0.40,
    "LIQUOR LAW VIOLATION":             0.20,
    "PUBLIC PEACE VIOLATION":           0.30,
    "CRIMINAL TRESPASS":                0.30,
    "INTERFERENCE WITH PUBLIC OFFICER": 0.25,
    "DECEPTIVE PRACTICE":               0.35,
    "FRAUD":                            0.40,
    "OBSCENITY":                        0.20,
    "PUBLIC INDECENCY":                 0.20,
    "NON-CRIMINAL":                     0.05,
    "NON - CRIMINAL":                   0.05,
    "OTHER OFFENSE":                    0.30,
    "CONCEALED CARRY LICENSE VIOLATION":0.35,
    "RITUALISM":                        0.20,
}
DEFAULT_SEVERITY = 0.30


os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def log(msg):
    print(msg)


def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Clean and engineer features for one chunk."""

    # ── Parse datetime ────────────────────────────────────────────────────────
    # Support two formats:
    #   1. Combined: "01/15/2015 09:30:00 PM"  (Chicago-style)
    #   2. Separate Date + Time columns: "01/15/2015" + "09:30:00"  (Indian CSV)
    if "Time" in chunk.columns and chunk["Date"].str.len().median() <= 10:
        # Indian format: combine Date + Time
        combined = chunk["Date"].str.strip() + " " + chunk["Time"].str.strip()
        chunk["Date"] = pd.to_datetime(combined, format="%m/%d/%Y %H:%M:%S", errors="coerce")
    else:
        chunk["Date"] = pd.to_datetime(
            chunk["Date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
        )
        if chunk["Date"].isna().all():
            chunk["Date"] = pd.to_datetime(chunk["Date"], errors="coerce")

    chunk.dropna(subset=["Date"], inplace=True)

    # ── Filter years ──────────────────────────────────────────────────────────
    chunk = chunk[(chunk["Date"].dt.year >= YEAR_MIN) &
                  (chunk["Date"].dt.year <= YEAR_MAX)].copy()
    if chunk.empty:
        return chunk

    # ── Temporal features ─────────────────────────────────────────────────────
    chunk["year"]        = chunk["Date"].dt.year
    chunk["month"]       = chunk["Date"].dt.month        # 1-12
    chunk["day_of_week"] = chunk["Date"].dt.dayofweek    # 0=Mon … 6=Sun
    chunk["hour"]        = chunk["Date"].dt.hour          # 0-23
    chunk["is_weekend"]  = chunk["day_of_week"].isin([5, 6]).astype(int)

    # Peak-hour risk multiplier
    def hour_risk(h):
        if   0  <= h <  6:  return 0.85    # late night
        elif 6  <= h < 10:  return 0.30    # early morning
        elif 10 <= h < 14:  return 0.35    # late morning
        elif 14 <= h < 18:  return 0.50    # afternoon
        elif 18 <= h < 21:  return 0.70    # evening
        else:               return 0.80    # night (21-24)

    chunk["hour_risk_weight"] = chunk["hour"].map(hour_risk)

    # ── Spatial features ──────────────────────────────────────────────────────
    chunk["Community Area"] = pd.to_numeric(chunk.get("Community Area", np.nan),
                                             errors="coerce")
    chunk["Beat"]           = pd.to_numeric(chunk.get("Beat", np.nan),
                                             errors="coerce")
    # Preserve district as string
    if "District" in chunk.columns:
        chunk["District"] = chunk["District"].astype(str)
    chunk["Latitude"]       = pd.to_numeric(chunk.get("Latitude", np.nan),
                                             errors="coerce")
    chunk["Longitude"]      = pd.to_numeric(chunk.get("Longitude", np.nan),
                                             errors="coerce")

    # Drop rows with no spatial anchor at all
    chunk.dropna(subset=["Community Area"], inplace=True)

    # ── Severity weight ───────────────────────────────────────────────────────
    primary = chunk.get("Primary Type", pd.Series(dtype=str))
    chunk["Primary Type"] = primary.str.strip().str.upper()
    chunk["severity_weight"] = chunk["Primary Type"].map(SEVERITY_MAP).fillna(DEFAULT_SEVERITY)

    # ── Arrest / Domestic flags ───────────────────────────────────────────────
    if "Arrest" in chunk.columns:
        chunk["Arrest"] = chunk["Arrest"].astype(bool).astype(int)
    else:
        chunk["Arrest"] = 0
    if "Domestic" in chunk.columns:
        chunk["Domestic"] = chunk["Domestic"].astype(bool).astype(int)
    else:
        chunk["Domestic"] = 0

    # ── Select only needed columns ────────────────────────────────────────────
    keep = [
        "Date", "year", "month", "day_of_week", "hour", "is_weekend",
        "hour_risk_weight", "severity_weight",
        "Primary Type", "Description", "Location Description",
        "Community Area", "Beat", "District", "Ward",
        "Latitude", "Longitude", "Arrest", "Domestic",
        "ID", "IUCR", "FBI Code",
    ]
    keep = [c for c in keep if c in chunk.columns]
    return chunk[keep]


def run():
    if not os.path.exists(CSV_PATH):
        print(f"\n❌  CSV not found: {CSV_PATH}")
        sys.exit(1)

    log("=" * 60)
    log("UCMS  ▸  Step 02: Preprocessing")
    log("=" * 60)
    log(f"  Source  : {CSV_PATH}")
    log(f"  Years   : {YEAR_MIN} – {YEAR_MAX}")
    log(f"  Chunk Sz: {CHUNK_SZ:,} rows\n")

    t_start   = time.time()
    chunks_in = 0
    total_in  = 0
    total_out = 0
    parts     = []

    reader = pd.read_csv(CSV_PATH, chunksize=CHUNK_SZ, low_memory=False)

    for chunk in reader:
        chunks_in += 1
        total_in  += len(chunk)
        processed  = process_chunk(chunk)
        total_out += len(processed)
        if not processed.empty:
            parts.append(processed)
        if chunks_in % 5 == 0:
            log(f"  Chunk {chunks_in:3d} | rows in: {total_in:>10,} | kept: {total_out:>9,}")

    log(f"\n  Finished reading — total chunks: {chunks_in}")
    log(f"  Rows input : {total_in:,}")
    log(f"  Rows kept  : {total_out:,}  ({total_out/total_in*100:.1f}%)")

    log("\n  Concatenating chunks …")
    df = pd.concat(parts, ignore_index=True)

    # ── Final type cleanup ────────────────────────────────────────────────────
    df["Community Area"] = df["Community Area"].astype("Int64")
    df["Beat"]           = df["Beat"].astype("Int64")
    if "District" in df.columns:
        try:
            df["District"] = pd.to_numeric(df["District"]).astype("Int64")
        except Exception:
            df["District"] = df["District"].astype(str)
    df["Ward"]           = df["Ward"] if "Ward" in df.columns else np.nan

    # ── Save ──────────────────────────────────────────────────────────────────
    log(f"\n  Saving → {OUT_PARQ}")
    df.to_parquet(OUT_PARQ, index=False, engine="pyarrow", compression="snappy")
    log(f"  Saving → {OUT_PARQ_IND}")
    df.to_parquet(OUT_PARQ_IND, index=False, engine="pyarrow", compression="snappy")
    fsize = os.path.getsize(OUT_PARQ) / (1024 ** 2)
    elapsed = time.time() - t_start

    log(f"  Parquet size : {fsize:.1f} MB")
    log(f"  Time elapsed : {elapsed/60:.1f} min")
    log(f"\n  Shape  : {df.shape}")
    log(f"\n  dtypes :\n{df.dtypes.to_string()}")

    # ── Quick sanity ──────────────────────────────────────────────────────────
    log(f"\n  Year distribution:\n{df['year'].value_counts().sort_index().to_string()}")
    log(f"\n  Top Primary Types:\n{df['Primary Type'].value_counts().head(15).to_string()}")

    log("\n[OK] STEP 02 COMPLETE  —  run  03_eda_analysis.py  next.\n")


if __name__ == "__main__":
    run()