"""
UCMS - Step 04: Historical Risk Score Generation (CCRP Component)
===================================================================
Purpose : Convert cleaned crime records into the HISTORICAL CRIME PREDICTION
          SCORE (CPS) per zone (Community Area) × time-slot.

          This is the "ground-truth" historical component that feeds into:
            • Training the GAT model  (node features)
            • Bayesian prior initialisation
            • Baseline comparison (vs KDE, RF)

Formula used (CCRP Definition — historical stream):
  ─────────────────────────────────────────────────
  raw_score(zone, slot) =
        Σ  [  frequency_weight
             × severity_weight
             × temporal_risk_weight
             × recency_weight  ]

  Normalised CPS ∈ [0, 1] using min-max per zone.

Outputs:
  outputs/zone_risk_scores.parquet       — zone × time-slot risk matrix
  outputs/zone_annual_profile.parquet    — annual aggregate per zone
  outputs/crime_type_weights.csv         — calibrated type weights
  outputs/04_risk_summary.txt
"""

import os, warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR   = "outputs"
PARQ_PATH = os.path.join(OUT_DIR, "chicago_crimes_clean.parquet")

# Time-slot granularity  (6 = four 6-hour slots/day)
TIME_SLOT_HOURS = 6     # choices: 1, 3, 6, 12, 24

# Recency decay : crimes from recent years weigh more
DECAY_BASE = 0.92       # per-year decay  (1.0 = no decay)
REFERENCE_YEAR = 2025

# ──────────────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)


def assign_time_slot(hour: pd.Series, slot_hours: int = 6) -> pd.Series:
    """Map hour → slot index (0, 1, 2, …)"""
    return (hour // slot_hours).astype(int)


def recency_weight(year: pd.Series, ref: int = REFERENCE_YEAR,
                   base: float = DECAY_BASE) -> pd.Series:
    """Exponential recency weight: base^(ref - year)"""
    return base ** (ref - year)


def build_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute composite historical risk score per (Community Area, year, time_slot).
    Returns a long-format DataFrame.
    """
    df = df.copy()
    df["time_slot"]     = assign_time_slot(df["hour"], TIME_SLOT_HOURS)
    df["recency_wt"]    = recency_weight(df["year"])

    # ── Composite score per individual incident ───────────────────────────────
    # frequency_weight = 1.0 per incident (aggregated below)
    df["incident_score"] = (
        df["severity_weight"] *
        df["hour_risk_weight"] *
        df["recency_wt"]
    )

    # ── Aggregate by zone × year × time_slot ─────────────────────────────────
    grp = df.groupby(["Community Area", "year", "time_slot"]).agg(
        crime_count       = ("ID",              "count"),
        severity_sum      = ("severity_weight",  "sum"),
        severity_mean     = ("severity_weight",  "mean"),
        temporal_wt_mean  = ("hour_risk_weight", "mean"),
        recency_wt_mean   = ("recency_wt",       "mean"),
        incident_score_sum= ("incident_score",   "sum"),
        arrest_rate       = ("Arrest",           "mean"),
        domestic_rate     = ("Domestic",         "mean"),
        lat_mean          = ("Latitude",         "mean"),
        lon_mean          = ("Longitude",        "mean"),
    ).reset_index()

    # ── Raw CCRP historical score ─────────────────────────────────────────────
    # Combines: how often × how severe × when × how recent
    grp["raw_cps"] = grp["incident_score_sum"]

    # ── Normalise per Community Area ──────────────────────────────────────────
    scaler = MinMaxScaler()
    grp["cps_norm"] = (
        grp.groupby("Community Area")["raw_cps"]
            .transform(lambda x: scaler.fit_transform(x.values.reshape(-1,1)).flatten())
    )

    # ── Frequency rank within year (zone vs all zones that year) ─────────────
    grp["freq_rank_pct"] = (
        grp.groupby(["year", "time_slot"])["crime_count"]
            .rank(pct=True)
    )

    # ── Final CPS (blended: 70% score + 30% frequency rank) ──────────────────
    grp["cps_final"] = 0.70 * grp["cps_norm"] + 0.30 * grp["freq_rank_pct"]
    grp["cps_final"] = grp["cps_final"].clip(0, 1)

    return grp


def build_annual_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Zone-level annual profile — one row per (Community Area, year).
    Used as static node features in the GAT graph.
    """
    prof = df.groupby(["Community Area", "year"]).agg(
        total_crimes      = ("crime_count",  "sum"),
        avg_severity      = ("severity_mean","mean"),
        avg_arrest_rate   = ("arrest_rate",  "mean"),
        avg_domestic_rate = ("domestic_rate","mean"),
        lat               = ("lat_mean",     "mean"),
        lon               = ("lon_mean",     "mean"),
        peak_slot         = ("crime_count",  "idxmax"),   # slot with most crimes
        max_cps           = ("cps_final",    "max"),
        mean_cps          = ("cps_final",    "mean"),
    ).reset_index()

    # Year-over-year crime change (%)
    prof.sort_values(["Community Area", "year"], inplace=True)
    prof["yoy_change_pct"] = (
        prof.groupby("Community Area")["total_crimes"]
            .pct_change() * 100
    ).round(2)

    return prof


def build_crime_type_weights(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute calibrated type-level weights = mean CPS for that type's incidents.
    Useful for the CCRP fusion stage.
    """
    tw = df_raw.groupby("Primary Type").agg(
        count          = ("ID",              "count"),
        severity_mean  = ("severity_weight", "mean"),
        arrest_rate    = ("Arrest",          "mean"),
    ).reset_index()
    tw["frequency_share"] = tw["count"] / tw["count"].sum()
    tw["calibrated_weight"] = (
        0.60 * tw["severity_mean"] +
        0.25 * (1 - tw["arrest_rate"]) +   # un-arrested = harder to deter
        0.15 * tw["frequency_share"].rank(pct=True)
    ).clip(0, 1)
    tw.sort_values("calibrated_weight", ascending=False, inplace=True)
    return tw


def run():
    print("=" * 60)
    print("UCMS  ▸  Step 04: Risk Score Generation")
    print("=" * 60)

    print("  Loading cleaned data …")
    df = pd.read_parquet(PARQ_PATH)
    print(f"  Rows: {len(df):,}")

    # ── Build scores ──────────────────────────────────────────────────────────
    print("\n  Building zone × time-slot risk matrix …")
    risk = build_risk_scores(df)
    print(f"  Risk matrix shape: {risk.shape}")

    print("\n  Building annual zone profiles …")
    annual = build_annual_profile(risk)
    print(f"  Annual profile shape: {annual.shape}")

    print("\n  Building crime type weights …")
    type_w = build_crime_type_weights(df)

    # ── Save ──────────────────────────────────────────────────────────────────
    risk_path   = os.path.join(OUT_DIR, "zone_risk_scores.parquet")
    annual_path = os.path.join(OUT_DIR, "zone_annual_profile.parquet")
    type_path   = os.path.join(OUT_DIR, "crime_type_weights.csv")

    risk.to_parquet(risk_path,   index=False, compression="snappy")
    annual.to_parquet(annual_path, index=False, compression="snappy")
    type_w.to_csv(type_path, index=False)

    print(f"\n  [OK] Zone risk scores     → {risk_path}")
    print(f"  [OK] Annual zone profiles  → {annual_path}")
    print(f"  [OK] Crime type weights    → {type_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    top5 = (risk.groupby("Community Area")["cps_final"]
                .mean()
                .sort_values(ascending=False)
                .head(5))
    summary = [
        f"Zones scored             : {risk['Community Area'].nunique()}",
        f"Years covered            : {risk['year'].min()} – {risk['year'].max()}",
        f"Time slots/day           : {24 // TIME_SLOT_HOURS}",
        f"Total (zone,year,slot)   : {len(risk):,}",
        f"Avg CPS (all zones)      : {risk['cps_final'].mean():.4f}",
        f"Top-5 high-risk zones    :\n{top5.to_string()}",
    ]
    report_path = os.path.join(OUT_DIR, "04_risk_summary.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(summary))
    print("\n  Summary:")
    for l in summary:
        print(f"    {l}")

    print("\n[OK] STEP 04 COMPLETE  —  run  05_feature_matrix.py  next.\n")


if __name__ == "__main__":
    run()