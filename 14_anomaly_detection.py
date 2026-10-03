"""
UCMS - Step 14: Anomaly Detection
===================================
Phase II Addition.

Purpose : Detect unusual crime-risk spikes in zone × slot CCRP time-series
          using two complementary methods:
            1. Isolation Forest  — unsupervised spatial outlier detection
            2. Z-score rolling   — per-zone temporal spike detection (|z| > 2.5)

          Results are merged and exported for the API and dashboard.

Inputs  : ccrp_fusion/outputs/ccrp_final_master.parquet
Outputs :
  outputs/anomaly_flags.parquet   — zone × year × slot with anomaly scores + flags
  plots/anomaly_timeline.png      — top-5 zone anomaly timelines

Usage   : python 14_anomaly_detection.py [--dry-run]
"""

import os, sys, warnings, argparse
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MASTER_PATH = os.path.join(BASE_DIR, "ccrp_fusion", "outputs", "ccrp_final_master.parquet")
OUT_DIR     = os.path.join(BASE_DIR, "outputs")
PLOT_DIR    = os.path.join(BASE_DIR, "plots")

CONTAMINATION = 0.05   # expected anomaly rate (~5%)
Z_THRESHOLD   = 2.5    # Z-score threshold for temporal spikes
WINDOW        = 3      # rolling window for Z-score (years)
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Isolation Forest anomaly detection
# ─────────────────────────────────────────────────────────────────────────────
def run_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit Isolation Forest on cross-sectional features per (year, slot).
    Returns df with column 'if_score' and 'if_anomaly' (bool).
    """
    feature_cols = ["ccrp_bayesian", "crowd_density", "lighting_risk",
                    "social_media", "suspicious_activity", "cctv_feed",
                    "historical_cps"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].fillna(0))

    iso = IsolationForest(
        n_estimators=150,
        contamination=CONTAMINATION,
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X)

    df = df.copy()
    df["if_score"]   = -iso.score_samples(X)          # higher = more anomalous
    df["if_anomaly"] = iso.predict(X) == -1            # True = anomaly
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Z-score rolling anomaly detection (temporal)
# ─────────────────────────────────────────────────────────────────────────────
def run_zscore_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-zone per-slot: flag records where ccrp_bayesian deviates > Z_THRESHOLD
    standard deviations from the rolling mean.
    """
    df = df.copy().sort_values(["Community Area", "time_slot", "year"])
    grp = df.groupby(["Community Area", "time_slot"])

    roll_mean = grp["ccrp_bayesian"].transform(
        lambda x: x.rolling(WINDOW, min_periods=1).mean()
    )
    roll_std  = grp["ccrp_bayesian"].transform(
        lambda x: x.rolling(WINDOW, min_periods=1).std().fillna(1e-6)
    )

    df["z_score"]    = ((df["ccrp_bayesian"] - roll_mean) / roll_std).abs()
    df["z_anomaly"]  = df["z_score"] > Z_THRESHOLD
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Merge & finalise
# ─────────────────────────────────────────────────────────────────────────────
def build_anomaly_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine IF + Z-score, produce final anomaly_score ∈ [0,1] and is_anomaly flag.
    """
    df = run_isolation_forest(df)
    df = run_zscore_temporal(df)

    # Normalise IF score to [0,1]
    ifs_min = df["if_score"].min(); ifs_max = df["if_score"].max()
    df["if_score_norm"] = (df["if_score"] - ifs_min) / max(ifs_max - ifs_min, 1e-9)

    # Normalise Z-score to [0,1] (cap at 5σ)
    df["z_score_norm"] = (df["z_score"].clip(0, 5) / 5.0)

    # Combined anomaly score (60% IF + 40% Z)
    df["anomaly_score"] = (0.60 * df["if_score_norm"] +
                           0.40 * df["z_score_norm"]).clip(0, 1).round(4)

    # Final flag: IF OR Z-score anomaly
    df["is_anomaly"] = df["if_anomaly"] | df["z_anomaly"]

    keep = ["Community Area", "year", "time_slot",
            "ccrp_bayesian", "anomaly_score", "is_anomaly",
            "if_score_norm", "z_score_norm"]
    keep = [c for c in keep if c in df.columns]

    # Also keep lat/lon if present
    for col in ["lat", "lon"]:
        if col in df.columns:
            keep.append(col)

    return df[keep].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────
def plot_anomaly_timeline(df_flags: pd.DataFrame, df_master: pd.DataFrame):
    """Anomaly timeline for top-5 zones with most anomaly events."""
    top5_zones = (df_flags[df_flags["is_anomaly"]]
                  .groupby("Community Area").size()
                  .sort_values(ascending=False)
                  .head(5).index.tolist())

    if not top5_zones:
        print("  No anomalies found — skipping plot.")
        return

    annual_anom = (df_flags.groupby(["Community Area", "year"])
                   .agg(mean_score=("anomaly_score", "mean"),
                        anom_count=("is_anomaly", "sum"))
                   .reset_index())

    fig, axes = plt.subplots(len(top5_zones), 1,
                             figsize=(12, 3 * len(top5_zones)),
                             facecolor="#0d1117")
    fig.suptitle("Anomaly Timeline — Top Zones", color="white",
                 fontsize=13, fontweight="bold")

    for ax, zone in zip(axes, top5_zones):
        sub = annual_anom[annual_anom["Community Area"] == zone].sort_values("year")
        anom_yrs = sub[sub["anom_count"] > 0]["year"].values
        normal_y  = sub["mean_score"].values

        ax.fill_between(sub["year"], normal_y, alpha=0.3, color="#388bfd")
        ax.plot(sub["year"], normal_y, color="#388bfd", linewidth=1.5)

        for yr in anom_yrs:
            ax.axvline(yr, color="#ff4d4f", linewidth=1.2, linestyle="--", alpha=0.7)

        ax.scatter(anom_yrs,
                   sub[sub["year"].isin(anom_yrs)]["mean_score"].values,
                   color="#ff4d4f", zorder=5, s=40)

        ax.set_title(f"Zone {zone}", color="white", fontsize=10)
        ax.set_facecolor("#0d1117"); ax.set_ylim(0, 1)
        ax.tick_params(colors="#484f58", labelsize=8)
        ax.spines[:].set_color("#21262d")
        ax.set_ylabel("Anomaly Score", color="#8b949e", fontsize=8)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "anomaly_timeline.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Plot -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run(dry_run: bool = False):
    print("=" * 60)
    print("UCMS Phase II -> Step 14: Anomaly Detection")
    print("=" * 60)

    if dry_run:
        print("  [DRY RUN] Skipping data load -- checking imports only.")
        print("  [OK] All imports OK. Run without --dry-run to execute fully.\n")
        return

    print(f"\n  Loading master parquet: {MASTER_PATH}")
    if not os.path.exists(MASTER_PATH):
        print(f"  [ERROR] File not found: {MASTER_PATH}")
        print("  Run steps 01-12 first to generate ccrp_final_master.parquet\n")
        sys.exit(1)

    df = pd.read_parquet(MASTER_PATH)
    print(f"  Loaded {len(df):,} records  |  Zones: {df['Community Area'].nunique()}")

    # -- Anomaly detection -----------------------------------------------------
    print("\n  Running Isolation Forest + Z-score detection ...")
    df_flags = build_anomaly_flags(df)

    n_anomalies = int(df_flags["is_anomaly"].sum())
    n_total     = len(df_flags)
    pct         = n_anomalies / n_total * 100

    print(f"\n  Total records   : {n_total:,}")
    print(f"  Anomaly records : {n_anomalies:,}  ({pct:.2f}%)")

    top_zones = (df_flags[df_flags["is_anomaly"]]
                 .groupby("Community Area").size()
                 .sort_values(ascending=False).head(5))
    print(f"\n  Top-5 zones by anomaly count:")
    for zone, cnt in top_zones.items():
        print(f"    Zone {zone:3d}  ->  {cnt} anomalous records")

    # -- Save -----------------------------------------------------------------
    out_path = os.path.join(OUT_DIR, "anomaly_flags.parquet")
    df_flags.to_parquet(out_path, index=False, compression="snappy")
    print(f"\n  Saved -> {out_path}")

    # -- Plot ------------------------------------------------------------------
    print("\n  Plotting anomaly timelines ...")
    plot_anomaly_timeline(df_flags, df)

    print("\n[OK] STEP 14 COMPLETE - Anomaly flags ready for Phase II API\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Check imports only, skip heavy computation")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
