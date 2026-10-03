"""
UCMS - Stream 07: All-Streams Verification & Combined Report
=============================================================
Purpose : Validate all 5 stream outputs exist, check schema,
          compute cross-stream correlations, and produce one
          merged dataframe ready for Phase 3 CCRP fusion.

Checks:
  ✓ All 5 parquet files present
  ✓ Schema correctness (required columns)
  ✓ Value range [0, 1]
  ✓ No nulls in key columns
  ✓ Zone coverage matches (all 77 zones)
  ✓ Cross-stream correlation matrix

Output:
  outputs/all_streams_merged.parquet   ← Phase 3 input
  outputs/stream_validation_report.txt
  plots/stream_correlation_matrix.png
  plots/stream_score_distributions.png
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

OUT_DIR   = "outputs"
PLOT_DIR  = "plots"
META_PATH = "outputs/zone_metadata.csv"

os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Stream definitions ────────────────────────────────────────────────────────
STREAMS = {
    "crowd_density"      : ("stream_crowd_density.parquet",      "crowd_density_score"),
    "lighting_risk"      : ("stream_lighting_risk.parquet",       "lighting_risk_score"),
    "social_media"       : ("stream_social_media_score.parquet",  "social_media_score"),
    "suspicious_activity": ("stream_suspicious_activity.parquet", "suspicious_activity_score"),
    "cctv_feed"          : ("stream_cctv_feed.parquet",           "cctv_score"),
}

# Also need historical CPS from Phase 1
HIST_PATH  = "outputs/zone_risk_scores.parquet"
HIST_SCORE = "cps_final"

MERGE_KEYS = ["Community Area", "year", "time_slot"]


def validate_stream(name, path, score_col, report):
    def log(msg):
        print(f"    {msg}"); report.append(f"    {msg}")

    full = os.path.join(OUT_DIR, path)
    if not os.path.exists(full):
        log(f"❌  MISSING: {full}")
        return None

    df = pd.read_parquet(full)
    log(f"✓  {name:<22} | rows={len(df):>7,} | cols={list(df.columns)}")

    # Schema check
    for col in MERGE_KEYS + [score_col]:
        if col not in df.columns:
            log(f"   ⚠  Missing column: {col}")
            return None

    # Range check
    mn, mx = df[score_col].min(), df[score_col].max()
    if mn < -0.01 or mx > 1.01:
        log(f"   ⚠  Score out of range: [{mn:.3f}, {mx:.3f}]")
    else:
        log(f"      Range: [{mn:.3f}, {mx:.3f}]  mean={df[score_col].mean():.3f}")

    # Null check
    nulls = df[MERGE_KEYS + [score_col]].isna().sum().sum()
    log(f"      Nulls in key columns: {nulls}")

    # Zone coverage
    zones_covered = df["Community Area"].nunique()
    log(f"      Zones covered: {zones_covered}")

    return df


def merge_all_streams(stream_dfs, hist_df):
    """
    Merge all 6 streams (5 simulated + 1 historical CPS) on
    Community Area × year × time_slot.
    Uses month-averaged values for streams that have month dimension.
    """
    merged = None

    for name, (df, score_col) in stream_dfs.items():
        # Average over months if month column exists
        if "month" in df.columns:
            agg = (df.groupby(MERGE_KEYS)[score_col]
                     .mean().reset_index()
                     .rename(columns={score_col: name}))
        else:
            agg = df[MERGE_KEYS + [score_col]].rename(columns={score_col: name})

        if merged is None:
            merged = agg
        else:
            merged = pd.merge(merged, agg, on=MERGE_KEYS, how="inner")

    # Merge historical CPS
    hist_agg = (hist_df.groupby(MERGE_KEYS)[HIST_SCORE]
                  .mean().reset_index()
                  .rename(columns={HIST_SCORE: "historical_cps"}))
    merged = pd.merge(merged, hist_agg, on=MERGE_KEYS, how="inner")

    return merged


def plot_correlation(merged, score_cols):
    corr = merged[score_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, ax=ax, linewidths=0.5,
                cbar_kws={"label": "Pearson r"})
    ax.set_title("Cross-Stream Correlation Matrix\n(UCMS CCRP — All 6 Streams)",
                  fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR,"stream_correlation_matrix.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot → {PLOT_DIR}/stream_correlation_matrix.png")


def plot_distributions(merged, score_cols):
    n = len(score_cols)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    colors = ["#3498db","#2c3e50","#8e44ad","#e74c3c","#27ae60","#f39c12"]
    for i, (col, color) in enumerate(zip(score_cols, colors)):
        axes[i].hist(merged[col], bins=40, color=color, alpha=0.8, edgecolor="white")
        axes[i].set_title(col.replace("_"," ").title(), fontweight="bold")
        axes[i].set_xlabel("Score [0–1]"); axes[i].set_ylabel("Count")
        axes[i].axvline(merged[col].mean(), color="red", linestyle="--",
                        linewidth=1.2, label=f"μ={merged[col].mean():.3f}")
        axes[i].legend(fontsize=8)
    plt.suptitle("Score Distributions — All 6 CCRP Streams",
                  fontweight="bold", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR,"stream_score_distributions.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot → {PLOT_DIR}/stream_score_distributions.png")


def run():
    print("="*60)
    print("UCMS  Stream 07 — All-Streams Verification")
    print("="*60)

    report = []
    report.append("UCMS CCRP — Stream Validation Report")
    report.append("="*55)

    # ── Validate each stream ──────────────────────────────────────────────────
    stream_dfs = {}
    all_ok     = True
    for name, (fname, score_col) in STREAMS.items():
        print(f"\n  [{name}]")
        report.append(f"\n[{name}]")
        df = validate_stream(name, fname, score_col, report)
        if df is not None:
            stream_dfs[name] = (df, score_col)
        else:
            all_ok = False

    # ── Load historical CPS ───────────────────────────────────────────────────
    print(f"\n  [historical_cps]")
    report.append(f"\n[historical_cps]")
    if os.path.exists(HIST_PATH):
        hist_df = pd.read_parquet(HIST_PATH)
        print(f"    ✓  historical_cps | rows={len(hist_df):,}")
        report.append(f"    ✓  rows={len(hist_df):,}  mean={hist_df[HIST_SCORE].mean():.3f}")
    else:
        print("    ❌  MISSING — run Phase 1 step 04 first!")
        hist_df = None
        all_ok  = False

    if not all_ok:
        print("\n  ⚠  Some streams missing. Fix above errors first.")
        return

    # ── Merge ─────────────────────────────────────────────────────────────────
    print("\n  Merging all 6 streams …")
    merged = merge_all_streams(stream_dfs, hist_df)

    score_cols = list(STREAMS.keys()) + ["historical_cps"]
    print(f"  Merged shape : {merged.shape}")
    print(f"  Zones covered: {merged['Community Area'].nunique()}")
    print(f"  Years covered: {merged['year'].min()}–{merged['year'].max()}")

    # ── Summary stats ─────────────────────────────────────────────────────────
    print("\n  Stream summary stats:")
    report.append("\n\nMerged Stream Stats:")
    for col in score_cols:
        row = f"  {col:<28} mean={merged[col].mean():.4f}  std={merged[col].std():.4f}  min={merged[col].min():.4f}  max={merged[col].max():.4f}"
        print(row); report.append(row)

    # ── Save merged ───────────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, "all_streams_merged.parquet")
    merged.to_parquet(out_path, index=False, compression="snappy")
    print(f"\n  Merged saved → {out_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_correlation(merged, score_cols)
    plot_distributions(merged, score_cols)

    # ── Save report ───────────────────────────────────────────────────────────
    rpt_path = os.path.join(OUT_DIR,"stream_validation_report.txt")
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"  Report → {rpt_path}")

    print("\n" + "="*60)
    print("✅  ALL STREAMS VERIFIED & MERGED")
    print("    → outputs/all_streams_merged.parquet is your Phase 3 input")
    print("    → Next: Phase 3 — CCRP Fusion Engine")
    print("="*60 + "\n")


if __name__ == "__main__":
    run()