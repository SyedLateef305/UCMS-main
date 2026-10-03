"""
UCMS - Step 11: Bayesian Recalibration Layer
=============================================
Purpose : Apply Bayesian inference to recalibrate the raw CCRP fused score.

          Why Bayesian?
          ─────────────
          The GAT + fusion weights give a point estimate CCRP score.
          Bayesian recalibration adds:
            1. Uncertainty quantification  (credible intervals)
            2. Prior injection  (historical crime knowledge)
            3. Evidence updating  (Bayes' theorem per zone)
            4. Posterior smoothing  (reduces noise spikes)

Method  : Beta-Binomial conjugate model per zone
          ──────────────────────────────────────
          Prior  : Beta(α₀, β₀)  — from historical crime rate
          Evidence: zone's CCRP score → treated as success rate
          Posterior: Beta(α₀ + successes, β₀ + failures)
          Recalibrated score = posterior mean = α/(α+β)

          This is computationally cheap (closed-form) and
          interpretable — suitable for the paper.

Outputs:
  outputs/ccrp_bayesian.parquet        ← recalibrated scores + uncertainty
  outputs/bayesian_report.txt
  plots/bayesian_uncertainty_map.png
  plots/calibration_curve.png
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta as beta_dist

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FUSION_DIR = os.path.join(BASE_DIR, "outputs")
HIST_DIR = os.path.join(BASE_DIR, "..", "outputs")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

CCRP_PATH  = os.path.join(FUSION_DIR, "ccrp_scores.parquet")
RISK_PATH  = os.path.join(HIST_DIR,   "zone_risk_scores.parquet")
META_PATH  = os.path.join(HIST_DIR,   "zone_metadata.csv")

YEAR_MIN, YEAR_MAX = 2010, 2025

# Bayesian prior strength  (higher = more weight on historical prior)
PRIOR_STRENGTH = 10.0    # equivalent sample size of prior
# ──────────────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


def compute_zone_priors(risk_df, zones):
    """
    Compute Beta prior parameters per zone from historical CPS.
    Prior mean = historical_cps_mean
    Prior strength = PRIOR_STRENGTH (controls how strongly we trust history)

    Beta(α, β) where:
      α = mean × strength
      β = (1 - mean) × strength
    """
    zone_hist = (risk_df.groupby("Community Area")["cps_final"]
                   .mean().reset_index()
                   .rename(columns={"cps_final":"hist_mean"}))
    zone_hist["Community Area"] = zone_hist["Community Area"].astype(int)
    zone_hist["prior_alpha"] = zone_hist["hist_mean"] * PRIOR_STRENGTH
    zone_hist["prior_beta"]  = (1 - zone_hist["hist_mean"]) * PRIOR_STRENGTH
    # Clip to avoid degenerate priors
    zone_hist["prior_alpha"] = zone_hist["prior_alpha"].clip(0.5, 20)
    zone_hist["prior_beta"]  = zone_hist["prior_beta"].clip(0.5, 20)
    return zone_hist.set_index("Community Area")


def bayesian_update(ccrp_score, prior_alpha, prior_beta,
                     n_obs=5, credible_level=0.90):
    """
    Update Beta prior with CCRP score as likelihood.

    Treat score as proportion of 'crime successes' in n_obs virtual observations.
    Returns: posterior_mean, lower_ci, upper_ci, uncertainty
    """
    # Evidence from CCRP score
    successes = ccrp_score * n_obs
    failures  = (1 - ccrp_score) * n_obs

    # Posterior parameters
    post_alpha = prior_alpha + successes
    post_beta  = prior_beta  + failures

    # Posterior mean (recalibrated score)
    post_mean = post_alpha / (post_alpha + post_beta)

    # Credible interval
    lo = (1 - credible_level) / 2
    hi = 1 - lo
    lower_ci = beta_dist.ppf(lo, post_alpha, post_beta)
    upper_ci = beta_dist.ppf(hi, post_alpha, post_beta)

    # Uncertainty = width of credible interval
    uncertainty = upper_ci - lower_ci

    return float(post_mean), float(lower_ci), float(upper_ci), float(uncertainty)


def recalibrate(df, prior_df):
    """Apply Bayesian update to every record."""
    records = []
    for _, row in df.iterrows():
        zone = int(row["Community Area"])
        if zone in prior_df.index:
            pa = prior_df.loc[zone, "prior_alpha"]
            pb = prior_df.loc[zone, "prior_beta"]
        else:
            pa, pb = 2.0, 2.0    # uninformative fallback

        score = float(row["ccrp_score"])
        pm, lo, hi, unc = bayesian_update(score, pa, pb)

        records.append({
            "Community Area"       : zone,
            "year"                 : int(row["year"]),
            "time_slot"            : int(row["time_slot"]),
            "ccrp_raw"             : round(score, 4),
            "ccrp_bayesian"        : round(pm,    4),
            "credible_lower_90"    : round(lo,    4),
            "credible_upper_90"    : round(hi,    4),
            "uncertainty"          : round(unc,   4),
            "prior_alpha"          : round(float(pa), 4),
            "prior_beta"           : round(float(pb), 4),
        })
    return pd.DataFrame(records)


def plot_uncertainty_map(df_bay, meta):
    """Bubble map: zone centroid sized by uncertainty, coloured by score."""
    zone_agg = (df_bay.groupby("Community Area")
                  .agg(ccrp_bay=("ccrp_bayesian","mean"),
                       uncertainty=("uncertainty","mean"))
                  .reset_index())
    if "Community Area" not in meta.columns and "zone_id" in meta.columns:
        meta = meta.rename(columns={"zone_id":"Community Area"})
    elif "Community Area" in meta.columns and "zone_id" in meta.columns:
        meta = meta.drop(columns=["zone_id"])
    merged = zone_agg.merge(meta, on="Community Area", how="left")
    merged = merged.dropna(subset=["lat","lon"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 8), facecolor="#0d1117")
    for ax in axes: ax.set_facecolor("#0d1117")

    # Left: Bayesian score map
    sc = axes[0].scatter(merged["lon"], merged["lat"],
                         c=merged["ccrp_bay"], cmap="YlOrRd",
                         s=80 + 300*merged["ccrp_bay"],
                         alpha=0.85, edgecolors="white", linewidths=0.3)
    plt.colorbar(sc, ax=axes[0]).set_label("Bayesian CCRP", color="white")
    axes[0].set_title("Bayesian Recalibrated Risk Score",
                       color="white", fontweight="bold")
    axes[0].tick_params(colors="white")
    axes[0].set_xlabel("Longitude", color="white")
    axes[0].set_ylabel("Latitude", color="white")

    # Right: Uncertainty map
    sc2 = axes[1].scatter(merged["lon"], merged["lat"],
                          c=merged["uncertainty"], cmap="Blues",
                          s=80 + 400*merged["uncertainty"],
                          alpha=0.85, edgecolors="white", linewidths=0.3)
    plt.colorbar(sc2, ax=axes[1]).set_label("Uncertainty (CI width)", color="white")
    axes[1].set_title("Prediction Uncertainty (90% CI Width)",
                       color="white", fontweight="bold")
    axes[1].tick_params(colors="white")
    axes[1].set_xlabel("Longitude", color="white")

    plt.suptitle("UCMS-CCRP Bayesian Recalibration — Chicago Zones",
                  color="white", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR,"bayesian_uncertainty_map.png"),
                dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Plot → {PLOT_DIR}/bayesian_uncertainty_map.png")


def plot_calibration_curve(df_bay):
    """Reliability diagram: raw CCRP vs Bayesian CCRP."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Calibration scatter
    sample = df_bay.sample(min(5000, len(df_bay)), random_state=42)
    axes[0].scatter(sample["ccrp_raw"], sample["ccrp_bayesian"],
                    alpha=0.3, s=5, color="#9b59b6")
    axes[0].plot([0,1],[0,1],"k--",linewidth=1,label="No change")
    axes[0].set_title("Raw CCRP vs Bayesian Recalibrated Score",
                       fontweight="bold")
    axes[0].set_xlabel("Raw CCRP"); axes[0].set_ylabel("Bayesian CCRP")
    axes[0].legend()

    # Shift distribution
    shift = df_bay["ccrp_bayesian"] - df_bay["ccrp_raw"]
    axes[1].hist(shift, bins=50, color="#1abc9c", edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].axvline(shift.mean(), color="orange", linestyle="-",
                    linewidth=1.5, label=f"Mean shift={shift.mean():.4f}")
    axes[1].set_title("Calibration Shift Distribution\n(Bayesian − Raw)",
                       fontweight="bold")
    axes[1].set_xlabel("Score shift"); axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR,"calibration_curve.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot → {PLOT_DIR}/calibration_curve.png")


def run():
    print("="*60)
    print("UCMS  Step 11 — Bayesian Recalibration Layer")
    print("="*60)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n  Loading CCRP fused scores …")
    df   = pd.read_parquet(CCRP_PATH)
    print(f"  Records: {len(df):,}  |  CCRP mean: {df['ccrp_score'].mean():.4f}")

    print("  Loading historical risk for priors …")
    risk_df = pd.read_parquet(RISK_PATH)
    meta    = pd.read_csv(META_PATH)

    # ── Compute priors ────────────────────────────────────────────────────────
    print("  Computing Beta priors per zone …")
    prior_df = compute_zone_priors(risk_df, df["Community Area"].unique())
    print(f"  Prior α range: [{prior_df['prior_alpha'].min():.2f}, "
          f"{prior_df['prior_alpha'].max():.2f}]")

    # ── Recalibrate ───────────────────────────────────────────────────────────
    print("  Applying Bayesian update … (this may take ~30 sec)")
    df_bay = recalibrate(df, prior_df)

    # ── Summary stats ─────────────────────────────────────────────────────────
    shift = df_bay["ccrp_bayesian"] - df_bay["ccrp_raw"]
    unc   = df_bay["uncertainty"]

    print(f"\n  Bayesian Recalibration Summary:")
    print(f"    Mean score (raw)      : {df_bay['ccrp_raw'].mean():.4f}")
    print(f"    Mean score (Bayesian) : {df_bay['ccrp_bayesian'].mean():.4f}")
    print(f"    Mean shift            : {shift.mean():+.4f}")
    print(f"    Std shift             : {shift.std():.4f}")
    print(f"    Avg uncertainty (90%) : {unc.mean():.4f}")
    print(f"    Max uncertainty       : {unc.max():.4f}")

    # Top 5 high-risk zones (Bayesian)
    top5 = (df_bay.groupby("Community Area")["ccrp_bayesian"]
              .mean().sort_values(ascending=False).head(5))
    print(f"\n  Top 5 High-Risk Zones (Bayesian CCRP):")
    for zone, score in top5.items():
        unc_val = df_bay[df_bay["Community Area"]==zone]["uncertainty"].mean()
        print(f"    Zone {zone:3d}  score={score:.4f}  uncertainty={unc_val:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, "ccrp_bayesian.parquet")
    df_bay.to_parquet(out_path, index=False, compression="snappy")
    print(f"\n  Saved → {out_path}")

    report = [
        "UCMS CCRP — Bayesian Recalibration Report",
        "="*50,
        f"Prior strength        : {PRIOR_STRENGTH}",
        f"Credible interval     : 90%",
        f"Records processed     : {len(df_bay):,}",
        f"Mean score (raw)      : {df_bay['ccrp_raw'].mean():.4f}",
        f"Mean score (Bayesian) : {df_bay['ccrp_bayesian'].mean():.4f}",
        f"Mean shift            : {shift.mean():+.4f}",
        f"Avg uncertainty       : {unc.mean():.4f}",
        f"\nTop 5 High-Risk Zones:",
    ] + [f"  Zone {z}: {s:.4f}" for z,s in top5.items()]

    with open(os.path.join(OUT_DIR,"bayesian_report.txt"),"w") as f:
        f.write("\n".join(report))

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_uncertainty_map(df_bay, meta)
    plot_calibration_curve(df_bay)

    print("\n[OK]  STEP 11 DONE  →  run  12_final_ccrp_score.py  next\n")


if __name__ == "__main__":
    run()