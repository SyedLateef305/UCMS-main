"""
UCMS - Stream 06: CCTV Feed Simulator
======================================
Generates synthetic CCTV-based activity flags per (Community Area, year, time_slot).

Logic:
  • Camera coverage density per zone (denser = more data = more detections)
  • Motion activity index: pedestrian + vehicle movement patterns
  • Incident flag probability based on zone risk × time
  • Camera health: malfunction/outage events (Poisson)
  • Frame quality degradation at night → lower confidence

Output:
  outputs/stream_cctv_feed.parquet
  plots/cctv_feed_plot.png
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

OUT_DIR   = "outputs"
PLOT_DIR  = "plots"
META_PATH = "outputs/zone_metadata.csv"
RISK_PATH = "outputs/zone_risk_scores.parquet"

YEAR_MIN, YEAR_MAX = 2010, 2025
N_SLOTS = 4
SEED    = 42

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
np.random.seed(SEED)

# CCTV camera density per zone (cameras per sq km proxy, normalised)
# Wealthier/downtown zones = more cameras
CA_CAMERA_DENSITY = {
    1:0.55,2:0.40,3:0.60,4:0.48,5:0.35,6:0.75,7:0.58,8:0.90,
    9:0.65,10:0.42,11:0.70,12:0.38,13:0.30,14:0.52,15:0.60,
    16:0.45,17:0.72,18:0.65,19:0.36,20:0.50,21:0.34,22:0.44,
    23:0.58,24:0.28,25:0.85,26:0.55,27:0.38,28:0.68,29:0.48,
    30:0.32,31:0.70,32:0.63,33:0.36,34:0.50,35:0.68,36:0.42,
    37:0.56,38:0.38,39:0.52,40:0.26,41:0.78,42:0.72,43:0.50,
    44:0.60,45:0.38,46:0.46,47:0.64,48:0.30,49:0.54,50:0.76,
    51:0.36,52:0.60,53:0.48,54:0.70,55:0.32,56:0.82,57:0.42,
    58:0.62,59:0.46,60:0.34,61:0.52,62:0.28,63:0.66,64:0.32,
    65:0.50,66:0.40,67:0.58,68:0.36,69:0.72,70:0.24,71:0.54,
    72:0.80,73:0.38,74:0.66,75:0.44,76:0.30,77:0.52,
}

# Night degrades frame quality → higher false negative rate
SLOT_FRAME_QUALITY = {0: 0.55, 1: 0.90, 2: 0.95, 3: 0.75}

# Motion activity multiplier per slot (pedestrian + vehicle)
SLOT_MOTION        = {0: 0.30, 1: 0.70, 2: 0.85, 3: 0.75}

# Camera tech improvement over years (IR, resolution)
def tech_quality(year):
    return min(1.0, 0.55 + (year - YEAR_MIN) * 0.03)


def load_zone_cps(risk_path, zones):
    df   = pd.read_parquet(risk_path)
    agg  = df.groupby("Community Area")["cps_final"].mean().reset_index()
    mn, mx = agg["cps_final"].min(), agg["cps_final"].max()
    agg["risk_base"] = (agg["cps_final"] - mn) / (mx - mn + 1e-9)
    rmap = dict(zip(agg["Community Area"].astype(int), agg["risk_base"]))
    return {int(z): rmap.get(int(z), 0.45) for z in zones}


def camera_malfunction(density):
    """Simulate camera outage/malfunction — less coverage = worse reliability."""
    lam    = 0.20 * (1 - density) + 0.02
    outage = np.random.poisson(lam)
    return min(1.0, outage / 3.0)   # fraction of cameras down


def simulate(zones, years, rmap):
    records = []
    for zone in zones:
        risk_base = rmap.get(int(zone), 0.45)
        cam_dens  = CA_CAMERA_DENSITY.get(int(zone), 0.50)

        for year in years:
            tq = tech_quality(year)
            # More cameras installed over time in high-risk zones
            eff_density = min(1.0, cam_dens + (year - YEAR_MIN) * 0.008 * risk_base)

            for slot in range(N_SLOTS):
                for month in range(1, 13):
                    frame_q = SLOT_FRAME_QUALITY[slot] * tq
                    motion  = SLOT_MOTION[slot]

                    # Camera outage fraction this slot
                    outage_frac = camera_malfunction(eff_density)
                    coverage    = eff_density * (1 - outage_frac)

                    # Motion index: how much activity captured by working cameras
                    motion_index = motion * coverage * (1 + 0.3 * risk_base)

                    # Incident detection probability
                    # High risk zone + night + good camera = high detection
                    incident_prob = (0.40 * risk_base +
                                     0.30 * (1 - SLOT_FRAME_QUALITY[slot] + 0.5) +
                                     0.30 * coverage) * frame_q
                    incident_flag = float(np.random.binomial(1, min(0.95, incident_prob)))

                    # Confidence score of the detection
                    if incident_flag:
                        confidence = float(np.clip(
                            np.random.beta(4, 2) * frame_q * coverage, 0.1, 1.0))
                    else:
                        confidence = 0.0

                    # Final CCTV score
                    raw   = (0.40 * motion_index +
                             0.35 * incident_flag * confidence +
                             0.25 * coverage)
                    noise = np.random.normal(0, 0.03)
                    score = float(np.clip(raw + noise, 0, 1))

                    records.append({
                        "Community Area"    : zone,
                        "year"              : year,
                        "month"             : month,
                        "time_slot"         : slot,
                        "cctv_score"        : round(score, 4),
                        "motion_index"      : round(float(motion_index), 4),
                        "incident_flag"     : int(incident_flag),
                        "detection_confidence": round(confidence, 4),
                        "camera_coverage"   : round(float(coverage), 4),
                        "outage_fraction"   : round(float(outage_frac), 4),
                    })
    return pd.DataFrame(records)


def plot_results(df):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1. Coverage vs score scatter
    sample = df.sample(min(5000, len(df)), random_state=42)
    axes[0,0].scatter(sample["camera_coverage"], sample["cctv_score"],
                      alpha=0.3, s=5, color="#3498db")
    axes[0,0].set_title("Camera Coverage vs CCTV Score", fontweight="bold")
    axes[0,0].set_xlabel("Coverage"); axes[0,0].set_ylabel("CCTV Score")

    # 2. Incident flags by slot
    slots = ["Night","Morning","Afternoon","Evening"]
    iflags = [df[df["time_slot"]==s]["incident_flag"].mean() for s in range(4)]
    axes[0,1].bar(slots, iflags,
                  color=["#1a1a2e","#3498db","#f39c12","#c0392b"], edgecolor="white")
    axes[0,1].set_title("Incident Detection Rate by Slot", fontweight="bold")
    axes[0,1].set_ylabel("Fraction with Incident"); axes[0,1].tick_params(axis="x",rotation=20)

    # 3. Year trend (technology improvement)
    yr = df.groupby("year")[["cctv_score","camera_coverage"]].mean()
    axes[1,0].plot(yr.index, yr["cctv_score"],     "o-", label="CCTV Score",    color="#e74c3c")
    axes[1,0].plot(yr.index, yr["camera_coverage"],"s--",label="Coverage",      color="#27ae60")
    axes[1,0].set_title("Year Trend — Score & Coverage", fontweight="bold")
    axes[1,0].set_xlabel("Year"); axes[1,0].legend()

    # 4. Top 20 high-CCTV-score zones
    top20 = df.groupby("Community Area")["cctv_score"].mean().sort_values(ascending=False).head(20)
    axes[1,1].bar(top20.index.astype(str), top20.values,
                  color="#9b59b6", edgecolor="white")
    axes[1,1].set_title("Top 20 Zones by CCTV Score", fontweight="bold")
    axes[1,1].set_xlabel("Community Area"); axes[1,1].set_ylabel("Avg CCTV Score")
    axes[1,1].tick_params(axis="x", rotation=60, labelsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR,"cctv_feed_plot.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def run():
    print("="*55)
    print("UCMS  Stream 06 — CCTV Feed Simulator")
    print("="*55)

    meta  = pd.read_csv(META_PATH)
    zones = meta["zone_id"].dropna().astype(int).values
    years = list(range(YEAR_MIN, YEAR_MAX+1))

    print("  Loading zone risk base …")
    rmap = load_zone_cps(RISK_PATH, zones)

    print(f"  Generating for {len(zones)} zones × {len(years)} years …")
    df = simulate(zones, years, rmap)

    print(f"  Records          : {len(df):,}")
    print(f"  Mean CCTV score  : {df['cctv_score'].mean():.3f}")
    print(f"  Incident rate    : {df['incident_flag'].mean():.2%}")
    print(f"  Avg coverage     : {df['camera_coverage'].mean():.3f}")

    out = os.path.join(OUT_DIR, "stream_cctv_feed.parquet")
    df.to_parquet(out, index=False, compression="snappy")
    print(f"  Saved  → {out}")

    plot_results(df)
    print(f"  Plot   → {PLOT_DIR}/cctv_feed_plot.png")
    print("\n✅  Stream 06 DONE")
    print("    → All 5 streams complete. Run sim_07_verify_all_streams.py next.\n")


if __name__ == "__main__":
    run()