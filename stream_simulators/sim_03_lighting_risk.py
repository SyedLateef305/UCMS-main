"""
UCMS - Stream 03: Lighting Risk Simulator
==========================================
Generates synthetic lighting risk scores per (Community Area, year, time_slot).

Logic:
  • Darkness window based on Chicago sunrise/sunset (seasonal)
  • Zone infrastructure rating (wealthy zones = better lighting)
  • Random streetlight outage events (Poisson process)
  • Night slot = highest risk, afternoon = lowest

Output:
  outputs/stream_lighting_risk.parquet
  plots/lighting_risk_profile.png
"""

import os, warnings, math
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

OUT_DIR  = os.path.join(BASE_DIR, "outputs")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
META_PATH = os.path.join(BASE_DIR, "outputs", "zone_metadata.csv")
YEAR_MIN, YEAR_MAX = 2010, 2025
N_SLOTS = 4
SEED    = 42

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
np.random.seed(SEED)

# Load zone lighting infrastructure quality from zone_metadata.csv if available
CA_INFRA_QUALITY = {}
if os.path.exists(META_PATH):
    try:
        _mdf = pd.read_csv(META_PATH)
        if "infra_quality" in _mdf.columns:
            CA_INFRA_QUALITY = dict(zip(_mdf["zone_id"], _mdf["infra_quality"]))
    except Exception:
        pass

if not CA_INFRA_QUALITY:
    CA_INFRA_QUALITY = {i: 0.60 for i in range(1, 78)}

# Indian Standard Time (Delhi latitude ~28.6°N) approximate daylight hours per month
# Format: (avg_sunrise_hour, avg_sunset_hour)
MONTH_DAYLIGHT = {
    1:(7.2,17.7), 2:(7.0,18.1), 3:(6.5,18.5), 4:(5.9,18.8),
    5:(5.5,19.1), 6:(5.4,19.3), 7:(5.5,19.3), 8:(5.8,19.0),
    9:(6.1,18.4), 10:(6.4,17.8), 11:(6.8,17.4), 12:(7.1,17.4),
}

# Slot midpoint hours for darkness calculation
SLOT_MID_HOUR = {0: 3.0, 1: 9.0, 2: 15.0, 3: 21.0}


def darkness_factor(slot, month):
    """How dark is it during this slot in this month? (0=full daylight, 1=full dark)"""
    sunrise, sunset = MONTH_DAYLIGHT[month]
    mid = SLOT_MID_HOUR[slot]
    if sunrise <= mid <= sunset:
        # Daylight — some residual from cloud/overcast
        return 0.05
    elif mid < sunrise:
        # Pre-dawn
        gap = sunrise - mid
        return min(1.0, 0.4 + gap * 0.15)
    else:
        # Post-sunset
        gap = mid - sunset
        return min(1.0, 0.5 + gap * 0.12)


def outage_probability(infra_quality):
    """Probability of streetlight outage in a zone based on infrastructure quality."""
    # Poor infra = higher outage rate (Poisson lambda: 0.02 to 0.25)
    lam = 0.25 * (1.0 - infra_quality) + 0.02
    return min(1.0, np.random.poisson(lam) / 3.0)


def simulate(zones, years):
    records = []
    for zone in zones:
        infra = CA_INFRA_QUALITY.get(int(zone), 0.55)
        for year in years:
            # Infrastructure degrades ~0.5% per year if not in wealthy zone
            yr_degrade = max(0, (year - YEAR_MIN) * 0.005 * (1 - infra))
            eff_infra  = max(0.1, infra - yr_degrade)
            for slot in range(N_SLOTS):
                for month in range(1, 13):
                    dark    = darkness_factor(slot, month)
                    outage  = outage_probability(eff_infra)
                    # Lighting risk = darkness × (1 - infrastructure) + outage component
                    raw     = 0.55 * dark + 0.30 * (1 - eff_infra) + 0.15 * outage
                    noise   = np.random.normal(0, 0.03)
                    score   = float(np.clip(raw + noise, 0.0, 1.0))
                    records.append({"Community Area":zone,"year":year,
                                    "month":month,"time_slot":slot,
                                    "lighting_risk_score":round(score,4)})
    return pd.DataFrame(records)


def plot_profile(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # LEFT: Placeholder (since month removed)
    axes[0].axis('off')
    axes[0].set_title("Monthly view removed (aggregated data)")

    # RIGHT: Time slot analysis (THIS is important)
    slots = ["Night(0-6)", "Morning(6-12)", "Afternoon(12-18)", "Evening(18-24)"]
    slot_means = [df[df["time_slot"] == s]["lighting_risk_score"].mean() for s in range(4)]

    axes[1].bar(slots, slot_means)
    axes[1].set_title("Avg Lighting Risk by Time Slot", fontweight="bold")
    axes[1].set_ylabel("Lighting Risk Score")
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "lighting_risk_profile.png"), dpi=150, bbox_inches="tight")
    plt.close()


def run():
    print("="*55)
    print("UCMS  Stream 03 — Lighting Risk Simulator")
    print("="*55)
    meta  = pd.read_csv(META_PATH)
    zones = meta["zone_id"].dropna().astype(int).values
    years = list(range(YEAR_MIN, YEAR_MAX+1))
    df    = simulate(zones, years)
    df = (df.groupby(["Community Area", "year", "time_slot"])
        ["lighting_risk_score"]
        .mean()
        .reset_index())
    print(f"  Records : {len(df):,}  |  mean={df['lighting_risk_score'].mean():.3f}")
    out = os.path.join(OUT_DIR,"stream_lighting_risk.parquet")
    df.to_parquet(out, index=False, compression="snappy")
    print(f"  Saved   → {out}")
    plot_profile(df)
    print(f"  Plot    → {PLOT_DIR}/lighting_risk_profile.png")
    print("\n[OK]  Stream 03 DONE\n")

if __name__ == "__main__":
    run()