"""
UCMS - Stream 04: Social Media Threat Score Simulator
======================================================
Generates synthetic social media threat scores per (Community Area, year, time_slot).

Logic:
  • Keyword frequency model: threat keywords spike near high-crime zones
  • Base threat level proportional to historical CPS score of zone
  • Evening/night slots → more social media crime-related activity
  • Spike events: random viral incidents (Poisson process, ~3/year citywide)
  • BERT-style sentiment proxy: negative sentiment ratio per zone/slot

Output:
  outputs/stream_social_media_score.parquet
  plots/social_media_score_plot.png
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
RISK_PATH = "outputs/zone_risk_scores.parquet"

YEAR_MIN, YEAR_MAX = 2010, 2025
N_SLOTS = 4
SEED    = 42

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
np.random.seed(SEED)

# Social media activity peaks by time slot (people post more at night)
SLOT_SOCIAL_MULT = {0: 0.85, 1: 0.35, 2: 0.50, 3: 0.90}

# Social media adoption growth by year (platform usage grew rapidly post-2010)
YEAR_ADOPTION = {
    2010:0.30, 2011:0.38, 2012:0.46, 2013:0.54, 2014:0.61,
    2015:0.67, 2016:0.72, 2017:0.76, 2018:0.80, 2019:0.83,
    2020:0.88, 2021:0.90, 2022:0.91, 2023:0.92, 2024:0.93, 2025:0.94,
}

# Month multiplier — more outdoor activity/events in warm months = more posts
MONTH_MULT = {
    1:0.70, 2:0.72, 3:0.75, 4:0.80, 5:0.85, 6:0.92,
    7:0.95, 8:0.93, 9:0.85, 10:0.80, 11:0.74, 12:0.78,
}


def load_zone_cps(risk_path, zones):
    """Load historical CPS as base threat level per zone."""
    df  = pd.read_parquet(risk_path)
    cps = (df.groupby("Community Area")["cps_final"]
             .mean()
             .reset_index()
             .rename(columns={"cps_final":"base_threat"}))
    # Normalise
    mn, mx = cps["base_threat"].min(), cps["base_threat"].max()
    cps["base_threat"] = (cps["base_threat"] - mn) / (mx - mn + 1e-9)
    zone_map = dict(zip(cps["Community Area"].astype(int), cps["base_threat"]))
    return {int(z): zone_map.get(int(z), 0.50) for z in zones}


def generate_spike_events(years, n_zones, rate=3):
    """Generate random citywide spike events (viral incidents)."""
    spikes = {}
    for year in years:
        n_spikes = np.random.poisson(rate)
        for _ in range(n_spikes):
            month  = np.random.randint(1, 13)
            slot   = np.random.randint(0, 4)
            zone   = np.random.randint(0, n_zones)
            radius = np.random.randint(1, 6)   # affects nearby zones too
            impact = np.random.uniform(0.2, 0.5)
            for r in range(radius):
                affected = (zone + r) % n_zones
                spikes[(affected, year, month, slot)] = impact
    return spikes


def simulate(zones, years, cps_map):
    spike_events = generate_spike_events(years, len(zones))
    records = []
    for zi, zone in enumerate(zones):
        base = cps_map.get(int(zone), 0.50)
        for year in years:
            adoption = YEAR_ADOPTION.get(year, 0.85)
            for slot in range(N_SLOTS):
                for month in range(1, 13):
                    social_mult = SLOT_SOCIAL_MULT[slot]
                    month_mult  = MONTH_MULT[month]

                    # Keyword threat = base × adoption × time × season
                    keyword_score = base * adoption * social_mult * month_mult

                    # Sentiment score: negative ratio correlated with crime base
                    sentiment_neg = 0.3 * base + 0.1 * social_mult + np.random.beta(2, 5) * 0.3

                    # Spike injection
                    spike = spike_events.get((zi, year, month, slot), 0.0)

                    # Final combined score
                    raw   = 0.60 * keyword_score + 0.25 * sentiment_neg + 0.15 * spike
                    noise = np.random.normal(0, 0.04)
                    score = float(np.clip(raw + noise, 0.0, 1.0))

                    records.append({
                        "Community Area"       : zone,
                        "year"                 : year,
                        "month"                : month,
                        "time_slot"            : slot,
                        "social_media_score"   : round(score, 4),
                        "keyword_score"        : round(keyword_score, 4),
                        "sentiment_neg_ratio"  : round(float(np.clip(sentiment_neg,0,1)), 4),
                        "spike_detected"       : int(spike > 0),
                    })
    return pd.DataFrame(records)


def plot_results(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Year trend
    yr = df.groupby("year")["social_media_score"].mean()
    axes[0].plot(yr.index, yr.values, "o-", color="#8e44ad", linewidth=2, markersize=6)
    axes[0].fill_between(yr.index, yr.values, alpha=0.15, color="#8e44ad")
    axes[0].set_title("Social Media Score Trend (2010–2025)", fontweight="bold")
    axes[0].set_xlabel("Year"); axes[0].set_ylabel("Avg Score")

    # 2. By slot
    slots = ["Night","Morning","Afternoon","Evening"]
    sv = [df[df["time_slot"]==s]["social_media_score"].mean() for s in range(4)]
    axes[1].bar(slots, sv, color=["#1a1a2e","#3498db","#f39c12","#8e44ad"],
                edgecolor="white")
    axes[1].set_title("Avg Score by Time Slot", fontweight="bold")
    axes[1].set_ylabel("Score"); axes[1].tick_params(axis="x",rotation=20)

    # 3. Spike detection distribution
    spike_counts = df.groupby("Community Area")["spike_detected"].sum().sort_values(ascending=False).head(20)
    axes[2].bar(spike_counts.index.astype(str), spike_counts.values,
                color="#e74c3c", edgecolor="white")
    axes[2].set_title("Spike Events — Top 20 Zones", fontweight="bold")
    axes[2].set_xlabel("Community Area"); axes[2].set_ylabel("Total Spikes")
    axes[2].tick_params(axis="x", rotation=60, labelsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR,"social_media_score_plot.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def run():
    print("="*55)
    print("UCMS  Stream 04 — Social Media Threat Score")
    print("="*55)

    meta  = pd.read_csv(META_PATH)
    zones = meta["zone_id"].dropna().astype(int).values
    years = list(range(YEAR_MIN, YEAR_MAX+1))

    print("  Loading historical CPS for base threat calibration …")
    cps_map = load_zone_cps(RISK_PATH, zones)

    print(f"  Zones: {len(zones)}  |  Years: {len(years)}  |  Generating …")
    df = simulate(zones, years, cps_map)

    total_spikes = df["spike_detected"].sum()
    print(f"  Records   : {len(df):,}")
    print(f"  Mean score: {df['social_media_score'].mean():.3f}")
    print(f"  Spikes    : {total_spikes:,}")

    out = os.path.join(OUT_DIR, "stream_social_media_score.parquet")
    df.to_parquet(out, index=False, compression="snappy")
    print(f"  Saved  → {out}")

    plot_results(df)
    print(f"  Plot   → {PLOT_DIR}/social_media_score_plot.png")
    print("\n✅  Stream 04 DONE\n")


if __name__ == "__main__":
    run()