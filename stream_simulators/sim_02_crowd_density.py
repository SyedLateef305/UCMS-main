"""
UCMS - Stream 02: Crowd Density Simulator
==========================================
Generates synthetic crowd density scores per (Community Area, year, time_slot).

Logic:
  • Base density from population ratios (Chicago census-based)
  • Time-of-day peaks: commute / lunch / evening
  • Seasonal variation: summer higher, winter lower
  • Gaussian noise ±10%

Output:
  outputs/stream_crowd_density.parquet
  plots/crowd_density_heatmap.png
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

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

# Load population density ratios from zone_metadata.csv if available
CA_POP_RATIO = {}
if os.path.exists(META_PATH):
    try:
        _mdf = pd.read_csv(META_PATH)
        if "pop_density_ratio" in _mdf.columns:
            CA_POP_RATIO = dict(zip(_mdf["zone_id"], _mdf["pop_density_ratio"]))
    except Exception:
        pass

# Fallback defaults
if not CA_POP_RATIO:
    CA_POP_RATIO = {i: 0.70 for i in range(1, 78)}

# Slot 0=night(0-6), 1=morning(6-12), 2=afternoon(12-18), 3=evening(18-24)
SLOT_CROWD   = {0:0.20, 1:0.65, 2:0.80, 3:0.75}
MONTH_SEASON = {1:0.70,2:0.72,3:0.78,4:0.82,5:0.88,6:0.95,
                7:1.00,8:0.97,9:0.87,10:0.83,11:0.75,12:0.73}


def simulate(zones, years):
    records = []
    for zone in zones:
        pop = CA_POP_RATIO.get(int(zone), 0.70)
        for year in years:
            yr_factor = 1.0 + (year - YEAR_MIN) * 0.003   # slight growth
            for slot in range(N_SLOTS):
                for month in range(1, 13):
                    raw   = pop * SLOT_CROWD[slot] * MONTH_SEASON[month] * yr_factor
                    score = float(np.clip(raw + np.random.normal(0, 0.05), 0.01, 1.0))
                    records.append({"Community Area":zone,"year":year,
                                    "month":month,"time_slot":slot,
                                    "crowd_density_score":score})
    return pd.DataFrame(records)


def plot_heatmap(df):
    piv = df.groupby(["Community Area","time_slot"])["crowd_density_score"].mean().unstack()
    piv.columns = ["Night(0-6)","Morn(6-12)","Aft(12-18)","Eve(18-24)"]
    fig, ax = plt.subplots(figsize=(12, 14))
    sns.heatmap(piv, cmap="YlOrRd", ax=ax, cbar_kws={"label":"Crowd Density Score"})
    ax.set_title("Indian Urban Zones — Diurnal Crowd Density Profile", fontsize=13, fontweight="bold")
    ax.set_ylabel("Zone ID")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR,"crowd_density_heatmap.png"), dpi=120)
    plt.close()


def run():
    print("="*55)
    print("UCMS  Stream 02 — Crowd Density Simulator")
    print("="*55)
    meta  = pd.read_csv(META_PATH)
    zones = meta["zone_id"].dropna().astype(int).values
    years = list(range(YEAR_MIN, YEAR_MAX+1))
    df    = simulate(zones, years)
    df = (df.groupby(["Community Area", "year", "time_slot"])
        ["crowd_density_score"]
        .mean()
        .reset_index())
    print(f"  Records : {len(df):,}  |  mean={df['crowd_density_score'].mean():.3f}")
    out = os.path.join(OUT_DIR,"stream_crowd_density.parquet")
    df.to_parquet(out, index=False, compression="snappy")
    print(f"  Saved   → {out}")
    plot_heatmap(df)
    print(f"  Plot    → {PLOT_DIR}/crowd_density_heatmap.png")
    print("\n[OK] Stream 02 DONE\n")

print("META_PATH:", META_PATH)
print("Exists?", os.path.exists(META_PATH))

if __name__ == "__main__":
    run()