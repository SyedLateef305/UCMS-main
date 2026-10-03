"""
UCMS - Step 03: Exploratory Data Analysis
==========================================
Purpose : Deep-dive analysis of cleaned crime data.  Generates publication-
          quality plots + aggregated tables that:
          1. Validate data quality
          2. Reveal temporal & spatial patterns used by CCRP
          3. Produce visualisations for the research paper / final report

Usage   : python 03_eda_analysis.py
Input   : outputs/chicago_crimes_clean.parquet
Outputs : plots/  (PNG files)
          outputs/03_eda_summary.txt
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR   = "outputs"
PLOT_DIR  = "plots"
PARQ_PATH = os.path.join(OUT_DIR, "chicago_crimes_clean.parquet")

os.makedirs(PLOT_DIR, exist_ok=True)

PALETTE   = "YlOrRd"
FIG_DPI   = 150
# ──────────────────────────────────────────────────────────────────────────────


def save_fig(name):
    path = os.path.join(PLOT_DIR, name)
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def run():
    print("=" * 60)
    print("UCMS  ▸  Step 03: EDA")
    print("=" * 60)

    df = pd.read_parquet(PARQ_PATH)
    print(f"  Loaded {len(df):,} rows")

    report_lines = []

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Annual crime volume trend
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[1] Annual trend …")
    annual = df.groupby("year").size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(annual["year"], annual["count"], color="#c0392b", edgecolor="white", linewidth=0.5)
    ax.plot(annual["year"], annual["count"], "o--", color="#2c3e50", linewidth=1.5)
    ax.set_title("Indian Urban Crime Volume by Year (2010–2025)", fontweight="bold", fontsize=13)
    ax.set_xlabel("Year"); ax.set_ylabel("Number of Crimes")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.tight_layout(); save_fig("01_annual_crime_trend.png")
    report_lines.append(f"Peak year : {annual.loc[annual['count'].idxmax(), 'year']} "
                         f"({annual['count'].max():,} crimes)")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Crime by hour of day
    # ─────────────────────────────────────────────────────────────────────────
    print("[2] Hour-of-day distribution …")
    hourly = df.groupby("hour").size()
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [("#e74c3c" if h in [0,1,2,3,21,22,23] else
               "#e67e22" if h in [18,19,20] else "#3498db") for h in range(24)]
    ax.bar(hourly.index, hourly.values, color=colors, edgecolor="white")
    ax.set_title("Crime Frequency by Hour of Day", fontweight="bold", fontsize=13)
    ax.set_xlabel("Hour (0–23)"); ax.set_ylabel("Number of Crimes")
    ax.set_xticks(range(24))
    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#e74c3c", label="High Risk (Night 21–3)"),
        Patch(color="#e67e22", label="Medium Risk (Evening 18–20)"),
        Patch(color="#3498db", label="Lower Risk"),
    ], fontsize=9)
    plt.tight_layout(); save_fig("02_hour_of_day.png")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Crime by day of week
    # ─────────────────────────────────────────────────────────────────────────
    print("[3] Day-of-week …")
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow  = df.groupby("day_of_week").size().reindex(range(7))
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(days, dow.values,
                  color=["#e74c3c" if i in [4,5,6] else "#3498db" for i in range(7)],
                  edgecolor="white")
    ax.set_title("Crime Frequency by Day of Week", fontweight="bold", fontsize=13)
    ax.set_ylabel("Number of Crimes")
    plt.tight_layout(); save_fig("03_day_of_week.png")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Top crime types
    # ─────────────────────────────────────────────────────────────────────────
    print("[4] Top crime types …")
    top_types = df["Primary Type"].value_counts().head(15)
    fig, ax   = plt.subplots(figsize=(10, 6))
    top_types.sort_values().plot(kind="barh", ax=ax, color="#c0392b", edgecolor="white")
    ax.set_title("Top 15 Indian Crime Types (IPC / BNS)", fontweight="bold", fontsize=13)
    ax.set_xlabel("Count")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout(); save_fig("04_top_crime_types.png")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Community-area heatmap (top 30 areas)
    # ─────────────────────────────────────────────────────────────────────────
    print("[5] Urban zone heatmap …")
    ca_year = (df.groupby(["Community Area", "year"])
                 .size()
                 .reset_index(name="count"))
    top_cas  = df["Community Area"].value_counts().head(30).index
    pivot    = (ca_year[ca_year["Community Area"].isin(top_cas)]
                  .pivot(index="Community Area", columns="year", values="count")
                  .fillna(0))
    fig, ax  = plt.subplots(figsize=(14, 9))
    sns.heatmap(pivot, cmap=PALETTE, linewidths=0.3,
                fmt=".0f", annot=False, ax=ax,
                cbar_kws={"label": "Crime Count"})
    ax.set_title("Crime Counts: Top 30 Urban Zones × Year", fontweight="bold", fontsize=13)
    ax.set_xlabel("Year"); ax.set_ylabel("Zone ID")
    plt.tight_layout(); save_fig("05_community_area_heatmap.png")

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Spatial scatter (lat/lon density cloud)
    # ─────────────────────────────────────────────────────────────────────────
    print("[6] Spatial crime density scatter …")
    sample = df[["Latitude", "Longitude"]].dropna().sample(min(200_000, len(df)),
                                                            random_state=42)
    fig, ax = plt.subplots(figsize=(8, 10), facecolor="#1a1a2e")
    ax.scatter(sample["Longitude"], sample["Latitude"],
               s=0.5, alpha=0.4, c="#e74c3c", linewidths=0)
    ax.set_facecolor("#1a1a2e")
    ax.set_title("Delhi NCT / Indian Urban Crime Spatial Density", color="white", fontweight="bold", fontsize=14)
    ax.set_xlabel("Longitude", color="white"); ax.set_ylabel("Latitude", color="white")
    ax.tick_params(colors="white")
    plt.tight_layout(); save_fig("06_spatial_density.png")

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Arrest rate by crime type
    # ─────────────────────────────────────────────────────────────────────────
    print("[7] Arrest rate …")
    arrest_rate = (df.groupby("Primary Type")["Arrest"]
                     .mean()
                     .sort_values(ascending=False)
                     .head(20))
    fig, ax = plt.subplots(figsize=(10, 6))
    arrest_rate.sort_values().plot(kind="barh", ax=ax, color="#27ae60", edgecolor="white")
    ax.set_title("Arrest Rate by Crime Type (Top 20)", fontweight="bold", fontsize=13)
    ax.set_xlabel("Arrest Rate (0–1)")
    ax.axvline(x=arrest_rate.mean(), color="red", linestyle="--",
               label=f"Mean={arrest_rate.mean():.2f}")
    ax.legend()
    plt.tight_layout(); save_fig("07_arrest_rate.png")

    # ─────────────────────────────────────────────────────────────────────────
    # 8. Monthly seasonality (averaged across years)
    # ─────────────────────────────────────────────────────────────────────────
    print("[8] Monthly seasonality …")
    monthly = df.groupby(["year", "month"]).size().reset_index(name="count")
    monthly_avg = monthly.groupby("month")["count"].mean()
    months = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(months, monthly_avg.values, "o-", color="#c0392b", linewidth=2, markersize=7)
    ax.fill_between(range(12), monthly_avg.values, alpha=0.15, color="#c0392b")
    ax.set_title("Average Monthly Crime Volume (Seasonality)", fontweight="bold", fontsize=13)
    ax.set_xlabel("Month"); ax.set_ylabel("Avg Crime Count")
    ax.set_xticks(range(12)); ax.set_xticklabels(months)
    plt.tight_layout(); save_fig("08_monthly_seasonality.png")

    # ─────────────────────────────────────────────────────────────────────────
    # 9. Severity weight distribution
    # ─────────────────────────────────────────────────────────────────────────
    print("[9] Severity distribution …")
    fig, ax = plt.subplots(figsize=(8, 4))
    df["severity_weight"].hist(bins=30, ax=ax, color="#9b59b6", edgecolor="white")
    ax.set_title("Distribution of Severity Weights Across All Crimes",
                 fontweight="bold", fontsize=13)
    ax.set_xlabel("Severity Weight"); ax.set_ylabel("Count")
    plt.tight_layout(); save_fig("09_severity_distribution.png")

    # ─────────────────────────────────────────────────────────────────────────
    # 10. Hour × Day-of-week pivot heatmap
    # ─────────────────────────────────────────────────────────────────────────
    print("[10] Hour × DOW heatmap …")
    hd = df.groupby(["hour", "day_of_week"]).size().reset_index(name="count")
    hd_pivot = hd.pivot(index="hour", columns="day_of_week", values="count").fillna(0)
    hd_pivot.columns = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(hd_pivot, cmap=PALETTE, ax=ax, linewidths=0.3,
                cbar_kws={"label": "Crime Count"})
    ax.set_title("Crime Frequency: Hour × Day of Week", fontweight="bold", fontsize=13)
    ax.set_ylabel("Hour of Day"); ax.set_xlabel("Day of Week")
    plt.tight_layout(); save_fig("10_hour_dow_heatmap.png")

    # ─────────────────────────────────────────────────────────────────────────
    # Summary report
    # ─────────────────────────────────────────────────────────────────────────
    report_lines += [
        f"Total crimes (2010–2025) : {len(df):,}",
        f"Community areas          : {df['Community Area'].nunique()}",
        f"Unique crime types       : {df['Primary Type'].nunique()}",
        f"Avg severity weight      : {df['severity_weight'].mean():.3f}",
        f"Overall arrest rate      : {df['Arrest'].mean():.2%}",
        f"Peak crime hour          : {df['hour'].value_counts().idxmax()}:00",
    ]
    report_path = os.path.join(OUT_DIR, "03_eda_summary.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n  EDA report → {report_path}")

    print("\n[OK] STEP 03 COMPLETE  —  run  04_risk_score_generation.py  next.\n")


if __name__ == "__main__":
    run()