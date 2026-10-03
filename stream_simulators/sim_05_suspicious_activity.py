"""
UCMS - Stream 05: Suspicious Activity Detector (YOLOv8 + UCF-Crime Proxy)
==========================================================================
Two-mode operation:
  MODE A (DEFAULT) — Synthetic simulation (no GPU needed, runs immediately)
                     Generates realistic suspicious activity scores using
                     behavioural pattern modelling calibrated to UCF-Crime stats.

  MODE B           — Real YOLOv8 inference on UCF-Crime video clips.
                     Set USE_YOLO = True  and point UCF_CLIP_DIR to your clips.
                     Requires:  pip install ultralytics opencv-python

Logic (Mode A):
  • Base suspicion from zone's historical violent crime rate
  • Behaviour patterns: loitering, abnormal movement, group gathering
  • Time-slot risk: night = highest detection probability
  • Confidence score = detection_count × avg_confidence

Output:
  outputs/stream_suspicious_activity.parquet
  plots/suspicious_activity_plot.png
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR   = "outputs"
PLOT_DIR  = "plots"
META_PATH = "outputs/zone_metadata.csv"
RISK_PATH = "outputs/zone_risk_scores.parquet"

YEAR_MIN, YEAR_MAX = 2010, 2025
N_SLOTS = 4
SEED    = 42

# ── MODE B config (only if you have UCF-Crime clips) ─────────────────────────
USE_YOLO     = False         # ← Set True only if you have clips + GPU
UCF_CLIP_DIR = "data/ucf_crime_clips"
YOLO_MODEL   = "yolov8n.pt"  # nano = fastest on CPU
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
np.random.seed(SEED)

# Suspicious behaviour types with base detection rates
BEHAVIOURS = {
    "loitering"         : 0.35,
    "abnormal_movement" : 0.28,
    "group_gathering"   : 0.22,
    "running_panic"     : 0.18,
    "object_left"       : 0.12,
    "aggressive_gesture": 0.20,
}

# Time-slot detection multipliers
SLOT_DETECTION = {0: 0.80, 1: 0.25, 2: 0.35, 3: 0.70}

# Month: more outdoor loitering in warm months
MONTH_OUTDOOR  = {
    1:0.55, 2:0.57, 3:0.62, 4:0.68, 5:0.75, 6:0.85,
    7:0.90, 8:0.88, 9:0.78, 10:0.70, 11:0.60, 12:0.56,
}


def load_zone_violence_rate(risk_path, zones):
    """Use historical CPS as proxy for violent activity base rate."""
    df   = pd.read_parquet(risk_path)
    agg  = (df.groupby("Community Area")["severity_mean"].mean()
              .reset_index())
    mn, mx = agg["severity_mean"].min(), agg["severity_mean"].max()
    agg["violence_base"] = (agg["severity_mean"] - mn) / (mx - mn + 1e-9)
    vmap = dict(zip(agg["Community Area"].astype(int), agg["violence_base"]))
    return {int(z): vmap.get(int(z), 0.40) for z in zones}


def simulate_yolo_proxy(violence_base, slot, month, year):
    """
    Simulate what YOLOv8 would output for a zone's CCTV feed.
    Returns: (detection_count, avg_confidence, activity_score)
    """
    base_rate = violence_base * SLOT_DETECTION[slot] * MONTH_OUTDOOR[month]
    # Technology improvement: better detection accuracy over the years
    tech_factor = 1.0 + (year - YEAR_MIN) * 0.015

    # Simulate detections per frame (Poisson)
    lam             = max(0.1, base_rate * 5 * tech_factor)
    detection_count = np.random.poisson(lam)

    # Simulate confidence scores for each detection (Beta distribution)
    if detection_count > 0:
        confidences    = np.random.beta(3, 2, size=detection_count) * 0.7 + 0.2
        avg_confidence = float(confidences.mean())
    else:
        avg_confidence = 0.0

    # Activity score = normalised combination
    activity_score = (0.50 * min(1.0, detection_count / 8.0) +
                      0.50 * avg_confidence * base_rate)
    return detection_count, avg_confidence, float(np.clip(activity_score, 0, 1))


def run_yolo_on_clips(clip_dir, zone_id):
    """
    MODE B: Real YOLOv8 inference.
    Only runs if USE_YOLO=True and clips exist.
    """
    try:
        from ultralytics import YOLO
        import cv2
    except ImportError:
        print("  ⚠  ultralytics/cv2 not installed. Falling back to simulation.")
        return None

    model = YOLO(YOLO_MODEL)
    clips = [f for f in os.listdir(clip_dir) if f.endswith((".mp4",".avi"))]
    if not clips:
        return None

    total_det, total_conf, n_frames = 0, 0, 0
    for clip in clips[:5]:    # sample 5 clips max
        cap = cv2.VideoCapture(os.path.join(clip_dir, clip))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            results = model(frame, verbose=False)
            for r in results:
                # Filter: person class (0) with conf > 0.4
                persons = [b for b in r.boxes if int(b.cls)==0 and float(b.conf)>0.4]
                total_det  += len(persons)
                total_conf += sum(float(b.conf) for b in persons)
                n_frames   += 1
            if n_frames > 100: break   # sample 100 frames max per clip
        cap.release()

    if n_frames == 0:
        return None
    avg_conf = total_conf / max(total_det, 1)
    det_rate = total_det / n_frames
    return min(1.0, 0.5 * det_rate + 0.5 * avg_conf)


def simulate(zones, years, vmap):
    records = []
    for zone in zones:
        vbase = vmap.get(int(zone), 0.40)
        for year in years:
            for slot in range(N_SLOTS):
                for month in range(1, 13):

                    if USE_YOLO:
                        # MODE B: real inference (returns a single score)
                        score = run_yolo_on_clips(UCF_CLIP_DIR, zone)
                        if score is None:
                            score = simulate_yolo_proxy(vbase, slot, month, year)[2]
                        det_count, avg_conf = -1, -1.0
                    else:
                        # MODE A: simulation
                        det_count, avg_conf, score = simulate_yolo_proxy(
                            vbase, slot, month, year)

                    noise = np.random.normal(0, 0.03)
                    score = float(np.clip(score + noise, 0, 1))

                    records.append({
                        "Community Area"          : zone,
                        "year"                    : year,
                        "month"                   : month,
                        "time_slot"               : slot,
                        "suspicious_activity_score": round(score, 4),
                        "yolo_detection_count"    : det_count,
                        "yolo_avg_confidence"     : round(avg_conf, 4),
                    })
    return pd.DataFrame(records)


def plot_results(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Top zones by avg suspicious score
    top20 = (df.groupby("Community Area")["suspicious_activity_score"]
               .mean().sort_values(ascending=False).head(20))
    axes[0].bar(top20.index.astype(str), top20.values,
                color="#e74c3c", edgecolor="white")
    axes[0].set_title("Top 20 Zones — Suspicious Activity", fontweight="bold")
    axes[0].set_xlabel("Community Area"); axes[0].set_ylabel("Avg Score")
    axes[0].tick_params(axis="x", rotation=70, labelsize=7)

    # 2. By time slot
    slots = ["Night","Morning","Afternoon","Evening"]
    sv = [df[df["time_slot"]==s]["suspicious_activity_score"].mean() for s in range(4)]
    colors = ["#1a1a2e","#3498db","#f39c12","#c0392b"]
    axes[1].bar(slots, sv, color=colors, edgecolor="white")
    axes[1].set_title("Score by Time Slot", fontweight="bold")
    axes[1].set_ylabel("Avg Score"); axes[1].tick_params(axis="x", rotation=20)

    # 3. Year trend
    yr = df.groupby("year")["suspicious_activity_score"].mean()
    axes[2].plot(yr.index, yr.values, "s-", color="#c0392b", linewidth=2, markersize=6)
    axes[2].set_title("Year Trend (tech improvement visible)", fontweight="bold")
    axes[2].set_xlabel("Year"); axes[2].set_ylabel("Avg Score")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR,"suspicious_activity_plot.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def run():
    print("="*55)
    print("UCMS  Stream 05 — Suspicious Activity Detector")
    print(f"  Mode: {'YOLOv8 Real Inference' if USE_YOLO else 'Simulation (Mode A)'}")
    print("="*55)

    meta  = pd.read_csv(META_PATH)
    zones = meta["zone_id"].dropna().astype(int).values
    years = list(range(YEAR_MIN, YEAR_MAX+1))

    print("  Loading violence base rates …")
    vmap = load_zone_violence_rate(RISK_PATH, zones)

    print(f"  Generating for {len(zones)} zones × {len(years)} years …")
    df = simulate(zones, years, vmap)

    print(f"  Records   : {len(df):,}")
    print(f"  Mean score: {df['suspicious_activity_score'].mean():.3f}")
    print(f"  Max score : {df['suspicious_activity_score'].max():.3f}")

    out = os.path.join(OUT_DIR, "stream_suspicious_activity.parquet")
    df.to_parquet(out, index=False, compression="snappy")
    print(f"  Saved  → {out}")

    plot_results(df)
    print(f"  Plot   → {PLOT_DIR}/suspicious_activity_plot.png")

    if not USE_YOLO:
        print("\n  ℹ  To use real YOLOv8 inference:")
        print("     1. pip install ultralytics opencv-python")
        print("     2. Download UCF-Crime clips → data/ucf_crime_clips/")
        print("     3. Set USE_YOLO = True at top of this file")

    print("\n✅  Stream 05 DONE\n")


if __name__ == "__main__":
    run()