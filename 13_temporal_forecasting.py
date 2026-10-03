"""
UCMS - Step 13: Temporal Forecasting
====================================
Phase II Addition.

Purpose : Train per-zone sequence models on historical CCRP time-series (2010-2025)
          and predict 2026 risk scores for every zone × slot.

Model   : PyTorch LSTM if torch is installed; otherwise high-performance
          Gradient Boosting / Ridge Autoregressive forecaster.

Inputs  : ccrp_fusion/outputs/ccrp_final_master.parquet
Outputs :
  outputs/forecasts_2026.parquet   — zone × slot × predicted_cps_2026
  outputs/forecast_metrics.json    — per-zone RMSE / MAE (on 2023-2025 holdout)
  plots/forecast_trend.png         — top-10 zone forecast sparklines

Usage   : python 13_temporal_forecasting.py [--dry-run]
"""

import os, sys, json, warnings, argparse
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")
np.random.seed(42)

# Check PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MASTER_PATH = os.path.join(BASE_DIR, "ccrp_fusion", "outputs", "ccrp_final_master.parquet")
OUT_DIR     = os.path.join(BASE_DIR, "outputs")
PLOT_DIR    = os.path.join(BASE_DIR, "plots")

YEAR_MIN    = 2010
YEAR_MAX    = 2025
FORECAST_YR = 2026
SEQ_LEN     = 5        # look-back window (years)
TRAIN_END   = 2022     # train on 2010-2022, validate 2023-2025
N_SLOTS     = 4
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch LSTM Model (Optional)
# ─────────────────────────────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    class ZoneLSTM(nn.Module):
        def __init__(self, n_zones: int, embed_dim: int = 8, hidden: int = 32):
            super().__init__()
            self.embed   = nn.Embedding(n_zones, embed_dim)
            self.lstm    = nn.LSTM(
                input_size=1 + embed_dim, hidden_size=hidden,
                num_layers=1, batch_first=True
            )
            self.head = nn.Sequential(
                nn.Linear(hidden, 16),
                nn.ELU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
            self._init()

        def _init(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, zone_idx, seq):
            e = self.embed(zone_idx)
            e = e.unsqueeze(1).expand(-1, seq.size(1), -1)
            x = torch.cat([seq, e], dim=-1)
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])


# ─────────────────────────────────────────────────────────────────────────────
# Feature preparation
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset(df, zones, seq_len=SEQ_LEN, train_end=TRAIN_END):
    pivot = (df.groupby(["Community Area", "year", "time_slot"])["ccrp_bayesian"]
               .mean()
               .reset_index())

    all_years = list(range(YEAR_MIN, YEAR_MAX + 1))
    zone_to_idx = {z: i for i, z in enumerate(zones)}

    train_X, train_Z, train_Y, train_S = [], [], [], []
    val_X,   val_Z,   val_Y,   val_S   = [], [], [], []
    full_X,  full_Z,  full_S            = [], [], []

    for zi, zone in enumerate(zones):
        for slot in range(N_SLOTS):
            sub = pivot[(pivot["Community Area"] == zone) &
                        (pivot["time_slot"] == slot)].set_index("year")

            series = np.array([
                float(sub.loc[y, "ccrp_bayesian"]) if y in sub.index else 0.0
                for y in all_years
            ], dtype=np.float32)

            for t in range(seq_len, len(all_years)):
                seq = series[t - seq_len: t]
                tgt = series[t]
                yr  = all_years[t]

                if yr <= train_end:
                    train_X.append(seq); train_Z.append(zi); train_Y.append(tgt); train_S.append(slot)
                else:
                    val_X.append(seq); val_Z.append(zi); val_Y.append(tgt); val_S.append(slot)

            full_X.append(series[-seq_len:])
            full_Z.append(zi)
            full_S.append(slot)

    return (np.array(train_X), np.array(train_Z), np.array(train_Y), np.array(train_S)), \
           (np.array(val_X),   np.array(val_Z),   np.array(val_Y),   np.array(val_S)),   \
           (np.array(full_X),  np.array(full_Z),  np.array(full_S))


# ─────────────────────────────────────────────────────────────────────────────
# Training and Prediction
# ─────────────────────────────────────────────────────────────────────────────
def train_and_forecast_sklearn(train_data, val_data, full_data, zones):
    X_tr, Z_tr, Y_tr, S_tr = train_data
    X_vl, Z_vl, Y_vl, S_vl = val_data
    X_fc, Z_fc, S_fc       = full_data

    # Feature matrix: [lags | zone_id | slot_id | mean_lag | std_lag | min_lag | max_lag]
    def make_feats(X, Z, S):
        mean_l = X.mean(axis=1, keepdims=True)
        std_l  = X.std(axis=1, keepdims=True)
        min_l  = X.min(axis=1, keepdims=True)
        max_l  = X.max(axis=1, keepdims=True)
        return np.hstack([X, Z[:, None], S[:, None], mean_l, std_l, min_l, max_l])

    F_tr = make_feats(X_tr, Z_tr, S_tr)
    F_vl = make_feats(X_vl, Z_vl, S_vl)
    F_fc = make_feats(X_fc, Z_fc, S_fc)

    model = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, random_state=42)
    model.fit(F_tr, Y_tr)

    val_preds = model.predict(F_vl)
    val_preds = np.clip(val_preds, 0, 1)

    # Per-zone metrics
    zone_metrics = {}
    for zi, zone in enumerate(zones):
        mask = (Z_vl == zi)
        if mask.sum() == 0:
            continue
        p, t = val_preds[mask], Y_vl[mask]
        rmse = float(((p - t) ** 2).mean() ** 0.5)
        mae  = float(np.abs(p - t).mean())
        zone_metrics[int(zone)] = {"rmse": round(rmse, 4), "mae": round(mae, 4)}

    # Refit on all data for 2026 forecast
    F_all = np.vstack([F_tr, F_vl])
    Y_all = np.concatenate([Y_tr, Y_vl])
    model.fit(F_all, Y_all)

    fc_preds = np.clip(model.predict(F_fc), 0, 1)

    rows = []
    for i in range(len(Z_fc)):
        zi   = Z_fc[i]
        slot = S_fc[i]
        rows.append({
            "Community Area": int(zones[zi]),
            "year": FORECAST_YR,
            "time_slot": int(slot),
            "predicted_cps": round(float(fc_preds[i]), 4),
        })

    return pd.DataFrame(rows), zone_metrics


def plot_forecast_trends(master_df, forecast_df, zones):
    top10 = (forecast_df.groupby("Community Area")["predicted_cps"]
                         .mean()
                         .sort_values(ascending=False)
                         .head(10)
                         .index.tolist())

    pivot_hist = (master_df.groupby(["Community Area", "year"])["ccrp_bayesian"]
                            .mean().reset_index())

    fig, axes = plt.subplots(2, 5, figsize=(16, 6), facecolor="#0d1117")
    fig.suptitle(f"CCRP Trend + {FORECAST_YR} Forecast — Top 10 Risk Zones",
                 color="white", fontsize=13, fontweight="bold")

    for ax, zone in zip(axes.flatten(), top10):
        hist = pivot_hist[pivot_hist["Community Area"] == zone].sort_values("year")
        fc   = forecast_df[(forecast_df["Community Area"] == zone)]["predicted_cps"].mean()

        ax.plot(hist["year"], hist["ccrp_bayesian"],
                color="#388bfd", linewidth=1.8, marker="o", markersize=3)
        ax.plot([hist["year"].iloc[-1], FORECAST_YR],
                [hist["ccrp_bayesian"].iloc[-1], fc],
                color="#ff4d4f", linewidth=1.8, linestyle="--", marker="o", markersize=4)

        ax.set_title(f"Zone {zone}", color="white", fontsize=10)
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#484f58", labelsize=7)
        ax.spines[:].set_color("#21262d")
        ax.set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "forecast_trend.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Plot -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run(dry_run: bool = False):
    print("=" * 60)
    print("UCMS Phase II -> Step 13: Temporal Forecasting")
    print("=" * 60)

    if dry_run:
        print("  [DRY RUN] Checking imports only.")
        print("  [OK] All imports OK. Run without --dry-run to execute fully.\n")
        return

    print(f"\n  Loading master parquet: {MASTER_PATH}")
    if not os.path.exists(MASTER_PATH):
        print(f"  [ERROR] File not found: {MASTER_PATH}")
        sys.exit(1)

    df = pd.read_parquet(MASTER_PATH)
    print(f"  Loaded {len(df):,} records")

    zones = np.sort(df["Community Area"].dropna().unique().astype(int))
    n_zones = len(zones)
    print(f"  Zones: {n_zones}  |  Slots: {N_SLOTS}  |  Years: {YEAR_MIN}-{YEAR_MAX}")

    print("\n  Building time-series sequences ...")
    train_data, val_data, full_data = build_dataset(df, zones, SEQ_LEN, TRAIN_END)
    print(f"  Train samples: {len(train_data[0]):,}  |  Val samples: {len(val_data[0]):,}")

    print("\n  Training Forecaster ...")
    forecast_df, zone_metrics = train_and_forecast_sklearn(train_data, val_data, full_data, zones)

    all_rmse = [v["rmse"] for v in zone_metrics.values()]
    print(f"  Mean RMSE (all zones): {np.mean(all_rmse):.4f}")
    print(f"  Max  RMSE:             {np.max(all_rmse):.4f}")

    # Top 5
    top5 = (forecast_df.groupby("Community Area")["predicted_cps"]
                        .mean().sort_values(ascending=False).head(5))
    print(f"\n  Top-5 predicted high-risk zones ({FORECAST_YR}):")
    for zone, cps in top5.items():
        print(f"    Zone {zone:3d}  ->  {cps:.4f}")

    # Save outputs
    fc_path = os.path.join(OUT_DIR, "forecasts_2026.parquet")
    forecast_df.to_parquet(fc_path, index=False, compression="snappy")
    print(f"\n  Saved -> {fc_path}")

    metrics_path = os.path.join(OUT_DIR, "forecast_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"zone_metrics": zone_metrics,
                   "mean_rmse": round(float(np.mean(all_rmse)), 4),
                   "forecast_year": FORECAST_YR}, f, indent=2)
    print(f"  Saved -> {metrics_path}")

    # Plot
    print("\n  Plotting forecast trends ...")
    plot_forecast_trends(df, forecast_df, zones)

    print("\n[OK] STEP 13 COMPLETE - Forecasts ready for Phase II API\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
