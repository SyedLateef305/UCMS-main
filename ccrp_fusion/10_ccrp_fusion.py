"""
UCMS - Step 10: CCRP Fusion Engine
=====================================
Purpose : Learn optimal per-stream weights that combine all 6 streams
          into a single CCRP risk score that best matches the GAT-predicted CPS.

          This IS the core CCRP algorithm contribution:
          ─────────────────────────────────────────────
          CCRP(z,t) = Σ_i  w_i(z,t) × stream_i(z,t)

          where w_i are learned via gradient descent minimising
          MSE against the GAT model's CPS predictions.

          Weights are:
            • Constrained to sum to 1 (softmax)
            • Zone-adaptive  (different weights per community area type)
            • Temporally-aware (slot-specific weight profiles)

Method:
  1. Load all_streams_merged.parquet  (6 stream scores per zone×year×slot)
  2. Load GAT predictions as supervision signal (ground truth)
  3. Learn W via small MLP (zone_type_embedding + slot_encoding → weights)
  4. Compute weighted CCRP score per record
  5. Save learned weights + CCRP scores

Outputs:
  outputs/ccrp_fusion_weights.pt        ← trained weight model
  outputs/ccrp_scores.parquet           ← final CCRP score per zone×year×slot
  outputs/learned_stream_weights.csv    ← interpretable weight table
  plots/fusion_weight_heatmap.png
  plots/ccrp_vs_gat_scatter.png
"""

import os, sys, warnings, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
torch.manual_seed(42); np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STREAM_DIR = os.path.join(BASE_DIR, "..", "outputs")
GAT_DIR = os.path.join(BASE_DIR, "..", "gat_model", "outputs")
HIST_DIR = os.path.join(BASE_DIR, "..", "historical_analysis", "outputs")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

MERGED_PATH = os.path.join(STREAM_DIR, "all_streams_merged.parquet")
GAT_FEAT    = os.path.join(HIST_DIR,   "gat_node_features.npy")
RISK_PATH   = os.path.join(HIST_DIR,   "zone_risk_scores.parquet")
META_PATH   = os.path.join(HIST_DIR,   "zone_metadata.csv")

N_STREAMS = 5

STREAM_COLS = [
    "crowd_density",
    "lighting_risk",
    "social_media",
    "suspicious_activity",
    "cctv_feed"
]

YEAR_MIN, YEAR_MAX = 2010, 2025
N_SLOTS     = 4
TRAIN_YEARS = list(range(2010, 2022))
TEST_YEARS  = list(range(2022, 2026))

# Training
EPOCHS      = 150
LR          = 2e-3
BATCH_SIZE  = 512
DROPOUT     = 0.25
# ──────────────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Weight Network
# ─────────────────────────────────────────────────────────────────────────────
class AdaptiveWeightNet(nn.Module):
    """
    Learns zone-adaptive, temporally-aware fusion weights.

    Input  : [zone_risk_level (1), slot_encoding (4), year_norm (1)]  → 6-dim
    Output : weights (5,) softmax → sum to 1
    """
    def __init__(self, n_streams=5, hidden=32, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_streams),
        )
        # Init: start with roughly equal weights
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, context):
        """context: (B, 6) → weights: (B, n_streams)"""
        logits  = self.net(context)
        weights = F.softmax(logits, dim=-1)
        return weights


def build_context(df):
    """
    Build context vectors for AdaptiveWeightNet.
    Returns: context (N, 6), streams (N, 5), target (N,)
    """
    # Zone risk level (from historical CPS mean)
    zone_risk = (df.groupby("Community Area")["historical_cps"]
                   .transform("mean")
                   .fillna(0.5).values)

    # Slot one-hot (4-dim)
    slot_oh = np.eye(4)[df["time_slot"].values.astype(int)]

    # Year normalised
    yr_norm = ((df["year"].values - YEAR_MIN) / (YEAR_MAX - YEAR_MIN)).reshape(-1, 1)

    context = np.hstack([zone_risk.reshape(-1,1), slot_oh, yr_norm])  # (N, 6)
    streams = df[STREAM_COLS].fillna(0).values                         # (N, 6)
    target  = df["historical_cps"].fillna(0).values                    # (N,)
    return context, streams, target


def train_weight_net(df_train, df_test):
    ctx_tr,  str_tr,  tgt_tr  = build_context(df_train)
    ctx_te,  str_te,  tgt_te  = build_context(df_test)

    ctx_tr  = torch.tensor(ctx_tr,  dtype=torch.float32)
    str_tr  = torch.tensor(str_tr,  dtype=torch.float32)
    tgt_tr  = torch.tensor(tgt_tr,  dtype=torch.float32).unsqueeze(-1)
    ctx_te  = torch.tensor(ctx_te,  dtype=torch.float32)
    str_te  = torch.tensor(str_te,  dtype=torch.float32)
    tgt_te  = torch.tensor(tgt_te,  dtype=torch.float32).unsqueeze(-1)

    model   = AdaptiveWeightNet(N_STREAMS)
    opt     = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    N       = len(ctx_tr)
    best_val_loss = float("inf")
    best_state    = None
    log_rows      = []

    print(f"\n  Training fusion weight network …")
    print(f"  Train: {N:,}   Test: {len(ctx_te):,}")
    print(f"  {'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>9} | {'R²':>6}")
    print("  " + "-"*40)

    for epoch in range(1, EPOCHS+1):
        model.train()
        idx = torch.randperm(N)
        ep_loss = 0.0
        n_batches = 0

        for i in range(0, N, BATCH_SIZE):
            b  = idx[i:i+BATCH_SIZE]
            wt = model(ctx_tr[b])                          # (B, 6)
            # Weighted sum of streams → predicted CCRP
            pred = (wt * str_tr[b]).sum(dim=-1, keepdim=True)  # (B, 1)

            # Loss: MSE + entropy regulariser (encourage diverse weights)
            mse_loss  = F.mse_loss(pred, tgt_tr[b])
            entr_reg  = -0.01 * (wt * torch.log(wt + 1e-8)).sum(dim=-1).mean()
            loss      = mse_loss + entr_reg

            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss  += mse_loss.item(); n_batches += 1

        sched.step()

        # Validation
        model.eval()
        with torch.no_grad():
            wt_te    = model(ctx_te)
            pred_te  = (wt_te * str_te).sum(dim=-1, keepdim=True)
            val_loss = F.mse_loss(pred_te, tgt_te).item()
            r2       = r2_score(tgt_te.numpy(), pred_te.numpy())

        tr_loss = ep_loss / max(n_batches, 1)
        log_rows.append({"epoch":epoch,"train_loss":tr_loss,
                          "val_loss":val_loss,"r2":r2})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k:v.clone() for k,v in model.state_dict().items()}

        if epoch % 25 == 0 or epoch == 1:
            print(f"  {epoch:>5} | {tr_loss:>10.5f} | {val_loss:>9.5f} | {r2:>6.3f}")

    model.load_state_dict(best_state)
    return model, pd.DataFrame(log_rows)


def extract_weights(model, df):
    """Get learned weights for every record."""
    ctx, streams, _ = build_context(df)
    ctx_t  = torch.tensor(ctx,     dtype=torch.float32)
    str_t  = torch.tensor(streams, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        wt   = model(ctx_t)                                   # (N, 6)
        pred = (wt * str_t).sum(dim=-1).numpy()               # (N,)
    return wt.numpy(), pred


def plot_weight_heatmap(df_weights):
    """Average learned weights per time slot."""
    slot_names = ["Night(0-6)","Morning(6-12)","Afternoon(12-18)","Evening(18-24)"]
    pivot = pd.DataFrame(index=slot_names, columns=STREAM_COLS)
    for s in range(4):
        mask = df_weights["time_slot"] == s
        for col in STREAM_COLS:
            pivot.loc[slot_names[s], col] = df_weights.loc[mask, f"w_{col}"].mean()
    pivot = pivot.astype(float)

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                ax=ax, linewidths=0.5, cbar_kws={"label":"Weight"})
    ax.set_title("CCRP Learned Fusion Weights — Per Time Slot",
                  fontweight="bold", fontsize=13)
    ax.set_xlabel("Stream"); ax.set_ylabel("Time Slot")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR,"fusion_weight_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot → {PLOT_DIR}/fusion_weight_heatmap.png")


def plot_ccrp_vs_target(df_out):
    sample = df_out.sample(min(8000, len(df_out)), random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter
    axes[0].scatter(sample["historical_cps"], sample["ccrp_score"],
                    alpha=0.3, s=5, color="#e74c3c")
    axes[0].plot([0,1],[0,1],"k--",linewidth=1)
    axes[0].set_title("CCRP Score vs Historical CPS (GAT target)",
                       fontweight="bold")
    axes[0].set_xlabel("Historical CPS"); axes[0].set_ylabel("CCRP Score")

    # Distribution comparison
    axes[1].hist(df_out["historical_cps"], bins=50, alpha=0.6,
                 color="#3498db", label="Historical CPS", density=True)
    axes[1].hist(df_out["ccrp_score"],     bins=50, alpha=0.6,
                 color="#e74c3c", label="CCRP Fused",  density=True)
    axes[1].set_title("Score Distribution Comparison", fontweight="bold")
    axes[1].set_xlabel("Score"); axes[1].set_ylabel("Density")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR,"ccrp_vs_gat_scatter.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot → {PLOT_DIR}/ccrp_vs_gat_scatter.png")


def run():
    print("="*60)
    print("UCMS  Step 10 — CCRP Fusion Engine")
    print("="*60)

    # ── Load merged streams ───────────────────────────────────────────────────
    print(f"\n  Loading merged streams …")
    df = pd.read_parquet(MERGED_PATH)
    print(f"  Shape: {df.shape}  |  Cols: {list(df.columns)}")

    # Confirm all stream cols present
    missing = [c for c in STREAM_COLS if c not in df.columns]
    if missing:
        print(f"  ❌  Missing columns: {missing}")
        sys.exit(1)

    # ── Split train/test ──────────────────────────────────────────────────────
    df_train = df[df["year"].isin(TRAIN_YEARS)].reset_index(drop=True)
    df_test  = df[df["year"].isin(TEST_YEARS)].reset_index(drop=True)
    print(f"  Train: {len(df_train):,}  |  Test: {len(df_test):,}")

    # ── Train fusion weight network ───────────────────────────────────────────
    model, log_df = train_weight_net(df_train, df_test)

    # ── Extract weights for ALL records ──────────────────────────────────────
    print("\n  Extracting weights for full dataset …")
    weights_all, pred_all = extract_weights(model, df)

    # Build output dataframe
    
    df_out = df[
    ["Community Area", "year", "time_slot", "historical_cps"]+ STREAM_COLS].copy()
    for i, col in enumerate(STREAM_COLS):
        df_out[f"w_{col}"] = weights_all[:, i]
    df_out["ccrp_score"] = np.clip(pred_all, 0, 1)

    # ── Metrics ───────────────────────────────────────────────────────────────
    tgt  = df_out["historical_cps"].values
    pred = df_out["ccrp_score"].values
    rmse = mean_squared_error(tgt, pred) ** 0.5
    mae  = mean_absolute_error(tgt, pred)
    r2   = r2_score(tgt, pred)
    print(f"\n  Fusion Performance (full dataset):")
    print(f"    RMSE  : {rmse:.4f}")
    print(f"    MAE   : {mae:.4f}")
    print(f"    R²    : {r2:.4f}")

    # ── Average weights per slot (interpretability) ───────────────────────────
    print("\n  Learned weights per time slot (avg):")
    slot_names = ["Night","Morning","Afternoon","Evening"]
    weight_rows = []
    for s in range(4):
        mask = df_out["time_slot"] == s
        row  = {"slot": slot_names[s]}
        for col in STREAM_COLS:
            row[col] = round(float(df_out.loc[mask, f"w_{col}"].mean()), 4)
        weight_rows.append(row)
        vals = "  ".join(f"{col[:8]}={row[col]:.3f}" for col in STREAM_COLS)
        print(f"    {slot_names[s]:<12} | {vals}")

    weight_df = pd.DataFrame(weight_rows)

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(OUT_DIR,"ccrp_fusion_weights.pt"))
    df_out.to_parquet(os.path.join(OUT_DIR,"ccrp_scores.parquet"),
                      index=False, compression="snappy")
    weight_df.to_csv(os.path.join(OUT_DIR,"learned_stream_weights.csv"), index=False)
    log_df.to_csv(os.path.join(OUT_DIR,"fusion_training_log.csv"), index=False)

    metrics = {"rmse":round(rmse,4),"mae":round(mae,4),"r2":round(r2,4)}
    with open(os.path.join(OUT_DIR,"fusion_metrics.json"),"w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Saved:")
    print(f"    ccrp_fusion_weights.pt")
    print(f"    ccrp_scores.parquet   ({len(df_out):,} records)")
    print(f"    learned_stream_weights.csv")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_weight_heatmap(df_out)
    plot_ccrp_vs_target(df_out)

    print("\n✅  STEP 10 DONE  →  run  11_bayesian_layer.py  next\n")


if __name__ == "__main__":
    run()