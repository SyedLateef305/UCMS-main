"""
UCMS - Step 07: GAT Training Loop (CPU-Optimized)
==================================================
Purpose : Train the UCMSGATModel on historical crime risk scores.

Data split strategy:
  Train  : 2010–2020  (11 years)
  Val    : 2021–2022  (2 years)
  Test   : 2023–2025  (held out — for final evaluation in step 09)

Training strategy (CPU-friendly):
  • Iterate over time steps (year × slot) — one forward pass per step
  • Mini-epoch = all zones for one time step
  • Batch over multiple time steps  (batch_size = 8 time steps)
  • Shuffle time steps each epoch
  • Early stopping on val loss (patience=15)
  • LR scheduler: ReduceLROnPlateau

Loss: MSE + 0.1 × MAE  (combined for stable gradients)

Outputs:
  outputs/gat_best_model.pt         — best checkpoint (lowest val loss)
  outputs/gat_final_model.pt        — final epoch model
  outputs/training_log.csv          — epoch-level metrics
  plots/training_curves.png
"""

import os, sys, time, json, warnings
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gat_model_06 import UCMSGATModel, make_temporal_encoding

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR      = "./outputs"
PLOT_DIR     = "./plots"
MODEL_DIR    = "outputs"

FEAT_PATH    = os.path.join(OUT_DIR, "gat_node_features.npy")
ADJ_PATH     = os.path.join(OUT_DIR, "gat_adjacency.npy")
EDGE_PATH    = os.path.join(OUT_DIR, "gat_edge_index.npy")
EW_PATH      = os.path.join(OUT_DIR, "gat_edge_weight.npy")
RISK_PATH    = os.path.join(OUT_DIR, "zone_risk_scores.parquet")
META_PATH    = os.path.join(OUT_DIR, "zone_metadata.csv")

# Training hyper-params
TRAIN_YEARS  = list(range(2010, 2021))
VAL_YEARS    = [2021, 2022]
TEST_YEARS   = [2023, 2024, 2025]
N_SLOTS      = 4
YEAR_MIN     = 2010
YEAR_MAX     = 2025

EPOCHS       = 120
BATCH_STEPS  = 8       # time steps per mini-batch
LR           = 3e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 15      # early stopping
DROPOUT      = 0.3

SEED         = 42
# ──────────────────────────────────────────────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_data():
    """Load numpy arrays and build target lookup."""
    feats      = np.load(FEAT_PATH)          # (N, T, F)
    adj        = np.load(ADJ_PATH)           # (N, N)
    risk_df = pd.read_parquet(RISK_PATH, engine="fastparquet")
    meta       = pd.read_csv(META_PATH)

    zones      = meta["zone_id"].values
    all_years  = list(range(YEAR_MIN, YEAR_MAX + 1))

    # Build target lookup: (zone_idx, year, slot) → cps_final
    target = {}
    for _, row in risk_df.iterrows():
        z = int(row["Community Area"])
        y = int(row["year"])
        s = int(row["time_slot"])
        if z in zones:
            zi = int(np.where(zones == z)[0][0])
            target[(zi, y, s)] = float(row["cps_final"])

    return feats, adj, zones, all_years, target


def build_time_steps(years, n_slots):
    """All (year, slot) combinations for a set of years."""
    return [(y, s) for y in years for s in range(n_slots)]


def batch_forward(model, feats_t, adj_t, year, slot, n_nodes):
    """Single forward pass for one time step."""
    x   = torch.tensor(feats_t, dtype=torch.float32)   # (N, F)
    t   = make_temporal_encoding(year, slot, YEAR_MIN, YEAR_MAX, n_nodes)
    out = model(x, adj_t, t)                            # (N, 1)
    return out


def get_targets(target_dict, year, slot, n_nodes):
    """Build target tensor (N,1) for a time step."""
    y = torch.zeros(n_nodes, 1, dtype=torch.float32)
    for zi in range(n_nodes):
        key = (zi, year, slot)
        if key in target_dict:
            y[zi, 0] = target_dict[key]
    return y


def combined_loss(pred, target):
    mse = F.mse_loss(pred, target)
    mae = F.l1_loss(pred, target)
    return mse + 0.1 * mae


import torch.nn.functional as F


def train_epoch(model, optimizer, feats, adj_t, time_steps, target_dict,
                n_nodes, batch_size):
    model.train()
    steps    = list(time_steps)
    np.random.shuffle(steps)
    total_loss = 0.0
    n_batches  = 0

    for i in range(0, len(steps), batch_size):
        batch = steps[i: i + batch_size]
        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0)

        for (year, slot) in batch:
            # Feature slice: find year index
            yr_idx  = year - YEAR_MIN
            if yr_idx < 0 or yr_idx >= feats.shape[1]:
                continue
            feats_t = feats[:, yr_idx, :]        # (N, F)
            pred    = batch_forward(model, feats_t, adj_t, year, slot, n_nodes)
            tgt     = get_targets(target_dict, year, slot, n_nodes)
            batch_loss = batch_loss + combined_loss(pred, tgt)

        if batch_loss.item() > 0:
            (batch_loss / len(batch)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += batch_loss.item() / len(batch)
            n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, feats, adj_t, time_steps, target_dict, n_nodes):
    model.eval()
    all_pred, all_tgt = [], []

    for (year, slot) in time_steps:
        yr_idx = year - YEAR_MIN
        if yr_idx < 0 or yr_idx >= feats.shape[1]:
            continue
        feats_t = feats[:, yr_idx, :]
        pred    = batch_forward(model, feats_t, adj_t, year, slot, n_nodes)
        tgt     = get_targets(target_dict, year, slot, n_nodes)
        all_pred.append(pred)
        all_tgt.append(tgt)

    pred_all = torch.cat(all_pred)
    tgt_all  = torch.cat(all_tgt)
    mse  = F.mse_loss(pred_all, tgt_all).item()
    mae  = F.l1_loss(pred_all,  tgt_all).item()
    rmse = mse ** 0.5
    # R²
    ss_res = ((tgt_all - pred_all) ** 2).sum().item()
    ss_tot = ((tgt_all - tgt_all.mean()) ** 2).sum().item()
    r2 = 1 - ss_res / max(ss_tot, 1e-9)

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def plot_curves(log_df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(log_df["epoch"], log_df["train_loss"], label="Train", color="#e74c3c")
    axes[0].plot(log_df["epoch"], log_df["val_loss"],   label="Val",   color="#3498db")
    axes[0].set_title("Loss (MSE + 0.1·MAE)"); axes[0].legend()
    axes[0].set_xlabel("Epoch")

    axes[1].plot(log_df["epoch"], log_df["val_rmse"], color="#9b59b6")
    axes[1].set_title("Validation RMSE"); axes[1].set_xlabel("Epoch")

    axes[2].plot(log_df["epoch"], log_df["val_r2"], color="#27ae60")
    axes[2].axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    axes[2].set_title("Validation R²"); axes[2].set_xlabel("Epoch")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved → {path}")


def run():
    print("=" * 60)
    print("UCMS  ▸  Step 07: GAT Training (CPU)")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n  Loading data …")
    feats, adj, zones, all_years, target_dict = load_data()
    N, T, Ft = feats.shape
    print(f"  Nodes : {N}  |  Years : {T}  |  Features : {Ft}")
    print(f"  Target entries : {len(target_dict):,}")

    adj_t  = torch.tensor(adj, dtype=torch.float32)

    train_steps = build_time_steps(TRAIN_YEARS, N_SLOTS)
    val_steps   = build_time_steps(VAL_YEARS,   N_SLOTS)
    test_steps  = build_time_steps(TEST_YEARS,  N_SLOTS)
    print(f"  Train steps: {len(train_steps)}  |  Val: {len(val_steps)}  |  Test: {len(test_steps)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n  Initialising model …")
    model = UCMSGATModel(
        in_features = Ft,
        hidden1     = 64,
        hidden2     = 32,
        n_heads1    = 4,
        n_heads2    = 2,
        dropout     = DROPOUT,
        use_pyg     = False,
    )
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params : {params:,}")

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n  Training for up to {EPOCHS} epochs (early stopping: patience={PATIENCE})\n")
    print(f"  {'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>9} | "
          f"{'RMSE':>7} | {'R²':>6} | {'LR':>8} | {'Time':>6}")
    print("  " + "-" * 65)

    best_val    = float("inf")
    best_epoch  = 0
    no_improve  = 0
    log_rows    = []
    t_total     = time.time()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, optimizer, feats, adj_t,
                               train_steps, target_dict, N, BATCH_STEPS)
        val_m   = evaluate(model, feats, adj_t, val_steps, target_dict, N)
        val_loss = val_m["mse"]
        scheduler.step(val_loss)
        lr_now  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        log_rows.append({
            "epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss,
            "val_rmse": val_m["rmse"], "val_mae": val_m["mae"], "val_r2": val_m["r2"],
            "lr": lr_now,
        })

        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:>5} | {tr_loss:>10.5f} | {val_loss:>9.5f} | "
                  f"{val_m['rmse']:>7.4f} | {val_m['r2']:>6.3f} | "
                  f"{lr_now:>8.2e} | {elapsed:>5.1f}s")

        # Checkpoint
        if val_loss < best_val:
            best_val   = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_m,
                "config": {
                    "in_features": Ft, "hidden1": 64, "hidden2": 32,
                    "n_heads1": 4, "n_heads2": 2, "dropout": DROPOUT,
                }
            }, os.path.join(MODEL_DIR, "gat_best_model.pt"))
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"\n  ⏹  Early stopping at epoch {epoch} "
                  f"(best was epoch {best_epoch}, val_loss={best_val:.5f})")
            break

    # ── Save final + log ──────────────────────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "gat_final_model.pt"))
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(os.path.join(MODEL_DIR, "training_log.csv"), index=False)

    total_time = (time.time() - t_total) / 60
    print(f"\n  Training complete  —  {total_time:.1f} min total")
    print(f"  Best epoch : {best_epoch}  |  Best val loss : {best_val:.5f}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\n  Loading best model for test evaluation …")
    ckpt = torch.load(os.path.join(MODEL_DIR, "gat_best_model.pt"), map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    test_m = evaluate(model, feats, adj_t, test_steps, target_dict, N)
    print(f"\n  ── TEST RESULTS (2023–2025) ──────────────────")
    for k, v in test_m.items():
        print(f"     {k.upper():<8} : {v:.4f}")

    # Save test results
    with open(os.path.join(MODEL_DIR, "test_results.json"), "w") as f:
        json.dump({"best_epoch": best_epoch, "test": test_m,
                   "val_best": best_val}, f, indent=2)

    plot_curves(log_df)

    print("\n✅  STEP 07 COMPLETE  —  run  08_baseline_comparison.py  next.\n")


if __name__ == "__main__":
    run()