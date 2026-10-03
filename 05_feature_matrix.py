"""
UCMS - Step 05: GAT Feature Matrix Builder
===========================================
Purpose : Assemble the final node-feature matrix and adjacency information
          that feeds directly into the Graph Attention Network (GAT).

          Each node  = one Community Area  (77 nodes for Chicago)
          Each edge  = geographic proximity (k-NN on lat/lon centroids)
          Features   = [cps_final, avg_severity, arrest_rate, domestic_rate,
                        yoy_change_pct, lat_norm, lon_norm,
                        hour_risk_weights × 4 slots]   → 14-dim per node

Outputs:
  outputs/gat_node_features.npy       — (N_zones, T_years, F_features)
  outputs/gat_adjacency.npy           — (N_zones, N_zones) — weighted adj
  outputs/gat_edge_index.npy          — (2, E) COO format for PyG
  outputs/gat_edge_weight.npy         — (E,) edge weights
  outputs/zone_metadata.csv           — zone IDs + lat/lon centroids
  outputs/05_feature_report.txt
"""

import os, warnings
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR      = "outputs"
ANNUAL_PATH  = os.path.join(OUT_DIR, "zone_annual_profile.parquet")
RISK_PATH    = os.path.join(OUT_DIR, "zone_risk_scores.parquet")

K_NEIGHBORS  = 5          # edges per node (spatial proximity graph)
DIST_SIGMA   = 0.05       # Gaussian kernel sigma for edge weights (degrees)
N_SLOTS      = 4          # time slots per day (must match step 04)
# ──────────────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

SLOT_WEIGHTS = {0: 0.85, 1: 0.30, 2: 0.50, 3: 0.80}   # night/morning/afternoon/evening


def build_adjacency(centroids: np.ndarray, k: int, sigma: float):
    """
    Build weighted adjacency matrix using Gaussian kernel on Euclidean distance
    of (lat, lon) centroids.  Returns:
      adj  (N,N)  — full symmetric weighted adj
      edge_index (2,E)  — COO sparse format
      edge_weight (E,)  — corresponding weights
    """
    N = len(centroids)
    dist = cdist(centroids, centroids, metric="euclidean")
    # Gaussian weight
    W = np.exp(- (dist ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(W, 0)

    # k-NN masking — keep only top-k neighbours per row
    adj = np.zeros_like(W)
    for i in range(N):
        top_k = np.argsort(W[i])[-k:]
        adj[i, top_k] = W[i, top_k]

    # Symmetrise
    adj = np.maximum(adj, adj.T)

    # COO format
    src, dst = np.where(adj > 0)
    edge_index  = np.stack([src, dst], axis=0)
    edge_weight = adj[src, dst]

    return adj, edge_index, edge_weight


def build_node_features(annual: pd.DataFrame, risk: pd.DataFrame,
                         zones: np.ndarray, years: np.ndarray) -> np.ndarray:
    """
    Build feature tensor  (N_zones, N_years, F_features).

    Features (14 total):
      0  cps_final          (mean across slots)
      1  total_crimes_norm  (log-normalised)
      2  avg_severity
      3  avg_arrest_rate
      4  avg_domestic_rate
      5  yoy_change_pct     (clipped ±50%)
      6  lat_norm
      7  lon_norm
      8-11  cps per time slot (0..3)
      12  max_cps
      13  mean_cps
    """
    N  = len(zones)
    T  = len(years)
    F  = 14

    # Centroid lookup
    centroids = (annual.groupby("Community Area")[["lat","lon"]]
                        .mean()
                        .reindex(zones))
    lat_s = StandardScaler().fit(centroids[["lat"]])
    lon_s = StandardScaler().fit(centroids[["lon"]])
    centroids["lat_norm"] = lat_s.transform(centroids[["lat"]]).flatten()
    centroids["lon_norm"] = lon_s.transform(centroids[["lon"]]).flatten()

    feats = np.zeros((N, T, F), dtype=np.float32)

    # Per-slot pivot
    slot_pivot = (risk.groupby(["Community Area","year","time_slot"])["cps_final"]
                      .mean()
                      .reset_index()
                      .pivot_table(index=["Community Area","year"],
                                   columns="time_slot",
                                   values="cps_final")
                      .fillna(0))
    slot_pivot.columns = [int(c) for c in slot_pivot.columns]

    for zi, zone in enumerate(zones):
        for ti, year in enumerate(years):
            row = annual[(annual["Community Area"] == zone) &
                         (annual["year"] == year)]
            if row.empty:
                continue
            r = row.iloc[0]

            tc     = np.log1p(r.get("total_crimes", 0)) / np.log1p(500_000)
            yoy = float(np.clip(float(0 if pd.isna(r.get("yoy_change_pct")) else r.get("yoy_change_pct")), -50, 50)) / 50
            lat_n  = float(centroids.loc[zone, "lat_norm"]) if zone in centroids.index else 0.0
            lon_n  = float(centroids.loc[zone, "lon_norm"]) if zone in centroids.index else 0.0

            # Slot CPS
            slot_cps = np.zeros(4, dtype=np.float32)
            if (zone, year) in slot_pivot.index:
                sv = slot_pivot.loc[(zone, year)]
                for s in range(4):
                    slot_cps[s] = sv.get(s, 0.0)

            feats[zi, ti, :] = [
                r.get("mean_cps", 0),
                tc,
                r.get("avg_severity", 0),
                r.get("avg_arrest_rate", 0),
                r.get("avg_domestic_rate", 0),
                yoy,
                lat_n,
                lon_n,
                slot_cps[0], slot_cps[1], slot_cps[2], slot_cps[3],
                r.get("max_cps", 0),
                r.get("mean_cps", 0),
            ]

    return feats, centroids


def run():
    print("=" * 60)
    print("UCMS  ▸  Step 05: GAT Feature Matrix")
    print("=" * 60)

    print("  Loading data …")
    annual = pd.read_parquet(ANNUAL_PATH)
    risk   = pd.read_parquet(RISK_PATH)

    zones = np.sort(annual["Community Area"].dropna().unique().astype(int))
    years = np.sort(annual["year"].unique().astype(int))
    print(f"  Zones: {len(zones)}  |  Years: {len(years)}")

    # ── Node features ─────────────────────────────────────────────────────────
    print("\n  Building node feature tensor (N, T, F) …")
    feats, centroids = build_node_features(annual, risk, zones, years)
    print(f"  Feature tensor shape: {feats.shape}")

    # ── Adjacency ─────────────────────────────────────────────────────────────
    print("\n  Building spatial adjacency (k-NN Gaussian) …")
    centroid_arr = centroids[["lat","lon"]].fillna(0).values
    adj, edge_index, edge_weight = build_adjacency(centroid_arr, K_NEIGHBORS, DIST_SIGMA)
    print(f"  Adj shape   : {adj.shape}")
    print(f"  Edges       : {edge_index.shape[1]:,}")
    print(f"  Avg deg     : {(adj > 0).sum(axis=1).mean():.1f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(os.path.join(OUT_DIR, "gat_node_features.npy"),  feats)
    np.save(os.path.join(OUT_DIR, "gat_adjacency.npy"),      adj)
    np.save(os.path.join(OUT_DIR, "gat_edge_index.npy"),     edge_index)
    np.save(os.path.join(OUT_DIR, "gat_edge_weight.npy"),    edge_weight)

    # Zone metadata (preserving Indian zone names & districts)
    meta_existing_path = os.path.join(OUT_DIR, "zone_metadata.csv")
    existing_meta = pd.read_csv(meta_existing_path) if os.path.exists(meta_existing_path) else pd.DataFrame()

    meta = pd.DataFrame({"zone_id": zones})
    meta["lat"] = centroids["lat"].reindex(zones).values
    meta["lon"] = centroids["lon"].reindex(zones).values
    if not existing_meta.empty and "zone_name" in existing_meta.columns:
        cols_to_merge = [c for c in ["zone_id", "zone_name", "district", "pop_density_ratio", "infra_quality", "base_crime_multiplier"] if c in existing_meta.columns and c != "Community Area"]
        meta = meta.merge(existing_meta[cols_to_merge], on="zone_id", how="left")
    meta.to_csv(os.path.join(OUT_DIR, "zone_metadata.csv"), index=False)

    # Feature name reference
    feat_names = [
        "mean_cps", "log_crime_count_norm", "avg_severity",
        "avg_arrest_rate", "avg_domestic_rate", "yoy_change_pct_norm",
        "lat_norm", "lon_norm",
        "cps_slot0_night", "cps_slot1_morning", "cps_slot2_afternoon",
        "cps_slot3_evening", "max_cps", "mean_cps_dup",
    ]
    with open(os.path.join(OUT_DIR, "feature_names.txt"), "w") as f:
        for i, n in enumerate(feat_names):
            f.write(f"{i:2d}  {n}\n")

    print("\n  Saved:")
    print(f"    gat_node_features.npy  {feats.shape}")
    print(f"    gat_adjacency.npy      {adj.shape}")
    print(f"    gat_edge_index.npy     {edge_index.shape}")
    print(f"    gat_edge_weight.npy    {edge_weight.shape}")
    print(f"    zone_metadata.csv      {len(zones)} zones")
    print(f"    feature_names.txt      {len(feat_names)} features")

    # ── Quick stats ───────────────────────────────────────────────────────────
    print("\n  Feature tensor stats (latest year):")
    for i, n in enumerate(feat_names):
        vals = feats[:, -1, i]
        print(f"    [{i:2d}] {n:<35} mean={vals.mean():.3f}  std={vals.std():.3f}")

    print("\n[OK] STEP 05 COMPLETE")
    print("    → You now have everything to initialise the GAT model.")
    print("    → Next: src/models/gat_model.py\n")


if __name__ == "__main__":
    run()