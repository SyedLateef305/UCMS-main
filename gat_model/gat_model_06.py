"""
UCMS - Step 06: GAT Model Architecture (CPU-Optimized)
=======================================================
Purpose : Define the Graph Attention Network used in CCRP.
          Two GAT conv layers + temporal encoding + MLP head.

Architecture:
  Input  : Node features  (N=77 zones, F=14 features)
  Layer1 : GATConv(14 → 64, heads=4, concat=True)  → 256-dim
  Layer2 : GATConv(256 → 32, heads=2, concat=False) → 32-dim
  Temporal: Linear(32+4 → 32) — injects year-slot encoding
  Head   : MLP(32 → 16 → 1)  → CPS prediction ∈ [0,1]

CPU Optimisations:
  • No DataParallel / mixed precision
  • ELU activation (cheaper than GELU on CPU)
  • Dropout=0.3 (regularisation + faster convergence)
  • Batch over time-steps (not full graph batches)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Try to import PyTorch Geometric ──────────────────────────────────────────
try:
    from torch_geometric.nn import GATConv, BatchNorm
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("⚠  torch_geometric not found — falling back to manual GAT.")


# ─────────────────────────────────────────────────────────────────────────────
# Manual GAT Layer (fallback if PyG not installed)
# ─────────────────────────────────────────────────────────────────────────────
class ManualGATLayer(nn.Module):
    """
    Single-head Graph Attention Layer (Veličković et al., 2018).
    Works with a dense adjacency matrix — fine for N=77 nodes.
    """
    def __init__(self, in_features: int, out_features: int,
                 dropout: float = 0.3, alpha: float = 0.2):
        super().__init__()
        self.W      = nn.Linear(in_features, out_features, bias=False)
        self.a      = nn.Linear(2 * out_features, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leaky  = nn.LeakyReLU(alpha)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x   : (N, in_features)
        adj : (N, N)  — binary or weighted adjacency
        """
        N   = x.size(0)
        Wh  = self.W(x)                                  # (N, out)
        Wh1 = Wh.unsqueeze(1).expand(-1, N, -1)          # (N, N, out)
        Wh2 = Wh.unsqueeze(0).expand(N, -1, -1)          # (N, N, out)
        e   = self.leaky(self.a(torch.cat([Wh1, Wh2], dim=-1)).squeeze(-1))  # (N,N)
        mask = (adj == 0)
        e    = e.masked_fill(mask, float("-inf"))
        alpha = F.softmax(e, dim=-1)                     # (N, N)
        alpha = self.dropout(alpha)
        return F.elu(torch.matmul(alpha, Wh))            # (N, out)


class MultiHeadGATLayer(nn.Module):
    """Multi-head wrapper for ManualGATLayer."""
    def __init__(self, in_features: int, out_features: int,
                 n_heads: int = 4, concat: bool = True, dropout: float = 0.3):
        super().__init__()
        self.concat = concat
        self.heads  = nn.ModuleList([
            ManualGATLayer(in_features, out_features, dropout)
            for _ in range(n_heads)
        ])

    def forward(self, x, adj):
        outs = [h(x, adj) for h in self.heads]
        return torch.cat(outs, dim=-1) if self.concat else torch.mean(torch.stack(outs), dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# UCMS-CCRP GAT Model
# ─────────────────────────────────────────────────────────────────────────────
class UCMSGATModel(nn.Module):
    """
    Graph Attention Network for CCRP crime risk prediction.

    Input:
      x         : (N, F)      node feature matrix
      adj/edge  : adjacency info (dense adj OR PyG edge_index)
      t_encoding: (N, 4)      temporal encoding [sin_year, cos_year, sin_slot, cos_slot]

    Output:
      cps_pred  : (N, 1)      predicted CPS ∈ [0, 1]
      attn_weights: list of attention weights per layer (for explainability)
    """

    def __init__(self,
                 in_features: int  = 14,
                 hidden1    : int  = 64,
                 hidden2    : int  = 32,
                 n_heads1   : int  = 4,
                 n_heads2   : int  = 2,
                 dropout    : float = 0.3,
                 use_pyg    : bool  = False):
        super().__init__()

        self.use_pyg = use_pyg and PYG_AVAILABLE

        if self.use_pyg:
            self._build_pyg(in_features, hidden1, hidden2,
                            n_heads1, n_heads2, dropout)
        else:
            self._build_manual(in_features, hidden1, hidden2,
                               n_heads1, n_heads2, dropout)

        # Temporal injection: concat [gat_out(32) | t_enc(4)] → 32
        self.temporal_proj = nn.Sequential(
            nn.Linear(hidden2 + 4, hidden2),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # MLP prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden2, 16),
            nn.ELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(16, 1),
            nn.Sigmoid(),          # output ∈ (0, 1)
        )

        # Batch norms
        self.bn1 = nn.LayerNorm(hidden1 * n_heads1)
        self.bn2 = nn.LayerNorm(hidden2)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _build_manual(self, in_f, h1, h2, nh1, nh2, drop):
        self.gat1 = MultiHeadGATLayer(in_f,       h1, n_heads=nh1,
                                       concat=True,  dropout=drop)
        self.gat2 = MultiHeadGATLayer(h1 * nh1,   h2, n_heads=nh2,
                                       concat=False, dropout=drop)

    def _build_pyg(self, in_f, h1, h2, nh1, nh2, drop):
        self.gat1 = GATConv(in_f,    h1, heads=nh1, concat=True,
                            dropout=drop, add_self_loops=True)
        self.gat2 = GATConv(h1*nh1,  h2, heads=nh2, concat=False,
                            dropout=drop, add_self_loops=True)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, adj_or_edge, t_encoding=None):
        """
        x           : (N, F)
        adj_or_edge : (N,N) dense  OR  (2,E) edge_index for PyG
        t_encoding  : (N, 4) or None
        """
        if self.use_pyg:
            h = F.elu(self.gat1(x, adj_or_edge))
            h = self.bn1(h)
            h = self.dropout(h)
            h = F.elu(self.gat2(h, adj_or_edge))
            h = self.bn2(h)
        else:
            h = self.gat1(x, adj_or_edge)          # (N, h1*nh1)
            h = self.bn1(h)
            h = self.dropout(h)
            h = self.gat2(h, adj_or_edge)           # (N, h2)
            h = self.bn2(h)

        # Inject temporal encoding
        if t_encoding is not None:
            h = torch.cat([h, t_encoding], dim=-1)  # (N, h2+4)
            h = self.temporal_proj(h)               # (N, h2)

        return self.head(h)                         # (N, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Encoding Utility
# ─────────────────────────────────────────────────────────────────────────────
def make_temporal_encoding(year: int, time_slot: int,
                            year_min: int = 2010,
                            year_max: int = 2025,
                            n_nodes: int = 77) -> torch.Tensor:
    """
    Encode year + time_slot as 4-dim sinusoidal vector, broadcast to all nodes.
    Returns (N, 4)
    """
    yr_norm = (year - year_min) / max(year_max - year_min, 1)
    sl_norm = time_slot / 3.0

    enc = torch.tensor([
        torch.sin(torch.tensor(yr_norm * 3.14159)).item(),
        torch.cos(torch.tensor(yr_norm * 3.14159)).item(),
        torch.sin(torch.tensor(sl_norm * 3.14159)).item(),
        torch.cos(torch.tensor(sl_norm * 3.14159)).item(),
    ], dtype=torch.float32)

    return enc.unsqueeze(0).expand(n_nodes, -1)   # (N, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity-check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N, num_features = 77, 14
    model = UCMSGATModel(in_features=num_features, use_pyg=False)
    print(model)

    # Count params
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Trainable parameters : {params:,}")

    # Forward pass test
    x   = torch.randn(N, num_features)
    adj = torch.rand(N, N)
    adj = (adj + adj.T) / 2
    adj.fill_diagonal_(0)
    t   = make_temporal_encoding(2024, 2, n_nodes=N)

    out = model(x, adj, t)
    print(f"  Output shape         : {out.shape}")   # (77, 1)
    print(f"  Output range         : [{out.min().item():.3f}, {out.max().item():.3f}]")
    print("\n✅  Model architecture OK")