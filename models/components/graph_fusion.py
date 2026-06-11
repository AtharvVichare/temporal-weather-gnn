"""
models/components/graph_fusion.py
==================================
Fuses information from two graph streams:
  - Static  stream: fixed geographic k-NN graph
  - Dynamic stream: learned or computed graph (updated per step)

Three fusion modes
------------------
  "gated"     – element-wise gated combination:
                  h_out = g ⊙ h_static + (1-g) ⊙ h_dynamic   where g = σ(W h_cat)
  "attention" – multi-head cross-attention between the two representations.
  "residual"  – simple addition with a learned scalar per dimension.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


# ─────────────────────────────────────────────────────────────
#  Single-graph GNN branch
# ─────────────────────────────────────────────────────────────

class GraphBranch(nn.Module):
    """
    A lightweight GNN branch that runs message-passing on a given graph
    and returns updated node embeddings.
    """

    def __init__(
        self,
        in_dim:    int,
        out_dim:   int,
        n_heads:   int  = 4,
        n_layers:  int  = 2,
        dropout:   float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms  = nn.ModuleList()

        cur = in_dim
        for i in range(n_layers):
            head_out = out_dim // n_heads
            self.layers.append(
                GATConv(cur, head_out, heads=n_heads, dropout=dropout, concat=True)
            )
            self.norms.append(nn.LayerNorm(out_dim))
            cur = out_dim

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x:          torch.Tensor,    # [N, in_dim]
        edge_index: torch.Tensor,    # [2, E]
        edge_attr:  torch.Tensor | None = None,
    ) -> torch.Tensor:               # [N, out_dim]
        h = x
        for layer, norm in zip(self.layers, self.norms):
            h2 = layer(h, edge_index)
            h  = norm(h2 + (h if h.shape == h2.shape else 0))
            h  = self.dropout(F.gelu(h))
        return h


# ─────────────────────────────────────────────────────────────
#  Fusion module
# ─────────────────────────────────────────────────────────────

class GraphFusion(nn.Module):
    """
    Runs two GNN branches (static + dynamic) and fuses their outputs.

    Parameters
    ----------
    in_dim      : input node feature dimension
    hidden_dim  : branch hidden dimension
    method      : "gated" | "attention" | "residual"
    n_heads     : attention heads (used by GAT inside branches)
    n_layers    : GNN layers per branch
    dropout     : dropout rate
    """

    def __init__(
        self,
        in_dim:     int,
        hidden_dim: int,
        method:     str   = "gated",
        n_heads:    int   = 4,
        n_layers:   int   = 2,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.method = method

        self.static_branch  = GraphBranch(in_dim, hidden_dim, n_heads, n_layers, dropout)
        self.dynamic_branch = GraphBranch(in_dim, hidden_dim, n_heads, n_layers, dropout)

        if method == "gated":
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )

        elif method == "attention":
            self.cross_attn = nn.MultiheadAttention(
                embed_dim   = hidden_dim,
                num_heads   = n_heads,
                dropout     = dropout,
                batch_first = True,
            )
            self.out_norm = nn.LayerNorm(hidden_dim)

        elif method == "residual":
            # Learnable scalar mixing coefficients
            self.alpha = nn.Parameter(torch.tensor(0.5))

        else:
            raise ValueError(f"Unknown fusion method: {method}")

        # Final projection
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        x:                torch.Tensor,         # [N, in_dim]
        static_ei:        torch.Tensor,         # [2, E_s]
        static_ea:        torch.Tensor | None,  # [E_s, Fe_s]
        dynamic_ei:       torch.Tensor,         # [2, E_d]
        dynamic_weights:  torch.Tensor | None = None,  # [E_d]
    ) -> torch.Tensor:                          # [N, hidden_dim]
        """
        Forward pass through both branches then fusion.
        """
        h_s = self.static_branch(x, static_ei, static_ea)       # [N, H]
        h_d = self.dynamic_branch(x, dynamic_ei)                 # [N, H]

        if self.method == "gated":
            g = self.gate(torch.cat([h_s, h_d], dim=-1))        # [N, H]
            h_fused = g * h_s + (1.0 - g) * h_d

        elif self.method == "attention":
            # Use h_s as query, h_d as key/value
            # Treat N as sequence length, batch=1 (per graph)
            q = h_s.unsqueeze(0)    # [1, N, H]
            k = h_d.unsqueeze(0)    # [1, N, H]
            v = h_d.unsqueeze(0)    # [1, N, H]
            attn_out, _ = self.cross_attn(q, k, v)
            h_fused = self.out_norm(attn_out.squeeze(0) + h_s)  # [N, H]

        else:  # residual
            α = torch.sigmoid(self.alpha)
            h_fused = α * h_s + (1.0 - α) * h_d

        return self.proj(h_fused)                                 # [N, H]


# ─────────────────────────────────────────────────────────────
#  Ablation wrapper: static-only or dynamic-only branch
# ─────────────────────────────────────────────────────────────

class SingleBranchWrapper(nn.Module):
    """
    Mimics GraphFusion API but runs only one branch.
    Used for ablation: model.variant = "static" or "dynamic".
    """

    def __init__(self, branch: GraphBranch, use_static: bool):
        super().__init__()
        self.branch     = branch
        self.use_static = use_static

    def forward(
        self,
        x:               torch.Tensor,
        static_ei:       torch.Tensor,
        static_ea:       torch.Tensor | None,
        dynamic_ei:      torch.Tensor,
        dynamic_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_static:
            return self.branch(x, static_ei, static_ea)
        else:
            return self.branch(x, dynamic_ei)
