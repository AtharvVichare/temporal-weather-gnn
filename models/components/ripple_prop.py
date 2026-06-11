"""
models/components/ripple_prop.py
=================================
Ripple Propagation Message Passing
-----------------------------------
Inspiration: T-RippleGNN (spatio-temporal traffic forecasting) adapted for
weather graphs.

Core idea
---------
Instead of running full-graph message passing at every layer, we identify
"active" nodes (those with high state change or prediction uncertainty) and
propagate information outward in localised wavefronts (ripples).

Algorithm per ripple round r:
  1. Score each node → importance scalar s_i ∈ [0,1].
  2. Select "epicentre" nodes where s_i > threshold.
  3. Expand epicentres hop-by-hop (BFS on the edge list) up to `radius` hops.
     Nodes in the expanded set form the "active front".
  4. Run one round of graph attention ONLY on the active-front subgraph.
  5. Update node embeddings; inactive nodes carry forward their previous state.

This reduces the effective message-passing cost on large graphs from O(N)
to O(|front|) per ripple step.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import k_hop_subgraph


class NodeImportanceScorer(nn.Module):
    """
    Learns a scalar importance score per node from its hidden state.
    score_i = σ( w^T h_i + b )
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [N, D] → scores: [N]"""
        return torch.sigmoid(self.scorer(h)).squeeze(-1)


class RippleLayer(nn.Module):
    """
    One round of ripple-propagation:
      score → select epicentres → expand front → GAT on front → merge.
    """

    def __init__(
        self,
        hidden_dim:       int,
        n_heads:          int  = 4,
        radius:           int  = 3,
        threshold:        float = 0.1,
        dropout:          float = 0.1,
    ):
        super().__init__()
        self.radius    = radius
        self.threshold = threshold

        self.scorer    = NodeImportanceScorer(hidden_dim)

        # GAT for the active-front sub-graph
        self.gat = GATConv(
            in_channels  = hidden_dim,
            out_channels = hidden_dim // n_heads,
            heads        = n_heads,
            dropout      = dropout,
            concat       = True,
            add_self_loops = True,
        )
        self.norm  = nn.LayerNorm(hidden_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h:          torch.Tensor,         # [N, D]
        edge_index: torch.Tensor,         # [2, E]
        edge_attr:  torch.Tensor | None = None,  # [E, Fe] (unused by GAT here)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        h_new        : [N, D]  updated node embeddings
        importance   : [N]     node importance scores (for logging)
        """
        N = h.size(0)
        device = h.device

        # ── 1. Score every node ──
        importance = self.scorer(h)               # [N]

        # ── 2. Find epicentre nodes ──
        epicentres = (importance > self.threshold).nonzero(as_tuple=False).view(-1)

        if epicentres.numel() == 0:
            # No nodes above threshold → propagate everywhere (fall-back)
            front_mask = torch.ones(N, dtype=torch.bool, device=device)
        else:
            # ── 3. Expand epicentres by `radius` hops ──
            # k_hop_subgraph returns (subset_nodes, sub_edge_index, mapping, edge_mask)
            subset, _, _, _ = k_hop_subgraph(
                node_idx   = epicentres,
                num_hops   = self.radius,
                edge_index = edge_index,
                num_nodes  = N,
                relabel_nodes = False,
            )
            front_mask = torch.zeros(N, dtype=torch.bool, device=device)
            front_mask[subset] = True

        # ── 4. Build sub-graph for the active front ──
        front_nodes = front_mask.nonzero(as_tuple=False).view(-1)

        # Filter edges: both endpoints in the front
        src, dst = edge_index
        edge_in_front = front_mask[src] & front_mask[dst]
        sub_edge_index = edge_index[:, edge_in_front]

        # ── 5. Run GAT only on the sub-graph ──
        h_front = self.gat(h, sub_edge_index)          # [N, D]  (GAT still returns N rows)

        # Residual + LayerNorm only for active nodes
        h_new = h.clone()
        h_new[front_mask] = self.norm(h_front[front_mask] + h[front_mask])
        h_new[front_mask] = self.norm2(h_new[front_mask] + self.ffn(h_new[front_mask]))

        return h_new, importance


class RipplePropagator(nn.Module):
    """
    Stacks `n_rounds` RippleLayer passes.

    Optionally accumulates skip connections across all rounds
    for better gradient flow.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_rounds:   int   = 4,
        n_heads:    int   = 4,
        radius:     int   = 3,
        threshold:  float = 0.1,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            RippleLayer(hidden_dim, n_heads, radius, threshold, dropout)
            for _ in range(n_rounds)
        ])
        # Learnable weight for accumulating rounds
        self.round_weights = nn.Parameter(torch.ones(n_rounds) / n_rounds)

    def forward(
        self,
        h:          torch.Tensor,               # [N, D]
        edge_index: torch.Tensor,               # [2, E]
        edge_attr:  torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Returns
        -------
        h_out       : [N, D]
        importances : list of [N] tensors, one per round
        """
        importances = []
        round_outs  = []

        for layer in self.layers:
            h, imp = layer(h, edge_index, edge_attr)
            importances.append(imp)
            round_outs.append(h)

        # Weighted sum of all rounds (soft ensemble)
        w = F.softmax(self.round_weights, dim=0)
        h_out = sum(w[i] * round_outs[i] for i in range(len(round_outs)))

        return h_out, importances
