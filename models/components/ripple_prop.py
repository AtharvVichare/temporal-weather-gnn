from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import k_hop_subgraph


class NodeImportanceScorer(nn.Module):

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.scorer(h)).squeeze(-1)


class RippleLayer(nn.Module):

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
        h:          torch.Tensor,         
        edge_index: torch.Tensor,         
        edge_attr:  torch.Tensor | None = None,  
    ) -> tuple[torch.Tensor, torch.Tensor]:

      
        N = h.size(0)
        device = h.device

        importance = self.scorer(h)

        epicentres = (importance > self.threshold).nonzero(as_tuple=False).view(-1)

        if epicentres.numel() == 0:
            front_mask = torch.ones(N, dtype=torch.bool, device=device)
        else:
            subset, _, _, _ = k_hop_subgraph(
                node_idx   = epicentres,
                num_hops   = self.radius,
                edge_index = edge_index,
                num_nodes  = N,
                relabel_nodes = False,
            )
            front_mask = torch.zeros(N, dtype=torch.bool, device=device)
            front_mask[subset] = True

        front_nodes = front_mask.nonzero(as_tuple=False).view(-1)

        src, dst = edge_index
        edge_in_front = front_mask[src] & front_mask[dst]
        sub_edge_index = edge_index[:, edge_in_front]

        h_front = self.gat(h, sub_edge_index)      

        h_new = h.clone()
        h_new[front_mask] = self.norm(h_front[front_mask] + h[front_mask])
        h_new[front_mask] = self.norm2(h_new[front_mask] + self.ffn(h_new[front_mask]))

        return h_new, importance


class RipplePropagator(nn.Module):

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
        self.round_weights = nn.Parameter(torch.ones(n_rounds) / n_rounds)

    def forward(
        self,
        h:          torch.Tensor,               # [N, D]
        edge_index: torch.Tensor,               # [2, E]
        edge_attr:  torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
      
        importances = []
        round_outs  = []

        for layer in self.layers:
            h, imp = layer(h, edge_index, edge_attr)
            importances.append(imp)
            round_outs.append(h)
          
        w = F.softmax(self.round_weights, dim=0)
        h_out = sum(w[i] * round_outs[i] for i in range(len(round_outs)))

        return h_out, importances
