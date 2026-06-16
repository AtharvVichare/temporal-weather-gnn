from __future__ import annotations
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


class PersistenceModel:

    def predict(
        self,
        x_seq: np.ndarray,    
        n_steps: int,
    ) -> np.ndarray:           
        last = x_seq[-1:]
        return np.repeat(last, n_steps, axis=0)


class ClimatologyModel:
    def __init__(self, window: int = 720):
        self.window = window

    def predict(
        self,
        x_seq: np.ndarray,    
        n_steps: int,
    ) -> np.ndarray:          
        clim = x_seq[-self.window:].mean(axis=0, keepdims=True)
        return np.repeat(clim, n_steps, axis=0)


class StaticGNNLayer(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.gat  = GATConv(in_dim, out_dim // n_heads, heads=n_heads, dropout=dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.gat(x, edge_index)
        return self.norm(h + (x if x.shape == h.shape else 0))


class StaticGNN(nn.Module):

    def __init__(
        self,
        in_dim:      int,
        hidden_dim:  int,
        out_dim:     int,
        n_layers:    int  = 4,
        n_heads:     int  = 4,
        forecast_len: int = 6,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.forecast_len = forecast_len
      
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.gnn_layers = nn.ModuleList([
            StaticGNNLayer(hidden_dim, hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim * forecast_len),
        )
        self.out_dim = out_dim

    def forward(
        self,
        graph_seq:  List[Data],        
        static_ei:  torch.Tensor,   
        static_ea:  torch.Tensor,
    ) -> Tuple[torch.Tensor, list]:

      
        device = static_ei.device
        T_in = len(graph_seq)
        N = graph_seq[0].num_nodes
      
        x_seq = torch.stack([g.x for g in graph_seq], dim=0).to(device) 
        x = x_seq.mean(dim=0)                                       

        h = self.encoder(x)
        for layer in self.gnn_layers:
            h = layer(h, static_ei)

        out = self.decoder(h)             
        out = out.view(N, self.forecast_len, self.out_dim) 
        preds = out.permute(1, 0, 2)       
        return preds, []

    def forward_batch(
        self,
        graph_seqs:  List[List[Data]],
        static_ei:   torch.Tensor,
        static_ea:   torch.Tensor,
    ) -> Tuple[torch.Tensor, list]:
        all_preds = []
        for seq in graph_seqs:
            p, _ = self.forward(seq, static_ei, static_ea)
            all_preds.append(p)
        return torch.stack(all_preds, dim=0), []
