from __future__ import annotations
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from models.components.ripple_prop import RipplePropagator
from models.components.graph_fusion import GraphFusion, SingleBranchWrapper, GraphBranch
from models.components.temporal_encoding import make_temporal_encoder, CalendarEncoding
from utils.graph_utils import DynamicGraphBuilder


class TemporalEncoder(nn.Module):

    def __init__(
        self,
        in_dim:     int,
        hidden_dim: int,
        n_layers:   int   = 3,
        n_heads:    int   = 4,
        dropout:    float = 0.1,
        te_method:  str   = "sinusoidal",
        max_seq:    int   = 168,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)

        self.temporal_enc = make_temporal_encoder(te_method, hidden_dim, max_seq)
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = hidden_dim,
            nhead           = n_heads,
            dim_feedforward = hidden_dim * 2,
            dropout         = dropout,
            batch_first     = True,  
            norm_first      = True, 
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_norm    = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x_seq:  torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:          
        T, N, _ = x_seq.shape

        h = self.in_proj(x_seq)         

        te = self.temporal_enc(T, device) 
        h  = h + te.unsqueeze(1)          

        h = h.permute(1, 0, 2)          
        h = self.transformer(h)         
        h = h.mean(dim=1)              

        return self.out_norm(h)



class Decoder(nn.Module):
    
    def __init__(self, hidden_dim: int, out_dim: int, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        for i in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [N, H] → [N, V]"""
        return self.mlp(h)



class StateUpdater(nn.Module):


    def __init__(self, feat_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRUCell(feat_dim, hidden_dim)

    def forward(self, feat: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.gru(feat, h)


# ─────────────────────────────────────────────────────────────
#  Full Temporal Weather GNN
# ─────────────────────────────────────────────────────────────

class TemporalWeatherGNN(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        mc  = cfg["model"]
        gc  = cfg["graph"]

        self.variant       = mc["variant"]
        self.hidden_dim    = mc["hidden_dim"]
        self.n_vars        = mc["node_feat_dim"]
        self.forecast_len  = cfg["data"]["forecast_len"]
        self.history_len   = cfg["data"]["history_len"]

        in_dim = self.n_vars + 4
        self.encoder = TemporalEncoder(
            in_dim     = in_dim,
            hidden_dim = self.hidden_dim,
            n_layers   = mc["encoder_layers"],
            n_heads    = mc["n_heads"],
            dropout    = mc["encoder_dropout"],
            te_method  = mc["temporal_encoding"],
            max_seq    = mc["max_seq_len"],
        )

        if self.variant in ("dynamic", "fusion"):
            self.dyn_builder = DynamicGraphBuilder(
                node_dim   = self.hidden_dim,
                top_k      = gc["dynamic_top_k"],
                method     = gc["dynamic_method"],
                hidden_dim = 32,
            )
        else:
            self.dyn_builder = None

        if self.variant == "fusion":
            self.fusion = GraphFusion(
                in_dim     = self.hidden_dim,
                hidden_dim = self.hidden_dim,
                method     = mc["fusion_method"],
                n_heads    = mc["n_heads"],
                n_layers   = 2,
            )
        elif self.variant == "static":
            branch = GraphBranch(self.hidden_dim, self.hidden_dim, mc["n_heads"], n_layers=2)
            self.fusion = SingleBranchWrapper(branch, use_static=True)
        else:  
            branch = GraphBranch(self.hidden_dim, self.hidden_dim, mc["n_heads"], n_layers=2)
            self.fusion = SingleBranchWrapper(branch, use_static=False)

        self.processor = RipplePropagator(
            hidden_dim = self.hidden_dim,
            n_rounds   = mc["processor_layers"],
            n_heads    = mc["n_heads"],
            radius     = gc["ripple_radius"],
            threshold  = gc["ripple_threshold"],
        )

        self.decoder = Decoder(
            hidden_dim = self.hidden_dim,
            out_dim    = self.n_vars,
            n_layers   = mc["decoder_layers"],
        )
        self.state_updater = StateUpdater(self.n_vars, self.hidden_dim)

        self.feat_proj = nn.Linear(self.n_vars, self.hidden_dim)

    def _build_dynamic_edges(self, h: torch.Tensor, device: torch.device):
        if self.dyn_builder is None:
            return None, None
        ei, ew = self.dyn_builder(h)
        return ei.to(device), ew.to(device)

    def _dummy_dynamic_edges(self, N: int, device: torch.device):
        """Fallback: self-loop graph when dynamic builder not used."""
        idx = torch.arange(N, device=device)
        ei  = torch.stack([idx, idx], dim=0)
        ew  = torch.ones(N, device=device)
        return ei, ew

    def forward(
        self,
        graph_seq:   List[Data],          
        static_ei:   torch.Tensor,         
        static_ea:   torch.Tensor,          
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        device = static_ei.device
        T_in   = len(graph_seq)
        N      = graph_seq[0].num_nodes
        x_seq = torch.stack([g.x for g in graph_seq], dim=0).to(device) 

        h = self.encoder(x_seq, device)                               
        if self.dyn_builder is not None:
            dyn_ei, dyn_ew = self._build_dynamic_edges(h, device)
        else:
            dyn_ei, dyn_ew = self._dummy_dynamic_edges(N, device)
          
        h = self.fusion(h, static_ei, static_ea, dyn_ei, dyn_ew)         
        h, importances = self.processor(h, static_ei, static_ea)           

        preds = []
        h_ar  = h

        for step in range(self.forecast_len):
            pred = self.decoder(h_ar)                               
            preds.append(pred)

            if step < self.forecast_len - 1:
                if self.dyn_builder is not None:
                    dyn_ei_new, dyn_ew_new = self._build_dynamic_edges(h_ar, device)
                else:
                    dyn_ei_new, dyn_ew_new = dyn_ei, dyn_ew

                h_ar = self.state_updater(pred, h_ar)           
                h_ar = self.fusion(h_ar, static_ei, static_ea, dyn_ei_new, dyn_ew_new)
                h_ar, imp_ar = self.processor(h_ar, static_ei, static_ea)
                importances.extend(imp_ar)

        preds = torch.stack(preds, dim=0)                               
        return preds, importances

    def forward_batch(
        self,
        graph_seqs:  List[List[Data]],     
        static_ei:   torch.Tensor,
        static_ea:   torch.Tensor,
    ) -> Tuple[torch.Tensor, List]:

      
        all_preds = []
        all_imps  = []
        for seq in graph_seqs:
            p, imp = self.forward(seq, static_ei, static_ea)
            all_preds.append(p)
            all_imps.extend(imp)
        return torch.stack(all_preds, dim=0), all_imps
