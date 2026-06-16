from __future__ import annotations
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two lat/lon points."""
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def pairwise_haversine(coords: np.ndarray) -> np.ndarray:
    """
    coords : [N, 2]  (lat, lon in degrees)
    Returns: [N, N]  distance matrix in km
    """
    N = coords.shape[0]
    dist = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            d = haversine_km(coords[i, 0], coords[i, 1],
                             coords[j, 0], coords[j, 1])
            dist[i, j] = dist[j, i] = d
    return dist


def build_static_graph(
    coords:      np.ndarray,
    k:           int   = 8,
    max_dist_km: float = 500.0,
) -> dict:

  
    N = coords.shape[0]
    dist = pairwise_haversine(coords)

    src_list, dst_list, attr_list = [], [], []

    for i in range(N):
        dists_i = dist[i].copy()
        dists_i[i] = np.inf                         
        nn_idx = np.argsort(dists_i)[:k]            
        for j in nn_idx:
            if dists_i[j] > max_dist_km:
                continue
            lat_i, lon_i = coords[i]
            lat_j, lon_j = coords[j]
            d      = dists_i[j]
            dlat   = lat_j - lat_i
            dlon   = lon_j - lon_i
            bearing = math.atan2(
                math.sin(math.radians(dlon)) * math.cos(math.radians(lat_j)),
                math.cos(math.radians(lat_i)) * math.sin(math.radians(lat_j))
                - math.sin(math.radians(lat_i)) * math.cos(math.radians(lat_j))
                * math.cos(math.radians(dlon)),
            )
            src_list.append(i)
            dst_list.append(j)
            attr_list.append([d / max_dist_km, dlat / 20.0, dlon / 20.0,
                               bearing / math.pi])  

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(attr_list, dtype=torch.float32)

    
    edge_index, edge_attr = to_undirected(edge_index, edge_attr, num_nodes=N, reduce="mean")

    return {
        "edge_index":  edge_index,
        "edge_attr":   edge_attr,
        "dist_matrix": dist,
    }


def save_graph(graph: dict, path: str) -> None:
    torch.save(graph, path)


def load_graph(path: str) -> dict:
    return torch.load(path, weights_only=False)


class DynamicGraphBuilder(nn.Module):

    def __init__(
        self,
        node_dim:   int,
        top_k:      int = 6,
        method:     str = "learned",
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.top_k  = top_k
        self.method = method

        if method == "learned":
            self.scorer = nn.Sequential(
                nn.Linear(node_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(
        self,
        h: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        wind_uv: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

      
        N, D = h.shape

        if self.method == "correlation":
            h_norm = F.normalize(h, dim=-1)
            sim = h_norm @ h_norm.T                                

        elif self.method == "wind_direction" and wind_uv is not None:
            wn = F.normalize(wind_uv, dim=-1)
            sim = wn @ wn.T                                  

        else: 
            hi = h.unsqueeze(1).expand(N, N, D)                   
            hj = h.unsqueeze(0).expand(N, N, D)                    
            pairs = torch.cat([hi, hj], dim=-1).reshape(N * N, 2 * D)
            scores = self.scorer(pairs).reshape(N, N)              
            sim = torch.sigmoid(scores)

        sim.fill_diagonal_(-1e9)
        topk_vals, topk_idx = torch.topk(sim, k=min(self.top_k, N - 1), dim=-1)  
        topk_vals = torch.sigmoid(topk_vals)                        

        src = torch.arange(N, device=h.device).unsqueeze(1).expand(N, self.top_k).reshape(-1)
        dst = topk_idx.reshape(-1)
        edge_weight = topk_vals.reshape(-1)

        edge_index = torch.stack([src, dst], dim=0)                  
        return edge_index, edge_weight
