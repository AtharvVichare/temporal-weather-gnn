"""
utils/graph_utils.py
====================
Graph construction utilities:
  - build_static_graph  : geographic k-NN graph (built once)
  - DynamicGraphBuilder : learns or computes dynamic edges at runtime
  - haversine distance  : great-circle distance between lat/lon pairs
"""

from __future__ import annotations
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops


# ─────────────────────────────────────────────────────────────
#  Haversine distance
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
#  Static graph (built once from geography)
# ─────────────────────────────────────────────────────────────

def build_static_graph(
    coords:      np.ndarray,
    k:           int   = 8,
    max_dist_km: float = 500.0,
) -> dict:
    """
    Build a k-NN geographic graph from station coordinates.

    Parameters
    ----------
    coords      : [N, 2] (lat, lon)
    k           : number of nearest neighbours per node
    max_dist_km : maximum allowed edge distance

    Returns
    -------
    dict with:
      "edge_index" : LongTensor [2, E]
      "edge_attr"  : FloatTensor [E, 4]  (dist_km, Δlat, Δlon, bearing)
      "dist_matrix": np.ndarray [N, N]
    """
    N = coords.shape[0]
    dist = pairwise_haversine(coords)

    src_list, dst_list, attr_list = [], [], []

    for i in range(N):
        dists_i = dist[i].copy()
        dists_i[i] = np.inf                         # exclude self
        nn_idx = np.argsort(dists_i)[:k]            # k nearest
        for j in nn_idx:
            if dists_i[j] > max_dist_km:
                continue
            lat_i, lon_i = coords[i]
            lat_j, lon_j = coords[j]
            d      = dists_i[j]
            dlat   = lat_j - lat_i
            dlon   = lon_j - lon_i
            # Approximate bearing
            bearing = math.atan2(
                math.sin(math.radians(dlon)) * math.cos(math.radians(lat_j)),
                math.cos(math.radians(lat_i)) * math.sin(math.radians(lat_j))
                - math.sin(math.radians(lat_i)) * math.cos(math.radians(lat_j))
                * math.cos(math.radians(dlon)),
            )
            src_list.append(i)
            dst_list.append(j)
            attr_list.append([d / max_dist_km, dlat / 20.0, dlon / 20.0,
                               bearing / math.pi])  # normalised

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(attr_list, dtype=torch.float32)

    # Make undirected
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


# ─────────────────────────────────────────────────────────────
#  Dynamic Graph Builder (learnable adjacency)
# ─────────────────────────────────────────────────────────────

class DynamicGraphBuilder(nn.Module):
    """
    Learns a sparse dynamic adjacency from node embeddings at runtime.

    Two methods:
      "learned"         – MLP-based pairwise compatibility score.
      "correlation"     – cosine similarity of node features.
      "wind_direction"  – directional similarity from u/v wind features.

    Only the top-k edges (per node) are retained for efficiency.
    """

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
            # Pairwise MLP: maps (h_i || h_j) → scalar score
            self.scorer = nn.Sequential(
                nn.Linear(node_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(
        self,
        h: torch.Tensor,       # [N, D]  node embeddings
        coords: Optional[torch.Tensor] = None,  # [N, 2]
        wind_uv: Optional[torch.Tensor] = None, # [N, 2]  u,v wind
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        edge_index : [2, E]
        edge_weight: [E]   (soft weights in [0,1])
        """
        N, D = h.shape

        if self.method == "correlation":
            # Cosine similarity matrix
            h_norm = F.normalize(h, dim=-1)
            sim = h_norm @ h_norm.T                                  # [N, N]

        elif self.method == "wind_direction" and wind_uv is not None:
            # Dot product of wind vectors → directional alignment
            wn = F.normalize(wind_uv, dim=-1)
            sim = wn @ wn.T                                          # [N, N]

        else:  # learned
            # Build all-pairs: stack (h_i, h_j) for each pair
            hi = h.unsqueeze(1).expand(N, N, D)                     # [N, N, D]
            hj = h.unsqueeze(0).expand(N, N, D)                     # [N, N, D]
            pairs = torch.cat([hi, hj], dim=-1).reshape(N * N, 2 * D)
            scores = self.scorer(pairs).reshape(N, N)                # [N, N]
            sim = torch.sigmoid(scores)

        # Mask self-loops
        sim.fill_diagonal_(-1e9)

        # Top-k per row
        topk_vals, topk_idx = torch.topk(sim, k=min(self.top_k, N - 1), dim=-1)  # [N, k]
        topk_vals = torch.sigmoid(topk_vals)                         # re-scale to [0,1]

        src = torch.arange(N, device=h.device).unsqueeze(1).expand(N, self.top_k).reshape(-1)
        dst = topk_idx.reshape(-1)
        edge_weight = topk_vals.reshape(-1)

        edge_index = torch.stack([src, dst], dim=0)                  # [2, N*k]
        return edge_index, edge_weight
