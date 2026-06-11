"""
models/components/temporal_encoding.py
=======================================
Positional / temporal encodings for the Temporal Weather GNN.

Three strategies
----------------
  "sinusoidal" – fixed sine/cosine encoding over time step index.
                 No learnable parameters; instantly generalises to any sequence length.
  "learned"    – nn.Embedding table indexed by time step.
                 Fast but limited to sequences seen during training.
  "relative"   – relative time difference encoding between pairs of steps.
                 Useful for irregular time series.

All encoders produce vectors of shape [T, D] or [N, D] depending on usage.
"""

from __future__ import annotations
import math

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────
#  Sinusoidal (fixed)
# ─────────────────────────────────────────────────────────────

class SinusoidalEncoding(nn.Module):
    """
    Classic Transformer sinusoidal positional encoding.

    PE(pos, 2i)   = sin(pos / 10000^(2i/D))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/D))
    """

    def __init__(self, dim: int, max_len: int = 1000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)            # [max_len, D]

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Returns [seq_len, D] encoding."""
        return self.dropout(self.pe[:seq_len].to(device))


# ─────────────────────────────────────────────────────────────
#  Learned
# ─────────────────────────────────────────────────────────────

class LearnedEncoding(nn.Module):
    """Learned embedding per time-step index."""

    def __init__(self, dim: int, max_len: int = 1000, dropout: float = 0.0):
        super().__init__()
        self.emb     = nn.Embedding(max_len, dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.emb.weight, std=0.02)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(seq_len, device=device)
        return self.dropout(self.emb(idx))              # [seq_len, D]


# ─────────────────────────────────────────────────────────────
#  Relative time encoding
# ─────────────────────────────────────────────────────────────

class RelativeTimeEncoding(nn.Module):
    """
    Encodes the relative time gap between each time step and the current
    (most recent) step, using a small MLP.

    Useful when the time grid is irregular (e.g., missing observations).
    """

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        # Input: scalar relative time gap (can be normalised hours)
        self.mlp = nn.Sequential(
            nn.Linear(1, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_len: int, device: torch.device, dt: float = 1.0) -> torch.Tensor:
        """
        seq_len : number of time steps
        dt      : time step size in hours (for normalisation)
        Returns [seq_len, D]
        """
        # Gaps relative to the last time step (negative, oldest first)
        gaps = torch.arange(seq_len - 1, -1, -1, dtype=torch.float, device=device)
        gaps = (gaps * dt / 24.0).unsqueeze(-1)        # [T, 1]  normalise by day
        return self.dropout(self.mlp(gaps))             # [T, D]


# ─────────────────────────────────────────────────────────────
#  Calendar feature encoding (appended to node features)
# ─────────────────────────────────────────────────────────────

class CalendarEncoding(nn.Module):
    """
    Encodes hour-of-day and day-of-year as sine/cosine pairs,
    optionally projecting to a higher-dimensional space.

    This is also computed in the Dataset (as raw scalars), but
    this module can re-project them in a learnable way.
    """

    def __init__(self, out_dim: int):
        super().__init__()
        # 4 input features: sin_h, cos_h, sin_doy, cos_doy
        self.proj = nn.Sequential(
            nn.Linear(4, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, cal_feats: torch.Tensor) -> torch.Tensor:
        """
        cal_feats : [*, 4]  (sin_h, cos_h, sin_doy, cos_doy)
        Returns   : [*, out_dim]
        """
        return self.proj(cal_feats)


# ─────────────────────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────────────────────

def make_temporal_encoder(method: str, dim: int, max_len: int = 1000) -> nn.Module:
    if method == "sinusoidal":
        return SinusoidalEncoding(dim, max_len)
    elif method == "learned":
        return LearnedEncoding(dim, max_len)
    elif method == "relative":
        return RelativeTimeEncoding(dim)
    else:
        raise ValueError(f"Unknown temporal encoding: {method}")
