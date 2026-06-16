from __future__ import annotations
import math

import torch
import torch.nn as nn


class SinusoidalEncoding(nn.Module):
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


class LearnedEncoding(nn.Module):

    def __init__(self, dim: int, max_len: int = 1000, dropout: float = 0.0):
        super().__init__()
        self.emb     = nn.Embedding(max_len, dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.emb.weight, std=0.02)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(seq_len, device=device)
        return self.dropout(self.emb(idx))            



class RelativeTimeEncoding(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_len: int, device: torch.device, dt: float = 1.0) -> torch.Tensor:

        gaps = torch.arange(seq_len - 1, -1, -1, dtype=torch.float, device=device)
        gaps = (gaps * dt / 24.0).unsqueeze(-1)        
        return self.dropout(self.mlp(gaps))          


class CalendarEncoding(nn.Module):

    def __init__(self, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(4, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, cal_feats: torch.Tensor) -> torch.Tensor:
        return self.proj(cal_feats)


def make_temporal_encoder(method: str, dim: int, max_len: int = 1000) -> nn.Module:
    if method == "sinusoidal":
        return SinusoidalEncoding(dim, max_len)
    elif method == "learned":
        return LearnedEncoding(dim, max_len)
    elif method == "relative":
        return RelativeTimeEncoding(dim)
    else:
        raise ValueError(f"Unknown temporal encoding: {method}")
