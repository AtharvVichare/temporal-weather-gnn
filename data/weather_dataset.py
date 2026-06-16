from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data

try:
    import lightning as L
except ImportError:
    import pytorch_lightning as L   

class Normalizer:

    def __init__(self, method: str = "z-score"):
        self.method = method
        self.mean: Optional[np.ndarray] = None
        self.std:  Optional[np.ndarray] = None
        self.min:  Optional[np.ndarray] = None
        self.max:  Optional[np.ndarray] = None

    def fit(self, features: np.ndarray) -> "Normalizer":
        """features: [T, N, V]"""
        flat = features.reshape(-1, features.shape[-1]) 
        if self.method == "z-score":
            self.mean = flat.mean(axis=0)
            self.std  = flat.std(axis=0) + 1e-8
        else:
            self.min = flat.min(axis=0)
            self.max = flat.max(axis=0) + 1e-8
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.method == "z-score":
            return (features - self.mean) / self.std
        return (features - self.min) / (self.max - self.min)

    def inverse_transform(self, features: np.ndarray) -> np.ndarray:
        if self.method == "z-score":
            return features * self.std + self.mean
        return features * (self.max - self.min) + self.min

    def save(self, path: str) -> None:
        np.savez(path, method=np.array([self.method]),
                 mean=self.mean, std=self.std, min=self.min, max=self.max)

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        d = np.load(path, allow_pickle=True)
        obj = cls(str(d["method"][0]))
        obj.mean = d.get("mean")
        obj.std  = d.get("std")
        obj.min  = d.get("min")
        obj.max  = d.get("max")
        return obj



class WeatherGraphDataset(Dataset):

    def __init__(
        self,
        features:     np.ndarray,
        coords:       np.ndarray,
        times:        pd.DatetimeIndex,
        static_graph: dict,
        history_len:  int = 12,
        forecast_len: int = 6,
        normalizer:   Optional[Normalizer] = None,
    ):
        self.features     = features.astype(np.float32)
        self.coords       = torch.tensor(coords, dtype=torch.float32)
        self.times        = times
        self.static_ei    = static_graph["edge_index"]   # [2, E]
        self.static_ea    = static_graph["edge_attr"]    # [E, Fe]
        self.history_len  = history_len
        self.forecast_len = forecast_len
        self.normalizer   = normalizer
        self.n_nodes      = features.shape[1]
        self.n_vars       = features.shape[2]

        if normalizer is not None:
            self.features = normalizer.transform(features).astype(np.float32)

        total_len = history_len + forecast_len
        self.indices = list(range(len(features) - total_len + 1))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[List[Data], torch.Tensor]:
        start = self.indices[idx]
        x_seq = self.features[start : start + self.history_len]        
        y_seq = self.features[start + self.history_len :
                              start + self.history_len + self.forecast_len] 

        graphs = []
        for t in range(self.history_len):
            node_feat = torch.tensor(x_seq[t], dtype=torch.float32)   
            time_obj  = self.times[start + t]
            hour      = time_obj.hour
            doy       = time_obj.dayofyear

            # Temporal scalars appended to node features
            t_enc = torch.tensor([
                np.sin(2 * np.pi * hour / 24),
                np.cos(2 * np.pi * hour / 24),
                np.sin(2 * np.pi * doy  / 365),
                np.cos(2 * np.pi * doy  / 365),
            ], dtype=torch.float32).unsqueeze(0).expand(self.n_nodes, -1)

            x = torch.cat([node_feat, t_enc], dim=-1)                  

            g = Data(
                x          = x,
                edge_index = self.static_ei,
                edge_attr  = self.static_ea,
                pos        = self.coords,
                num_nodes  = self.n_nodes,
                t_step     = torch.tensor(t, dtype=torch.long),
            )
            graphs.append(g)

        target = torch.tensor(y_seq, dtype=torch.float32)             
        return graphs, target


def collate_fn(batch):
    graph_seqs = [item[0] for item in batch]
    targets    = torch.stack([item[1] for item in batch], dim=0)
    return graph_seqs, targets
  

class WeatherDataModule(L.LightningDataModule):

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.normalizer: Optional[Normalizer] = None

    def prepare_data(self) -> None:
        """Download/generate data if not already present."""
        cache = Path(self.cfg["data"]["cache_dir"])
        if not (cache / "features.npy").exists():
            from data.prepare_data import generate_synthetic_data
            generate_synthetic_data(
                n_stations = self.cfg["data"]["n_stations"],
                n_timesteps= 8760,
                dt_hours   = self.cfg["data"]["dt_hours"],
                variables  = self.cfg["data"]["variables"],
                out_dir    = str(cache),
            )

    def setup(self, stage: Optional[str] = None) -> None:
        cache = Path(self.cfg["data"]["cache_dir"])

        features  = np.load(cache / "features.npy")     
        coords    = np.load(cache / "coords.npy")        
        times_raw = pd.read_csv(cache / "times.csv", header=0).iloc[:, 0]
        times     = pd.DatetimeIndex(pd.to_datetime(times_raw))

        T = len(features)
        train_end = int(T * self.cfg["data"]["train_frac"])
        val_end   = train_end + int(T * self.cfg["data"]["val_frac"])

        train_feat = features[:train_end]
        val_feat   = features[train_end:val_end]
        test_feat  = features[val_end:]
        train_times = times[:train_end]
        val_times   = times[train_end:val_end]
        test_times  = times[val_end:]

        if self.cfg["data"]["normalize"]:
            self.normalizer = Normalizer(self.cfg["data"]["norm_method"])
            self.normalizer.fit(train_feat)
            norm_path = str(cache / "normalizer.npz")
            self.normalizer.save(norm_path)
        else:
            self.normalizer = None

        from utils.graph_utils import build_static_graph
        static_graph = build_static_graph(
            coords      = coords,
            k           = self.cfg["graph"]["static_k"],
            max_dist_km = self.cfg["graph"]["static_max_dist_km"],
        )

        kw = dict(
            coords       = coords,
            static_graph = static_graph,
            history_len  = self.cfg["data"]["history_len"],
            forecast_len = self.cfg["data"]["forecast_len"],
            normalizer   = self.normalizer,
        )
        self.train_ds = WeatherGraphDataset(train_feat, times=train_times, **kw)
        self.val_ds   = WeatherGraphDataset(val_feat,   times=val_times,   **kw)
        self.test_ds  = WeatherGraphDataset(test_feat,  times=test_times,  **kw)
        self.coords   = coords
        self.static_graph = static_graph

        print(f"[DataModule] Train={len(self.train_ds)} Val={len(self.val_ds)} Test={len(self.test_ds)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size  = self.cfg["data"]["batch_size"],
            shuffle     = True,
            num_workers = self.cfg["data"]["num_workers"],
            pin_memory  = self.cfg["data"]["pin_memory"],
            collate_fn  = collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size  = self.cfg["data"]["batch_size"],
            shuffle     = False,
            num_workers = self.cfg["data"]["num_workers"],
            pin_memory  = self.cfg["data"]["pin_memory"],
            collate_fn  = collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size  = 1,
            shuffle     = False,
            num_workers = self.cfg["data"]["num_workers"],
            pin_memory  = False,
            collate_fn  = collate_fn,
        )
