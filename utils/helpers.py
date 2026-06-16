from __future__ import annotations

import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import numpy as np
import torch
import yaml


def load_config(path: str, overrides: dict | None = None) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if overrides:
        for k, v in overrides.items():
            keys = k.split(".")
            d = cfg
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            d[keys[-1]] = v

    return cfg


def save_config(cfg: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)



def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(name: str = "weather_gnn", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count (trainable) parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_param_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


@contextmanager
def Timer(label: str = "") -> Generator[None, None, None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        tag = f"[{label}] " if label else ""
        print(f"{tag}Elapsed: {elapsed:.3f}s")


def benchmark_inference(
    model:       torch.nn.Module,
    graph_seq:   list,
    static_ei:   torch.Tensor,
    static_ea:   torch.Tensor,
    n_runs:      int = 20,
    device:      str = "cpu",
) -> dict:

    model.eval()
    model.to(device)
    static_ei = static_ei.to(device)
    static_ea = static_ea.to(device)

    
    with torch.no_grad():
        for _ in range(3):
            model.forward(graph_seq, static_ei, static_ea)

    times = []
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            model.forward(graph_seq, static_ei, static_ea)
            if device != "cpu":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    peak_mb = 0.0
    if device != "cpu" and torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms":  float(np.std(times)),
        "peak_mb": peak_mb,
    }
