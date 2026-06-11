"""
evaluate.py
===========
Evaluation script for trained Temporal Weather GNN models.

Outputs
-------
  1. Console: pretty-printed metric table (all variables × lead times).
  2. results/metrics.json  – full results dict.
  3. results/metrics_table.txt  – ASCII table.
  4. results/plots/*.png   – RMSE vs. lead-time curves per variable.
  5. results/scalability.txt – inference time and memory report.

Usage
-----
  python evaluate.py --checkpoint checkpoints/best.ckpt
  python evaluate.py --checkpoint checkpoints/best.ckpt --config configs/default.yaml
  python evaluate.py --checkpoint none   # runs with random weights (sanity check)
"""

from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

try:
    import lightning as L
except ImportError:
    import pytorch_lightning as L

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from data.weather_dataset import WeatherDataModule
from models.temporal_gnn import TemporalWeatherGNN
from train import WeatherGNNModule
from utils.helpers import load_config, set_seed, get_logger, benchmark_inference, count_parameters, format_param_count
from utils.metrics import (
    compute_all_metrics,
    format_metrics_table,
    persistence_forecast,
    climatology_forecast,
)

logger = get_logger("evaluate")


# ─────────────────────────────────────────────────────────────
#  Evaluation loop
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def run_evaluation(
    model_module:   WeatherGNNModule,
    dm:             WeatherDataModule,
    cfg:            dict,
    device:         torch.device,
    out_dir:        str = "results",
) -> dict:
    """
    Run the model on the test set and collect predictions.

    Returns full metrics dict.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{out_dir}/plots").mkdir(exist_ok=True)

    model_module.eval()
    model_module.to(device)

    static_ei = dm.static_graph["edge_index"].to(device)
    static_ea = dm.static_graph["edge_attr"].to(device)

    model_module._static_ei = static_ei
    model_module._static_ea = static_ea

    test_loader = dm.test_dataloader()
    variables   = cfg["data"]["variables"]
    dt_hours    = cfg["data"]["dt_hours"]
    forecast_len = cfg["data"]["forecast_len"]

    all_preds      = []
    all_targets    = []
    all_persist    = []
    all_clim       = []
    inference_times = []

    logger.info("Running evaluation on test set...")
    for batch_idx, (graph_seqs, targets) in enumerate(test_loader):
        targets = targets.numpy()           # [B, T_out, N, V]
        B = len(graph_seqs)

        # Persistence baseline: last known frame repeated
        # We grab the last step of the input sequence
        persist_batch = np.stack([
            persistence_forecast(
                np.stack([g.x.numpy() for g in seq], axis=0)[:, :, :len(variables)],
                n_steps=forecast_len,
            )
            for seq in graph_seqs
        ])  # [B, T_out, N, V]

        # Climatology baseline (use target as proxy for rolling mean)
        clim_batch = np.stack([
            climatology_forecast(
                targets[b],
                n_steps=forecast_len,
            )
            for b in range(B)
        ])

        # Model forward
        t0 = time.perf_counter()
        preds, _ = model_module.model.forward_batch(
            [seq for seq in graph_seqs], static_ei, static_ea
        )
        inference_times.append((time.perf_counter() - t0) * 1000)

        preds_np = preds.cpu().numpy()      # [B, T_out, N, V]

        all_preds.append(preds_np)
        all_targets.append(targets)
        all_persist.append(persist_batch)
        all_clim.append(clim_batch)

        if batch_idx % 20 == 0:
            logger.info(f"  Batch {batch_idx}/{len(test_loader)}")

    all_preds   = np.concatenate(all_preds,   axis=0)   # [T_test, T_out, N, V]
    all_targets = np.concatenate(all_targets, axis=0)
    all_persist = np.concatenate(all_persist, axis=0)
    all_clim    = np.concatenate(all_clim,    axis=0)

    logger.info(f"Test set size: {all_preds.shape}")

    # ── Compute metrics ──
    lead_times = cfg["evaluation"].get("lead_times", list(range(1, forecast_len + 1)))
    lead_times = [lt for lt in lead_times if lt <= forecast_len]

    results = compute_all_metrics(
        preds       = all_preds,
        targets     = all_targets,
        persistence = all_persist,
        climatology = all_clim,
        variables   = variables,
        lead_times  = lead_times,
        dt_hours    = dt_hours,
    )

    # ── Scalability metrics ──
    mean_inf_ms = float(np.mean(inference_times))
    std_inf_ms  = float(np.std(inference_times))
    n_params    = count_parameters(model_module.model)
    scalability = {
        "n_parameters":     n_params,
        "n_parameters_fmt": format_param_count(n_params),
        "inference_mean_ms": mean_inf_ms,
        "inference_std_ms":  std_inf_ms,
        "n_test_samples":    int(all_preds.shape[0]),
        "n_nodes":           int(all_preds.shape[2]),
        "forecast_len":      int(all_preds.shape[1]),
    }
    results["_scalability"] = scalability

    # ── Print table ──
    table_str = format_metrics_table(
        {k: v for k, v in results.items() if not k.startswith("_")},
        variables
    )
    print("\n" + "=" * 70)
    print("  TEMPORAL WEATHER GNN  —  EVALUATION RESULTS")
    print("=" * 70)
    print(table_str)
    print()
    print(f"  Inference: {mean_inf_ms:.1f} ± {std_inf_ms:.1f} ms/batch")
    print(f"  Model parameters: {format_param_count(n_params)}")
    print("=" * 70 + "\n")

    # ── Save ──
    table_path = f"{out_dir}/metrics_table.txt"
    with open(table_path, "w") as f:
        f.write(table_str)
    logger.info(f"Metrics table saved → {table_path}")

    metrics_json = f"{out_dir}/metrics.json"
    with open(metrics_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Metrics JSON saved → {metrics_json}")

    scalability_path = f"{out_dir}/scalability.txt"
    with open(scalability_path, "w") as f:
        for k, v in scalability.items():
            f.write(f"{k}: {v}\n")
    logger.info(f"Scalability report → {scalability_path}")

    # ── Plots ──
    _plot_rmse_curves(results, variables, lead_times, dt_hours, out_dir)
    _plot_skill_heatmap(results, variables, lead_times, dt_hours, out_dir)
    _plot_sample_predictions(all_preds, all_targets, variables, dt_hours, out_dir)

    return results


# ─────────────────────────────────────────────────────────────
#  Plotting helpers
# ─────────────────────────────────────────────────────────────

def _plot_rmse_curves(results: dict, variables, lead_times, dt_hours, out_dir) -> None:
    """RMSE vs. lead time per variable."""
    fig, axes = plt.subplots(1, len(variables), figsize=(4 * len(variables), 4), squeeze=False)
    fig.suptitle("RMSE vs. Lead Time", fontsize=14, fontweight="bold")

    for vi, var in enumerate(variables):
        ax = axes[0][vi]
        rmse_model  = []
        rmse_persist = []
        rmse_clim   = []
        hours = []

        for lt in lead_times:
            key = f"{var}_lead{int(lt * dt_hours)}h"
            if key not in results:
                continue
            m = results[key]
            hours.append(int(lt * dt_hours))
            rmse_model.append(m["rmse"])
            # Reconstruct baseline RMSE from skill scores
            ss_p = m["skill_persistence"]
            ss_c = m["skill_climatology"]
            rmse_persist.append(m["rmse"] / max(1 - ss_p, 1e-6))
            rmse_clim.append(m["rmse"] / max(1 - ss_c, 1e-6))

        ax.plot(hours, rmse_model,   "b-o", label="TempGNN",     lw=2)
        ax.plot(hours, rmse_persist, "r--s", label="Persistence", lw=1.5)
        ax.plot(hours, rmse_clim,    "g--^", label="Climatology", lw=1.5)
        ax.set_title(var.replace("_", " ").title())
        ax.set_xlabel("Lead time (h)")
        ax.set_ylabel("RMSE (normalised)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{out_dir}/plots/rmse_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"RMSE curves saved → {path}")


def _plot_skill_heatmap(results: dict, variables, lead_times, dt_hours, out_dir) -> None:
    """Skill score heatmap: variables × lead times."""
    skill_mat = np.zeros((len(variables), len(lead_times)))

    for vi, var in enumerate(variables):
        for li, lt in enumerate(lead_times):
            key = f"{var}_lead{int(lt * dt_hours)}h"
            if key in results:
                skill_mat[vi, li] = results[key]["skill_persistence"]

    fig, ax = plt.subplots(figsize=(max(6, len(lead_times) * 1.2), max(3, len(variables) * 0.8)))
    sns.heatmap(
        skill_mat,
        xticklabels=[f"{int(lt * dt_hours)}h" for lt in lead_times],
        yticklabels=[v.replace("_", " ").title() for v in variables],
        annot=True, fmt=".2f", cmap="RdYlGn", center=0,
        vmin=-0.5, vmax=1.0, ax=ax,
    )
    ax.set_title("Skill Score vs. Persistence", fontsize=13, fontweight="bold")
    ax.set_xlabel("Lead Time")
    ax.set_ylabel("Variable")
    plt.tight_layout()
    path = f"{out_dir}/plots/skill_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Skill heatmap saved → {path}")


def _plot_sample_predictions(preds, targets, variables, dt_hours, out_dir) -> None:
    """Plot predicted vs true time series for a sample node and variable."""
    T_test, T_out, N, V = preds.shape
    sample_node = N // 2     # middle node

    fig, axes = plt.subplots(V, 1, figsize=(12, 3 * V), squeeze=False)
    fig.suptitle(f"Sample Predictions — Node {sample_node}", fontsize=13)

    for vi, var in enumerate(variables):
        ax = axes[vi][0]
        # Show first 50 test windows, lead-1 prediction
        n_show = min(50, T_test)
        t_ax   = np.arange(n_show)

        ax.plot(t_ax, targets[:n_show, 0, sample_node, vi], "k-",  label="Target",    lw=1.5)
        ax.plot(t_ax, preds  [:n_show, 0, sample_node, vi], "b--", label="Prediction", lw=1.5)
        ax.set_ylabel(var.replace("_", " ").title(), fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1][0].set_xlabel("Test sample index")
    plt.tight_layout()
    path = f"{out_dir}/plots/sample_predictions.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Sample predictions saved → {path}")


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Temporal Weather GNN")
    parser.add_argument("--config",     type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="none",
                        help="Path to .ckpt file, or 'none' to use random weights")
    parser.add_argument("--out_dir",    type=str, default="results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── DataModule ──
    dm = WeatherDataModule(cfg)
    dm.prepare_data()
    dm.setup("test")

    variable_names = cfg["data"]["variables"]
    n_vars         = len(variable_names)
    cfg["model"]["node_feat_dim"] = n_vars

    # ── Load model ──
    if args.checkpoint.lower() == "none":
        logger.warning("No checkpoint provided – using randomly-initialised weights.")
        model_module = WeatherGNNModule(cfg, n_vars, variable_names)
    else:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        model_module = WeatherGNNModule.load_from_checkpoint(
            args.checkpoint,
            cfg            = cfg,
            n_vars         = n_vars,
            variable_names = variable_names,
        )

    # ── Evaluate ──
    results = run_evaluation(model_module, dm, cfg, device, out_dir=args.out_dir)
    logger.info("Evaluation complete.")
