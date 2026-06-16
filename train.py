from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from data.weather_dataset import WeatherDataModule
from models.temporal_gnn import TemporalWeatherGNN
from utils.helpers import load_config, set_seed, get_logger, count_parameters, format_param_count
from utils.metrics import rmse as metric_rmse, mae as metric_mae


logger = get_logger("train")


class WeatherGNNModule(L.LightningModule):

    def __init__(self, cfg: dict, n_vars: int, variable_names: List[str]):
        super().__init__()
        self.save_hyperparameters()
        self.cfg            = cfg
        self.variable_names = variable_names
        self.n_vars         = n_vars

        cfg["model"]["node_feat_dim"] = n_vars
        self.model = TemporalWeatherGNN(cfg)

        loss_fn = cfg["training"]["loss"]
        if loss_fn == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_fn == "mae":
            self.loss_fn = nn.L1Loss()
        elif loss_fn == "huber":
            self.loss_fn = nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss: {loss_fn}")

        self._static_ei: Optional[torch.Tensor] = None
        self._static_ea: Optional[torch.Tensor] = None

    def _get_static_graph(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._static_ei is None:
            dm = self.trainer.datamodule
            self._static_ei = dm.static_graph["edge_index"].to(self.device)
            self._static_ea = dm.static_graph["edge_attr"].to(self.device)
        return self._static_ei, self._static_ea


    def _step(self, batch, stage: str) -> torch.Tensor:
        graph_seqs, targets = batch     
        targets = targets.to(self.device)
        B = targets.size(0)

        static_ei, static_ea = self._get_static_graph()

        preds, importances = self.model.forward_batch(graph_seqs, static_ei, static_ea)

        loss = self.loss_fn(preds, targets)
      
        phys_w = self.cfg["training"].get("physics_loss_weight", 0.0)
        if phys_w > 0:
            if preds.size(1) > 1:
                dt_pred = preds[:, 1:] - preds[:, :-1] 
                dt_tgt  = targets[:, 1:] - targets[:, :-1]
                phys_loss = F.mse_loss(dt_pred, dt_tgt)
                loss = loss + phys_w * phys_loss

        rollout_w = self.cfg["training"].get("rollout_loss_weight", 1.0)
        if rollout_w != 1.0:
            T_out = preds.size(1)
            weights = torch.linspace(1.0, rollout_w, T_out, device=self.device)
            per_step = ((preds - targets) ** 2).mean(dim=(0, 2, 3))  # [T_out]
            loss = (per_step * weights).mean()

        self.log(f"{stage}/loss", loss, on_step=(stage == "train"),
                 on_epoch=True, prog_bar=True, batch_size=B)

        with torch.no_grad():
            for vi, var in enumerate(self.variable_names):
                p = preds[:, 0, :, vi].cpu().numpy().ravel()
                t = targets[:, 0, :, vi].cpu().numpy().ravel()
                self.log(f"{stage}/rmse_{var}", metric_rmse(p, t),
                         on_epoch=True, prog_bar=(stage == "val"), batch_size=B)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")


    def configure_optimizers(self):
        tc  = self.cfg["training"]
        opt = torch.optim.AdamW(
            self.parameters(),
            lr           = tc["lr"],
            weight_decay = tc["weight_decay"],
        )

        sched_name = tc.get("scheduler", "cosine")
        if sched_name == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=tc["epochs"] - tc.get("warmup_epochs", 0)
            )
        elif sched_name == "step":
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
        else:
            return opt

        # Warmup wrapper
        warmup = tc.get("warmup_epochs", 0)
        if warmup > 0:
            def lr_lambda(epoch):
                if epoch < warmup:
                    return epoch / max(1, warmup)
                return 1.0
            warmup_sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
            return [opt], [
                {"scheduler": warmup_sched, "interval": "epoch", "name": "warmup"},
                {"scheduler": sched,        "interval": "epoch", "name": "cosine"},
            ]

        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


def train(cfg: dict) -> None:
    set_seed(cfg["experiment"]["seed"])
    log_cfg = cfg["logging"]
    tc      = cfg["training"]

    dm = WeatherDataModule(cfg)
    dm.prepare_data()
    dm.setup("fit")

    variable_names = cfg["data"]["variables"]
    n_vars         = len(variable_names)

    model_module = WeatherGNNModule(cfg, n_vars, variable_names)
    n_params = count_parameters(model_module.model)
    logger.info(f"Model parameters: {format_param_count(n_params)}")

    if log_cfg["backend"] == "wandb":
        try:
            exp_logger = WandbLogger(
                project = log_cfg["wandb_project"],
                entity  = log_cfg.get("wandb_entity"),
                name    = cfg["experiment"]["name"],
            )
        except Exception:
            logger.warning("WandB not available, falling back to TensorBoard.")
            exp_logger = TensorBoardLogger("logs", name=cfg["experiment"]["name"])
    else:
        exp_logger = TensorBoardLogger("logs", name=cfg["experiment"]["name"])

    monitor = tc.get("monitor_metric", "val/rmse_temperature")
    ckpt_cb = ModelCheckpoint(
        dirpath    = tc.get("ckpt_dir", "checkpoints"),
        filename   = "best-{epoch:02d}-{val/loss:.4f}",
        monitor    = "val/loss",
        mode       = "min",
        save_top_k = tc.get("save_top_k", 3),
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    early_stop_cb = EarlyStopping(
        monitor  = "val/loss",
        patience = 10,
        mode     = "min",
    )
    trainer = L.Trainer(
        max_epochs          = tc["epochs"],
        logger              = exp_logger,
        callbacks           = [ckpt_cb, lr_cb, early_stop_cb],
        gradient_clip_val   = tc.get("gradient_clip", 1.0),
        log_every_n_steps   = log_cfg.get("log_every_n_steps", 10),
        accelerator         = "auto",
        devices             = 1,
        enable_progress_bar = True,
        deterministic       = False,
        precision           = "32",
    )

    logger.info(f"Starting training: {cfg['experiment']['name']}")
    trainer.fit(model_module, datamodule=dm)
    logger.info(f"Best checkpoint: {ckpt_cb.best_model_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Temporal Weather GNN")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("overrides", nargs="*",
                        help="Override config values: section.key=value")
    args = parser.parse_args()

    cfg = load_config(args.config)

    for ov in args.overrides:
        if "=" not in ov:
            logger.warning(f"Ignoring malformed override: {ov}")
            continue
        k, v = ov.split("=", 1)
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                if v.lower() in ("true", "false"):
                    v = v.lower() == "true"
        keys = k.split(".")
        d = cfg
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = v
        logger.info(f"Override: {k} = {v}")

    train(cfg)
