"""
utils/metrics.py
================
All weather forecasting metrics.

Functions
---------
  rmse            : Root Mean Squared Error
  mae             : Mean Absolute Error
  bias            : Mean Error (signed)
  acc             : Anomaly Correlation Coefficient
  skill_score     : Skill Score vs a baseline (e.g., persistence)
  compute_all     : Full metric table [n_vars, n_lead_times]
  persistence_forecast : Simple t+0 → t+k baseline
  climatology_forecast : Rolling mean baseline

All inputs are numpy arrays with shape [T, N, V] (time, nodes, variables).
"""

from __future__ import annotations

import numpy as np
from typing import Sequence


# ─────────────────────────────────────────────────────────────
#  Scalar metrics (operate on arbitrary arrays)
# ─────────────────────────────────────────────────────────────

def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def bias(pred: np.ndarray, target: np.ndarray) -> float:
    """Signed mean error (positive = over-prediction)."""
    return float(np.mean(pred - target))


def acc(
    pred:   np.ndarray,
    target: np.ndarray,
    clim:   np.ndarray,
) -> float:
    """
    Anomaly Correlation Coefficient.

    ACC = Σ (pred_anom * target_anom) / sqrt(Σ pred_anom² * Σ target_anom²)

    pred, target, clim : same shape.  clim is the climatological mean.
    """
    p_anom = pred   - clim
    t_anom = target - clim
    num    = np.sum(p_anom * t_anom)
    den    = np.sqrt(np.sum(p_anom ** 2) * np.sum(t_anom ** 2)) + 1e-10
    return float(num / den)


def skill_score(
    pred:     np.ndarray,
    target:   np.ndarray,
    baseline: np.ndarray,
    metric:   str = "mse",
) -> float:
    """
    Skill score vs a baseline.

    SS = 1 - score_model / score_baseline

    metric : "mse" | "mae"
    """
    if metric == "mse":
        s_model    = np.mean((pred     - target) ** 2)
        s_baseline = np.mean((baseline - target) ** 2)
    elif metric == "mae":
        s_model    = np.mean(np.abs(pred     - target))
        s_baseline = np.mean(np.abs(baseline - target))
    else:
        raise ValueError(metric)
    return float(1.0 - s_model / (s_baseline + 1e-10))


# ─────────────────────────────────────────────────────────────
#  Baseline forecasts
# ─────────────────────────────────────────────────────────────

def persistence_forecast(
    x: np.ndarray,   # [T_in, N, V]  – the input window
    n_steps: int,
) -> np.ndarray:     # [n_steps, N, V]
    """Repeat the last observed frame for all forecast steps."""
    last = x[-1:, :, :]                          # [1, N, V]
    return np.repeat(last, n_steps, axis=0)


def climatology_forecast(
    history: np.ndarray,  # [T_history, N, V]
    n_steps: int,
    window:  int = 720,   # use last `window` steps to estimate clim
) -> np.ndarray:          # [n_steps, N, V]
    """Rolling-window mean as the climatological baseline."""
    clim = history[-window:].mean(axis=0, keepdims=True)  # [1, N, V]
    return np.repeat(clim, n_steps, axis=0)


# ─────────────────────────────────────────────────────────────
#  Full metric table
# ─────────────────────────────────────────────────────────────

def compute_all_metrics(
    preds:       np.ndarray,            # [T_test, T_out, N, V]
    targets:     np.ndarray,            # [T_test, T_out, N, V]
    persistence: np.ndarray,            # [T_test, T_out, N, V]
    climatology: np.ndarray,            # [T_test, T_out, N, V]
    variables:   Sequence[str],
    lead_times:  Sequence[int],         # forecast steps (e.g., [1, 3, 6, 12])
    dt_hours:    float = 1.0,
) -> dict:
    """
    Compute RMSE / MAE / Bias / ACC / SkillScore for every
    (variable, lead_time) combination.

    Returns
    -------
    dict with keys like "temperature_lead3h":
      {rmse, mae, bias, acc, skill_persistence, skill_climatology}
    """
    n_test, T_out, N, V = preds.shape
    results = {}

    # Climatological mean per node per variable (used for ACC)
    clim_mean = climatology.mean(axis=(0, 1), keepdims=True)
    clim_mean = np.broadcast_to(clim_mean, preds.shape)

    for vi, var in enumerate(variables):
        for li, lt in enumerate(lead_times):
            if lt - 1 >= T_out:
                continue
            step = lt - 1      # 0-indexed step

            p  = preds      [:, step, :, vi].ravel()
            t  = targets    [:, step, :, vi].ravel()
            pb = persistence[:, step, :, vi].ravel()
            cb = climatology[:, step, :, vi].ravel()
            cm = clim_mean  [:, step, :, vi].ravel()

            key = f"{var}_lead{int(lt * dt_hours)}h"
            results[key] = {
                "variable":          var,
                "lead_time_h":       int(lt * dt_hours),
                "rmse":              rmse(p, t),
                "mae":               mae(p, t),
                "bias":              bias(p, t),
                "acc":               acc(p, t, cm),
                "skill_persistence": skill_score(p, t, pb),
                "skill_climatology": skill_score(p, t, cb),
                "n_samples":         len(p),
            }

    return results


def format_metrics_table(results: dict, variables: Sequence[str]) -> str:
    """
    Render a pretty ASCII table of all metrics.

    Returns a string (can be printed or saved to file).
    """
    try:
        from tabulate import tabulate
        have_tabulate = True
    except ImportError:
        have_tabulate = False

    rows = []
    headers = ["Variable", "Lead (h)", "RMSE", "MAE", "Bias", "ACC",
               "SS-Persist", "SS-Clim"]

    for key, m in sorted(results.items(), key=lambda x: (x[1]["variable"], x[1]["lead_time_h"])):
        rows.append([
            m["variable"],
            m["lead_time_h"],
            f"{m['rmse']:.4f}",
            f"{m['mae']:.4f}",
            f"{m['bias']:+.4f}",
            f"{m['acc']:.4f}",
            f"{m['skill_persistence']:.4f}",
            f"{m['skill_climatology']:.4f}",
        ])

    if have_tabulate:
        return tabulate(rows, headers=headers, tablefmt="rounded_outline")
    else:
        # Fallback manual formatting
        col_w = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
        def fmt_row(row):
            return "  ".join(str(v).ljust(w) for v, w in zip(row, col_w))
        sep  = "-" * (sum(col_w) + 2 * len(col_w))
        lines = [sep, fmt_row(headers), sep] + [fmt_row(r) for r in rows] + [sep]
        return "\n".join(lines)
