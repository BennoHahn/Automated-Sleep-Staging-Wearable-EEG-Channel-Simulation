"""Step 5b: Whole-Night Clinical Sleep Parameters & Bland-Altman Analysis (`src/evaluation/clinical_sleep.py`).

Implements whole-night clinical sleep architecture parameters and agreement metrics from:
- van der Aar et al. (2025), Pages 317-321 (Table 3 & Fig. 3: TST, SE%, SOL, WASO, stage durations,
  Bland-Altman plots, and Pearson correlation `r`).
- Markov et al. (2025), Pages 18-20 (Standardized clinical parameter definitions).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent.parent.parent / ".mplconfig"),
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

STAGE_MAP: dict[int, str] = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def compute_hypnogram_parameters(
    hypnogram: np.ndarray,
    epoch_sec: float = 30.0,
) -> dict[str, float]:
    """Compute clinical sleep architecture parameters from a single-night hypnogram (0=W, 1=N1, 2=N2, 3=N3, 4=REM).

    Definitions (AASM / van der Aar et al., 2025 / Markov et al., 2025):
    - `TRT_min`: Total Recording Time (Time in Bed) in minutes = `N_epochs * (epoch_sec / 60)`.
    - `TST_min`: Total Sleep Time in minutes = total duration of all non-Wake (`1, 2, 3, 4`) epochs.
    - `SE_pct`: Sleep Efficiency (%) = `(TST_min / TRT_min) * 100`.
    - `SOL_min`: Sleep Onset Latency in minutes = elapsed time from start of recording (`epoch 0`)
      to the first epoch of any sleep stage (`N1, N2, N3, or REM`). If no sleep occurs, `SOL_min = TRT_min`.
    - `WASO_min`: Wake After Sleep Onset in minutes = total duration of `Wake (0)` epochs occurring
      after `SOL` (sleep onset) up to the end of the recording.
    - `REML_min`: REM Latency in minutes = elapsed time from sleep onset to the first `REM (4)` epoch
      (or `NaN` if no REM epoch occurs).
    - Stage Durations (`Wake_min`, `N1_min`, `N2_min`, `N3_min`, `REM_min`) in minutes.
    - Stage Percentages:
      - `Wake_pct_TRT`: `%` of Total Recording Time spent in Wake.
      - `N1_pct_TST`, `N2_pct_TST`, `N3_pct_TST`, `REM_pct_TST`: `%` of Total Sleep Time spent in each sleep stage.
    """
    hyp = np.asarray(hypnogram, dtype=np.int64).ravel()
    valid = (hyp >= 0) & (hyp <= 4)
    hyp = hyp[valid]

    epoch_min = float(epoch_sec) / 60.0
    n_epochs = len(hyp)
    trt_min = n_epochs * epoch_min

    if n_epochs == 0:
        return {
            "TRT_min": 0.0,
            "TST_min": 0.0,
            "SE_pct": 0.0,
            "SOL_min": 0.0,
            "WASO_min": 0.0,
            "REML_min": float("nan"),
            "Wake_min": 0.0,
            "N1_min": 0.0,
            "N2_min": 0.0,
            "N3_min": 0.0,
            "REM_min": 0.0,
            "Wake_pct_TRT": 0.0,
            "N1_pct_TST": 0.0,
            "N2_pct_TST": 0.0,
            "N3_pct_TST": 0.0,
            "REM_pct_TST": 0.0,
        }

    is_sleep = hyp > 0
    sleep_indices = np.where(is_sleep)[0]
    tst_min = float(np.sum(is_sleep)) * epoch_min
    se_pct = (tst_min / trt_min) * 100.0 if trt_min > 0 else 0.0

    if len(sleep_indices) > 0:
        first_sleep_idx = int(sleep_indices[0])
        sol_min = first_sleep_idx * epoch_min
        # Wake After Sleep Onset (WASO): Wake epochs after first sleep onset
        waso_epochs = int(np.sum(hyp[first_sleep_idx:] == 0))
        waso_min = waso_epochs * epoch_min

        rem_indices = np.where(hyp[first_sleep_idx:] == 4)[0]
        reml_min = float(rem_indices[0]) * epoch_min if len(rem_indices) > 0 else float("nan")
    else:
        sol_min = trt_min
        waso_min = 0.0
        reml_min = float("nan")

    wake_min = float(np.sum(hyp == 0)) * epoch_min
    n1_min = float(np.sum(hyp == 1)) * epoch_min
    n2_min = float(np.sum(hyp == 2)) * epoch_min
    n3_min = float(np.sum(hyp == 3)) * epoch_min
    rem_min = float(np.sum(hyp == 4)) * epoch_min

    wake_pct_trt = (wake_min / trt_min) * 100.0 if trt_min > 0 else 0.0
    n1_pct_tst = (n1_min / tst_min) * 100.0 if tst_min > 0 else 0.0
    n2_pct_tst = (n2_min / tst_min) * 100.0 if tst_min > 0 else 0.0
    n3_pct_tst = (n3_min / tst_min) * 100.0 if tst_min > 0 else 0.0
    rem_pct_tst = (rem_min / tst_min) * 100.0 if tst_min > 0 else 0.0

    return {
        "TRT_min": float(trt_min),
        "TST_min": float(tst_min),
        "SE_pct": float(se_pct),
        "SOL_min": float(sol_min),
        "WASO_min": float(waso_min),
        "REML_min": float(reml_min),
        "Wake_min": float(wake_min),
        "N1_min": float(n1_min),
        "N2_min": float(n2_min),
        "N3_min": float(n3_min),
        "REM_min": float(rem_min),
        "Wake_pct_TRT": float(wake_pct_trt),
        "N1_pct_TST": float(n1_pct_tst),
        "N2_pct_TST": float(n2_pct_tst),
        "N3_pct_TST": float(n3_pct_tst),
        "REM_pct_TST": float(rem_pct_tst),
    }


def compute_bland_altman_agreement(
    subject_Records: list[dict[str, Any]],
    epoch_sec: float = 30.0,
) -> dict[str, dict[str, float]]:
    """Compute Bland-Altman bias, 95% Limits of Agreement (LoA), MAE, and Pearson `r` across nights.

    Args:
        subject_Records: List of dicts, each containing `'subject_id'`, `'y_true'`, `'y_pred'`.
        epoch_sec: Duration of one epoch in seconds (30.0).

    Returns:
        Dictionary mapping each clinical parameter name (`TST_min`, `SE_pct`, `SOL_min`, `WASO_min`,
        `N1_min`, `N2_min`, `N3_min`, `REM_min`, etc.) to its agreement statistics:
          - `'true_mean'`, `'true_std'`, `'pred_mean'`, `'pred_std'`
          - `'bias_mean'` (`pred - true`), `'bias_std'`
          - `'loa_lower'` (`bias - 1.96 * std`), `'loa_upper'` (`bias + 1.96 * std`)
          - `'mae'` (Mean Absolute Error)
          - `'pearson_r'`, `'pearson_p'`
    """
    true_params_list = [
        compute_hypnogram_parameters(rec["y_true"], epoch_sec=epoch_sec)
        for rec in subject_Records
    ]
    pred_params_list = [
        compute_hypnogram_parameters(rec["y_pred"], epoch_sec=epoch_sec)
        for rec in subject_Records
    ]

    if not true_params_list:
        return {}

    param_keys = [
        "TST_min",
        "SE_pct",
        "SOL_min",
        "WASO_min",
        "Wake_min",
        "N1_min",
        "N2_min",
        "N3_min",
        "REM_min",
        "N1_pct_TST",
        "N2_pct_TST",
        "N3_pct_TST",
        "REM_pct_TST",
    ]

    agreement: dict[str, dict[str, float]] = {}
    for k in param_keys:
        t_vals = np.array([p[k] for p in true_params_list], dtype=np.float64)
        p_vals = np.array([p[k] for p in pred_params_list], dtype=np.float64)
        valid = np.isfinite(t_vals) & np.isfinite(p_vals)
        t_vals = t_vals[valid]
        p_vals = p_vals[valid]

        if len(t_vals) == 0:
            continue

        diffs = p_vals - t_vals  # Pred - PSG Ground Truth
        bias_mean = float(np.mean(diffs))
        bias_std = float(np.std(diffs, ddof=1 if len(diffs) > 1 else 0))
        loa_lower = bias_mean - 1.96 * bias_std
        loa_upper = bias_mean + 1.96 * bias_std
        mae = float(np.mean(np.abs(diffs)))

        if len(t_vals) >= 2 and np.std(t_vals) > 0 and np.std(p_vals) > 0:
            r_val, p_val = pearsonr(t_vals, p_vals)
            pearson_r = float(r_val)
            pearson_p = float(p_val)
        else:
            pearson_r = 0.0
            pearson_p = 1.0

        agreement[k] = {
            "true_mean": float(np.mean(t_vals)),
            "true_std": float(np.std(t_vals)),
            "pred_mean": float(np.mean(p_vals)),
            "pred_std": float(np.std(p_vals)),
            "bias_mean": bias_mean,
            "bias_std": bias_std,
            "loa_lower": float(loa_lower),
            "loa_upper": float(loa_upper),
            "mae": mae,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
        }

    return agreement


def plot_bland_altman_grid(
    subject_records: list[dict[str, Any]],
    parameters: list[str] | None = None,
    epoch_sec: float = 30.0,
    title: str = "Bland-Altman Agreement (Predicted vs. PSG Ground Truth)",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Generate a grid of Bland-Altman plots for key clinical sleep parameters (van der Aar 2025, Fig. 3)."""
    if parameters is None:
        parameters = ["TST_min", "SE_pct", "SOL_min", "WASO_min"]

    true_list = [
        compute_hypnogram_parameters(rec["y_true"], epoch_sec=epoch_sec)
        for rec in subject_records
    ]
    pred_list = [
        compute_hypnogram_parameters(rec["y_pred"], epoch_sec=epoch_sec)
        for rec in subject_records
    ]

    n_params = len(parameters)
    ncols = 2
    nrows = (n_params + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.2 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()

    for idx, param in enumerate(parameters):
        ax = axes_flat[idx]
        t_vals = np.array([p[param] for p in true_list], dtype=np.float64)
        p_vals = np.array([p[param] for p in pred_list], dtype=np.float64)
        means = 0.5 * (t_vals + p_vals)
        diffs = p_vals - t_vals

        bias = float(np.mean(diffs))
        sd = float(np.std(diffs, ddof=1 if len(diffs) > 1 else 0))
        loa_hi = bias + 1.96 * sd
        loa_lo = bias - 1.96 * sd

        ax.scatter(means, diffs, alpha=0.75, edgecolors="k", s=45)
        ax.axhline(bias, color="tab:blue", linestyle="-", label=f"Bias: {bias:+.1f}")
        ax.axhline(loa_hi, color="tab:red", linestyle="--", label=f"+1.96 SD: {loa_hi:+.1f}")
        ax.axhline(loa_lo, color="tab:red", linestyle="--", label=f"-1.96 SD: {loa_lo:+.1f}")
        ax.axhline(0.0, color="gray", linestyle=":", alpha=0.6)
        ax.set_title(param)
        ax.set_xlabel("Mean of Predicted & PSG")
        ax.set_ylabel("Difference (Pred - PSG)")
        ax.legend(loc="best", fontsize=8)

    for idx in range(n_params, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_hypnogram_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subject_id: str = "Subject",
    config_name: str = "Config",
    epoch_sec: float = 30.0,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot full-night manual PSG hypnogram vs. model-predicted hypnogram."""
    y_t = np.asarray(y_true, dtype=np.int64).ravel()
    y_p = np.asarray(y_pred, dtype=np.int64).ravel()
    hours = np.arange(len(y_t)) * (epoch_sec / 3600.0)

    # Clinical hypnogram display order (top to bottom): Wake (0), REM (4), N1 (1), N2 (2), N3 (3)
    display_order = {0: 4, 4: 3, 1: 2, 2: 1, 3: 0}
    yt_disp = [display_order.get(int(v), 0) for v in y_t]
    yp_disp = [display_order.get(int(v), 0) for v in y_p]

    fig, axes = plt.subplots(2, 1, figsize=(12, 4.5), sharex=True, sharey=True)
    axes[0].step(hours, yt_disp, where="post", color="black", linewidth=1.0)
    axes[0].set_title(f"{subject_id} — Manual PSG Ground Truth Hypnogram")
    axes[0].set_yticks([0, 1, 2, 3, 4])
    axes[0].set_yticklabels(["N3", "N2", "N1", "REM", "Wake"])

    axes[1].step(hours, yp_disp, where="post", color="tab:blue", linewidth=1.0)
    axes[1].set_title(f"{subject_id} — Predicted Hypnogram ({config_name})")
    axes[1].set_xlabel("Time (Hours)")
    axes[1].set_yticks([0, 1, 2, 3, 4])
    axes[1].set_yticklabels(["N3", "N2", "N1", "REM", "Wake"])

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
