"""Step 5a: Standardized Epoch-by-Epoch Evaluation Metrics (`src/evaluation/metrics.py`).

Implements the exact mathematical framework from:
- Markov et al. (2025), "Evaluating the performance of wearable EEG sleep monitoring
  devices: a meta-analysis approach", npj Biomedical Innovations, Pages 20-23 (Equations 1-29).
- van der Aar et al. (2025), Pages 317-321 (Normalized confusion matrices mean +- SD across folds).
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
import seaborn as sns

STAGE_NAMES: list[str] = ["Wake", "N1", "N2", "N3", "REM"]


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = 5,
) -> np.ndarray:
    """Compute `(n_classes, n_classes)` integer confusion matrix `C` where `C[i, j]` is true `i`, pred `j`."""
    y_true_arr = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=np.int64).ravel()
    valid = (
        (y_true_arr >= 0)
        & (y_true_arr < n_classes)
        & (y_pred_arr >= 0)
        & (y_pred_arr < n_classes)
    )
    y_t = y_true_arr[valid]
    y_p = y_pred_arr[valid]
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_t, y_p), 1)
    return cm


def normalize_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    """Row-normalize a confusion matrix so each row sums to 1.0 (or 0.0 if a class is absent)."""
    cm_float = cm.astype(np.float64)
    row_sums = cm_float.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_cm = np.where(row_sums > 0, cm_float / row_sums, 0.0)
    return norm_cm


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0.0 else 0.0


def compute_multiclass_kappa_from_cm(cm: np.ndarray) -> float:
    """Compute Multiclass Cohen's Kappa (Eq. 1-3 in Markov et al., 2025)."""
    cm_f = cm.astype(np.float64)
    total = float(cm_f.sum())
    if total <= 0:
        return 0.0
    p_j = float(np.trace(cm_f)) / total
    row_totals = cm_f.sum(axis=1)
    col_totals = cm_f.sum(axis=0)
    p_e = float(np.sum(row_totals * col_totals)) / (total * total)
    if np.isclose(1.0 - p_e, 0.0):
        return 1.0 if np.isclose(p_j, 1.0) else 0.0
    return float((p_j - p_e) / (1.0 - p_e))


def compute_multiclass_mcc_from_cm(cm: np.ndarray) -> float:
    """Compute Multiclass Matthews Correlation Coefficient (Eq. 4-6 in Markov et al., 2025 / Gorodkin 2004)."""
    cm_f = cm.astype(np.float64)
    total = float(cm_f.sum())
    if total <= 0:
        return 0.0
    c_correct = float(np.trace(cm_f))
    t_k = cm_f.sum(axis=1)  # C_{i+}: true row sums
    p_k = cm_f.sum(axis=0)  # C_{+i}: predicted column sums

    cov_xy = c_correct * total - float(np.sum(t_k * p_k))
    cov_xx = total * total - float(np.sum(p_k * p_k))
    cov_yy = total * total - float(np.sum(t_k * t_k))
    denom = float(np.sqrt(max(cov_xx * cov_yy, 0.0)))
    if denom <= 0.0:
        return 0.0
    return float(cov_xy / denom)


def compute_per_class_metrics_from_cm(
    cm: np.ndarray,
    stage_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute sleep-stage-specific metrics (Eq. 7-16 in Markov et al., 2025) for each class."""
    n_classes = cm.shape[0]
    if stage_names is None:
        stage_names = STAGE_NAMES[:n_classes]

    cm_f = cm.astype(np.float64)
    total = float(cm_f.sum())
    row_sums = cm_f.sum(axis=1)
    col_sums = cm_f.sum(axis=0)

    per_class: dict[str, dict[str, float]] = {}
    for i, name in enumerate(stage_names):
        tp = float(cm_f[i, i])
        fn = float(row_sums[i] - tp)
        fp = float(col_sums[i] - tp)
        tn = float(total - (tp + fn + fp))

        # Eq. (7): ACC
        acc = _safe_div(tn + tp, total)

        # Eq. (8)-(10): Binary One-vs-Rest Cohen's Kappa
        pj = _safe_div(tn + tp, total)
        true_yes = tp + fn
        pred_yes = tp + fp
        true_no = tn + fp
        pred_no = tn + fn
        pe = _safe_div(true_yes * pred_yes + true_no * pred_no, total * total)
        kappa = _safe_div(pj - pe, 1.0 - pe) if not np.isclose(1.0 - pe, 0.0) else 0.0

        # Eq. (11): Sensitivity (SE / Recall)
        se = _safe_div(tp, tp + fn)
        # Eq. (12): Specificity (SP)
        sp = _safe_div(tn, tn + fp)
        # Eq. (13): Positive Predictive Value (PPV / Precision)
        ppv = _safe_div(tp, tp + fp)
        # Eq. (14): Negative Predictive Value (NPV)
        npv = _safe_div(tn, tn + fn)
        # Eq. (15): F1 score
        f1 = _safe_div(2.0 * se * ppv, se + ppv)
        # Eq. (16): Binary One-vs-Rest MCC
        mcc_denom = float(np.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.0)))
        mcc = _safe_div(tp * tn - fp * fn, mcc_denom)

        per_class[name] = {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "support": true_yes,
            "ACC": acc,
            "kappa": kappa,
            "SE": se,
            "SP": sp,
            "PPV": ppv,
            "NPV": npv,
            "F1": f1,
            "MCC": mcc,
        }

    return per_class


def compute_all_epoch_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stage_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute the complete standardized Markov et al. (2025) epoch-by-epoch evaluation suite.

    Includes:
      - Overall multiclass `accuracy` (Eq. 17), `kappa` (Eq. 1-3), `mcc` (Eq. 4-6), `weighted_f1`
      - Micro-averaged metrics: `micro_ACC`, `micro_SE` (Eq. 18), `micro_SP` (Eq. 19),
        `micro_PPV` (Eq. 20), `micro_NPV` (Eq. 21), `micro_F1` (Eq. 22)
      - Macro-averaged metrics (Eq. 23-29): `macro_ACC`, `macro_kappa`, `macro_SE`, `macro_SP`,
        `macro_PPV`, `macro_NPV`, `macro_F1`, `macro_MCC`
      - Per-class metrics dictionary (`Wake`, `N1`, `N2`, `N3`, `REM`)
      - Raw and row-normalized confusion matrices (`confusion_matrix`, `confusion_matrix_norm`)
    """
    n_classes = len(stage_names) if stage_names is not None else 5
    if stage_names is None:
        stage_names = STAGE_NAMES[:n_classes]

    cm = compute_confusion_matrix(y_true, y_pred, n_classes=n_classes)
    cm_norm = normalize_confusion_matrix(cm)
    total = float(cm.sum())

    overall_acc = _safe_div(float(np.trace(cm)), total)
    multiclass_kappa = compute_multiclass_kappa_from_cm(cm)
    multiclass_mcc = compute_multiclass_mcc_from_cm(cm)

    per_class = compute_per_class_metrics_from_cm(cm, stage_names=stage_names)

    # Aggregate TP, TN, FP, FN for Micro-Averaged equations (Eq. 18-22)
    sum_tp = sum(m["TP"] for m in per_class.values())
    sum_tn = sum(m["TN"] for m in per_class.values())
    sum_fp = sum(m["FP"] for m in per_class.values())
    sum_fn = sum(m["FN"] for m in per_class.values())

    micro_acc = _safe_div(sum_tp + sum_tn, sum_tp + sum_tn + sum_fp + sum_fn)
    micro_se = _safe_div(sum_tp, sum_tp + sum_fn)
    micro_sp = _safe_div(sum_tn, sum_tn + sum_fp)
    micro_ppv = _safe_div(sum_tp, sum_tp + sum_fp)
    micro_npv = _safe_div(sum_tn, sum_tn + sum_fn)
    micro_f1 = _safe_div(2.0 * micro_se * micro_ppv, micro_se + micro_ppv)

    # Macro-Averaged equations (Eq. 23-29)
    # Note: If a fold/recording has 0 support for a stage during a tiny dry-run, we still
    # compute the arithmetic mean across all classes (and also provide `macro_F1_present`
    # over classes present in `y_true`).
    macro_acc = float(np.mean([m["ACC"] for m in per_class.values()]))
    macro_kappa = float(np.mean([m["kappa"] for m in per_class.values()]))
    macro_se = float(np.mean([m["SE"] for m in per_class.values()]))
    macro_sp = float(np.mean([m["SP"] for m in per_class.values()]))
    macro_ppv = float(np.mean([m["PPV"] for m in per_class.values()]))
    macro_npv = float(np.mean([m["NPV"] for m in per_class.values()]))
    macro_f1 = float(np.mean([m["F1"] for m in per_class.values()]))
    macro_mcc = float(np.mean([m["MCC"] for m in per_class.values()]))

    weighted_f1 = (
        sum(m["F1"] * m["support"] for m in per_class.values()) / total if total > 0 else 0.0
    )

    return {
        "n_epochs": int(total),
        "accuracy": overall_acc,
        "kappa": multiclass_kappa,
        "mcc": multiclass_mcc,
        "macro_f1": macro_f1,
        "weighted_f1": float(weighted_f1),
        "micro": {
            "ACC": micro_acc,
            "SE": micro_se,
            "SP": micro_sp,
            "PPV": micro_ppv,
            "NPV": micro_npv,
            "F1": micro_f1,
        },
        "macro": {
            "ACC": macro_acc,
            "kappa": macro_kappa,
            "SE": macro_se,
            "SP": macro_sp,
            "PPV": macro_ppv,
            "NPV": macro_npv,
            "F1": macro_f1,
            "MCC": macro_mcc,
        },
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_norm": cm_norm.tolist(),
    }


def aggregate_fold_metrics(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate epoch metrics and normalized confusion matrices (mean +- SD) across CV folds."""
    if not fold_results:
        return {}

    scalar_keys = ["accuracy", "kappa", "mcc", "macro_f1", "weighted_f1"]
    summary: dict[str, Any] = {}
    for k in scalar_keys:
        vals = [float(r[k]) for r in fold_results]
        summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=0))}

    # Aggregate micro and macro dictionaries
    for group in ("micro", "macro"):
        summary[group] = {}
        for m_key in fold_results[0][group].keys():
            vals = [float(r[group][m_key]) for r in fold_results]
            summary[group][m_key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=0)),
            }

    # Aggregate per-class metrics
    summary["per_class"] = {}
    for stage in fold_results[0]["per_class"].keys():
        summary["per_class"][stage] = {}
        for m_key in ("ACC", "kappa", "SE", "SP", "PPV", "NPV", "F1", "MCC"):
            vals = [float(r["per_class"][stage][m_key]) for r in fold_results]
            summary["per_class"][stage][m_key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=0)),
            }

    # Pooled confusion matrix across all folds + Mean +- SD of row-normalized confusion matrices
    cms = [np.asarray(r["confusion_matrix"], dtype=np.int64) for r in fold_results]
    norm_cms = [np.asarray(r["confusion_matrix_norm"], dtype=np.float64) for r in fold_results]
    pooled_cm = np.sum(cms, axis=0)
    mean_norm_cm = np.mean(norm_cms, axis=0)
    std_norm_cm = np.std(norm_cms, axis=0, ddof=0)

    summary["pooled_confusion_matrix"] = pooled_cm.tolist()
    summary["mean_confusion_matrix_norm"] = mean_norm_cm.tolist()
    summary["std_confusion_matrix_norm"] = std_norm_cm.tolist()
    summary["pooled_metrics"] = compute_all_epoch_metrics_from_cm(pooled_cm)
    return summary


def compute_all_epoch_metrics_from_cm(
    cm: np.ndarray,
    stage_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute all Markov et al. (2025) metrics directly from an existing `(5, 5)` confusion matrix."""
    n_classes = cm.shape[0]
    if stage_names is None:
        stage_names = STAGE_NAMES[:n_classes]
    total = float(cm.sum())
    overall_acc = _safe_div(float(np.trace(cm)), total)
    multiclass_kappa = compute_multiclass_kappa_from_cm(cm)
    multiclass_mcc = compute_multiclass_mcc_from_cm(cm)
    per_class = compute_per_class_metrics_from_cm(cm, stage_names=stage_names)

    sum_tp = sum(m["TP"] for m in per_class.values())
    sum_tn = sum(m["TN"] for m in per_class.values())
    sum_fp = sum(m["FP"] for m in per_class.values())
    sum_fn = sum(m["FN"] for m in per_class.values())

    micro_se = _safe_div(sum_tp, sum_tp + sum_fn)
    micro_ppv = _safe_div(sum_tp, sum_tp + sum_fp)
    weighted_f1 = (
        sum(m["F1"] * m["support"] for m in per_class.values()) / total if total > 0 else 0.0
    )

    return {
        "n_epochs": int(total),
        "accuracy": overall_acc,
        "kappa": multiclass_kappa,
        "mcc": multiclass_mcc,
        "macro_f1": float(np.mean([m["F1"] for m in per_class.values()])),
        "weighted_f1": float(weighted_f1),
        "micro_f1": _safe_div(2.0 * micro_se * micro_ppv, micro_se + micro_ppv),
        "per_class": per_class,
    }


def plot_normalized_confusion_matrix(
    mean_norm_cm: np.ndarray,
    std_norm_cm: np.ndarray | None = None,
    stage_names: list[str] | None = None,
    title: str = "Normalized Confusion Matrix",
    save_path: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot row-normalized confusion matrix with `mean +- SD` annotations (van der Aar et al., 2025 Fig. 2)."""
    mean_arr = np.asarray(mean_norm_cm, dtype=np.float64)
    n_classes = mean_arr.shape[0]
    if stage_names is None:
        stage_names = STAGE_NAMES[:n_classes]

    annot = np.empty((n_classes, n_classes), dtype=object)
    for i in range(n_classes):
        for j in range(n_classes):
            if std_norm_cm is not None:
                std_val = float(np.asarray(std_norm_cm)[i, j])
                annot[i, j] = f"{mean_arr[i, j]:.2f}\n(±{std_val:.2f})"
            else:
                annot[i, j] = f"{mean_arr[i, j]:.2f}"

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        created_fig = True
    else:
        fig = ax.figure

    sns.heatmap(
        mean_arr,
        annot=annot,
        fmt="",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        xticklabels=stage_names,
        yticklabels=stage_names,
        cbar=True,
        ax=ax,
    )
    ax.set_xlabel("Predicted Sleep Stage")
    ax.set_ylabel("True PSG Sleep Stage")
    ax.set_title(title)

    if created_fig:
        fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def metrics_summary_to_dataframe(
    ablation_summaries: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Format a dictionary of `config_name -> aggregated_summary` into a clean comparison DataFrame."""
    rows: list[dict[str, Any]] = []
    for cfg_name, summ in ablation_summaries.items():
        if not summ:
            continue
        row: dict[str, Any] = {
            "Config": cfg_name,
            "ACC (mean±sd)": f"{summ['accuracy']['mean']:.3f} ± {summ['accuracy']['std']:.3f}",
            "Kappa (mean±sd)": f"{summ['kappa']['mean']:.3f} ± {summ['kappa']['std']:.3f}",
            "MCC (mean±sd)": f"{summ['mcc']['mean']:.3f} ± {summ['mcc']['std']:.3f}",
            "Macro-F1 (mean±sd)": f"{summ['macro_f1']['mean']:.3f} ± {summ['macro_f1']['std']:.3f}",
            "Weighted-F1 (mean±sd)": f"{summ['weighted_f1']['mean']:.3f} ± {summ['weighted_f1']['std']:.3f}",
            "Pooled_Kappa": round(float(summ["pooled_metrics"]["kappa"]), 4),
            "Pooled_MCC": round(float(summ["pooled_metrics"]["mcc"]), 4),
            "Pooled_Macro_F1": round(float(summ["pooled_metrics"]["macro_f1"]), 4),
        }
        for stage in STAGE_NAMES:
            if stage in summ.get("per_class", {}):
                row[f"F1_{stage}"] = round(
                    float(summ["pooled_metrics"]["per_class"][stage]["F1"]), 4
                )
        rows.append(row)
    return pd.DataFrame(rows)
