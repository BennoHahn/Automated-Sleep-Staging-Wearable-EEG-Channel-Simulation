"""Step 4: Training, Validation, Early Stopping & 10-Fold Subject-Independent CV (`src/training/trainer.py`).

References:
- van der Aar et al. (2024), Section 2.3 (p. 4): 200 epochs, early stopping patience 50,
  80/10/10 subject-level split, Adam lr=1e-4.
- van der Aar et al. (2025), p. 317: 10-fold CV, subject-level splitting, clinical sleep evaluation.
- Supratak & Guo (2020), TinySleepNet: gradient clipping max_norm=5.0, weight_decay=1e-3.
- He et al. (2023), Section 6.3 (p. 7): Strict subject-independent splitting to prevent data leakage.

Caching & Resumability:
- Every completed fold is immediately saved to `results/<run_name>/folds/<config_name>/fold_<XX>.json`
  (including full-night `y_true` and `y_pred` hypnogram arrays, training history, epoch metrics,
  and clinical sleep parameters), and model weights are saved to
  `results/<run_name>/checkpoints/<config_name>/<config_name>_fold<XX>.pt`.
- When `resume=True` (the default) and `force_retrain=False`, already-completed folds and
  configurations are loaded from disk in milliseconds without retraining.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.channel_sim import apply_channel_config
from src.data.dataset import SleepStageDataset
from src.evaluation.clinical_sleep import (
    compute_bland_altman_agreement,
    compute_hypnogram_parameters,
)
from src.evaluation.metrics import (
    aggregate_fold_metrics,
    compute_all_epoch_metrics,
)
from src.models.tinysleepnet import TinySleepNet, build_tinysleepnet_from_config


def resolve_device(device_str: str = "auto") -> torch.device:
    """Resolve PyTorch execution device (`'auto'` -> CUDA if available, else DirectML discrete GPU, else CPU)."""
    req = str(device_str).strip().lower()
    if req in ("auto", "gpu", "dml", "directml"):
        if req not in ("dml", "directml") and torch.cuda.is_available():
            return torch.device("cuda")
        try:
            import torch_directml
            if torch_directml.is_available():
                n_devs = torch_directml.device_count()
                best_idx = 0
                for idx in range(n_devs):
                    name_low = torch_directml.device_name(idx).lower()
                    if any(k in name_low for k in ("nvidia", "geforce", "rtx", "gtx", "quadro", "rx ", "radeon pro")):
                        best_idx = idx
                        break
                return torch_directml.device(best_idx)
        except ImportError:
            pass
        if req in ("dml", "directml"):
            raise RuntimeError("DirectML requested but `torch-directml` is not available.")
        return torch.device("cpu")
    return torch.device(device_str)


def get_device_description(device: torch.device | None = None) -> str:
    """Return a human-readable string describing the active compute device (including GPU model name)."""
    if device is None:
        device = resolve_device("auto")
    dev_type = str(device)
    if dev_type.startswith("cuda") and torch.cuda.is_available():
        return f"{dev_type} ({torch.cuda.get_device_name(device)})"
    if dev_type.startswith("privateuseone"):
        try:
            import torch_directml
            idx = device.index if device.index is not None else 0
            return f"{dev_type} (DirectML GPU: {torch_directml.device_name(idx).strip()})"
        except Exception:
            return f"{dev_type} (DirectML GPU)"
    return dev_type


class DirectMLAdam(torch.optim.Optimizer):
    """Exact Adam optimizer using `mul_` + `add_` instead of `lerp_` so 100% of ops stay on DirectML GPU."""

    def __init__(
        self,
        params: Any,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-3,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            wd = float(group["weight_decay"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd != 0.0:
                    grad = grad.add(p, alpha=wd)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step_t = state["step"]

                # Use mul_ and add_ (100% native DirectML GPU ops, avoiding aten::lerp.Scalar_out fallback)
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - (beta1 ** step_t)
                bias_correction2 = 1.0 - (beta2 ** step_t)
                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


@torch.no_grad()
def _clip_grad_norm_dml_safe(
    parameters: Any,
    max_norm: float,
    is_dml: bool = False,
) -> None:
    """Clip gradient norm without triggering unsupported DirectML foreach/linalg fallbacks."""
    if not is_dml:
        nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
        return
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total_norm_sq = torch.zeros((), device=grads[0].device, dtype=torch.float32)
    for g in grads:
        total_norm_sq = total_norm_sq + torch.sum(g * g)
    total_norm = torch.sqrt(total_norm_sq)
    clip_coef = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    for g in grads:
        g.mul_(clip_coef)


def compute_class_weights(
    labels: np.ndarray,
    n_classes: int = 5,
    mode: str | bool = False,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    """Compute optional class weight tensor for `nn.CrossEntropyLoss`.

    Supported `mode` values:
    - `False` / `"none"`: Unweighted Cross-Entropy (`None`)
    - `"tinysleepnet"`: Supratak & Guo (2020) weights `[1.0, 1.5, 1.0, 1.0, 1.0]`
    - `"sqrt"`: Square-root damped inverse class frequency `(total / count) ** 0.5`
    - `True` / `"balanced"`: Full inverse class frequency `total / (n_classes * count)`
    """
    if not mode or str(mode).lower() in ("false", "none"):
        return None
    mode_str = str(mode).strip().lower()
    if mode_str == "tinysleepnet":
        # Supratak & Guo (2020) weight N1 by 1.5 and all other stages by 1.0
        w = np.array([1.0, 1.5, 1.0, 1.0, 1.0], dtype=np.float32)[:n_classes]
    else:
        counts = np.bincount(labels[(labels >= 0) & (labels < n_classes)], minlength=n_classes)
        total = max(float(counts.sum()), 1.0)
        inv_freq = np.where(counts > 0, total / (n_classes * np.maximum(counts, 1)), 1.0).astype(
            np.float32
        )
        if mode_str == "sqrt":
            w = np.sqrt(inv_freq).astype(np.float32)
        else:
            w = inv_freq
        w = w / np.mean(w)
    tensor_w = torch.tensor(w, dtype=torch.float32)
    return tensor_w.to(device) if device is not None else tensor_w


def predict_subject_hypnogram(
    model: TinySleepNet,
    subject_record: dict[str, Any],
    channel_config: dict[str, Any],
    device: torch.device,
    batch_size: int = 128,
    seq_len: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict the full-night hypnogram `(y_true, y_pred)` for a single subject recording.

    Handles both `use_context=False` (batch of isolated epochs) and `use_context=True`
    (contiguous sequence windows covering every epoch of the night).
    """
    model.eval()
    x_np, _ = apply_channel_config(subject_record, channel_config, normalize=True)
    y_true = subject_record["labels"].astype(np.int64)
    n_epochs = len(y_true)

    if n_epochs == 0:
        return y_true, np.array([], dtype=np.int64)

    preds_list: list[np.ndarray] = []
    with torch.no_grad():
        if not model.use_context:
            for start in range(0, n_epochs, batch_size):
                end = min(start + batch_size, n_epochs)
                bx = torch.from_numpy(x_np[start:end]).to(device)
                logits = model(bx)  # (B, n_classes)
                preds_list.append(logits.argmax(dim=-1).cpu().numpy())
            y_pred = np.concatenate(preds_list, axis=0)
        else:
            start = 0
            while start + seq_len <= n_epochs:
                seq_x = torch.from_numpy(x_np[start : start + seq_len]).unsqueeze(0).to(device)
                logits = model(seq_x)  # (1, seq_len, n_classes)
                preds_list.append(logits.squeeze(0).argmax(dim=-1).cpu().numpy())
                start += seq_len
            if start < n_epochs:
                rem = n_epochs - start
                if n_epochs >= seq_len:
                    tail_x = (
                        torch.from_numpy(x_np[n_epochs - seq_len : n_epochs])
                        .unsqueeze(0)
                        .to(device)
                    )
                    logits = model(tail_x).squeeze(0)
                    preds_list.append(logits[-rem:].argmax(dim=-1).cpu().numpy())
                else:
                    tail_x = torch.from_numpy(x_np[start:n_epochs]).unsqueeze(0).to(device)
                    logits = model(tail_x).squeeze(0)
                    preds_list.append(logits.argmax(dim=-1).cpu().numpy())
            y_pred = np.concatenate(preds_list, axis=0)

    return y_true, y_pred.astype(np.int64)


def evaluate_subject_list(
    model: TinySleepNet,
    subject_records: list[dict[str, Any]],
    channel_config: dict[str, Any],
    device: torch.device,
    batch_size: int = 128,
    seq_len: int = 20,
    epoch_sec: float = 30.0,
) -> dict[str, Any]:
    """Evaluate `model` across a list of subject recordings, returning epoch + clinical metrics."""
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    per_subject_hypnograms: list[dict[str, Any]] = []

    for rec in subject_records:
        y_t, y_p = predict_subject_hypnogram(
            model=model,
            subject_record=rec,
            channel_config=channel_config,
            device=device,
            batch_size=batch_size,
            seq_len=seq_len,
        )
        all_true.append(y_t)
        all_pred.append(y_p)
        per_subject_hypnograms.append(
            {
                "subject_id": rec["subject_id"],
                "y_true": y_t,
                "y_pred": y_p,
                "true_clinical": compute_hypnogram_parameters(y_t, epoch_sec=epoch_sec),
                "pred_clinical": compute_hypnogram_parameters(y_p, epoch_sec=epoch_sec),
            }
        )

    y_true_concat = np.concatenate(all_true, axis=0) if all_true else np.array([], dtype=np.int64)
    y_pred_concat = np.concatenate(all_pred, axis=0) if all_pred else np.array([], dtype=np.int64)
    epoch_metrics = compute_all_epoch_metrics(y_true_concat, y_pred_concat)
    clinical_agreement = compute_bland_altman_agreement(
        per_subject_hypnograms, epoch_sec=epoch_sec
    )

    return {
        "epoch_metrics": epoch_metrics,
        "clinical_agreement": clinical_agreement,
        "per_subject": per_subject_hypnograms,
    }


def _serialize_fold_result(fold_res: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy arrays in `fold_res` (`y_true`, `y_pred`) to JSON-serializable lists."""
    out = copy.deepcopy(fold_res)
    serial_per_subj = []
    for item in out.get("test_per_subject", []):
        serial_per_subj.append(
            {
                "subject_id": str(item["subject_id"]),
                "y_true": np.asarray(item["y_true"], dtype=np.int64).tolist(),
                "y_pred": np.asarray(item["y_pred"], dtype=np.int64).tolist(),
                "true_clinical": item["true_clinical"],
                "pred_clinical": item["pred_clinical"],
            }
        )
    out["test_per_subject"] = serial_per_subj
    return out


def _deserialize_fold_result(payload: dict[str, Any]) -> dict[str, Any]:
    """Restore numpy arrays (`y_true`, `y_pred`) in a loaded fold result dictionary."""
    out = copy.deepcopy(payload)
    restored_per_subj = []
    for item in out.get("test_per_subject", []):
        restored_per_subj.append(
            {
                "subject_id": str(item["subject_id"]),
                "y_true": np.asarray(item["y_true"], dtype=np.int64),
                "y_pred": np.asarray(item["y_pred"], dtype=np.int64),
                "true_clinical": item["true_clinical"],
                "pred_clinical": item["pred_clinical"],
            }
        )
    out["test_per_subject"] = restored_per_subj
    return out


def save_fold_result_json(fold_res: dict[str, Any], path: str | Path) -> Path:
    """Persist a single fold's complete evaluation result to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_serialize_fold_result(fold_res), f, indent=2)
    return path


def load_fold_result_json(path: str | Path) -> dict[str, Any]:
    """Load a single fold's complete evaluation result from JSON."""
    with open(Path(path), "r", encoding="utf-8") as f:
        payload = json.load(f)
    return _deserialize_fold_result(payload)


def normalize_class_weight_mode(mode: str | bool | None) -> str:
    """Return canonical lowercase string (`'none'`, `'sqrt'`, `'tinysleepnet'`, `'balanced'`) for class weight mode."""
    if mode is None or mode is False:
        return "none"
    if mode is True:
        return "balanced"
    s = str(mode).strip().lower()
    if s in ("", "false", "none", "0", "off"):
        return "none"
    if s in ("true", "balanced", "inverse"):
        return "balanced"
    if s in ("sqrt", "square_root"):
        return "sqrt"
    if s in ("tinysleepnet", "tsn"):
        return "tinysleepnet"
    return s


def build_settings_signature(
    config_name: str,
    channel_config: dict[str, Any],
    base_config: dict[str, Any],
    fold_split: dict[str, Any],
    use_context_override: bool | None = None,
    max_epochs_override: int | None = None,
    class_weight_mode_override: str | bool | None = None,
    batch_size_override: int | None = None,
    early_stopping_patience_override: int | None = None,
) -> dict[str, Any]:
    """Build a deterministic signature of all dataset, channel, and hyperparameter settings for a fold.

    Used to verify that a cached `fold_XX.json` or `.pt` checkpoint on disk was trained with
    the exact same cohort split, channel configuration, context setting, class-weight mode,
    batch size, max epochs, and early-stopping patience before resuming from it.
    """
    train_cfg = base_config.get("training", {})
    model_cfg = base_config.get("model", {})

    use_context = (
        bool(model_cfg.get("use_context", False))
        if use_context_override is None
        else bool(use_context_override)
    )
    seq_len = int(model_cfg.get("seq_len", 20)) if use_context else 1

    if batch_size_override is not None:
        effective_bs = int(batch_size_override)
    else:
        effective_bs = int(train_cfg.get("batch_size", 128))
        if use_context:
            effective_bs = min(effective_bs, int(train_cfg.get("seq_batch_size", 32)))

    max_epochs = (
        int(train_cfg.get("max_epochs", 200))
        if max_epochs_override is None
        else int(max_epochs_override)
    )
    patience = (
        int(train_cfg.get("early_stopping_patience", 50))
        if early_stopping_patience_override is None
        else int(early_stopping_patience_override)
    )
    raw_cw = (
        train_cfg.get("class_weighted_loss", False)
        if class_weight_mode_override is None
        else class_weight_mode_override
    )
    cw_mode = normalize_class_weight_mode(raw_cw)

    train_ids = sorted([str(x) for x in fold_split.get("train", [])])
    val_ids = sorted([str(x) for x in fold_split.get("val", [])])
    test_ids = sorted([str(x) for x in fold_split.get("test", [])])
    split_blob = json.dumps(
        {"train": train_ids, "val": val_ids, "test": test_ids},
        sort_keys=True,
    ).encode("utf-8")
    split_sha1 = hashlib.sha1(split_blob).hexdigest()[:16]

    channels_blob = json.dumps(channel_config.get("channels", []), sort_keys=True)

    return {
        "config_name": str(config_name),
        "fold": int(fold_split.get("fold", 0)),
        "channels_spec": channels_blob,
        "use_context": bool(use_context),
        "seq_len": int(seq_len),
        "class_weighted_loss": cw_mode,
        "max_epochs": int(max_epochs),
        "early_stopping_patience": int(patience),
        "batch_size": int(effective_bs),
        "lr": float(train_cfg.get("lr", 1e-4)),
        "weight_decay": float(train_cfg.get("weight_decay", 1e-3)),
        "monitor": str(train_cfg.get("monitor", "val_macro_f1")),
        "n_train_subjects": len(train_ids),
        "n_val_subjects": len(val_ids),
        "n_test_subjects": len(test_ids),
        "split_sha1": split_sha1,
    }


def settings_signatures_match(
    saved_sig: dict[str, Any] | None,
    expected_sig: dict[str, Any],
) -> tuple[bool, str]:
    """Compare a saved fold's `settings_signature` against `expected_sig`, returning `(is_match, reason)`."""
    if not saved_sig or not isinstance(saved_sig, dict):
        return False, "missing settings_signature in cached fold"
    for k, exp_v in expected_sig.items():
        if k not in saved_sig:
            return False, f"missing key '{k}'"
        sav_v = saved_sig[k]
        if isinstance(exp_v, float):
            if not np.isclose(float(sav_v), exp_v, rtol=1e-5, atol=1e-8):
                return False, f"{k} mismatch (saved={sav_v} vs current={exp_v})"
        elif sav_v != exp_v:
            return False, f"{k} mismatch (saved={sav_v} vs current={exp_v})"
    return True, "exact match"


def get_experiment_results_dir(
    root_dir: str | Path,
    base_config: dict[str, Any],
    n_subjects: int,
    use_context: bool | None = None,
    class_weight_mode: str | bool | None = None,
    batch_size: int | None = None,
    max_epochs: int | None = None,
    early_stopping_patience: int | None = None,
) -> Path:
    """Return a settings-specific directory inside `results/` so distinct runs never overwrite each other."""
    root_dir = Path(root_dir)
    base_results = root_dir / base_config.get("paths", {}).get("results_dir", "results")
    train_cfg = base_config.get("training", {})
    model_cfg = base_config.get("model", {})
    cf = base_config.get("cohort_filter", {})

    ctx = bool(model_cfg.get("use_context", True)) if use_context is None else bool(use_context)
    cw = normalize_class_weight_mode(
        train_cfg.get("class_weighted_loss", "sqrt") if class_weight_mode is None else class_weight_mode
    )
    bs = int(train_cfg.get("batch_size", 32)) if batch_size is None else int(batch_size)
    if ctx and batch_size is None:
        bs = min(bs, int(train_cfg.get("seq_batch_size", 32)))
    ep = int(train_cfg.get("max_epochs", 100)) if max_epochs is None else int(max_epochs)
    pat = (
        int(train_cfg.get("early_stopping_patience", 20))
        if early_stopping_patience is None
        else int(early_stopping_patience)
    )

    cohort_tag = "healthy" if bool(cf.get("enabled", True)) and bool(cf.get("healthy_only", True)) else "all"
    min_q = max(int(cf.get("min_eeg_quality_rating", 1) or 1), int(cf.get("min_eog_quality_rating", 1) or 1))
    if min_q > 1:
        cohort_tag += f"_q{min_q}"
    min_h = max(float(cf.get("min_recording_hours", 0.0) or 0.0), float(cf.get("min_usable_eeg_hours", 0.0) or 0.0))
    if min_h > 0:
        cohort_tag += f"_h{min_h:g}"

    run_folder = f"shhs1_{cohort_tag}_n{n_subjects}_ctx{ctx}_w{cw}_bs{bs}_ep{ep}_pat{pat}"
    return base_results / run_folder


def _recover_fold_result_from_checkpoint(
    ckpt_path: str | Path,
    test_records: list[dict[str, Any]],
    channel_config: dict[str, Any],
    base_config: dict[str, Any],
    use_context_override: bool | None = None,
) -> dict[str, Any]:
    """Rebuild a fold result dictionary from an existing `.pt` checkpoint without retraining."""
    ckpt_path = Path(ckpt_path)
    device = resolve_device(str(base_config.get("training", {}).get("device", "auto")))
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)

    in_channels = int(payload.get("in_channels", 1))
    use_context = (
        bool(payload.get("use_context", False))
        if use_context_override is None
        else bool(use_context_override)
    )
    model = build_tinysleepnet_from_config(
        base_config,
        in_channels=in_channels,
        use_context_override=use_context,
    ).to(device)
    model.load_from_checkpoint_payload(payload)

    batch_size = int(base_config.get("training", {}).get("batch_size", 128))
    seq_len = int(base_config.get("model", {}).get("seq_len", 20))
    epoch_sec = float(base_config.get("preprocessing", {}).get("epoch_sec", 30.0))

    test_eval = evaluate_subject_list(
        model=model,
        subject_records=test_records,
        channel_config=channel_config,
        device=device,
        batch_size=batch_size,
        seq_len=seq_len,
        epoch_sec=epoch_sec,
    )

    return {
        "config_name": str(payload.get("config_name", ckpt_path.parent.name)),
        "fold": int(payload.get("fold", 0)),
        "use_context": use_context,
        "in_channels": in_channels,
        "channel_names": list(payload.get("channel_names", [])),
        "max_epochs_configured": int(payload.get("max_epochs_configured", 0)),
        "best_epoch": int(payload.get("best_epoch", 0)),
        "best_val_score": float(payload.get("best_val_score", 0.0)),
        "elapsed_sec": 0.0,
        "checkpoint_path": str(ckpt_path),
        "settings_signature": payload.get("settings_signature"),
        "train_subjects": list(payload.get("train_subjects", [])),
        "val_subjects": list(payload.get("val_subjects", [])),
        "test_subjects": list(payload.get("test_subjects", [r["subject_id"] for r in test_records])),
        "history": [],
        "test_epoch_metrics": test_eval["epoch_metrics"],
        "test_clinical_agreement": test_eval["clinical_agreement"],
        "test_per_subject": test_eval["per_subject"],
    }


def train_one_fold(
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    channel_config: dict[str, Any],
    base_config: dict[str, Any],
    fold_idx: int = 0,
    config_name: str = "default",
    use_context_override: bool | None = None,
    max_epochs_override: int | None = None,
    class_weight_mode_override: str | bool | None = None,
    batch_size_override: int | None = None,
    early_stopping_patience_override: int | None = None,
    settings_signature: dict[str, Any] | None = None,
    checkpoint_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train TinySleepNet on one CV fold with validation early stopping, then evaluate on test set."""
    seed = int(base_config.get("seed", 42)) + fold_idx
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_cfg = base_config.get("training", {})
    model_cfg = base_config.get("model", {})
    prep_cfg = base_config.get("preprocessing", {})

    device = resolve_device(str(train_cfg.get("device", "auto")))
    use_context = (
        bool(model_cfg.get("use_context", False))
        if use_context_override is None
        else bool(use_context_override)
    )
    seq_len = int(model_cfg.get("seq_len", 20))
    if batch_size_override is not None:
        batch_size = int(batch_size_override)
    else:
        batch_size = int(train_cfg.get("batch_size", 128))
        if use_context:
            batch_size = min(batch_size, int(train_cfg.get("seq_batch_size", 32)))

    max_epochs = (
        int(train_cfg.get("max_epochs", 200))
        if max_epochs_override is None
        else int(max_epochs_override)
    )
    patience = (
        int(train_cfg.get("early_stopping_patience", 50))
        if early_stopping_patience_override is None
        else int(early_stopping_patience_override)
    )
    cw_mode = (
        train_cfg.get("class_weighted_loss", False)
        if class_weight_mode_override is None
        else class_weight_mode_override
    )
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-3))
    grad_clip = float(train_cfg.get("grad_clip", 5.0))
    monitor_key = str(train_cfg.get("monitor", "val_macro_f1"))
    epoch_sec = float(prep_cfg.get("epoch_sec", 30.0))

    if settings_signature is None:
        settings_signature = build_settings_signature(
            config_name=config_name,
            channel_config=channel_config,
            base_config=base_config,
            fold_split={
                "fold": fold_idx,
                "train": [r["subject_id"] for r in train_records],
                "val": [r["subject_id"] for r in val_records],
                "test": [r["subject_id"] for r in test_records],
            },
            use_context_override=use_context,
            max_epochs_override=max_epochs,
            class_weight_mode_override=cw_mode,
            batch_size_override=batch_size,
            early_stopping_patience_override=patience,
        )

    train_ds = SleepStageDataset(
        subject_records=train_records,
        channel_config=channel_config,
        use_context=use_context,
        seq_len=seq_len,
        is_train=True,
        augmentation_cfg=train_cfg.get("augmentation", {}),
    )
    n_channels = train_ds.n_channels
    channel_names = list(train_ds.channel_names)

    model = build_tinysleepnet_from_config(
        base_config,
        in_channels=n_channels,
        use_context_override=use_context,
    ).to(device)

    all_train_labels = np.concatenate([r["labels"] for r in train_records], axis=0)
    class_weights = compute_class_weights(
        all_train_labels,
        n_classes=model.n_classes,
        mode=cw_mode,
        device=device,
    )
    is_dml = str(device).startswith("privateuseone")
    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.backends.cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    if is_dml:
        optimizer: torch.optim.Optimizer = DirectMLAdam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
        )

    # Pre-stage contiguous training tensors in RAM/VRAM for small-to-medium non-context datasets
    aug_cfg = train_cfg.get("augmentation", {})
    aug_enabled = bool(aug_cfg.get("enabled", True))
    max_shift_frac = float(aug_cfg.get("max_shift_frac", 0.1))
    sign_flip_prob = float(aug_cfg.get("sign_flip_prob", 0.5))
    max_shift_samples = int(round(train_ds.epoch_samples * max_shift_frac)) if aug_enabled else 0

    if not use_context and len(train_records) <= 50:
        x_train_all = torch.from_numpy(
            np.concatenate(train_ds.subject_epochs, axis=0).astype(np.float32, copy=False)
        )
        y_train_all = torch.from_numpy(
            np.concatenate(train_ds.subject_labels, axis=0).astype(np.int64, copy=False)
        )
        n_train_items = len(y_train_all)
        train_loader = None
    else:
        x_train_all = None
        y_train_all = None
        n_train_items = len(train_ds)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=is_cuda,
        )

    best_score = -float("inf")
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    epochs_no_improve = 0
    history: list[dict[str, float]] = []

    t_start = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        if not use_context and x_train_all is not None and y_train_all is not None:
            perm = torch.randperm(n_train_items)
            for start_idx in range(0, n_train_items, batch_size):
                batch_idx = perm[start_idx : start_idx + batch_size]
                bx = x_train_all[batch_idx]
                by = y_train_all[batch_idx]

                # Fast vectorized batch augmentation on CPU before a single contiguous GPU transfer
                if aug_enabled:
                    if max_shift_samples > 0:
                        shift = int(np.random.randint(-max_shift_samples, max_shift_samples + 1))
                        if shift != 0:
                            bx = torch.roll(bx, shifts=shift, dims=-1)
                    if sign_flip_prob > 0:
                        signs = torch.where(
                            torch.rand((bx.shape[0], 1, 1)) < sign_flip_prob, -1.0, 1.0
                        )
                        bx = bx * signs

                bx = bx.to(device, non_blocking=is_cuda)
                by = by.to(device, non_blocking=is_cuda)
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                if grad_clip > 0:
                    _clip_grad_norm_dml_safe(
                        model.parameters(),
                        max_norm=grad_clip,
                        is_dml=is_dml,
                    )
                optimizer.step()
                running_loss += float(loss.item())
                n_batches += 1
        else:
            train_ds.on_epoch_start()
            assert train_loader is not None
            for bx, by in train_loader:
                bx = bx.to(device, non_blocking=is_cuda)
                by = by.to(device, non_blocking=is_cuda)
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits.reshape(-1, model.n_classes), by.reshape(-1))
                loss.backward()
                if grad_clip > 0:
                    _clip_grad_norm_dml_safe(
                        model.parameters(),
                        max_norm=grad_clip,
                        is_dml=is_dml,
                    )
                optimizer.step()
                running_loss += float(loss.item())
                n_batches += 1

        train_loss = running_loss / max(n_batches, 1)

        # Validation evaluation
        val_eval = evaluate_subject_list(
            model=model,
            subject_records=val_records,
            channel_config=channel_config,
            device=device,
            batch_size=batch_size,
            seq_len=seq_len,
            epoch_sec=epoch_sec,
        )
        val_m = val_eval["epoch_metrics"]
        val_macro_f1 = float(val_m["macro_f1"])
        val_weighted_f1 = float(val_m["weighted_f1"])
        val_kappa = float(val_m["kappa"])
        val_acc = float(val_m["accuracy"])

        if monitor_key == "val_weighted_f1":
            current_score = val_weighted_f1
        elif monitor_key == "val_kappa":
            current_score = val_kappa
        elif monitor_key == "val_accuracy":
            current_score = val_acc
        else:
            current_score = val_macro_f1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": val_acc,
                "val_kappa": val_kappa,
                "val_macro_f1": val_macro_f1,
                "val_weighted_f1": val_weighted_f1,
            }
        )

        if current_score > best_score + 1e-6:
            best_score = current_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == max_epochs):
            print(
                f"  [{config_name} | Fold {fold_idx:02d}] Epoch {epoch:03d}/{max_epochs:03d} "
                f"| loss={train_loss:.4f} | val_acc={val_acc:.3f} | val_kappa={val_kappa:.3f} "
                f"| val_macro_f1={val_macro_f1:.3f} (best={best_score:.3f} @ ep {best_epoch})"
            )

        if epochs_no_improve >= patience:
            if verbose:
                print(
                    f"  [{config_name} | Fold {fold_idx:02d}] Early stopping triggered at epoch {epoch} "
                    f"(best epoch: {best_epoch}, {monitor_key}={best_score:.4f})."
                )
            break

    # Free training dataset buffers before test evaluation
    del train_ds
    if x_train_all is not None:
        del x_train_all, y_train_all
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Restore best validation model and evaluate on held-out Test subject(s)
    model.load_state_dict(best_state)
    test_eval = evaluate_subject_list(
        model=model,
        subject_records=test_records,
        channel_config=channel_config,
        device=device,
        batch_size=batch_size,
        seq_len=seq_len,
        epoch_sec=epoch_sec,
    )

    checkpoint_path_str: str | None = None
    if checkpoint_dir is not None:
        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{config_name}_fold{fold_idx:02d}.pt"
        payload = model.get_checkpoint_payload(
            extra_metadata={
                "config_name": config_name,
                "fold": fold_idx,
                "best_epoch": best_epoch,
                "best_val_score": best_score,
                "max_epochs_configured": max_epochs,
                "channel_names": channel_names,
                "settings_signature": settings_signature,
                "train_subjects": [r["subject_id"] for r in train_records],
                "val_subjects": [r["subject_id"] for r in val_records],
                "test_subjects": [r["subject_id"] for r in test_records],
            }
        )
        torch.save(payload, ckpt_path)
        checkpoint_path_str = str(ckpt_path)

    elapsed_sec = time.time() - t_start
    if verbose:
        tm = test_eval["epoch_metrics"]
        n_test_s = len(test_records)
        subj_preview = (
            str([r["subject_id"] for r in test_records])
            if n_test_s <= 3
            else f"{n_test_s} subjects ({test_records[0]['subject_id']}..{test_records[-1]['subject_id']})"
        )
        print(
            f"  -> [{config_name} | Fold {fold_idx:02d} DONE in {elapsed_sec:.1f}s] "
            f"Test {subj_preview}: "
            f"ACC={tm['accuracy']:.3f} | Kappa={tm['kappa']:.3f} | MCC={tm['mcc']:.3f} | Macro-F1={tm['macro_f1']:.3f}"
        )

    return {
        "config_name": config_name,
        "fold": fold_idx,
        "use_context": use_context,
        "in_channels": n_channels,
        "channel_names": channel_names,
        "max_epochs_configured": max_epochs,
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "elapsed_sec": elapsed_sec,
        "checkpoint_path": checkpoint_path_str,
        "settings_signature": settings_signature,
        "train_subjects": [r["subject_id"] for r in train_records],
        "val_subjects": [r["subject_id"] for r in val_records],
        "test_subjects": [r["subject_id"] for r in test_records],
        "history": history,
        "test_epoch_metrics": test_eval["epoch_metrics"],
        "test_clinical_agreement": test_eval["clinical_agreement"],
        "test_per_subject": test_eval["test_per_subject"] if "test_per_subject" in test_eval else test_eval["per_subject"],
    }


def run_cross_validation_for_config(
    all_records: list[dict[str, Any]],
    splits: list[dict[str, Any]],
    channel_config: dict[str, Any],
    base_config: dict[str, Any],
    config_name: str,
    use_context_override: bool | None = None,
    max_epochs_override: int | None = None,
    class_weight_mode_override: str | bool | None = None,
    batch_size_override: int | None = None,
    early_stopping_patience_override: int | None = None,
    max_folds: int | None = None,
    results_dir: str | Path | None = None,
    resume: bool = True,
    force_retrain: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run subject-independent K-fold cross-validation for one channel configuration with strict settings-matched caching.

    Args:
        all_records: List of preprocessed subject dicts (eager or lazy metadata dicts).
        splits: List of fold split dicts (`train`, `val`, `test` subject IDs).
        channel_config: Channel specification dict from `ablation_channels.yaml`.
        base_config: Base configuration dict from `base_config.yaml`.
        config_name: Identifier of the channel configuration (e.g. `'dual_eeg'`).
        use_context_override: Override `model.use_context`.
        max_epochs_override: Override `training.max_epochs`.
        class_weight_mode_override: Override `training.class_weighted_loss` (`'sqrt'`, `'tinysleepnet'`, `'balanced'`, `False`).
        batch_size_override: Override `training.batch_size`.
        early_stopping_patience_override: Override `training.early_stopping_patience`.
        max_folds: Optional limit on number of folds to run (`None` = all 10 folds).
        results_dir: Directory where checkpoints, per-fold JSONs, and aggregated summaries are saved.
        resume: If True (default) and `force_retrain=False`, loads any already-completed fold from
            `results_dir/folds/{config_name}/fold_{fold_idx:02d}.json` ONLY if its `settings_signature`
            matches the current run settings (cohort split, channels, context, class weights, batch size, epochs).
        force_retrain: If True, ignores cached fold files and retrains all folds from scratch.
        verbose: Print progress messages.
    """
    subj_map = {str(r["subject_id"]): r for r in all_records}
    active_splits = splits[:max_folds] if max_folds is not None else splits

    fold_outputs: list[dict[str, Any]] = []
    all_test_subject_hypnograms: list[dict[str, Any]] = []

    out_dir = Path(results_dir) if results_dir is not None else None
    ckpt_dir = out_dir / "checkpoints" / config_name if out_dir else None
    folds_cache_dir = out_dir / "folds" / config_name if out_dir else None

    for sp in active_splits:
        fold_idx = int(sp["fold"])
        expected_sig = build_settings_signature(
            config_name=config_name,
            channel_config=channel_config,
            base_config=base_config,
            fold_split=sp,
            use_context_override=use_context_override,
            max_epochs_override=max_epochs_override,
            class_weight_mode_override=class_weight_mode_override,
            batch_size_override=batch_size_override,
            early_stopping_patience_override=early_stopping_patience_override,
        )

        fold_cache_file = (
            folds_cache_dir / f"fold_{fold_idx:02d}.json"
            if folds_cache_dir is not None
            else None
        )
        ckpt_file = (
            ckpt_dir / f"{config_name}_fold{fold_idx:02d}.pt"
            if ckpt_dir is not None
            else None
        )

        loaded_from_cache = False

        # 1. Check if this fold's JSON result is already saved on disk AND matches current settings
        if (
            resume
            and not force_retrain
            and fold_cache_file is not None
            and fold_cache_file.exists()
        ):
            cached_fold_res = load_fold_result_json(fold_cache_file)
            is_match, reason = settings_signatures_match(
                cached_fold_res.get("settings_signature"),
                expected_sig,
            )
            if is_match:
                fold_res = cached_fold_res
                loaded_from_cache = True
                if verbose:
                    tm = fold_res["test_epoch_metrics"]
                    print(
                        f"  [{config_name} | Fold {fold_idx:02d}] Loaded verified saved result from {fold_cache_file.name} "
                        f"(best_ep={fold_res.get('best_epoch')}, Test ACC={tm['accuracy']:.3f}, "
                        f"Kappa={tm['kappa']:.3f}, MCC={tm['mcc']:.3f}, Macro-F1={tm['macro_f1']:.3f})"
                    )
            elif verbose:
                print(
                    f"  [{config_name} | Fold {fold_idx:02d}] Ignoring cached {fold_cache_file.name} "
                    f"because settings changed ({reason}) -> retraining fold."
                )

        # 2. Check if a trained .pt model checkpoint exists AND matches current settings
        if (
            not loaded_from_cache
            and resume
            and not force_retrain
            and ckpt_file is not None
            and ckpt_file.exists()
        ):
            try:
                ckpt_meta = torch.load(ckpt_file, map_location="cpu", weights_only=False)
                is_match, reason = settings_signatures_match(
                    ckpt_meta.get("settings_signature"),
                    expected_sig,
                )
            except Exception as e:
                is_match, reason = False, f"checkpoint read error ({e})"

            if is_match:
                test_recs = [subj_map[sid] for sid in sp["test"] if sid in subj_map]
                fold_res = _recover_fold_result_from_checkpoint(
                    ckpt_path=ckpt_file,
                    test_records=test_recs,
                    channel_config=channel_config,
                    base_config=base_config,
                    use_context_override=use_context_override,
                )
                loaded_from_cache = True
                if fold_cache_file is not None:
                    save_fold_result_json(fold_res, fold_cache_file)
                if verbose:
                    tm = fold_res["test_epoch_metrics"]
                    print(
                        f"  [{config_name} | Fold {fold_idx:02d}] Recovered verified checkpoint {ckpt_file.name} "
                        f"(best_ep={fold_res.get('best_epoch')}, Test ACC={tm['accuracy']:.3f}, "
                        f"Kappa={tm['kappa']:.3f}, MCC={tm['mcc']:.3f}, Macro-F1={tm['macro_f1']:.3f})"
                    )
            elif verbose:
                print(
                    f"  [{config_name} | Fold {fold_idx:02d}] Ignoring checkpoint {ckpt_file.name} "
                    f"because settings changed ({reason}) -> retraining fold."
                )

        # 3. Otherwise train this fold and immediately save both .pt and fold_XX.json
        if not loaded_from_cache:
            train_recs = [subj_map[sid] for sid in sp["train"] if sid in subj_map]
            val_recs = [subj_map[sid] for sid in sp["val"] if sid in subj_map]
            test_recs = [subj_map[sid] for sid in sp["test"] if sid in subj_map]

            fold_res = train_one_fold(
                train_records=train_recs,
                val_records=val_recs,
                test_records=test_recs,
                channel_config=channel_config,
                base_config=base_config,
                fold_idx=fold_idx,
                config_name=config_name,
                use_context_override=use_context_override,
                max_epochs_override=max_epochs_override,
                class_weight_mode_override=class_weight_mode_override,
                batch_size_override=batch_size_override,
                early_stopping_patience_override=early_stopping_patience_override,
                settings_signature=expected_sig,
                checkpoint_dir=ckpt_dir,
                verbose=verbose,
            )
            if fold_cache_file is not None:
                save_fold_result_json(fold_res, fold_cache_file)

        fold_outputs.append(fold_res)
        all_test_subject_hypnograms.extend(fold_res["test_per_subject"])

        # Continuously update the aggregated `metrics_{config_name}.json` after each fold
        # so partial progress is also saved if interrupted.
        if out_dir is not None:
            _save_config_result_bundle(
                out_dir=out_dir,
                config_name=config_name,
                channel_config=channel_config,
                base_config=base_config,
                fold_outputs=fold_outputs,
                all_test_subject_hypnograms=all_test_subject_hypnograms,
            )

    return _build_config_result_bundle(
        config_name=config_name,
        channel_config=channel_config,
        base_config=base_config,
        fold_outputs=fold_outputs,
        all_test_subject_hypnograms=all_test_subject_hypnograms,
    )


def _build_config_result_bundle(
    config_name: str,
    channel_config: dict[str, Any],
    base_config: dict[str, Any],
    fold_outputs: list[dict[str, Any]],
    all_test_subject_hypnograms: list[dict[str, Any]],
) -> dict[str, Any]:
    fold_epoch_metrics = [f["test_epoch_metrics"] for f in fold_outputs]
    aggregated_epoch = aggregate_fold_metrics(fold_epoch_metrics)
    epoch_sec = float(base_config.get("preprocessing", {}).get("epoch_sec", 30.0))
    pooled_clinical_agreement = compute_bland_altman_agreement(
        all_test_subject_hypnograms, epoch_sec=epoch_sec
    )
    return {
        "config_name": config_name,
        "description": channel_config.get("description", ""),
        "category": channel_config.get("category", ""),
        "mastoid": channel_config.get("mastoid", None),
        "n_folds_run": len(fold_outputs),
        "aggregated_metrics": aggregated_epoch,
        "pooled_clinical_agreement": pooled_clinical_agreement,
        "folds": fold_outputs,
    }


def _save_config_result_bundle(
    out_dir: Path,
    config_name: str,
    channel_config: dict[str, Any],
    base_config: dict[str, Any],
    fold_outputs: list[dict[str, Any]],
    all_test_subject_hypnograms: list[dict[str, Any]],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = _build_config_result_bundle(
        config_name=config_name,
        channel_config=channel_config,
        base_config=base_config,
        fold_outputs=fold_outputs,
        all_test_subject_hypnograms=all_test_subject_hypnograms,
    )
    serializable_bundle = {
        "config_name": bundle["config_name"],
        "description": bundle["description"],
        "category": bundle["category"],
        "mastoid": bundle["mastoid"],
        "n_folds_run": bundle["n_folds_run"],
        "aggregated_metrics": bundle["aggregated_metrics"],
        "pooled_clinical_agreement": bundle["pooled_clinical_agreement"],
        "folds": [_serialize_fold_result(f) for f in bundle["folds"]],
    }
    json_path = out_dir / f"metrics_{config_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable_bundle, f, indent=2)
    return json_path


def load_saved_ablation_results(
    results_dir: str | Path,
    config_names: list[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load saved `metrics_<config_name>.json` files from `results_dir` without running any training.

    Restores full `ablation_results` (including per-fold `y_true` and `y_pred` numpy arrays for
    hypnogram and Bland-Altman plotting) and `ablation_summaries` (for summary tables and confusion
    matrix plotting).

    Args:
        results_dir: Directory containing `metrics_*.json` files.
        config_names: Optional list of config names to load (if None, loads all found in `results_dir`).

    Returns:
        Tuple of `(ablation_results, ablation_summaries)`.
    """
    res_dir = Path(results_dir)
    ablation_results: dict[str, dict[str, Any]] = {}
    ablation_summaries: dict[str, dict[str, Any]] = {}

    if not res_dir.exists():
        return ablation_results, ablation_summaries

    if config_names is not None:
        files = [res_dir / f"metrics_{name}.json" for name in config_names if (res_dir / f"metrics_{name}.json").exists()]
    else:
        files = sorted(res_dir.glob("metrics_*.json"))

    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            payload = json.load(f)
        cfg_name = str(payload["config_name"])
        payload["folds"] = [_deserialize_fold_result(fd) for fd in payload.get("folds", [])]
        ablation_results[cfg_name] = payload
        ablation_summaries[cfg_name] = payload["aggregated_metrics"]

    return ablation_results, ablation_summaries
