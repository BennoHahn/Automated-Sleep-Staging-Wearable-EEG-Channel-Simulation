"""Step 2b: PyTorch Dataset & Subject-Independent Split Utilities (`src/data/dataset.py`).

Supports both:
- Model A (`use_context = False`, One-to-One): Returns single 30s epochs `(C, 3000)` and scalar label `y`.
- Model B (`use_context = True`, Many-to-Many LSTM): Returns contiguous sequences of `M` epochs
  `(M, C, 3000)` and sequence labels `(M,)` from within a single subject recording (never crossing
  subject boundaries).

Also implements the data augmentations from TinySleepNet (Supratak & Guo, 2020) and `PROJECT_SPEC.md`:
1. Random signal roll / time-shift of up to `+-10%` of epoch length (`+-300` samples at 100 Hz).
2. Random sign flip (multiplying the signal by `-1` with 50% probability) during training.
3. Random sequence start offset (`0..5` epochs) when `use_context = True` during training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.channel_sim import apply_channel_config
from src.data.preprocessing import (
    load_preprocessed_subject,
    load_preprocessed_subject_metadata,
)


def generate_subject_cv_splits(
    subject_ids: list[str],
    n_folds: int = 10,
    seed: int = 42,
    save_path: str | Path | None = None,
) -> list[dict[str, list[str]]]:
    """Generate strict subject-independent K-fold cross-validation splits (80% train, 10% val, 10% test).

    For `N = 10` subjects and `n_folds = 10`, each fold assigns:
      - 1 subject to `test` (10%)
      - 1 subject to `val` (10%)
      - 8 subjects to `train` (80%)
    Every subject serves as the held-out test subject in exactly one fold, and as the validation
    subject in exactly one fold. For general `N` subjects, subjects are partitioned into `n_folds`
    disjoint groups; fold `k` uses group `k` for `test`, group `(k + 1) % n_folds` for `val`,
    and the remaining `n_folds - 2` groups for `train`.

    Args:
        subject_ids: Sorted list of unique subject IDs.
        n_folds: Number of cross-validation folds (default 10).
        seed: Random seed for reproducible subject shuffling.
        save_path: Optional JSON file path to persist the splits.

    Returns:
        List of `n_folds` dictionaries, each with `'fold'`, `'train'`, `'val'`, and `'test'` lists.
    """
    unique_ids = sorted(list(dict.fromkeys(subject_ids)))
    n_subj = len(unique_ids)
    if n_subj < 3:
        raise ValueError(
            f"At least 3 subjects are required for train/val/test subject splitting, got {n_subj}."
        )
    effective_folds = min(n_folds, n_subj)

    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(unique_ids))

    # Partition subjects into `effective_folds` near-equal buckets
    buckets: list[list[str]] = [
        [str(x) for x in arr] for arr in np.array_split(shuffled, effective_folds)
    ]

    splits: list[dict[str, Any]] = []
    for fold_idx in range(effective_folds):
        test_ids = sorted(buckets[fold_idx])
        val_ids = sorted(buckets[(fold_idx + 1) % effective_folds])
        train_ids = sorted(
            [
                s
                for b_idx, bucket in enumerate(buckets)
                if b_idx not in (fold_idx, (fold_idx + 1) % effective_folds)
                for s in bucket
            ]
        )
        # Assert strict disjointness (zero subject leakage)
        assert set(train_ids).isdisjoint(val_ids)
        assert set(train_ids).isdisjoint(test_ids)
        assert set(val_ids).isdisjoint(test_ids)

        splits.append(
            {
                "fold": fold_idx,
                "train": train_ids,
                "val": val_ids,
                "test": test_ids,
            }
        )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "seed": seed,
                    "n_folds": effective_folds,
                    "n_subjects": n_subj,
                    "subject_ids": unique_ids,
                    "splits": splits,
                },
                f,
                indent=2,
            )

    return splits


def load_subject_cv_splits(split_path: str | Path) -> list[dict[str, Any]]:
    """Load saved subject-independent CV splits from JSON."""
    with open(Path(split_path), "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["splits"]


def ensure_subject_cv_splits(
    subject_ids: list[str],
    split_path: str | Path,
    n_folds: int = 10,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load saved CV splits from `split_path` ONLY if they match `subject_ids`, `n_folds`, and `seed`; otherwise regenerate."""
    split_path = Path(split_path)
    unique_ids = sorted(list(dict.fromkeys(subject_ids)))
    effective_folds = min(n_folds, len(unique_ids))

    if split_path.exists():
        try:
            with open(split_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            splits = payload.get("splits", [])
            if splits:
                sp0 = splits[0]
                saved_ids = sorted(list(set(sp0["train"]) | set(sp0["val"]) | set(sp0["test"])))
                if (
                    saved_ids == unique_ids
                    and int(payload.get("n_folds", len(splits))) == effective_folds
                    and int(payload.get("seed", seed)) == int(seed)
                ):
                    return splits
        except Exception:
            pass

    return generate_subject_cv_splits(
        subject_ids=unique_ids,
        n_folds=n_folds,
        seed=seed,
        save_path=split_path,
    )


class SleepStageDataset(Dataset):
    """PyTorch Dataset supporting both single-epoch (`use_context=False`) and sequence (`use_context=True`) modes.

    Attributes:
        subject_epochs: List of per-subject float32 arrays `(N_i, C, 3000)`.
        subject_labels: List of per-subject int64 arrays `(N_i,)`.
        subject_ids: List of subject ID strings.
        use_context: If False, each item is `(C, 3000), scalar_label`.
                     If True, each item is `(seq_len, C, 3000), (seq_len,)`.
    """

    def __init__(
        self,
        subject_records: list[dict[str, Any]],
        channel_config: dict[str, Any],
        use_context: bool = False,
        seq_len: int = 20,
        is_train: bool = False,
        augmentation_cfg: dict[str, Any] | None = None,
        storage_dtype: Any = None,
        defer_augmentation_to_batch: bool = False,
    ) -> None:
        super().__init__()
        self.use_context = bool(use_context)
        self.seq_len = int(seq_len)
        self.is_train = bool(is_train)
        self.defer_augmentation_to_batch = bool(defer_augmentation_to_batch)

        aug = augmentation_cfg or {}
        self.aug_enabled = bool(aug.get("enabled", True)) and self.is_train
        self.max_shift_frac = float(aug.get("max_shift_frac", 0.1))
        self.sign_flip_prob = float(aug.get("sign_flip_prob", 0.5))

        # Use float16 in RAM when indexing large cohorts (>50 subjects) so only the active
        # channel(s) in float16 are materialized in memory, and convert to float32 per batch.
        if storage_dtype is None:
            storage_dtype = np.float16 if len(subject_records) > 50 else np.float32
        self.storage_dtype = storage_dtype

        self.subject_epochs: list[np.ndarray] = []
        self.subject_labels: list[np.ndarray] = []
        self.subject_ids: list[str] = []
        self.channel_names: list[str] = []

        for rec in subject_records:
            derived_x, ch_names = apply_channel_config(
                rec,
                channel_config,
                normalize=True,
                out_dtype=self.storage_dtype,
            )
            self.subject_epochs.append(derived_x)
            self.subject_labels.append(rec["labels"].astype(np.int64))
            self.subject_ids.append(str(rec["subject_id"]))
            if not self.channel_names:
                self.channel_names = ch_names

        self.n_channels = self.subject_epochs[0].shape[1] if self.subject_epochs else 0
        self.epoch_samples = self.subject_epochs[0].shape[2] if self.subject_epochs else 3000

        # Build index map: list of (subj_idx, start_epoch_idx)
        self.index_map: list[tuple[int, int]] = []
        self._build_index_map(random_offset=False)

    def _build_index_map(self, random_offset: bool = False) -> None:
        """Build lookup table of `(subject_idx, epoch_idx_or_seq_start)`."""
        self.index_map.clear()
        if not self.use_context:
            for s_idx, x in enumerate(self.subject_epochs):
                for ep_idx in range(len(x)):
                    self.index_map.append((s_idx, ep_idx))
        else:
            for s_idx, x in enumerate(self.subject_epochs):
                n_ep = len(x)
                offset = int(np.random.randint(0, min(6, max(1, n_ep)))) if random_offset else 0
                start = offset
                while start + self.seq_len <= n_ep:
                    self.index_map.append((s_idx, start))
                    start += self.seq_len
                # During evaluation (`not self.is_train`), include any trailing epochs
                # via a final window aligned to the end if needed, or handle full night in evaluator.

    def on_epoch_start(self) -> None:
        """Optionally re-jitter sequence start offsets at the beginning of a training epoch."""
        if self.use_context and self.is_train and self.aug_enabled:
            self._build_index_map(random_offset=True)

    def _augment_single_epoch(self, x: np.ndarray) -> np.ndarray:
        """Apply random time-shift (`+-max_shift_frac`) and random sign flip to an epoch `(C, T)`."""
        out = x.astype(np.float32, copy=True)
        if self.max_shift_frac > 0:
            max_shift = int(round(self.epoch_samples * self.max_shift_frac))
            if max_shift > 0:
                shift = int(np.random.randint(-max_shift, max_shift + 1))
                if shift != 0:
                    out = np.roll(out, shift=shift, axis=-1)
        if self.sign_flip_prob > 0 and np.random.rand() < self.sign_flip_prob:
            out = -out
        return out

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s_idx, ep_idx = self.index_map[idx]
        subj_x = self.subject_epochs[s_idx]
        subj_y = self.subject_labels[s_idx]

        if not self.use_context:
            if self.defer_augmentation_to_batch:
                return torch.from_numpy(subj_x[ep_idx]), torch.tensor(
                    subj_y[ep_idx], dtype=torch.long
                )
            x = subj_x[ep_idx].astype(np.float32, copy=False)  # (C, 3000)
            y = subj_y[ep_idx]  # scalar int64
            if self.aug_enabled:
                x = self._augment_single_epoch(x)
            return torch.from_numpy(x), torch.tensor(
                y, dtype=torch.long
            )
        else:
            y_seq = subj_y[ep_idx : ep_idx + self.seq_len]  # (M,)
            if self.defer_augmentation_to_batch:
                return torch.from_numpy(subj_x[ep_idx : ep_idx + self.seq_len]), torch.from_numpy(
                    y_seq.astype(np.int64, copy=False)
                )
            x_seq = subj_x[ep_idx : ep_idx + self.seq_len].astype(np.float32, copy=False)  # (M, C, 3000)
            if self.aug_enabled:
                x_seq = np.stack([self._augment_single_epoch(ep) for ep in x_seq], axis=0)
            return torch.from_numpy(x_seq), torch.from_numpy(
                y_seq.astype(np.int64, copy=False)
            )


def load_subject_records_from_dir(
    processed_dir: str | Path,
    subject_ids: list[str] | None = None,
    lazy: bool | None = None,
) -> list[dict[str, Any]]:
    """Load preprocessed `.npz` subject records from `processed_dir`.

    Args:
        processed_dir: Directory containing `{subject_id}.npz` files.
        subject_ids: Optional whitelist of subject IDs to load (e.g. from `select_shhs_cohort`).
            Only existing `.npz` files in `processed_dir` matching `subject_ids` are loaded.
        lazy: If True, loads lightweight metadata + labels (`epochs=None`) and defers signal
            loading to `apply_channel_config`. If `None` (default), automatically enables
            `lazy=True` when `> 50` subjects are present so large cohorts (`1,771` subjects)
            consume `< 15 MB` RAM instead of `> 40 GB`.
    """
    processed_dir = Path(processed_dir)
    if subject_ids is None:
        npz_files = sorted(processed_dir.glob("*.npz"))
    else:
        npz_files = [
            processed_dir / f"{sid}.npz"
            for sid in subject_ids
            if (processed_dir / f"{sid}.npz").exists()
        ]

    if lazy is None:
        lazy = len(npz_files) > 50

    if lazy:
        return [load_preprocessed_subject_metadata(p) for p in npz_files]
    return [load_preprocessed_subject(p) for p in npz_files]
