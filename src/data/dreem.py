"""Dreem Open Datasets (DOD-H & DOD-O) Zenodo Downloader and HDF5 Preprocessor.

Handles open-access ingestion from Zenodo record `15900394` (`dod-h` 25 healthy subjects
at IRBA France, `dod-o` 55 scored OSA subjects at Stanford; 80 scored subjects total),
HDF5 (`signals/eeg/*`, `hypnogram`) extraction, cohort-aware mains notch filtering,
resampling to 100 Hz, 30-second epoch segmentation, z-score normalization, and `.npz`
caching matching the existing SHHS1 pipeline format.
"""

from __future__ import annotations

import json
from pathlib import Path
import tarfile
from typing import Any
import urllib.request
import zipfile

import h5py
import numpy as np

from src.data.dataset import load_subject_cv_splits
from src.data.preprocessing import (
    _download_url_to_file,
    normalize_epochs,
    preprocess_channel,
    save_preprocessed_subject,
)

ZENODO_RECORD_ID: str = "15900394"
EXCLUDED_DOD_SUBJECTS: frozenset[str] = frozenset({
    "5d8e93a5-d26a-436d-bc2c-e9527dc92cbf",
})

# Shared across all 80 scored subjects (25 DOD-H + 55 DOD-O)
DREEM_SHARED_EEG_CHANNELS: tuple[str, ...] = (
    "C3-M2",
    "C4-M1",
    "F3-M2",
    "F3-F4",
    "F3-O1",
    "F4-O2",
)
# Present only in DOD-H (25 healthy subjects)
DODH_ONLY_EEG_CHANNELS: tuple[str, ...] = (
    "FP1-F3",
    "FP1-M2",
    "FP1-O1",
    "FP2-F4",
    "FP2-M1",
    "FP2-O2",
)
# Present only in DOD-O (55 OSA subjects)
DODO_ONLY_EEG_CHANNELS: tuple[str, ...] = (
    "O1-M2",
    "O2-M1",
)
DREEM_ALL_EEG_CHANNELS: tuple[str, ...] = (
    *DREEM_SHARED_EEG_CHANNELS,
    *DODH_ONLY_EEG_CHANNELS,
    *DODO_ONLY_EEG_CHANNELS,
)

DEFAULT_COHORT_MAINS_HZ: dict[str, float] = {
    "dod-h": 50.0,  # IRBA Bretigny-sur-Orge, France
    "dod-o": 60.0,  # Stanford Sleep Medicine Center, USA
}

_CANONICAL_CHANNEL_MAP: dict[str, str] = {
    ch.upper().replace("_", "-"): ch for ch in DREEM_ALL_EEG_CHANNELS
}


def _decode_attr(val: Any) -> Any:
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8", errors="replace").strip()
    if isinstance(val, np.ndarray) and val.ndim == 0:
        return _decode_attr(val.item())
    return val


def _normalize_channel_key(raw_key: str) -> str:
    """Map Dreem HDF5 channel keys (e.g. `C3_M2`, `Fp1_F3`) to canonical names (`C3-M2`, `FP1-F3`)."""
    cleaned = str(raw_key).strip().split("/")[-1]
    lookup = cleaned.upper().replace("_", "-")
    return _CANONICAL_CHANNEL_MAP.get(lookup, cleaned.replace("_", "-"))


def _infer_cohort_from_path(h5_path: Path) -> str | None:
    parts = [p.lower() for p in h5_path.parts]
    for part in reversed(parts[:-1]):
        if part in {"dod-h", "dodh", "dod_h"}:
            return "dod-h"
        if part in {"dod-o", "dodo", "dod_o"}:
            return "dod-o"
    stem_lower = h5_path.stem.lower()
    if stem_lower.startswith("dodh"):
        return "dod-h"
    if stem_lower.startswith("dodo"):
        return "dod-o"
    return None


def _make_subject_id(h5_path: Path, cohort: str) -> str:
    stem = h5_path.stem
    prefix = "dodh_" if cohort == "dod-h" else "dodo_"
    if stem.lower().startswith(prefix):
        return stem
    return f"{prefix}{stem}"


def _is_excluded_subject(h5_path: Path, excluded: frozenset[str]) -> bool:
    name_l = h5_path.name.lower()
    stem_l = h5_path.stem.lower()
    return any(ex.lower() in {name_l, stem_l} or ex.lower() in name_l for ex in excluded)


def discover_dreem_h5_files(
    raw_dir: str | Path,
    dodh_dir: str | Path | None = None,
    dodo_dir: str | Path | None = None,
    excluded_subjects: set[str] | frozenset[str] | list[str] | None = None,
) -> list[dict[str, Any]]:
    """Discover valid DOD-H and DOD-O `.h5` files, excluding known unscored recordings."""
    excluded = (
        EXCLUDED_DOD_SUBJECTS
        if excluded_subjects is None
        else frozenset(str(s) for s in excluded_subjects)
    )
    raw_root = Path(raw_dir)
    search_roots: list[tuple[Path, str | None]] = []
    if dodh_dir is not None and Path(dodh_dir).exists():
        search_roots.append((Path(dodh_dir), "dod-h"))
    if dodo_dir is not None and Path(dodo_dir).exists():
        search_roots.append((Path(dodo_dir), "dod-o"))
    if raw_root.exists():
        search_roots.append((raw_root, None))

    seen_paths: set[Path] = set()
    entries: list[dict[str, Any]] = []
    for root, default_cohort in search_roots:
        for h5_path in sorted(root.rglob("*.h5")):
            resolved = h5_path.resolve()
            if resolved in seen_paths or _is_excluded_subject(h5_path, excluded):
                continue
            seen_paths.add(resolved)
            cohort = default_cohort or _infer_cohort_from_path(h5_path) or "unknown"
            entries.append(
                {
                    "h5_path": h5_path,
                    "cohort": cohort,
                    "subject_id": _make_subject_id(h5_path, cohort) if cohort != "unknown" else h5_path.stem,
                }
            )
    entries.sort(key=lambda item: item["subject_id"])
    return entries


def _unit_scale_to_uv(unit_val: Any) -> float:
    if unit_val is None:
        return 1.0
    u = str(_decode_attr(unit_val)).strip().lower()
    if u in {"v", "volt", "volts"}:
        return 1e6
    if u in {"mv", "millivolt", "millivolts"}:
        return 1e3
    return 1.0


def _read_attr_float(obj: h5py.Group | h5py.Dataset, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in obj.attrs:
            val = _decode_attr(obj.attrs[key])
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def _read_attr_unit(obj: h5py.Group | h5py.Dataset) -> str | None:
    for key in ("unit", "units", "physical_dim"):
        if key in obj.attrs:
            return str(_decode_attr(obj.attrs[key]))
    return None


def _extract_h5_hypnogram(h5_file: h5py.File) -> np.ndarray | None:
    for path in ("hypnogram", "stages", "annotations/hypnogram"):
        if path in h5_file:
            arr = np.asarray(h5_file[path][...]).ravel()
            if arr.size == 0:
                return None
            return np.rint(arr).astype(np.int64)
    return None


def _find_eeg_group(h5_file: h5py.File) -> h5py.Group | None:
    if "signals" in h5_file and isinstance(h5_file["signals"], h5py.Group):
        sig_grp = h5_file["signals"]
        for key in sig_grp.keys():
            if key.lower() == "eeg" and isinstance(sig_grp[key], h5py.Group):
                return sig_grp[key]
    for key in h5_file.keys():
        if key.lower() == "eeg" and isinstance(h5_file[key], h5py.Group):
            return h5_file[key]
    return None


def read_dreem_h5_recording(
    h5_path: str | Path,
    target_channels: list[str] | tuple[str, ...] | None = None,
    default_fs: float = 250.0,
) -> tuple[dict[str, np.ndarray], dict[str, float], np.ndarray | None, str]:
    """Read EEG channels, sampling frequencies, and consensus hypnogram from a Dreem `.h5` file."""
    wanted = set(target_channels) if target_channels is not None else None
    signals: dict[str, np.ndarray] = {}
    fs_map: dict[str, float] = {}

    with h5py.File(h5_path, "r") as f:
        hypnogram = _extract_h5_hypnogram(f)
        root_fs = _read_attr_float(f, ("fs", "sampling_rate", "frequency", "sfreq"))
        root_unit = _read_attr_unit(f)

        eeg_grp = _find_eeg_group(f)
        if eeg_grp is None:
            raise KeyError(f"Group 'signals/eeg' not found in {h5_path}")

        grp_fs = _read_attr_float(eeg_grp, ("fs", "sampling_rate", "frequency", "sfreq"))
        grp_unit = _read_attr_unit(eeg_grp)

        for raw_key in eeg_grp.keys():
            ds = eeg_grp[raw_key]
            if not isinstance(ds, h5py.Dataset):
                continue
            canon = _normalize_channel_key(raw_key)
            if wanted is not None and canon not in wanted:
                continue

            orig_fs = (
                _read_attr_float(ds, ("fs", "sampling_rate", "frequency", "sfreq"))
                or grp_fs
                or root_fs
                or float(default_fs)
            )
            unit = _read_attr_unit(ds) or grp_unit or root_unit
            scale = _unit_scale_to_uv(unit)

            arr = np.asarray(ds[...], dtype=np.float64).ravel() * scale
            if not np.all(np.isfinite(arr)):
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            signals[canon] = arr
            fs_map[canon] = float(orig_fs)

    inferred_cohort = _infer_cohort_from_path(Path(h5_path))
    if inferred_cohort is None:
        if any(ch.startswith("FP") for ch in signals):
            inferred_cohort = "dod-h"
        elif any(ch in DODO_ONLY_EEG_CHANNELS for ch in signals):
            inferred_cohort = "dod-o"
        else:
            inferred_cohort = "unknown"

    return signals, fs_map, hypnogram, inferred_cohort


def _resolve_notch_freq(
    cohort: str,
    config: dict[str, Any],
) -> float | None:
    prep = config.get("preprocessing", {})
    notch_cfg = prep.get("notch", {})
    if not bool(notch_cfg.get("enabled", True)):
        return None
    explicit_notch = notch_cfg.get("freq_hz", None)
    if explicit_notch is not None:
        return float(explicit_notch)

    ds_cfg = config.get("dataset", {})
    cohort_map = ds_cfg.get("cohort_mains_freq_hz", DEFAULT_COHORT_MAINS_HZ)
    if cohort in cohort_map and cohort_map[cohort] is not None:
        return float(cohort_map[cohort])
    if ds_cfg.get("mains_freq_hz", None) is not None:
        return float(ds_cfg["mains_freq_hz"])
    return DEFAULT_COHORT_MAINS_HZ.get(cohort, None)


def preprocess_dreem_subject(
    h5_path: str | Path,
    subject_id: str,
    config: dict[str, Any],
    cohort: str | None = None,
) -> dict[str, Any] | None:
    """Preprocess one Dreem `.h5` night into epoch tensors `(N_valid, C, T)` ready for `.npz` caching.

    Returns `None` when a recording lacks consensus hypnogram annotations or contains zero valid epochs.
    """
    prep = config.get("preprocessing", {})
    target_channels = list(prep.get("channels", DREEM_ALL_EEG_CHANNELS))
    target_fs = float(prep.get("target_fs", 100.0))
    epoch_sec = float(prep.get("epoch_sec", 30.0))
    bp = prep.get("bandpass", {})
    low_hz = float(bp.get("low_hz", 0.2))
    high_hz = float(bp.get("high_hz", 49.0))
    order = int(bp.get("order", 5))
    zero_phase = bool(bp.get("zero_phase", True))
    notch_q = float(prep.get("notch", {}).get("q", 30.0))
    norm_mode = str(prep.get("normalization", "zscore_per_recording"))
    storage_dtype = str(prep.get("storage_dtype", "float32"))

    signals, fs_map, hypnogram, detected_cohort = read_dreem_h5_recording(
        h5_path=h5_path,
        target_channels=target_channels,
    )
    resolved_cohort = cohort if cohort and cohort != "unknown" else detected_cohort
    if hypnogram is None or not signals:
        return None

    ordered_channels = [ch for ch in target_channels if ch in signals]
    if not ordered_channels:
        return None

    notch_freq = _resolve_notch_freq(resolved_cohort, config)
    samples_per_epoch = int(round(target_fs * epoch_sec))

    processed_traces: list[np.ndarray] = []
    for ch in ordered_channels:
        sig_out = preprocess_channel(
            raw_signal=signals[ch],
            orig_fs=fs_map[ch],
            target_fs=target_fs,
            low_hz=low_hz,
            high_hz=high_hz,
            order=order,
            zero_phase=zero_phase,
            notch_freq=notch_freq,
            notch_q=notch_q,
        )
        processed_traces.append(sig_out)

    n_sig_epochs = min(len(tr) // samples_per_epoch for tr in processed_traces)
    n_common_epochs = min(n_sig_epochs, len(hypnogram))
    if n_common_epochs <= 0:
        return None

    n_samples = n_common_epochs * samples_per_epoch
    stacked = np.stack(
        [
            tr[:n_samples].reshape(n_common_epochs, samples_per_epoch)
            for tr in processed_traces
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    labels_all = hypnogram[:n_common_epochs]
    valid_mask = (labels_all >= 0) & (labels_all <= 4)
    n_valid = int(np.sum(valid_mask))
    if n_valid == 0:
        return None

    epochs_valid = stacked[valid_mask]
    labels_valid = labels_all[valid_mask].astype(np.int64)
    valid_epoch_indices = np.where(valid_mask)[0].astype(np.int64)

    epochs_norm = normalize_epochs(epochs_valid, mode=norm_mode)

    return {
        "subject_id": str(subject_id),
        "cohort": resolved_cohort,
        "epochs": epochs_norm,
        "labels": labels_valid,
        "valid_epoch_indices": valid_epoch_indices,
        "channel_names": ordered_channels,
        "fs": int(round(target_fs)),
        "epoch_sec": float(epoch_sec),
        "normalization": norm_mode,
        "is_normalized": norm_mode.lower() != "none",
        "storage_dtype": storage_dtype,
    }


def _interleave_cohorts_for_limit(
    entries: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    dodh = [e for e in entries if e["cohort"] == "dod-h"]
    dodo = [e for e in entries if e["cohort"] == "dod-o"]
    other = [e for e in entries if e["cohort"] not in {"dod-h", "dod-o"}]
    selected: list[dict[str, Any]] = []
    idx = 0
    while len(selected) < limit and (idx < len(dodh) or idx < len(dodo)):
        if idx < len(dodh) and len(selected) < limit:
            selected.append(dodh[idx])
        if idx < len(dodo) and len(selected) < limit:
            selected.append(dodo[idx])
        idx += 1
    for e in other:
        if len(selected) >= limit:
            break
        selected.append(e)
    return selected


def preprocess_all_dreem_recordings(
    config: dict[str, Any],
    overwrite: bool = False,
    limit: int | None = None,
    verbose: bool = True,
) -> list[Path]:
    """Discover and preprocess all DOD-H + DOD-O `.h5` files into `paths.processed_dir/*.npz`."""
    paths = config.get("paths", {})
    raw_dir = paths.get("raw_dir", "data/raw/dreem")
    dodh_dir = paths.get("dodh_dir", None)
    dodo_dir = paths.get("dodo_dir", None)
    processed_dir = Path(paths.get("processed_dir", "data/processed/dreem_dod"))
    processed_dir.mkdir(parents=True, exist_ok=True)

    excluded = config.get("dataset", {}).get("excluded_subjects", list(EXCLUDED_DOD_SUBJECTS))
    prep = config.get("preprocessing", {})
    storage_dtype = str(prep.get("storage_dtype", "float32"))
    compressed = bool(prep.get("compressed", False))

    entries = discover_dreem_h5_files(
        raw_dir=raw_dir,
        dodh_dir=dodh_dir,
        dodo_dir=dodo_dir,
        excluded_subjects=excluded,
    )
    if not entries:
        raise FileNotFoundError(
            f"No Dreem DOD .h5 files found under {raw_dir}. "
            "Download DOD-H and DOD-O from Zenodo record 15900394 or pass --download."
        )

    if limit is not None and limit < len(entries):
        entries = _interleave_cohorts_for_limit(entries, int(limit))

    saved_paths: list[Path] = []
    n_skipped_existing = 0
    n_skipped_unscored = 0

    for entry in entries:
        sid = entry["subject_id"]
        out_path = processed_dir / f"{sid}.npz"
        if out_path.exists() and not overwrite:
            saved_paths.append(out_path)
            n_skipped_existing += 1
            continue

        subj_dict = preprocess_dreem_subject(
            h5_path=entry["h5_path"],
            subject_id=sid,
            cohort=entry["cohort"],
            config=config,
        )
        if subj_dict is None:
            n_skipped_unscored += 1
            if verbose:
                print(f"  [Dreem Prep] Skipped unscored/empty file: {entry['h5_path'].name}")
            continue

        saved = save_preprocessed_subject(
            subj_dict,
            out_dir=processed_dir,
            storage_dtype=storage_dtype,
            compressed=compressed,
        )
        saved_paths.append(saved)
        if verbose:
            n_ep, n_ch, _ = subj_dict["epochs"].shape
            print(
                f"  [Dreem Prep] Saved {sid}.npz ({entry['cohort']}, "
                f"{n_ep} valid epochs, {n_ch} EEG channels)",
                flush=True,
            )

    if verbose:
        print(
            f"  [Dreem Prep Summary] total={len(saved_paths)} ready "
            f"(cached={n_skipped_existing}, skipped_unscored={n_skipped_unscored}) in {processed_dir}"
        )
    return saved_paths


def _match_cohort_for_filename(filename: str) -> str | None:
    low = filename.lower()
    if "dodh" in low or "dod-h" in low or "dod_h" in low:
        return "dod-h"
    if "dodo" in low or "dod-o" in low or "dod_o" in low:
        return "dod-o"
    return None


def _extract_archive_into_cohort_dir(archive_path: Path, cohort_dir: Path) -> int:
    """Extract `.zip` or `.tar*` archive and move `.h5` files cleanly into `cohort_dir`."""
    cohort_dir.mkdir(parents=True, exist_ok=True)
    tmp_extract = cohort_dir / "_extract_tmp"
    tmp_extract.mkdir(parents=True, exist_ok=True)

    lower = archive_path.name.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(tmp_extract)
    elif lower.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(tmp_extract)
    else:
        raise ValueError(f"Unsupported Dreem archive format: {archive_path}")

    moved = 0
    for h5_file in tmp_extract.rglob("*.h5"):
        dest = cohort_dir / h5_file.name
        if not dest.exists():
            h5_file.replace(dest)
            moved += 1
        else:
            h5_file.unlink()
            moved += 1

    # Clean empty staging folders
    for sub in sorted(tmp_extract.rglob("*"), reverse=True):
        try:
            if sub.is_dir():
                sub.rmdir()
            else:
                sub.unlink()
        except OSError:
            pass
    try:
        tmp_extract.rmdir()
    except OSError:
        pass
    return moved


def download_dreem_zenodo(
    raw_dir: str | Path = "data/raw/dreem",
    record_id: str = ZENODO_RECORD_ID,
    cohorts: tuple[str, ...] = ("dod-h", "dod-o"),
    force: bool = False,
    verbose: bool = True,
) -> dict[str, Path]:
    """Download and extract DOD-H and DOD-O HDF5 files from Zenodo REST API (`15900394`)."""
    raw_root = Path(raw_dir)
    raw_root.mkdir(parents=True, exist_ok=True)
    wanted_cohorts = set(cohorts)
    cohort_dirs: dict[str, Path] = {c: raw_root / c for c in wanted_cohorts}

    # Fast skip if .h5 files already exist for all requested cohorts
    if not force and all(
        cdir.exists() and any(cdir.rglob("*.h5")) for cdir in cohort_dirs.values()
    ):
        if verbose:
            counts = {c: len(list(cdir.rglob("*.h5"))) for c, cdir in cohort_dirs.items()}
            print(f"  [Dreem Download] Existing .h5 files found: {counts}. Skipping download.")
        return cohort_dirs

    api_url = f"https://zenodo.org/api/records/{record_id}"
    req = urllib.request.Request(api_url, headers={"User-Agent": "SleepStageReducedSensors/1.0"})
    with urllib.request.urlopen(req, timeout=60.0) as resp:
        metadata = json.loads(resp.read().decode("utf-8"))

    files = metadata.get("files", [])
    if not files:
        raise RuntimeError(f"Zenodo record {record_id} returned no downloadable files.")

    archives_dir = raw_root / "_archives"
    archives_dir.mkdir(parents=True, exist_ok=True)

    for entry in files:
        fname = str(entry.get("key") or entry.get("filename") or "")
        cohort = _match_cohort_for_filename(fname)
        if cohort not in wanted_cohorts:
            continue
        cdir = cohort_dirs[cohort]
        if not force and cdir.exists() and any(cdir.rglob("*.h5")):
            if verbose:
                print(f"  [Dreem Download] {cohort} already populated in {cdir}; skipping {fname}.")
            continue

        download_url = (
            entry.get("links", {}).get("self")
            or f"https://zenodo.org/api/records/{record_id}/files/{fname}/content"
        )
        lower_name = fname.lower()
        if lower_name.endswith(".h5"):
            dest = cdir / Path(fname).name
            cdir.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(f"  [Dreem Download] Fetching {fname} -> {dest}", flush=True)
            _download_url_to_file(download_url, dest)
        else:
            archive_dest = archives_dir / Path(fname).name
            if verbose:
                size_mb = float(entry.get("size", 0)) / (1024 * 1024)
                print(
                    f"  [Dreem Download] Fetching {fname} ({size_mb:.1f} MB) -> {archive_dest}",
                    flush=True,
                )
            _download_url_to_file(download_url, archive_dest)
            n_h5 = _extract_archive_into_cohort_dir(archive_dest, cdir)
            if verbose:
                print(f"  [Dreem Download] Extracted {n_h5} .h5 files into {cdir}", flush=True)

    return cohort_dirs


def generate_stratified_dreem_cv_splits(
    subject_ids: list[str],
    n_folds: int = 10,
    seed: int = 42,
    save_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Generate cohort-stratified subject-independent K-fold CV splits across DOD-H + DOD-O.

    Assigns `dodh_*` and `dodo_*` subjects in counter-balanced round-robin order so every
    fold contains nearly equal numbers of healthy (`DOD-H`) and OSA (`DOD-O`) test/val subjects
    AND filtering down to `DOD-H`-only or `DOD-O`-only keeps all folds non-empty when `N >= K`.
    """
    unique_ids = sorted(list(dict.fromkeys(str(s) for s in subject_ids)))
    n_subj = len(unique_ids)
    if n_subj < 3:
        raise ValueError(f"At least 3 subjects are required for CV splits, got {n_subj}.")
    effective_folds = min(n_folds, n_subj)

    rng = np.random.default_rng(seed)
    dodh_ids = [str(s) for s in rng.permutation([s for s in unique_ids if s.startswith("dodh_")])]
    dodo_ids = [str(s) for s in rng.permutation([s for s in unique_ids if s.startswith("dodo_")])]
    other_ids = [
        str(s)
        for s in rng.permutation(
            [s for s in unique_ids if not (s.startswith("dodh_") or s.startswith("dodo_"))]
        )
    ]

    buckets: list[list[str]] = [[] for _ in range(effective_folds)]
    for i, sid in enumerate(dodh_ids):
        buckets[i % effective_folds].append(sid)
    # Reverse offset for DOD-O balances exact total subjects per fold (25 + 55 = 80 -> 8/fold)
    for i, sid in enumerate(dodo_ids):
        buckets[(effective_folds - 1 - (i % effective_folds))].append(sid)
    for i, sid in enumerate(other_ids):
        buckets[i % effective_folds].append(sid)

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
        splits.append(
            {
                "fold": fold_idx,
                "train": train_ids,
                "val": val_ids,
                "test": test_ids,
            }
        )

    if save_path is not None:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        with open(save_p, "w", encoding="utf-8") as f:
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


def ensure_dreem_cv_splits(
    subject_ids: list[str],
    split_path: str | Path,
    n_folds: int = 10,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load persisted stratified Dreem splits if matching `subject_ids`, else regenerate."""
    split_p = Path(split_path)
    expected_ids = sorted(list(dict.fromkeys(str(s) for s in subject_ids)))
    effective_folds = min(n_folds, len(expected_ids))

    if split_p.exists():
        with open(split_p, "r", encoding="utf-8") as f:
            payload = json.load(f)
        saved_ids = sorted(str(s) for s in payload.get("subject_ids", []))
        if (
            saved_ids == expected_ids
            and int(payload.get("n_folds", -1)) == effective_folds
            and int(payload.get("seed", -1)) == int(seed)
        ):
            return load_subject_cv_splits(split_p)

    return generate_stratified_dreem_cv_splits(
        subject_ids=expected_ids,
        n_folds=effective_folds,
        seed=seed,
        save_path=split_p,
    )


def filter_splits_for_subjects(
    splits: list[dict[str, Any]],
    allowed_subject_ids: set[str] | list[str],
) -> list[dict[str, Any]]:
    """Restrict master CV splits to `allowed_subject_ids` (e.g. DOD-H-only or DOD-O-only channels)."""
    allowed = {str(s) for s in allowed_subject_ids}
    filtered: list[dict[str, Any]] = []
    for sp in splits:
        tr = [s for s in sp["train"] if s in allowed]
        va = [s for s in sp["val"] if s in allowed]
        te = [s for s in sp["test"] if s in allowed]
        if tr and va and te:
            filtered.append({"fold": int(sp["fold"]), "train": tr, "val": va, "test": te})
    return filtered
