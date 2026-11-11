"""Step 1: Data Ingestion & Standardized Preprocessing for PSG Recordings.

Implements standardized preprocessing following:
- van der Aar et al. (2024), Section 2.1.3 (p. 4)
- van der Aar et al. (2025), Section "PSG" (p. 317)
- Supratak & Guo (2020), TinySleepNet

Pipeline steps:
1. Read EDF physiological signals and Profusion XML (or general) annotations.
2. Map sleep stages to the 5 AASM classes:
   0: Wake, 1: N1, 2: N2, 3: N3 (merges R&K stages 3 and 4), 4: REM.
3. Apply notch filter (50/60 Hz when below Nyquist) and resample all channels to 100 Hz.
4. Apply zero-phase 5th-order Butterworth bandpass filter (0.2 - 49.0 Hz at 100 Hz).
5. Segment into non-overlapping 30-second epochs (3000 samples per epoch at 100 Hz).
6. Apply per-recording (subject-level) per-channel z-score normalization.
"""

from __future__ import annotations

import logging
from math import gcd
import os
from pathlib import Path
import shutil
import time
from typing import Any
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pyedflib
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt

logger = logging.getLogger(__name__)

# Canonical mapping from raw EDF labels (SHHS1, Dreem, BOAS, etc.) to internal names
CHANNEL_ALIASES: dict[str, list[str]] = {
    "C4-A1": ["EEG", "C4-A1", "C4-M1", "C4_M1", "EEG C4-A1", "EEG C4-M1"],
    "C3-A2": ["EEG(sec)", "EEG2", "EEG(SEC)", "C3-A2", "C3-M2", "C3_M2", "EEG C3-A2", "EEG C3-M2"],
    "EOG(L)": ["EOG(L)", "EOG-L", "EOG1", "LOC", "EOG Left", "EOG_L"],
    "EOG(R)": ["EOG(R)", "EOG-R", "EOG2", "ROC", "EOG Right", "EOG_R"],
    "EMG": ["EMG", "CHIN", "EMG Chin", "Chin EMG", "Submental EMG"],
    # 10-20 frontal & occipital channels for Phase 2 (BOAS ds005555 / Dreem DOD)
    "F3-M2": ["F3-M2", "F3_M2", "F3-A2", "EEG F3-M2", "EEG F3-A2", "F3"],
    "F4-M1": ["F4-M1", "F4_M1", "F4-A1", "EEG F4-M1", "EEG F4-A1"],
    "F4-M2": ["F4-M2", "F4_M2", "F4-A2", "EEG F4-M2", "F4"],
    "O1-M2": ["O1-M2", "O1_M2", "O1-A2", "EEG O1-M2", "O1"],
    "O2-M1": ["O2-M1", "O2_M1", "O2-A1", "EEG O2-M1", "O2"],
    "AF7": ["AF7", "EEG AF7"],
    "AF8": ["AF8", "EEG AF8"],
}

# Standard SHHS1 target channels (4-channel EEG+EOG baseline; EMG excluded per Early Fusion architecture)
DEFAULT_SHHS_CHANNELS: list[str] = ["C4-A1", "C3-A2", "EOG(L)", "EOG(R)"]

# Profusion XML integer stage code -> AASM 5-class integer (0=W, 1=N1, 2=N2, 3=N3, 4=REM, -1=Exclude)
PROFUSION_STAGE_MAP: dict[int, int] = {
    0: 0,  # Wake -> 0
    1: 1,  # Stage 1 -> 1 (N1)
    2: 2,  # Stage 2 -> 2 (N2)
    3: 3,  # Stage 3 -> 3 (N3)
    4: 3,  # Stage 4 (R&K) -> 3 (N3, merged per AASM standard)
    5: 4,  # REM -> 4
    6: -1,  # Movement -> exclude
    9: -1,  # Unscored / Unknown -> exclude
}

AASM_STAGE_NAMES: list[str] = ["Wake", "N1", "N2", "N3", "REM"]


def parse_profusion_xml(xml_path: str | Path) -> tuple[np.ndarray, float]:
    """Parse a Compumedics Profusion XML annotation file from SHHS.

    Args:
        xml_path: Path to `*-profusion.xml`.

    Returns:
        Tuple of:
          - stages: 1D numpy int64 array of mapped AASM labels (0..4, or -1 for unknown/movement)
          - epoch_sec: Epoch length in seconds (typically 30.0)
    """
    tree = ET.parse( Path(xml_path) )
    root = tree.getroot()
    epoch_el = root.find("EpochLength")
    epoch_sec = float(epoch_el.text) if epoch_el is not None and epoch_el.text else 30.0

    stage_elements = root.findall(".//SleepStages/SleepStage")
    raw_stages = [int(el.text.strip()) for el in stage_elements if el.text is not None]
    mapped_stages = np.array(
        [PROFUSION_STAGE_MAP.get(s, -1) for s in raw_stages], dtype=np.int64
    )
    return mapped_stages, epoch_sec


def apply_notch_filter(
    signal: np.ndarray,
    fs: float,
    notch_freq: float = 60.0,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """Apply zero-phase IIR notch filter if `notch_freq` is strictly below Nyquist (`fs / 2`)."""
    nyquist = fs / 2.0
    if notch_freq <= 0 or notch_freq >= nyquist * 0.98:
        return signal
    b, a = iirnotch(w0=notch_freq, Q=quality_factor, fs=fs)
    return filtfilt(b, a, signal).astype(np.float64, copy=False)


def resample_signal(signal: np.ndarray, orig_fs: float, target_fs: float = 100.0) -> np.ndarray:
    """Resample 1D continuous signal from `orig_fs` to `target_fs` using polyphase filtering."""
    if np.isclose(orig_fs, target_fs):
        return signal.astype(np.float64, copy=False)
    orig_int = int(round(orig_fs))
    target_int = int(round(target_fs))
    common = gcd(orig_int, target_int)
    up = target_int // common
    down = orig_int // common
    return resample_poly(signal, up=up, down=down).astype(np.float64, copy=False)


def apply_bandpass_filter(
    signal: np.ndarray,
    fs: float = 100.0,
    low_hz: float = 0.2,
    high_hz: float = 49.0,
    order: int = 5,
    zero_phase: bool = True,
) -> np.ndarray:
    """Apply 5th-order Butterworth bandpass filter between `low_hz` and `high_hz`."""
    nyquist = fs / 2.0
    clamped_high = min(high_hz, nyquist * 0.98)
    if low_hz <= 0 or low_hz >= clamped_high:
        raise ValueError(f"Invalid bandpass range [{low_hz}, {clamped_high}] Hz for fs={fs} Hz.")
    sos = butter(order, [low_hz, clamped_high], btype="bandpass", fs=fs, output="sos")
    if zero_phase:
        return sosfiltfilt(sos, signal).astype(np.float64, copy=False)
    from scipy.signal import sosfilt
    return sosfilt(sos, signal).astype(np.float64, copy=False)


def preprocess_channel(
    raw_signal: np.ndarray,
    orig_fs: float,
    target_fs: float = 100.0,
    low_hz: float = 0.2,
    high_hz: float = 49.0,
    order: int = 5,
    zero_phase: bool = True,
    notch_freq: float | None = 60.0,
    notch_q: float = 30.0,
) -> np.ndarray:
    """Filter and resample a single continuous PSG channel to `target_fs` (default 100 Hz).

    Note on anti-aliasing and Nyquist bounds:
    - If `notch_freq` (e.g. 60 Hz or 50 Hz) is below the original Nyquist (`orig_fs / 2`),
      the notch filter is applied at `orig_fs` before resampling.
    - The signal is resampled to `target_fs` (100 Hz) via polyphase filtering (`resample_poly`),
      which includes a built-in Kaiser anti-aliasing FIR lowpass filter.
    - Finally, the zero-phase 5th-order Butterworth bandpass filter (`0.2 - 49.0 Hz`, clamped
      to the effective Nyquist of `min(orig_fs, target_fs)`) is applied at `target_fs`.
    """
    sig = np.asarray(raw_signal, dtype=np.float64)
    if notch_freq is not None and notch_freq > 0:
        sig = apply_notch_filter(sig, fs=orig_fs, notch_freq=notch_freq, quality_factor=notch_q)

    sig = resample_signal(sig, orig_fs=orig_fs, target_fs=target_fs)

    # Effective upper frequency cannot exceed the Nyquist of the original or target sampling rate
    effective_nyquist = min(orig_fs, target_fs) / 2.0
    effective_high_hz = min(high_hz, effective_nyquist * 0.96)
    sig = apply_bandpass_filter(
        sig,
        fs=target_fs,
        low_hz=low_hz,
        high_hz=effective_high_hz,
        order=order,
        zero_phase=zero_phase,
    )
    return sig


def normalize_epochs(
    epochs: np.ndarray,
    mode: str = "zscore_per_recording",
    eps: float = 1e-8,
) -> np.ndarray:
    """Normalize segmented epochs of shape `(N_epochs, N_channels, Epoch_samples)`.

    Args:
        epochs: Array of shape `(N_epochs, C, T)`.
        mode: `'zscore_per_recording'` (subject-level per-channel zero mean, unit variance),
              `'zscore_per_epoch'` (epoch-level per-channel z-score), or `'none'`.
        eps: Numerical stability constant.

    Returns:
        Normalized float32 array of shape `(N_epochs, C, T)`.
    """
    if mode is None or mode.lower() == "none":
        return epochs.astype(np.float32, copy=False)

    mode_lower = mode.lower()
    if mode_lower == "zscore_per_recording":
        # Compute mean and std per channel across the entire recording (axes 0 and 2)
        mean = epochs.mean(axis=(0, 2), keepdims=True)
        std = epochs.std(axis=(0, 2), keepdims=True)
        return ((epochs - mean) / (std + eps)).astype(np.float32)
    elif mode_lower == "zscore_per_epoch":
        mean = epochs.mean(axis=2, keepdims=True)
        std = epochs.std(axis=2, keepdims=True)
        return ((epochs - mean) / (std + eps)).astype(np.float32)
    else:
        raise ValueError(f"Unsupported normalization mode: {mode}")


def read_edf_channels(
    edf_path: str | Path,
    requested_canonical_channels: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float], float]:
    """Read requested canonical channels from an EDF file.

    Args:
        edf_path: Path to `.edf` file.
        requested_canonical_channels: List of canonical channel names to extract
            (defaults to `DEFAULT_SHHS_CHANNELS`).

    Returns:
        Tuple of:
          - signals_dict: Mapping `canonical_name -> 1D numpy float64 array`
          - fs_dict: Mapping `canonical_name -> sampling frequency (Hz)`
          - duration_sec: Total file duration in seconds
    """
    if requested_canonical_channels is None:
        requested_canonical_channels = DEFAULT_SHHS_CHANNELS

    edf_path = Path(edf_path)
    reader = pyedflib.EdfReader(str(edf_path))
    try:
        raw_labels = reader.getSignalLabels()
        label_to_idx = {lbl.strip(): idx for idx, lbl in enumerate(raw_labels)}
        label_to_idx_upper = {lbl.strip().upper(): idx for idx, lbl in enumerate(raw_labels)}

        signals_dict: dict[str, np.ndarray] = {}
        fs_dict: dict[str, float] = {}
        duration_sec = float(reader.getFileDuration())

        for canon_name in requested_canonical_channels:
            aliases = CHANNEL_ALIASES.get(canon_name, [canon_name])
            matched_idx = None
            for alias in aliases:
                if alias in label_to_idx:
                    matched_idx = label_to_idx[alias]
                    break
                if alias.upper() in label_to_idx_upper:
                    matched_idx = label_to_idx_upper[alias.upper()]
                    break

            if matched_idx is not None:
                signals_dict[canon_name] = reader.readSignal(matched_idx).astype(np.float64)
                fs_dict[canon_name] = float(reader.getSampleFrequency(matched_idx))
    finally:
        reader.close()

    return signals_dict, fs_dict, duration_sec


def preprocess_subject(
    edf_path: str | Path,
    xml_path: str | Path,
    config: dict[str, Any],
    canonical_channels: list[str] | None = None,
    apply_normalization: bool = False,
) -> dict[str, Any]:
    """Run the full standardized preprocessing pipeline for one PSG subject recording.

    Note: By default (`apply_normalization=False`), filtered 100 Hz epochs are kept in
    physical units (uV) so that downstream mathematical channel simulations in
    `channel_sim.py` (such as bipolar subtraction `(F3-M2) - (F4-M2)` or `EOG(L) - EOG(R)`)
    cancel shared physical references accurately before per-recording z-score
    normalization is applied.

    Args:
        edf_path: Path to `.edf` recording.
        xml_path: Path to `*-profusion.xml` annotation file.
        config: Parsed dictionary from `configs/base_config.yaml`.
        canonical_channels: Canonical channels to extract (defaults to `DEFAULT_SHHS_CHANNELS`).
        apply_normalization: If True, normalizes immediately; if False, defers z-scoring to `channel_sim.py`.

    Returns:
        Dictionary with keys:
          - `'subject_id'`: str (e.g. `'shhs1-200001'`)
          - `'epochs'`: float32 ndarray of shape `(N_valid_epochs, N_channels, 3000)`
          - `'labels'`: int64 ndarray of shape `(N_valid_epochs,)` with values in `0..4`
          - `'valid_epoch_indices'`: int64 ndarray of original epoch indices `(N_valid_epochs,)`
          - `'channel_names'`: list[str] of canonical channel names
          - `'fs'`: int (`100`)
          - `'epoch_sec'`: float (`30.0`)
          - `'normalization'`: str (configured normalization mode)
          - `'is_normalized'`: bool
    """
    edf_path = Path(edf_path)
    xml_path = Path(xml_path)
    subject_id = edf_path.stem

    prep_cfg = config.get("preprocessing", {})
    dataset_cfg = config.get("dataset", {})

    target_fs = float(prep_cfg.get("target_fs", 100.0))
    epoch_sec = float(prep_cfg.get("epoch_sec", 30.0))
    samples_per_epoch = int(round(target_fs * epoch_sec))

    bp_cfg = prep_cfg.get("bandpass", {})
    low_hz = float(bp_cfg.get("low_hz", 0.2))
    high_hz = float(bp_cfg.get("high_hz", 49.0))
    order = int(bp_cfg.get("order", 5))
    zero_phase = bool(bp_cfg.get("zero_phase", True))

    notch_cfg = prep_cfg.get("notch", {})
    notch_enabled = bool(notch_cfg.get("enabled", True))
    notch_freq = notch_cfg.get("freq_hz")
    if notch_freq is None:
        notch_freq = float(dataset_cfg.get("mains_freq_hz", 60.0))
    if not notch_enabled:
        notch_freq = None
    notch_q = float(notch_cfg.get("q", 30.0))

    norm_mode = str(prep_cfg.get("normalization", "zscore_per_recording"))
    exclude_unknown = bool(prep_cfg.get("exclude_unknown_movement", True))
    storage_dtype = str(prep_cfg.get("storage_dtype", "float16"))

    if canonical_channels is None:
        cfg_channels = prep_cfg.get("channels")
        canonical_channels = list(cfg_channels) if cfg_channels else DEFAULT_SHHS_CHANNELS

    # 1. Read EDF signals and XML annotations
    signals_dict, fs_dict, _ = read_edf_channels(edf_path, canonical_channels)
    if not signals_dict:
        raise RuntimeError(f"No matching PSG channels found in {edf_path}")

    labels, xml_epoch_sec = parse_profusion_xml(xml_path)
    if not np.isclose(xml_epoch_sec, epoch_sec):
        logger.warning(
            "XML EpochLength (%s s) differs from config epoch_sec (%s s) in %s",
            xml_epoch_sec,
            epoch_sec,
            subject_id,
        )

    # 2. Filter and resample each channel to target_fs (100 Hz)
    channel_names = list(signals_dict.keys())
    processed_channels: list[np.ndarray] = []
    min_num_epochs = len(labels)

    for ch_name in channel_names:
        ch_sig = preprocess_channel(
            raw_signal=signals_dict[ch_name],
            orig_fs=fs_dict[ch_name],
            target_fs=target_fs,
            low_hz=low_hz,
            high_hz=high_hz,
            order=order,
            zero_phase=zero_phase,
            notch_freq=notch_freq,
            notch_q=notch_q,
        )
        n_ep = len(ch_sig) // samples_per_epoch
        min_num_epochs = min(min_num_epochs, n_ep)
        processed_channels.append(ch_sig)

    # 3. Truncate to common epoch count and reshape to (N_epochs, N_channels, samples_per_epoch)
    total_samples = min_num_epochs * samples_per_epoch
    stacked_epochs = np.stack(
        [
            ch[:total_samples].reshape(min_num_epochs, samples_per_epoch)
            for ch in processed_channels
        ],
        axis=1,
    ).astype(np.float32)  # Shape: (N_epochs, N_channels, 3000)
    labels = labels[:min_num_epochs]
    valid_indices = np.arange(min_num_epochs, dtype=np.int64)

    # 4. Exclude unknown/movement epochs if configured
    if exclude_unknown:
        valid_mask = (labels >= 0) & (labels <= 4)
        stacked_epochs = stacked_epochs[valid_mask]
        labels = labels[valid_mask]
        valid_indices = valid_indices[valid_mask]

    # 5. Apply normalization if requested immediately
    if apply_normalization:
        stacked_epochs = normalize_epochs(stacked_epochs, mode=norm_mode)

    return {
        "subject_id": subject_id,
        "epochs": stacked_epochs,
        "labels": labels,
        "valid_epoch_indices": valid_indices,
        "channel_names": channel_names,
        "fs": int(round(target_fs)),
        "epoch_sec": epoch_sec,
        "normalization": norm_mode,
        "is_normalized": apply_normalization,
        "storage_dtype": storage_dtype,
    }


def save_preprocessed_subject(
    data: dict[str, Any],
    out_dir: str | Path,
    storage_dtype: str | None = None,
    compressed: bool = False,
) -> Path:
    """Save a preprocessed subject dictionary to `out_dir/{subject_id}.npz`.

    By default, stores `epochs` in `float16` (`~23 MB` per 4-channel SHHS1 subject) using
    uncompressed `.npz` (`compressed=False`) for near-instant disk read/write speeds across
    large cohorts (`1,771` subjects).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{data['subject_id']}.npz"

    dtype_str = str(storage_dtype or data.get("storage_dtype", "float16")).strip().lower()
    epochs_to_save = (
        data["epochs"].astype(np.float16, copy=False)
        if dtype_str == "float16"
        else data["epochs"].astype(np.float32, copy=False)
    )

    save_fn = np.savez_compressed if compressed else np.savez
    save_fn(
        out_path,
        subject_id=data["subject_id"],
        epochs=epochs_to_save,
        labels=data["labels"],
        valid_epoch_indices=data["valid_epoch_indices"],
        channel_names=np.array(data["channel_names"], dtype=str),
        fs=np.int64(data["fs"]),
        epoch_sec=np.float64(data["epoch_sec"]),
        normalization=str(data.get("normalization", "zscore_per_recording")),
        is_normalized=bool(data.get("is_normalized", False)),
        storage_dtype=dtype_str,
    )
    return out_path


def load_preprocessed_subject(
    npz_path: str | Path,
    epochs_dtype: Any = np.float32,
) -> dict[str, Any]:
    """Load a preprocessed subject `.npz` file from disk (including full `epochs` array)."""
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as archive:
        norm_mode = (
            str(archive["normalization"])
            if "normalization" in archive.files
            else "zscore_per_recording"
        )
        is_norm = (
            bool(archive["is_normalized"])
            if "is_normalized" in archive.files
            else False
        )
        labels = archive["labels"].astype(np.int64)
        return {
            "subject_id": str(archive["subject_id"]),
            "epochs": archive["epochs"].astype(epochs_dtype, copy=False),
            "labels": labels,
            "valid_epoch_indices": archive["valid_epoch_indices"].astype(np.int64),
            "channel_names": [str(x) for x in archive["channel_names"].tolist()],
            "fs": int(archive["fs"]),
            "epoch_sec": float(archive["epoch_sec"]),
            "normalization": norm_mode,
            "is_normalized": is_norm,
            "n_epochs": int(len(labels)),
            "npz_path": str(npz_path),
        }


def load_preprocessed_subject_metadata(npz_path: str | Path) -> dict[str, Any]:
    """Load lightweight subject metadata and labels without reading the multi-MB `epochs` array into RAM.

    Ideal for indexing large cohorts (`1,771` subjects) using `< 15 MB` total RAM.
    `apply_channel_config` and `materialize_subject_record` automatically load `epochs`
    from `npz_path` on demand when needed.
    """
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as archive:
        norm_mode = (
            str(archive["normalization"])
            if "normalization" in archive.files
            else "zscore_per_recording"
        )
        is_norm = (
            bool(archive["is_normalized"])
            if "is_normalized" in archive.files
            else False
        )
        labels = archive["labels"].astype(np.int64)
        valid_indices = (
            archive["valid_epoch_indices"].astype(np.int64)
            if "valid_epoch_indices" in archive.files
            else np.arange(len(labels), dtype=np.int64)
        )
        return {
            "subject_id": str(archive["subject_id"]),
            "epochs": None,
            "labels": labels,
            "valid_epoch_indices": valid_indices,
            "channel_names": [str(x) for x in archive["channel_names"].tolist()],
            "fs": int(archive["fs"]),
            "epoch_sec": float(archive["epoch_sec"]),
            "normalization": norm_mode,
            "is_normalized": is_norm,
            "n_epochs": int(len(labels)),
            "npz_path": str(npz_path),
        }


def materialize_subject_record(
    record: dict[str, Any],
    epochs_dtype: Any = np.float32,
) -> dict[str, Any]:
    """Ensure `record['epochs']` is loaded in memory (returns a new dict if `record` was lazy)."""
    if record.get("epochs") is not None:
        return record
    npz_path = record.get("npz_path")
    if not npz_path:
        raise ValueError(f"Cannot materialize lazy subject record without 'npz_path': {record.get('subject_id')}")
    return load_preprocessed_subject(npz_path, epochs_dtype=epochs_dtype)


def select_shhs_cohort(
    config: dict[str, Any],
    root_dir: str | Path = ".",
    save_csv_path: str | Path | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Filter `Dataset_Information/shhs1-dataset-0.21.0.csv` according to `config['cohort_filter']`.

    Supported `cohort_filter` keys in `configs/base_config.yaml`:
      - `enabled`: bool (if False, returns all subjects in the metadata table)
      - `healthy_only`: bool (if True, requires `ahi_a0h3 < max_ahi` and `rdi3p < max_ahi`)
      - `max_ahi`: float (default `5.0` events/hour for healthy / no sleep apnea per AASM)
      - `min_recording_hours`: float (default `0.0`, filters `timebedp >= min_recording_hours * 60`)
      - `min_usable_eeg_hours`: float (default `0.0`, filters `eeg1dur` and `eeg2dur`)
      - `min_eeg_quality_rating`: int (default `1`, filters `qualeeg1` and `qualeeg2` in `1..4`)
      - `min_eog_quality_rating`: int (default `1`, filters `qualeog` in `1..4`)
      - `exclude_abnormal_eeg`: bool (default `False`, filters `abneeg == 0`)
      - `max_subjects`: int | None (optional cap for debugging)

    Returns:
      Tuple of `(selected_subject_ids, filtered_dataframe)`.
    """
    root_dir = Path(root_dir)
    cf = config.get("cohort_filter", {})
    csv_rel = cf.get("metadata_csv", "Dataset_Information/shhs1-dataset-0.21.0.csv")
    csv_path = root_dir / csv_rel
    if not csv_path.exists():
        raise FileNotFoundError(f"SHHS1 metadata CSV not found at {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    mask = pd.Series(True, index=df.index)

    if bool(cf.get("enabled", True)):
        if bool(cf.get("healthy_only", True)):
            max_ahi = float(cf.get("max_ahi", 5.0))
            ahi_ok = df["ahi_a0h3"].notna() & (df["ahi_a0h3"] < max_ahi)
            if "rdi3p" in df.columns:
                rdi_ok = df["rdi3p"].isna() | (df["rdi3p"] < max_ahi)
                mask &= ahi_ok & rdi_ok
            else:
                mask &= ahi_ok

        min_rec_hours = float(cf.get("min_recording_hours", 0.0) or 0.0)
        if min_rec_hours > 0.0 and "timebedp" in df.columns:
            mask &= df["timebedp"].notna() & (df["timebedp"] >= min_rec_hours * 60.0)

        min_eeg_hours = float(cf.get("min_usable_eeg_hours", 0.0) or 0.0)
        if min_eeg_hours > 0.0 and {"eeg1dur", "eeg2dur"}.issubset(df.columns):
            mask &= (
                df["eeg1dur"].notna()
                & (df["eeg1dur"] >= min_eeg_hours)
                & df["eeg2dur"].notna()
                & (df["eeg2dur"] >= min_eeg_hours)
            )

        min_eeg_qual = int(cf.get("min_eeg_quality_rating", 1) or 1)
        if min_eeg_qual > 1 and {"qualeeg1", "qualeeg2"}.issubset(df.columns):
            mask &= (
                df["qualeeg1"].notna()
                & (df["qualeeg1"] >= min_eeg_qual)
                & df["qualeeg2"].notna()
                & (df["qualeeg2"] >= min_eeg_qual)
            )

        min_eog_qual = int(cf.get("min_eog_quality_rating", 1) or 1)
        if min_eog_qual > 1 and "qualeog" in df.columns:
            mask &= df["qualeog"].notna() & (df["qualeog"] >= min_eog_qual)

        if bool(cf.get("exclude_abnormal_eeg", False)) and "abneeg" in df.columns:
            mask &= df["abneeg"].fillna(0) == 0

    filtered_df = df.loc[mask].copy()
    filtered_df["subject_id"] = filtered_df["nsrrid"].astype(int).map(lambda x: f"shhs1-{x}")
    filtered_df = filtered_df.sort_values("subject_id").reset_index(drop=True)

    max_subjects = cf.get("max_subjects", None)
    if max_subjects is not None:
        filtered_df = filtered_df.iloc[: int(max_subjects)].copy()

    if save_csv_path is None:
        splits_dir = root_dir / config.get("paths", {}).get("splits_dir", "data/splits")
        save_csv_path = splits_dir / "shhs1_selected_cohort.csv"
    else:
        save_csv_path = Path(save_csv_path)

    save_csv_path.parent.mkdir(parents=True, exist_ok=True)
    cols_to_keep = [
        c
        for c in [
            "subject_id",
            "nsrrid",
            "age_s1",
            "gender",
            "ahi_a0h3",
            "rdi3p",
            "overall_shhs1",
            "qualeeg1",
            "qualeeg2",
            "qualeog",
            "qualemg",
            "timebedp",
            "slpprdp",
            "eeg1dur",
            "eeg2dur",
            "abneeg",
        ]
        if c in filtered_df.columns
    ]
    filtered_df[cols_to_keep].to_csv(save_csv_path, index=False)

    return filtered_df["subject_id"].tolist(), filtered_df


def resolve_nsrr_token(
    nsrr_token: str | None = None,
    root_dir: str | Path = ".",
) -> str | None:
    """Resolve the user's NSRR download token from argument, `NSRR_TOKEN` env var, or `.nsrr_token` file."""
    if nsrr_token and str(nsrr_token).strip():
        return str(nsrr_token).strip()
    env_tok = os.environ.get("NSRR_TOKEN", "").strip()
    if env_tok:
        return env_tok
    token_file = Path(root_dir) / ".nsrr_token"
    if token_file.exists():
        content = token_file.read_text(encoding="utf-8").strip()
        if content:
            return content.splitlines()[0].strip()
    return None


def _download_url_to_file(
    url: str,
    dest_path: Path,
    max_retries: int = 3,
    timeout: float = 120.0,
) -> None:
    """Download `url` to `dest_path` atomically via a `.part` temporary file with retries."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "nsrr-py/shhs-sleep-staging-pipeline"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp_path, "wb") as out_f:
                shutil.copyfileobj(resp, out_f, length=1024 * 1024)
            tmp_path.replace(dest_path)
            return
        except urllib.error.HTTPError as e:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            # Do not retry permanent HTTP errors like 404 Not Found or 401/403 Forbidden
            if e.code in (401, 403, 404):
                raise
            last_err = e
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            last_err = e
        if attempt < max_retries:
            time.sleep(2.0 * attempt)
    raise RuntimeError(f"Failed to download {url} after {max_retries} attempts: {last_err}")


def download_shhs_subject_from_nsrr(
    subject_id: str,
    edf_dir: str | Path,
    annotation_dir: str | Path,
    nsrr_token: str,
) -> tuple[Path, Path]:
    """Download `{subject_id}.edf` and `{subject_id}-profusion.xml` from NSRR (`sleepdata.org`)."""
    edf_dir = Path(edf_dir)
    annotation_dir = Path(annotation_dir)
    edf_path = edf_dir / f"{subject_id}.edf"
    xml_path = annotation_dir / f"{subject_id}-profusion.xml"

    base_url = f"https://sleepdata.org/datasets/shhs/files/a/{nsrr_token}/m/nsrr-py/polysomnography"
    xml_url = f"{base_url}/annotations-events-profusion/shhs1/{subject_id}-profusion.xml"
    edf_url = f"{base_url}/edfs/shhs1/{subject_id}.edf"

    if not xml_path.exists() or xml_path.stat().st_size == 0:
        _download_url_to_file(xml_url, xml_path)
    if not edf_path.exists() or edf_path.stat().st_size == 0:
        _download_url_to_file(edf_url, edf_path)

    return edf_path, xml_path


def download_and_preprocess_shhs_cohort(
    config: dict[str, Any],
    subject_ids: list[str] | None = None,
    root_dir: str | Path = ".",
    nsrr_token: str | None = None,
    overwrite: bool = False,
    delete_raw_edf: bool | None = None,
    verbose: bool = True,
) -> tuple[list[Path], list[str]]:
    """Download (if needed) and preprocess the selected SHHS1 cohort one patient at a time.

    Workflow per patient (`subject_id`):
      1. If `data/processed/{subject_id}.npz` already exists and `overwrite=False`, reuse it immediately.
      2. Otherwise, check if `{subject_id}.edf` and `{subject_id}-profusion.xml` exist locally; if not,
         download them from NSRR using the token (`nsrr_token`, `NSRR_TOKEN` env var, or `.nsrr_token` file).
      3. Preprocess the configured channels (`[C4-A1, C3-A2, EOG(L), EOG(R)]`) at 100 Hz and save
         `data/processed/{subject_id}.npz` in `float16` (`~23 MB` per subject).
      4. Delete the raw `.edf` file (`~55 MB`) immediately after `.npz` is saved when
         `delete_raw_edf_after_preprocess` is enabled, keeping total disk usage around `~40 GB` for 1,771 subjects.

    Returns:
      Tuple of `(saved_npz_paths, skipped_subject_ids)`.
    """
    root_dir = Path(root_dir)
    paths_cfg = config.get("paths", {})
    prep_cfg = config.get("preprocessing", {})

    edf_dir = root_dir / paths_cfg.get("edf_dir", "data/raw/edf")
    ann_dir = root_dir / paths_cfg.get("annotation_dir", "data/raw/annotations-events-profusion")
    proc_dir = root_dir / paths_cfg.get("processed_dir", "data/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)

    if subject_ids is None:
        subject_ids, _ = select_shhs_cohort(config, root_dir=root_dir)

    if delete_raw_edf is None:
        delete_raw_edf = bool(prep_cfg.get("delete_raw_edf_after_preprocess", True))

    token = resolve_nsrr_token(nsrr_token=nsrr_token, root_dir=root_dir)

    saved_paths: list[Path] = []
    skipped_subjects: list[str] = []
    n_total = len(subject_ids)
    n_cached = 0
    n_newly_processed = 0
    t_cohort_start = time.time()

    for idx, sid in enumerate(subject_ids, start=1):
        out_path = proc_dir / f"{sid}.npz"
        if out_path.exists() and not overwrite:
            saved_paths.append(out_path)
            n_cached += 1
            continue

        t_subj = time.time()
        edf_path = edf_dir / f"{sid}.edf"
        xml_path = ann_dir / f"{sid}-profusion.xml"

        try:
            if not (edf_path.exists() and xml_path.exists()):
                if not token:
                    raise RuntimeError(
                        f"Raw EDF/XML for {sid} not found locally in {edf_dir} and no NSRR token was provided. "
                        f"Save your NSRR token in '{root_dir / '.nsrr_token'}' or set `NSRR_TOKEN`."
                    )
                edf_path, xml_path = download_shhs_subject_from_nsrr(
                    subject_id=sid,
                    edf_dir=edf_dir,
                    annotation_dir=ann_dir,
                    nsrr_token=token,
                )

            subj_data = preprocess_subject(edf_path, xml_path, config=config)
            saved_path = save_preprocessed_subject(subj_data, proc_dir)
            saved_paths.append(saved_path)
            n_newly_processed += 1

            if delete_raw_edf and edf_path.exists():
                edf_path.unlink(missing_ok=True)

            if verbose and (
                n_newly_processed <= 5
                or n_newly_processed % 10 == 0
                or idx == n_total
            ):
                elapsed_subj = time.time() - t_subj
                total_elapsed = time.time() - t_cohort_start
                avg_per_new = total_elapsed / max(n_newly_processed, 1)
                remaining = max(0, n_total - idx)
                eta_min = (remaining * avg_per_new) / 60.0
                size_mb = saved_path.stat().st_size / (1024 * 1024)
                print(
                    f"  [{idx:4d}/{n_total:4d}] {sid}: {len(subj_data['labels'])} epochs, "
                    f"channels={subj_data['channel_names']} -> {size_mb:.1f} MB ({elapsed_subj:.1f}s) "
                    f"| cached={n_cached}, new={n_newly_processed}, skipped={len(skipped_subjects)} "
                    f"| ETA ~{eta_min:.1f} min"
                )
        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.warning("Subject %s not found on NSRR (HTTP 404); skipping.", sid)
                skipped_subjects.append(sid)
                if verbose:
                    print(f"  [{idx:4d}/{n_total:4d}] {sid}: HTTP 404 (not available on NSRR) -> skipped.")
            elif e.code in (401, 403):
                raise RuntimeError(
                    f"NSRR authentication failed (HTTP {e.code}) for {sid}. "
                    f"Please check that your NSRR token in `.nsrr_token` is current and valid."
                ) from e
            else:
                logger.warning("HTTP error %s for subject %s: %s; skipping.", e.code, sid, e)
                skipped_subjects.append(sid)
        except Exception as e:
            if "no NSRR token was provided" in str(e):
                raise
            logger.warning("Error processing subject %s: %s; skipping.", sid, e)
            skipped_subjects.append(sid)
            if verbose:
                print(f"  [{idx:4d}/{n_total:4d}] {sid}: ERROR ({e}) -> skipped.")

    if verbose:
        print(
            f"Cohort preprocessing summary: {len(saved_paths)}/{n_total} subjects ready "
            f"(cached={n_cached}, newly_processed={n_newly_processed}, skipped={len(skipped_subjects)})."
        )
    return saved_paths, skipped_subjects


def find_shhs_pairs(
    edf_dir: str | Path,
    annotation_dir: str | Path,
) -> list[tuple[Path, Path]]:
    """Match `.edf` files in `edf_dir` with `*-profusion.xml` files in `annotation_dir`."""
    edf_dir = Path(edf_dir)
    annotation_dir = Path(annotation_dir)
    edf_files = sorted(edf_dir.glob("*.edf"))
    pairs: list[tuple[Path, Path]] = []
    for edf_path in edf_files:
        subject_id = edf_path.stem
        xml_candidate = annotation_dir / f"{subject_id}-profusion.xml"
        if not xml_candidate.exists():
            # Fallback search
            matches = list(annotation_dir.glob(f"{subject_id}*.xml"))
            if not matches:
                raise FileNotFoundError(
                    f"Missing Profusion XML annotation for {edf_path.name} in {annotation_dir}"
                )
            xml_candidate = matches[0]
        pairs.append((edf_path, xml_candidate))
    return pairs


def preprocess_all_recordings(
    config: dict[str, Any],
    root_dir: str | Path = ".",
    overwrite: bool = False,
    limit_subjects: int | None = None,
) -> list[Path]:
    """Preprocess all EDF/XML pairs found in `paths.edf_dir` and cache `.npz` files in `paths.processed_dir`."""
    root_dir = Path(root_dir)
    paths_cfg = config.get("paths", {})
    edf_dir = root_dir / paths_cfg.get("edf_dir", "data/raw/edf")
    ann_dir = root_dir / paths_cfg.get("annotation_dir", "data/raw/annotations-events-profusion")
    proc_dir = root_dir / paths_cfg.get("processed_dir", "data/processed")

    pairs = find_shhs_pairs(edf_dir, ann_dir)
    if limit_subjects is not None:
        pairs = pairs[:limit_subjects]

    saved_paths: list[Path] = []
    for edf_path, xml_path in pairs:
        subject_id = edf_path.stem
        out_path = proc_dir / f"{subject_id}.npz"
        if out_path.exists() and not overwrite:
            saved_paths.append(out_path)
            continue
        subj_data = preprocess_subject(edf_path, xml_path, config=config)
        saved_path = save_preprocessed_subject(subj_data, proc_dir)
        saved_paths.append(saved_path)
    return saved_paths
