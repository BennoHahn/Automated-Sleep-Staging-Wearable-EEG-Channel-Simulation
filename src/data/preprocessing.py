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
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
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

# Standard SHHS1 target channels
DEFAULT_SHHS_CHANNELS: list[str] = ["C4-A1", "C3-A2", "EOG(L)", "EOG(R)", "EMG"]

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
    }


def save_preprocessed_subject(data: dict[str, Any], out_dir: str | Path) -> Path:
    """Save a preprocessed subject dictionary to `out_dir/{subject_id}.npz`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{data['subject_id']}.npz"
    np.savez_compressed(
        out_path,
        subject_id=data["subject_id"],
        epochs=data["epochs"],
        labels=data["labels"],
        valid_epoch_indices=data["valid_epoch_indices"],
        channel_names=np.array(data["channel_names"], dtype=str),
        fs=np.int64(data["fs"]),
        epoch_sec=np.float64(data["epoch_sec"]),
        normalization=str(data.get("normalization", "zscore_per_recording")),
        is_normalized=bool(data.get("is_normalized", False)),
    )
    return out_path


def load_preprocessed_subject(npz_path: str | Path) -> dict[str, Any]:
    """Load a preprocessed subject `.npz` file from disk."""
    with np.load(Path(npz_path), allow_pickle=False) as archive:
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
        return {
            "subject_id": str(archive["subject_id"]),
            "epochs": archive["epochs"].astype(np.float32),
            "labels": archive["labels"].astype(np.int64),
            "valid_epoch_indices": archive["valid_epoch_indices"].astype(np.int64),
            "channel_names": [str(x) for x in archive["channel_names"].tolist()],
            "fs": int(archive["fs"]),
            "epoch_sec": float(archive["epoch_sec"]),
            "normalization": norm_mode,
            "is_normalized": is_norm,
        }


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
