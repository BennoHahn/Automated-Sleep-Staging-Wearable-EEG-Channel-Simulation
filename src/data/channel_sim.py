"""Step 2: Channel Ablation & Mathematical Wearable Simulation (`src/data/channel_sim.py`).

Implements both approaches to simulating wearable EEG setups from clinical PSG:
1. Option 1 (Location-Close PSG Proxies):
   - Single-lead mastoid-referenced EEG (`C4-A1`, `C3-A2`, `F3-M2`)
   - Periorbital / temple EOG proxies (`EOG(L)`, `EOG(L) + EOG(R)`) near `AF7/AF8/Fp1/Fp2`
     (Heremans et al., 2022; Huang et al., 2014).
2. Option 2 (Mathematical Channel Simulations via Re-referencing & Linear Combinations):
   - Midline averaging with mastoid reference:
     `Sim_Fz_M2 = (F3-M2 + F4-M2) / 2` (van der Aar et al., 2025)
     `Sim_EOG_Avg = (EOG(L) + EOG(R)) / 2` (Heremans et al., 2022)
     `Sim_EEG_Avg = ((C3-A2) + (C4-A1)) / 2`
   - Cross-head / cross-forehead bipolar subtraction without mastoid reference:
     `Sim_F3_F4 = (F3-M2) - (F4-M2)` (van der Aar et al., 2024; Dreem 2021)
     `Sim_EOG_Bipolar = EOG(L) - EOG(R)` (Heremans et al., 2022)
     `Sim_EEG_Bipolar = (C3-A2) - (C4-A1)`
   - Fronto-occipital / fronto-central differential without mastoid reference:
     `Sim_F3_O1 = (F3-M2) - (O1-M2)` (Dreem, Guillot & Thorey 2021; Markov et al. 2025)
     `Sim_C4_EOG = (C4-A1) - EOG(L)` (Supratak et al. 2017; Supratak & Guo 2020)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.data.preprocessing import normalize_epochs


def load_ablation_configs(
    yaml_path: str | Path = "configs/ablation_channels.yaml",
    section: str = "shhs_configs",
) -> dict[str, dict[str, Any]]:
    """Load channel simulation recipes from `configs/ablation_channels.yaml`.

    Args:
        yaml_path: Path to YAML config file.
        section: `'shhs_configs'`, `'full_10_20_configs'`, or `'all'`.

    Returns:
        Dictionary mapping `config_name -> config_spec_dict`.
    """
    with open(Path(yaml_path), "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if section == "all":
        merged: dict[str, dict[str, Any]] = {}
        merged.update(raw.get("shhs_configs", {}))
        merged.update(raw.get("full_10_20_configs", {}))
        return merged

    if section in raw:
        return raw[section]

    # Support legacy flat `configs:` key if present
    if "configs" in raw:
        return raw["configs"]

    raise KeyError(f"Section '{section}' not found in {yaml_path}")


def compute_channel_operation(
    epochs: np.ndarray,
    available_channels: list[str],
    op: str,
    inputs: list[str],
    weights: list[float] | None = None,
) -> np.ndarray:
    """Compute a single derived channel `(N_epochs, 1, T)` from `epochs` `(N_epochs, C, T)`.

    Args:
        epochs: Array of shape `(N_epochs, C, T)` in physical units (uV).
        available_channels: List of channel names corresponding to axis 1 of `epochs`.
        op: Operation type: `'identity'`, `'mean'`, `'diff'`, or `'linear'`.
        inputs: List of input channel names required by the operation.
        weights: Optional explicit weights when `op == 'linear'`.

    Returns:
        Derived channel array of shape `(N_epochs, 1, T)` in float32.
    """
    ch_to_idx = {name: i for i, name in enumerate(available_channels)}
    for req in inputs:
        if req not in ch_to_idx:
            raise KeyError(
                f"Required input channel '{req}' is not in available channels {available_channels}."
            )

    op_lower = op.lower()
    if op_lower == "identity":
        if len(inputs) != 1:
            raise ValueError(f"'identity' op requires exactly 1 input channel, got {inputs}")
        idx = ch_to_idx[inputs[0]]
        return epochs[:, idx : idx + 1, :].astype(np.float32, copy=False)

    elif op_lower == "mean":
        if len(inputs) < 2:
            raise ValueError(f"'mean' op requires at least 2 input channels, got {inputs}")
        indices = [ch_to_idx[ch] for ch in inputs]
        avg = np.mean(epochs[:, indices, :], axis=1, keepdims=True)
        return avg.astype(np.float32)

    elif op_lower == "diff":
        if len(inputs) != 2:
            raise ValueError(f"'diff' op requires exactly 2 input channels [ch1, ch2], got {inputs}")
        idx1, idx2 = ch_to_idx[inputs[0]], ch_to_idx[inputs[1]]
        diff = epochs[:, idx1 : idx1 + 1, :] - epochs[:, idx2 : idx2 + 1, :]
        return diff.astype(np.float32)

    elif op_lower == "linear":
        if weights is None or len(weights) != len(inputs):
            raise ValueError(f"'linear' op requires `weights` of same length as `inputs` ({inputs})")
        out = np.zeros((epochs.shape[0], 1, epochs.shape[2]), dtype=np.float32)
        for ch_name, w in zip(inputs, weights):
            idx = ch_to_idx[ch_name]
            out += float(w) * epochs[:, idx : idx + 1, :]
        return out

    else:
        raise ValueError(f"Unsupported channel simulation operation: '{op}'")


def is_channel_config_compatible(
    available_channels: list[str],
    channel_config: dict[str, Any],
) -> bool:
    """Return True if all input channels required by `channel_config` are present in `available_channels`."""
    avail_set = set(available_channels)
    for spec in channel_config.get("channels", []):
        if isinstance(spec, str):
            if spec not in avail_set:
                return False
        elif isinstance(spec, dict):
            for inp in spec.get("inputs", []):
                if inp not in avail_set:
                    return False
    return True


def apply_channel_config(
    subject_data: dict[str, Any],
    channel_config: dict[str, Any],
    normalize: bool = True,
    norm_mode: str | None = None,
    out_dtype: Any = np.float32,
) -> tuple[np.ndarray, list[str]]:
    """Extract or mathematically simulate the channels defined in `channel_config` for one subject.

    Crucially, linear operations (`diff`, `mean`, `linear`) are evaluated on the unnormalized
    physical signals (`uV`) so that shared reference electrodes cancel out properly before
    per-recording z-score normalization is applied to the resulting channel(s).

    Supports both eager records (`subject_data['epochs']` in RAM) and lazy records
    (`subject_data['epochs'] is None`, loaded on demand from `subject_data['npz_path']`).

    Args:
        subject_data: Dictionary returned by `preprocess_subject`, `load_preprocessed_subject`,
            or `load_preprocessed_subject_metadata`.
        channel_config: Single configuration dictionary from `ablation_channels.yaml`
            containing a `'channels'` list of channel specs (strings or dicts with `name, op, inputs`).
        normalize: Whether to normalize the resulting derived channels.
        norm_mode: Override normalization mode (defaults to `subject_data.get('normalization', 'zscore_per_recording')`).
        out_dtype: Output numpy dtype (default `np.float32`; use `np.float16` for memory-efficient large cohorts).

    Returns:
        Tuple of:
          - derived_epochs: array of shape `(N_epochs, N_out_channels, T)` with dtype `out_dtype`
          - derived_names: list[str] of output channel names
    """
    epochs = subject_data.get("epochs")
    if epochs is None:
        npz_path = subject_data.get("npz_path")
        if not npz_path:
            raise ValueError(
                f"Subject record '{subject_data.get('subject_id')}' has neither 'epochs' nor 'npz_path'."
            )
        with np.load(Path(npz_path), allow_pickle=False) as archive:
            epochs = archive["epochs"].astype(np.float32, copy=False)
    else:
        epochs = np.asarray(epochs, dtype=np.float32)

    available_channels = list(subject_data["channel_names"])
    already_normalized = bool(subject_data.get("is_normalized", False))

    if norm_mode is None:
        norm_mode = str(subject_data.get("normalization", "zscore_per_recording"))

    channel_specs = channel_config.get("channels", [])
    if not channel_specs:
        raise ValueError(f"Channel configuration has no 'channels' list: {channel_config}")

    derived_list: list[np.ndarray] = []
    derived_names: list[str] = []

    for spec in channel_specs:
        if isinstance(spec, str):
            ch_arr = compute_channel_operation(
                epochs=epochs,
                available_channels=available_channels,
                op="identity",
                inputs=[spec],
            )
            derived_list.append(ch_arr)
            derived_names.append(spec)
        elif isinstance(spec, dict):
            name = str(spec.get("name", "sim_ch"))
            op = str(spec.get("op", "identity"))
            inputs = list(spec.get("inputs", []))
            weights = spec.get("weights")
            ch_arr = compute_channel_operation(
                epochs=epochs,
                available_channels=available_channels,
                op=op,
                inputs=inputs,
                weights=weights,
            )
            derived_list.append(ch_arr)
            derived_names.append(name)
        else:
            raise TypeError(f"Invalid channel specification: {spec}")

    derived_epochs = np.concatenate(derived_list, axis=1)  # (N_epochs, C_out, T)

    # Apply normalization if requested (even if raw channels were already normalized,
    # subtracting or averaging two channels changes their variance, so re-normalizing
    # ensures zero mean and unit variance per derived channel).
    if normalize and norm_mode.lower() != "none":
        if not already_normalized or any(
            isinstance(s, dict) and s.get("op", "identity") != "identity" for s in channel_specs
        ):
            derived_epochs = normalize_epochs(derived_epochs, mode=norm_mode)

    return derived_epochs.astype(out_dtype, copy=False), derived_names


def evaluate_simulation_fidelity(
    sim_epochs: np.ndarray,
    real_wearable_epochs: np.ndarray,
) -> dict[str, float]:
    """Compare mathematically simulated PSG channels against simultaneously recorded wearable channels.

    Designed for Phase 2 validation (e.g., Bitbrain BOAS `ds005555` comparing PSG-simulated
    `Sim_F3_F4` or `Sim_Fz_M2` against real wearable `AF7-AF8` or `AF7`).

    Args:
        sim_epochs: Array of shape `(N_epochs, 1, T)` or `(N_epochs, T)`.
        real_wearable_epochs: Array of same shape as `sim_epochs`.

    Returns:
        Dictionary with mean epoch-wise Pearson `r`, `abs_r`, and `rmse`.
    """
    s = sim_epochs.reshape(sim_epochs.shape[0], -1).astype(np.float64)
    w = real_wearable_epochs.reshape(real_wearable_epochs.shape[0], -1).astype(np.float64)

    s_centered = s - s.mean(axis=1, keepdims=True)
    w_centered = w - w.mean(axis=1, keepdims=True)
    num = np.sum(s_centered * w_centered, axis=1)
    den = np.sqrt(np.sum(s_centered**2, axis=1) * np.sum(w_centered**2, axis=1)) + 1e-12
    r_per_epoch = num / den

    rmse = float(np.sqrt(np.mean((s - w) ** 2)))
    return {
        "mean_pearson_r": float(np.mean(r_per_epoch)),
        "mean_abs_pearson_r": float(np.mean(np.abs(r_per_epoch))),
        "rmse": rmse,
    }
