"""End-to-End Pipeline Runner for Dreem Open Datasets (DOD-H + DOD-O).

Supports Zenodo download (`record 15900394`), `.h5` -> `.npz` preprocessing, cohort-stratified
10-fold CV split generation, TinySleepNet training across singular EEG channel configurations
(all 6 shared DOD-H+DOD-O EEG channels at N=80, plus optional 6 DOD-H-only frontopolar channels
at N=25 and 2 DOD-O-only occipital channels at N=55), and standardized evaluation/plotting.

Usage examples:
    python run_dreem_training.py --dry-run
    python run_dreem_training.py --download --preprocess-only
    python run_dreem_training.py --download --channels shared
    python run_dreem_training.py --channels all
    python run_dreem_training.py --channels dod_f3_f4,dod_c3_m2
    python run_dreem_training.py --eval-only --channels all
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

# Enable expandable CUDA memory segments to mitigate fragmentation on GPU
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
import pandas as pd
import yaml

from src.data.channel_sim import is_channel_config_compatible, load_ablation_configs
from src.data.dataset import load_subject_records_from_dir
from src.data.dreem import (
    DREEM_ALL_EEG_CHANNELS,
    DODH_ONLY_EEG_CHANNELS,
    DODO_ONLY_EEG_CHANNELS,
    DREEM_SHARED_EEG_CHANNELS,
    _format_invalid_h5_error,
    _is_ignored_h5_artifact,
    download_dreem_zenodo,
    ensure_dreem_cv_splits,
    filter_splits_for_subjects,
    inspect_dreem_raw_files,
    preprocess_all_dreem_recordings,
)
from src.evaluation.clinical_sleep import (
    plot_bland_altman_grid,
    plot_hypnogram_comparison,
)
from src.evaluation.metrics import (
    compute_all_epoch_metrics_from_cm,
    compute_confusion_matrix,
    metrics_summary_to_dataframe,
    plot_normalized_confusion_matrix,
)
from src.training.trainer import (
    get_device_description,
    get_experiment_results_dir,
    load_saved_ablation_results,
    resolve_device,
    run_cross_validation_for_config,
)


def _create_synthetic_dreem_h5_dataset(
    raw_dir: Path,
    n_dodh: int = 2,
    n_dodo: int = 2,
    n_epochs: int = 45,
    fs: float = 250.0,
) -> None:
    """Create tiny valid DOD-H + DOD-O HDF5 files in `raw_dir` for offline `--dry-run` checks."""
    rng = np.random.default_rng(1234)
    n_samples = int(round(n_epochs * 30.0 * fs))
    t = np.arange(n_samples, dtype=np.float64) / fs

    def _write_h5(path: Path, channels: tuple[str, ...], mains_hz: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        hyp = np.tile(np.array([0, 1, 2, 3, 4], dtype=np.int64), n_epochs // 5 + 1)[:n_epochs]
        # Inject one unscored sentinel epoch to test -1 filtering
        if len(hyp) > 6:
            hyp[-1] = -1
        with h5py.File(path, "w") as f:
            sig_grp = f.create_group("signals")
            eeg_grp = sig_grp.create_group("eeg")
            eeg_grp.attrs["fs"] = fs
            eeg_grp.attrs["unit"] = "uV"
            for idx, ch in enumerate(channels):
                h5_key = ch.replace("-", "_")
                base = 25.0 * np.sin(2.0 * np.pi * (8.0 + 0.5 * idx) * t)
                mains = 5.0 * np.sin(2.0 * np.pi * mains_hz * t)
                noise = rng.normal(0.0, 5.0, size=n_samples)
                ds = eeg_grp.create_dataset(h5_key, data=(base + mains + noise).astype(np.float32))
                ds.attrs["fs"] = fs
                ds.attrs["unit"] = "uV"
            f.create_dataset("hypnogram", data=hyp)

    dodh_channels = (*DREEM_SHARED_EEG_CHANNELS, *DODH_ONLY_EEG_CHANNELS)
    dodo_channels = (*DREEM_SHARED_EEG_CHANNELS, *DODO_ONLY_EEG_CHANNELS)

    for i in range(n_dodh):
        _write_h5(raw_dir / "dod-h" / f"synth_h_{i:02d}.h5", dodh_channels, mains_hz=50.0)
    for i in range(n_dodo):
        _write_h5(raw_dir / "dod-o" / f"synth_o_{i:02d}.h5", dodo_channels, mains_hz=60.0)


def select_dreem_channel_configs(
    all_dreem_configs: dict[str, dict[str, Any]],
    selector: str | None,
) -> dict[str, dict[str, Any]]:
    """Filter `dreem_configs` by scope (`all`, `shared`, `dodh`, `dodo`) or explicit names."""
    if selector is None or selector.strip().lower() == "all":
        return dict(all_dreem_configs)

    sel_lower = selector.strip().lower()
    scope_aliases = {
        "shared": "shared",
        "dodh": "dodh_only",
        "dod-h": "dodh_only",
        "dodh_only": "dodh_only",
        "dodo": "dodo_only",
        "dod-o": "dodo_only",
        "dodo_only": "dodo_only",
    }
    if sel_lower in scope_aliases:
        target_scope = scope_aliases[sel_lower]
        selected = {
            k: v for k, v in all_dreem_configs.items() if v.get("cohort_scope") == target_scope
        }
        if not selected:
            raise ValueError(f"No configs matched scope '{selector}'.")
        return selected

    requested_tokens = [tok.strip() for tok in selector.split(",") if tok.strip()]
    selected: dict[str, dict[str, Any]] = {}
    for tok in requested_tokens:
        if tok in all_dreem_configs:
            selected[tok] = all_dreem_configs[tok]
            continue
        # Also allow canonical EEG channel names (e.g. 'F3-F4', 'C3_M2')
        tok_canon = tok.upper().replace("_", "-")
        matched = [
            (k, v)
            for k, v in all_dreem_configs.items()
            if any(
                str(ch.get("name", "")).upper().replace("_", "-") == tok_canon
                for ch in v.get("channels", [])
            )
        ]
        if not matched:
            available = ", ".join(all_dreem_configs.keys())
            raise ValueError(
                f"Unknown Dreem config/channel '{tok}'. Available configs: {available}"
            )
        for k, v in matched:
            selected[k] = v
    return selected


def _compute_cohort_subgroup_rows(
    ablation_results: dict[str, dict[str, Any]],
    selected_configs: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Compute pooled DOD-H (healthy) vs DOD-O (OSA) metrics from test hypnograms per config."""
    rows: list[dict[str, Any]] = []
    for cfg_name, bundle in ablation_results.items():
        if cfg_name not in selected_configs:
            continue
        per_subj: list[dict[str, Any]] = []
        for fold_res in bundle.get("folds", []):
            per_subj.extend(fold_res.get("test_per_subject", []))

        for subgroup_label, prefix in (("all", ""), ("dod-h", "dodh_"), ("dod-o", "dodo_")):
            sub = [
                rec
                for rec in per_subj
                if not prefix or str(rec.get("subject_id", "")).startswith(prefix)
            ]
            if not sub:
                continue
            y_true = np.concatenate([np.asarray(r["y_true"], dtype=np.int64) for r in sub])
            y_pred = np.concatenate([np.asarray(r["y_pred"], dtype=np.int64) for r in sub])
            cm = compute_confusion_matrix(y_true, y_pred, n_classes=5)
            ep_metrics = compute_all_epoch_metrics_from_cm(cm)
            rows.append(
                {
                    "Config": cfg_name,
                    "Subgroup": subgroup_label,
                    "N_Subjects": len(sub),
                    "N_Epochs": ep_metrics["n_epochs"],
                    "Pooled_ACC": round(float(ep_metrics["accuracy"]), 4),
                    "Pooled_Kappa": round(float(ep_metrics["kappa"]), 4),
                    "Pooled_MCC": round(float(ep_metrics["mcc"]), 4),
                    "Pooled_Macro_F1": round(float(ep_metrics["macro_f1"]), 4),
                    "F1_N1": round(float(ep_metrics["per_class"]["N1"]["F1"]), 4),
                    "F1_REM": round(float(ep_metrics["per_class"]["REM"]["F1"]), 4),
                }
            )
    return pd.DataFrame(rows)


def _export_dreem_evaluation_artifacts(
    results_dir: Path,
    selected_configs: dict[str, dict[str, Any]],
    ablation_results: dict[str, dict[str, Any]],
    ablation_summaries: dict[str, dict[str, Any]],
    compat_counts: dict[str, tuple[int, int, int]],
    epoch_sec: float = 30.0,
) -> pd.DataFrame:
    """Save ablation summary CSV, subgroup breakdown CSV, confusion matrices, and clinical plots."""
    df = metrics_summary_to_dataframe(ablation_summaries)
    if not df.empty:
        df.insert(
            1,
            "Cohort_Scope",
            [selected_configs.get(c, {}).get("cohort_scope", "") for c in df["Config"]],
        )
        df.insert(
            2,
            "Channel",
            [
                "+".join(
                    str(ch.get("name", "")) for ch in selected_configs.get(c, {}).get("channels", [])
                )
                for c in df["Config"]
            ],
        )
        df.insert(
            3,
            "N_Subjects",
            [compat_counts.get(c, (0, 0, 0))[0] for c in df["Config"]],
        )
        df.to_csv(results_dir / "ablation_summary_10fold.csv", index=False)

    sub_df = _compute_cohort_subgroup_rows(ablation_results, selected_configs)
    if not sub_df.empty:
        sub_df.to_csv(results_dir / "subgroup_summary_10fold.csv", index=False)

    import matplotlib.pyplot as plt

    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for cfg_name, bundle in ablation_results.items():
        summ = bundle.get("aggregated_metrics", {})
        if not summ:
            continue
        mean_cm = np.asarray(summ["mean_confusion_matrix_norm"], dtype=np.float64)
        std_cm = np.asarray(summ["std_confusion_matrix_norm"], dtype=np.float64)
        cm_fig = plot_normalized_confusion_matrix(
            mean_norm_cm=mean_cm,
            std_norm_cm=std_cm,
            title=f"Dreem DOD Confusion Matrix — {cfg_name}",
            save_path=fig_dir / f"cm_{cfg_name}.png",
        )
        plt.close(cm_fig)

        pooled_subj: list[dict[str, Any]] = []
        for fd in bundle.get("folds", []):
            pooled_subj.extend(fd.get("test_per_subject", []))
        if pooled_subj:
            ba_fig = plot_bland_altman_grid(
                pooled_subj,
                epoch_sec=epoch_sec,
                title=f"Dreem DOD Bland-Altman — {cfg_name}",
                save_path=fig_dir / f"bland_altman_{cfg_name}.png",
            )
            plt.close(ba_fig)
            first_sub = pooled_subj[0]
            hyp_fig = plot_hypnogram_comparison(
                y_true=first_sub["y_true"],
                y_pred=first_sub["y_pred"],
                subject_id=str(first_sub.get("subject_id", "Subject")),
                config_name=cfg_name,
                epoch_sec=epoch_sec,
                save_path=fig_dir / f"hypnogram_{cfg_name}.png",
            )
            plt.close(hyp_fig)

    return df


def _resolve_storage_paths(
    base_cfg: dict[str, Any],
    temp_root: Path | None,
) -> tuple[Path, Path, Path, Path, Path]:
    paths = base_cfg.get("paths", {})
    if temp_root is not None:
        raw_dir = temp_root / "raw"
        processed_dir = temp_root / "processed"
        splits_path = temp_root / "splits" / paths.get("splits_filename", "dreem_dod_10fold_splits.json")
        results_base = temp_root / "results"
        return raw_dir, processed_dir, splits_path, results_base, temp_root

    raw_dir = PROJECT_ROOT / paths.get("raw_dir", "data/raw/dreem")
    processed_dir = PROJECT_ROOT / paths.get("processed_dir", "data/processed/dreem_dod")
    splits_path = (
        PROJECT_ROOT
        / paths.get("splits_dir", "data/splits")
        / paths.get("splits_filename", "dreem_dod_10fold_splits.json")
    )
    results_base = PROJECT_ROOT / paths.get("results_dir", "results")
    return raw_dir, processed_dir, splits_path, results_base, PROJECT_ROOT


def run_dreem_pipeline(args: argparse.Namespace) -> tuple[Path, pd.DataFrame]:
    config_path = PROJECT_ROOT / args.config
    ablation_path = PROJECT_ROOT / args.ablation_config

    with open(config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    all_dreem_cfgs = load_ablation_configs(ablation_path, section="dreem_configs")
    selected_cfgs = select_dreem_channel_configs(all_dreem_cfgs, args.channels)

    use_context_override = False if args.no_context else None
    max_epochs_override = args.max_epochs
    max_folds_override = args.max_folds
    dry_cfg = base_cfg.get("training", {}).get("dry_run", {})

    raw_real = PROJECT_ROOT / base_cfg.get("paths", {}).get("raw_dir", "data/raw/dreem")
    proc_real = PROJECT_ROOT / base_cfg.get("paths", {}).get("processed_dir", "data/processed/dreem_dod")
    has_real_proc = proc_real.exists() and any(proc_real.glob("*.npz"))
    has_real_raw = raw_real.exists() and (
        any(not _is_ignored_h5_artifact(p) for p in raw_real.rglob("*.h5"))
        or any(raw_real.glob("*.zip"))
    )

    temp_ctx: tempfile.TemporaryDirectory[str] | None = None
    if args.dry_run and not args.download and not has_real_proc and not has_real_raw:
        # Offline synthetic dry-run in an isolated temp directory so data/ stays spotless
        temp_ctx = tempfile.TemporaryDirectory(prefix="dreem_dryrun_")
        temp_root = Path(temp_ctx.name)
        _create_synthetic_dreem_h5_dataset(
            raw_dir=temp_root / "raw",
            n_dodh=max(3, int(dry_cfg.get("n_subjects", 4)) // 2 + 1),
            n_dodo=max(3, int(dry_cfg.get("n_subjects", 4)) // 2 + 1),
            n_epochs=45,
        )
        base_cfg = dict(base_cfg)
        base_cfg["paths"] = {
            **base_cfg.get("paths", {}),
            "raw_dir": str(temp_root / "raw"),
            "dodh_dir": str(temp_root / "raw" / "dod-h"),
            "dodo_dir": str(temp_root / "raw" / "dod-o"),
            "processed_dir": str(temp_root / "processed"),
            "splits_dir": str(temp_root / "splits"),
            "results_dir": str(temp_root / "results"),
        }
        raw_dir, processed_dir, splits_path, _, cfg_root = _resolve_storage_paths(base_cfg, temp_root)
    else:
        raw_dir, processed_dir, splits_path, _, cfg_root = _resolve_storage_paths(base_cfg, None)

    try:
        if args.dry_run:
            if max_epochs_override is None:
                max_epochs_override = int(dry_cfg.get("epochs", 2))
            if max_folds_override is None:
                max_folds_override = 2
            print(
                f"[DRY-RUN] epochs={max_epochs_override}, folds={max_folds_override}"
                + (" (synthetic offline DOD-H + DOD-O HDF5)" if temp_ctx else " (real dataset)")
            )

        if args.download:
            print(f"[Step 1] Downloading Dreem DOD-H + DOD-O from Zenodo into {raw_dir}...")
            download_dreem_zenodo(
                raw_dir=raw_dir,
                record_id=str(base_cfg.get("dataset", {}).get("zenodo_record_id", "15900394")),
                cohorts=tuple(base_cfg.get("dataset", {}).get("cohorts", ["dod-h", "dod-o"])),
                force=getattr(args, "force_download", False),
            )

        if getattr(args, "check_raw", False):
            report = inspect_dreem_raw_files(
                raw_dir=raw_dir,
                dodh_dir=raw_dir / "dod-h",
                dodo_dir=raw_dir / "dod-o",
                excluded_subjects=base_cfg.get("dataset", {}).get("excluded_subjects"),
            )
            valid = report["valid_entries"]
            n_h = sum(1 for e in valid if e["cohort"] == "dod-h")
            n_o = sum(1 for e in valid if e["cohort"] == "dod-o")
            print(
                f"[Check Raw] valid_hdf5={len(valid)} (dod-h={n_h}, dod-o={n_o}), "
                f"invalid={len(report['invalid_files'])}, "
                f"appledouble_sidecars_skipped={report['sidecars_skipped']}"
            )
            if report["invalid_files"]:
                total = len(valid) + len(report["invalid_files"])
                print(_format_invalid_h5_error(report["invalid_files"], total, raw_dir))
            return raw_dir, pd.DataFrame()

        prep_needed = args.preprocess_only or args.overwrite_prep or not (
            processed_dir.exists() and any(processed_dir.glob("*.npz"))
        )
        if not args.eval_only and prep_needed:
            print(f"[Step 2] Preprocessing .h5 recordings into {processed_dir}...")
            prep_limit = args.limit
            if args.dry_run and prep_limit is None and temp_ctx is None:
                prep_limit = int(dry_cfg.get("n_subjects", 4))
            preprocess_all_dreem_recordings(
                config={
                    **base_cfg,
                    "paths": {
                        **base_cfg.get("paths", {}),
                        "raw_dir": str(raw_dir),
                        "dodh_dir": str(raw_dir / "dod-h"),
                        "dodo_dir": str(raw_dir / "dod-o"),
                        "processed_dir": str(processed_dir),
                    },
                },
                overwrite=args.overwrite_prep,
                limit=prep_limit,
                verbose=True,
            )

        if args.preprocess_only:
            npz_count = len(list(processed_dir.glob("*.npz")))
            print(f"[Done] Preprocessing complete: {npz_count} .npz files in {processed_dir}.")
            return processed_dir, pd.DataFrame()

        # Load preprocessed records without SHHS 4-channel filter so DOD-H + DOD-O stay intact
        all_records = load_subject_records_from_dir(
            processed_dir,
            lazy=False,
            required_channels=None,
        )
        if args.limit is not None and len(all_records) > args.limit:
            all_records = all_records[: args.limit]
        if not all_records:
            raise FileNotFoundError(f"No preprocessed .npz files found in {processed_dir}.")

        subject_ids = [str(r["subject_id"]) for r in all_records]
        n_dodh = sum(1 for s in subject_ids if s.startswith("dodh_"))
        n_dodo = sum(1 for s in subject_ids if s.startswith("dodo_"))
        print(
            f"[Step 3] Loaded {len(all_records)} Dreem subjects "
            f"(DOD-H={n_dodh}, DOD-O={n_dodo}) from {processed_dir}"
        )

        cv_cfg = base_cfg.get("cv", {})
        seed = int(base_cfg.get("seed", 42))
        n_folds = int(cv_cfg.get("n_folds", 10))
        master_splits = ensure_dreem_cv_splits(
            subject_ids=subject_ids,
            split_path=splits_path,
            n_folds=n_folds,
            seed=seed,
        )

        results_dir = get_experiment_results_dir(
            root_dir=cfg_root,
            base_config=base_cfg,
            n_subjects=len(all_records),
            use_context=use_context_override,
            class_weight_mode=args.class_weights,
            batch_size=args.batch_size,
            max_epochs=max_epochs_override,
            early_stopping_patience=args.early_stopping_patience,
        )
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Step 4] Results directory: {results_dir}")
        print(f"[Device] {get_device_description(resolve_device(base_cfg.get('training', {}).get('device', 'auto')))}")

        compat_counts: dict[str, tuple[int, int, int]] = {}
        ablation_results: dict[str, dict[str, Any]] = {}
        ablation_summaries: dict[str, dict[str, Any]] = {}

        if args.eval_only:
            ablation_results, ablation_summaries = load_saved_ablation_results(
                results_dir=results_dir,
                config_names=list(selected_cfgs.keys()),
            )
            for cfg_name, cfg_spec in selected_cfgs.items():
                compat = [r for r in all_records if is_channel_config_compatible(r["channel_names"], cfg_spec)]
                compat_counts[cfg_name] = (
                    len(compat),
                    sum(1 for r in compat if str(r["subject_id"]).startswith("dodh_")),
                    sum(1 for r in compat if str(r["subject_id"]).startswith("dodo_")),
                )
        else:
            for idx, (cfg_name, cfg_spec) in enumerate(selected_cfgs.items(), start=1):
                compat_recs = [
                    r for r in all_records if is_channel_config_compatible(r["channel_names"], cfg_spec)
                ]
                if len(compat_recs) < 3:
                    print(
                        f"  [Skip] {cfg_name}: only {len(compat_recs)} compatible subjects (need >= 3)."
                    )
                    continue

                compat_ids = {str(r["subject_id"]) for r in compat_recs}
                c_h = sum(1 for sid in compat_ids if sid.startswith("dodh_"))
                c_o = sum(1 for sid in compat_ids if sid.startswith("dodo_"))
                compat_counts[cfg_name] = (len(compat_recs), c_h, c_o)

                cfg_splits = filter_splits_for_subjects(master_splits, compat_ids)
                if not cfg_splits:
                    # Fallback for small dry-run/limited subsets where subset count < n_folds
                    from src.data.dreem import generate_stratified_dreem_cv_splits

                    cfg_splits = generate_stratified_dreem_cv_splits(
                        subject_ids=sorted(compat_ids),
                        n_folds=min(n_folds, len(compat_ids)),
                        seed=seed,
                        save_path=None,
                    )

                print(
                    f"\n[{idx}/{len(selected_cfgs)}] {cfg_name}: {cfg_spec.get('description', '')} "
                    f"(N={len(compat_recs)}: DOD-H={c_h}, DOD-O={c_o}, folds={len(cfg_splits)})"
                )
                bundle = run_cross_validation_for_config(
                    all_records=compat_recs,
                    splits=cfg_splits,
                    channel_config=cfg_spec,
                    base_config=base_cfg,
                    config_name=cfg_name,
                    use_context_override=use_context_override,
                    max_epochs_override=max_epochs_override,
                    class_weight_mode_override=args.class_weights,
                    batch_size_override=args.batch_size,
                    early_stopping_patience_override=args.early_stopping_patience,
                    max_folds=max_folds_override,
                    results_dir=results_dir,
                    resume=True,
                    force_retrain=args.force_retrain,
                    verbose=True,
                )
                ablation_results[cfg_name] = bundle
                ablation_summaries[cfg_name] = bundle["aggregated_metrics"]

        epoch_sec = float(base_cfg.get("preprocessing", {}).get("epoch_sec", 30.0))
        summary_df = _export_dreem_evaluation_artifacts(
            results_dir=results_dir,
            selected_configs=selected_cfgs,
            ablation_results=ablation_results,
            ablation_summaries=ablation_summaries,
            compat_counts=compat_counts,
            epoch_sec=epoch_sec,
        )
        if not summary_df.empty:
            print("\n=== Dreem DOD Singular EEG Summary ===")
            print(summary_df.to_string(index=False))

        # Copy CSV artifacts out if running synthetic temp dry-run? Just report results cleanly.
        return results_dir, summary_df
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download, preprocess, train, and evaluate TinySleepNet on Dreem DOD-H + DOD-O singular EEG channels."
    )
    parser.add_argument("--config", type=str, default="configs/dreem_config.yaml")
    parser.add_argument("--ablation-config", type=str, default="configs/ablation_channels.yaml")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and extract DOD-H and DOD-O .h5 files from Zenodo record 15900394.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download DOD-H and DOD-O archives even if local files exist.",
    )
    parser.add_argument(
        "--check-raw",
        action="store_true",
        help="Inspect data/raw/dreem .h5 files for valid HDF5 headers and exit.",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Only run downloading (if --download) and .h5 -> .npz preprocessing.",
    )
    parser.add_argument(
        "--overwrite-prep",
        action="store_true",
        help="Re-run .h5 -> .npz preprocessing even if cached .npz files exist.",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default="all",
        help="Which singular EEG configs to run: 'all' (all 14), 'shared' (6 in DOD-H+DOD-O), 'dodh' (6), 'dodo' (2), or comma-separated names.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of subjects.")
    parser.add_argument("--dry-run", action="store_true", help="Fast 2-epoch, 2-fold smoke run (uses synthetic HDF5 if no local data exists).")
    parser.add_argument("--no-context", action="store_true", help="Disable BiLSTM temporal sequence context (epoch-wise CNN mode).")
    parser.add_argument("--class-weights", type=str, default=None, help="Override class weights: 'sqrt', 'tinysleepnet', 'balanced', 'false'.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max training epochs.")
    parser.add_argument("--early-stopping-patience", type=int, default=None, help="Override early stopping patience.")
    parser.add_argument("--max-folds", type=int, default=None, help="Override number of CV folds to execute.")
    parser.add_argument("--force-retrain", action="store_true", help="Ignore cached fold JSONs/checkpoints and retrain.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and aggregate saved metrics_*.json in results_dir.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_dreem_pipeline(args)


if __name__ == "__main__":
    main()
