"""CLI Script: Run Channel Ablation & Wearable Simulation Study (Supports `--dry-run` and Full 10-Fold CV)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.channel_sim import load_ablation_configs
from src.data.dataset import (
    generate_subject_cv_splits,
    load_subject_cv_splits,
    load_subject_records_from_dir,
)
from src.data.preprocessing import preprocess_all_recordings
from src.evaluation.metrics import metrics_summary_to_dataframe
from src.training.trainer import run_cross_validation_for_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TinySleepNet Channel Ablation & Wearable Simulation Study."
    )
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--ablation-config", type=str, default="configs/ablation_channels.yaml")
    parser.add_argument(
        "--channels",
         nargs="*",
        default=None,
        help="Specific channel config names to run (default: all in shhs_configs)",
    )
    parser.add_argument(
        "--use-context",
        action="store_true",
        help="Toggle Model B (Context-Aware LSTM, use_context=True)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify pipeline on a tiny subset (3 subjects, 1 fold, 2 epochs) across configs and both context modes.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max training epochs per fold.",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Limit number of CV folds executed per config.",
    )
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    shhs_configs = load_ablation_configs(
        PROJECT_ROOT / args.ablation_config, section="shhs_configs"
    )
    if args.channels:
        selected_configs = {k: shhs_configs[k] for k in args.channels if k in shhs_configs}
    else:
        selected_configs = shhs_configs

    proc_dir = PROJECT_ROOT / base_config["paths"]["processed_dir"]
    splits_dir = PROJECT_ROOT / base_config["paths"]["splits_dir"]
    results_dir = PROJECT_ROOT / base_config["paths"]["results_dir"]

    if args.dry_run:
        print("=== DRY-RUN MODE: Verifying end-to-end pipeline on 3 subjects for 2 epochs ===")
        preprocess_all_recordings(base_config, root_dir=PROJECT_ROOT, limit_subjects=3)
        all_records = load_subject_records_from_dir(proc_dir)[:3]
        splits = generate_subject_cv_splits(
            [r["subject_id"] for r in all_records], n_folds=3, seed=42
        )
        dry_results_dir = results_dir / "dry_run"

        for ctx_flag in (False, True):
            mode_label = "Model B (LSTM, use_context=True)" if ctx_flag else "Model A (CNN, use_context=False)"
            print(f"\n--- Testing {mode_label} across {len(selected_configs)} channel configs ---")
            summaries = {}
            for cfg_name, cfg_spec in selected_configs.items():
                res = run_cross_validation_for_config(
                    all_records=all_records,
                    splits=splits,
                    channel_config=cfg_spec,
                    base_config=base_config,
                    config_name=cfg_name,
                    use_context_override=ctx_flag,
                    max_epochs_override=2,
                    max_folds=1,
                    results_dir=dry_results_dir / ("context_true" if ctx_flag else "context_false"),
                    verbose=True,
                )
                summaries[cfg_name] = res["aggregated_metrics"]
            df = metrics_summary_to_dataframe(summaries)
            print(f"\nDry-Run Summary ({mode_label}):")
            print(df.to_string(index=False))
        print("\n=== DRY-RUN COMPLETED SUCCESSFULLY WITH ZERO ERRORS ===")
        return

    # Full run on all 10 preprocessed SHHS recordings
    preprocess_all_recordings(base_config, root_dir=PROJECT_ROOT)
    all_records = load_subject_records_from_dir(proc_dir)
    split_file = splits_dir / "shhs_10fold_splits.json"
    if split_file.exists():
        splits = load_subject_cv_splits(split_file)
    else:
        splits = generate_subject_cv_splits(
            [r["subject_id"] for r in all_records],
            n_folds=int(base_config.get("cv", {}).get("n_folds", 10)),
            seed=int(base_config.get("seed", 42)),
            save_path=split_file,
        )

    use_context = True if args.use_context else bool(base_config["model"].get("use_context", False))
    summaries = {}
    for cfg_name, cfg_spec in selected_configs.items():
        print(f"\n==================================================================")
        print(f"Running Config: {cfg_name} — {cfg_spec.get('description', '')}")
        print(f"==================================================================")
        res = run_cross_validation_for_config(
            all_records=all_records,
            splits=splits,
            channel_config=cfg_spec,
            base_config=base_config,
            config_name=cfg_name,
            use_context_override=use_context,
            max_epochs_override=args.max_epochs,
            max_folds=args.max_folds,
            results_dir=results_dir,
            verbose=True,
        )
        summaries[cfg_name] = res["aggregated_metrics"]

    df = metrics_summary_to_dataframe(summaries)
    out_csv = results_dir / "ablation_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved ablation summary table to {out_csv}:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
