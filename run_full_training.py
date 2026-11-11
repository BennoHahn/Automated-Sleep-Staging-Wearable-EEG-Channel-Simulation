"""Standalone CLI runner for Full SHHS1 Cohort Download, Preprocessing & 10-Fold CV Training.

Executes the exact same pipeline as `phase1_pipeline.ipynb` (Steps 1, 4, and 5) so that
multi-hour runs on the NVIDIA DGX Spark can also be executed inside `tmux` or `nohup`
without depending on an active browser connection. All per-fold JSONs, checkpoints,
and summary tables are saved to the exact same `results/<run_dir>/` folder read by
`phase1_pipeline.ipynb`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import (
    select_shhs_cohort,
    download_and_preprocess_shhs_cohort,
)
from src.data.channel_sim import (
    load_ablation_configs,
    is_channel_config_compatible,
)
from src.data.dataset import (
    ensure_subject_cv_splits,
    load_subject_records_from_dir,
)
from src.training.trainer import (
    run_cross_validation_for_config,
    get_experiment_results_dir,
    resolve_device,
    get_device_description,
)
from src.evaluation.metrics import (
    metrics_summary_to_dataframe,
    plot_normalized_confusion_matrix,
)
from src.evaluation.clinical_sleep import (
    plot_bland_altman_grid,
    plot_hypnogram_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="SHHS1 Full Cohort Training Runner")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["dual_eeg"],
        help="Channel config(s) to train (e.g. dual_eeg single_eeg_c4 single_eeg_c3)",
    )
    parser.add_argument(
        "--class-weights",
        type=str,
        default="sqrt",
        help="Class weight mode: sqrt | tinysleepnet | balanced | none",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--max-epochs", type=int, default=100, help="Max epochs per fold")
    parser.add_argument(
        "--early-stopping-patience",
        "--patience",
        dest="patience",
        type=int,
        default=20,
        help="Early stopping patience in epochs (NOT number of patients)",
    )
    parser.add_argument("--max-folds", type=int, default=10, help="Number of CV folds to run")
    parser.add_argument("--no-context", action="store_true", help="Disable LSTM context (Model A)")
    parser.add_argument("--force-retrain", action="store_true", help="Ignore cached folds and retrain")
    args = parser.parse_args()

    use_context = not args.no_context

    with open(PROJECT_ROOT / "configs/base_config.yaml", "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    shhs_configs = load_ablation_configs(
        PROJECT_ROOT / "configs/ablation_channels.yaml", section="shhs_configs"
    )

    device = resolve_device(base_config["training"]["device"])
    print(f"==============================================================================")
    print(f"SHHS1 Full Pipeline Runner | Device: {get_device_description(device)}")
    print(f"Configs       : {args.configs}")
    print(f"Context LSTM  : {use_context} (seq_len={base_config['model']['seq_len']})")
    print(f"Class Weights : {args.class_weights} | Batch Size: {args.batch_size}")
    print(f"Max Epochs    : {args.max_epochs} | Early Stopping Patience: {args.patience}")
    print(f"==============================================================================\n")

    # Step 1: Select cohort, download from NSRR, preprocess into float16 .npz
    selected_subject_ids, _ = select_shhs_cohort(base_config, root_dir=PROJECT_ROOT)
    print(f"[Step 1] Selected {len(selected_subject_ids):,} healthy SHHS1 subjects.")

    download_and_preprocess_shhs_cohort(
        config=base_config,
        subject_ids=selected_subject_ids,
        root_dir=PROJECT_ROOT,
        overwrite=False,
        verbose=True,
    )

    all_records = load_subject_records_from_dir(
        PROJECT_ROOT / base_config["paths"]["processed_dir"],
        subject_ids=selected_subject_ids,
    )
    if not all_records:
        raise RuntimeError("No preprocessed subject records found in data/processed/.")

    total_epochs = sum(len(r["labels"]) for r in all_records)
    print(f"[Step 1] Loaded {len(all_records):,} subjects ({total_epochs:,} valid 30s epochs).\n")

    # Step 2 & 4: Ensure 10-fold subject-independent CV splits
    split_file = PROJECT_ROOT / base_config["paths"]["splits_dir"] / "shhs_10fold_splits.json"
    splits = ensure_subject_cv_splits(
        subject_ids=[r["subject_id"] for r in all_records],
        split_path=split_file,
        n_folds=10,
        seed=int(base_config["seed"]),
    )

    results_dir = get_experiment_results_dir(
        root_dir=PROJECT_ROOT,
        base_config=base_config,
        n_subjects=len(all_records),
        use_context=use_context,
        class_weight_mode=args.class_weights,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.patience,
    )
    print(f"[Step 4] Active Results Directory: {results_dir}")

    ablation_results = {}
    ablation_summaries = {}

    for cfg_name in args.configs:
        if cfg_name not in shhs_configs:
            raise KeyError(f"Unknown channel config '{cfg_name}'. Available: {list(shhs_configs.keys())}")
        cfg_spec = shhs_configs[cfg_name]
        if not is_channel_config_compatible(all_records[0]["channel_names"], cfg_spec):
            print(f"[Skipping {cfg_name}] Required channels not in {all_records[0]['channel_names']}")
            continue

        print(f"\n{'='*78}")
        print(f"Training Config: {cfg_name} ({cfg_spec['description']})")
        print(f"{'='*78}")
        res = run_cross_validation_for_config(
            all_records=all_records,
            splits=splits,
            channel_config=cfg_spec,
            base_config=base_config,
            config_name=cfg_name,
            use_context_override=use_context,
            max_epochs_override=args.max_epochs,
            class_weight_mode_override=args.class_weights,
            batch_size_override=args.batch_size,
            early_stopping_patience_override=args.patience,
            max_folds=args.max_folds,
            results_dir=results_dir,
            resume=True,
            force_retrain=args.force_retrain,
            verbose=True,
        )
        ablation_results[cfg_name] = res
        ablation_summaries[cfg_name] = res["aggregated_metrics"]

    # Step 5: Save Summary CSV & Figures
    df_ablation_summary = metrics_summary_to_dataframe(ablation_summaries)
    summary_csv_path = results_dir / "ablation_summary_10fold.csv"
    df_ablation_summary.to_csv(summary_csv_path, index=False)
    print(f"\n[Step 5] Saved Summary CSV to {summary_csv_path}")
    print(df_ablation_summary.to_string(index=False))

    active_cfgs = [c for c in args.configs if c in ablation_summaries]
    if active_cfgs:
        n_cfgs = len(active_cfgs)
        ncols = min(3, max(1, n_cfgs))
        nrows = (n_cfgs + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 5.2 * nrows))
        axes_flat = np.atleast_1d(axes).ravel()
        for idx, cfg_name in enumerate(active_cfgs):
            summ = ablation_summaries[cfg_name]
            plot_normalized_confusion_matrix(
                mean_norm_cm=np.array(summ["mean_confusion_matrix_norm"]),
                std_norm_cm=np.array(summ["std_confusion_matrix_norm"]),
                title=f"{cfg_name} (κ={summ['pooled_metrics']['kappa']:.2f}, {ablation_results[cfg_name]['n_folds_run']} folds)",
                ax=axes_flat[idx],
            )
        for idx in range(n_cfgs, len(axes_flat)):
            axes_flat[idx].axis("off")
        fig.tight_layout()
        cm_fig_path = results_dir / "figures" / "confusion_matrices_grid.png"
        cm_fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(cm_fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        target_cfg = active_cfgs[0]
        clin_agree = ablation_results[target_cfg]["pooled_clinical_agreement"]
        df_clinical = pd.DataFrame.from_dict(clin_agree, orient="index")
        df_clinical.to_csv(results_dir / f"clinical_agreement_{target_cfg}.csv")

        test_subject_records = [
            subj_rec
            for fold_out in ablation_results[target_cfg]["folds"]
            for subj_rec in fold_out["test_per_subject"]
        ]
        if test_subject_records:
            first_test = test_subject_records[0]
            plot_hypnogram_comparison(
                y_true=first_test["y_true"],
                y_pred=first_test["y_pred"],
                subject_id=first_test["subject_id"],
                config_name=target_cfg,
                save_path=results_dir / "figures" / f"hypnogram_{target_cfg}_{first_test['subject_id']}.png",
            )
            plt.close("all")
            if len(test_subject_records) >= 2:
                plot_bland_altman_grid(
                    subject_records=test_subject_records,
                    parameters=["TST_min", "SE_pct", "SOL_min", "WASO_min"],
                    title=f"Bland-Altman Agreement — {target_cfg} (N={len(test_subject_records)} Test Nights)",
                    save_path=results_dir / "figures" / f"bland_altman_{target_cfg}.png",
                )
                plt.close("all")
        print(f"[Step 5] Saved confusion matrix, clinical agreement CSV, hypnogram & Bland-Altman plots to {results_dir / 'figures'}")


if __name__ == "__main__":
    main()
