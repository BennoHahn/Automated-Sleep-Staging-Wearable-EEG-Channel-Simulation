"""CLI Script: Preprocess SHHS1 EDF & Profusion XML files into standardized 100 Hz `.npz` archives."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import generate_subject_cv_splits
from src.data.preprocessing import load_preprocessed_subject, preprocess_all_recordings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 1: Standardized PSG Preprocessing.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to base_config.yaml",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run preprocessing even if .npz files already exist",
    )
    parser.add_argument(
        "--limit-subjects",
        type=int,
        default=None,
        help="Limit number of subjects to process (e.g. for quick testing)",
    )
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("Step 1: Preprocessing EDF & Profusion XML recordings...")
    saved_paths = preprocess_all_recordings(
        config=config,
        root_dir=PROJECT_ROOT,
        overwrite=args.overwrite,
        limit_subjects=args.limit_subjects,
    )

    subject_ids: list[str] = []
    total_epochs = 0
    for p in saved_paths:
        rec = load_preprocessed_subject(p)
        sid = rec["subject_id"]
        n_ep = len(rec["labels"])
        total_epochs += n_ep
        subject_ids.append(sid)
        stage_counts = {
            stage: int((rec["labels"] == idx).sum())
            for idx, stage in enumerate(["W", "N1", "N2", "N3", "REM"])
        }
        print(
            f"  - {sid}: shape={rec['epochs'].shape}, channels={rec['channel_names']}, "
            f"stages={stage_counts}"
        )

    print(f"Total processed subjects: {len(saved_paths)} ({total_epochs} 30s epochs).")

    if len(subject_ids) >= 3:
        splits_dir = PROJECT_ROOT / config.get("paths", {}).get("splits_dir", "data/splits")
        split_file = splits_dir / "shhs_10fold_splits.json"
        n_folds = int(config.get("cv", {}).get("n_folds", 10))
        seed = int(config.get("seed", 42))
        splits = generate_subject_cv_splits(
            subject_ids=subject_ids,
            n_folds=n_folds,
            seed=seed,
            save_path=split_file,
        )
        print(f"Saved {len(splits)}-fold subject-independent CV splits to {split_file}")


if __name__ == "__main__":
    main()
