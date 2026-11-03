"""CLI Script: Aggregate and visualize saved evaluation metrics from `results/`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (
    metrics_summary_to_dataframe,
    plot_normalized_confusion_matrix,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and plot saved ablation results.")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    res_dir = PROJECT_ROOT / args.results_dir
    metric_files = sorted(res_dir.glob("metrics_*.json"))
    if not metric_files:
        print(f"No metrics_*.json files found in {res_dir}")
        return

    summaries = {}
    for mf in metric_files:
        with open(mf, "r", encoding="utf-8") as f:
            payload = json.load(f)
        cfg_name = payload["config_name"]
        summaries[cfg_name] = payload["aggregated_metrics"]

        mean_cm = np.array(payload["aggregated_metrics"]["mean_confusion_matrix_norm"])
        std_cm = np.array(payload["aggregated_metrics"]["std_confusion_matrix_norm"])
        fig_path = res_dir / "figures" / f"cm_{cfg_name}.png"
        plot_normalized_confusion_matrix(
            mean_norm_cm=mean_cm,
            std_norm_cm=std_cm,
            title=f"Normalized Confusion Matrix — {cfg_name}",
            save_path=fig_path,
        )

    df = metrics_summary_to_dataframe(summaries)
    print("\n=== Ablation Study Summary ===")
    print(df.to_string(index=False))
    df.to_csv(res_dir / "ablation_summary.csv", index=False)


if __name__ == "__main__":
    main()
