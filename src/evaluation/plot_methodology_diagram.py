"""Generate the high-resolution Methodology & Training Architecture block diagram PNG.

Usage:
    python src/evaluation/plot_methodology_diagram.py
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def generate_methodology_diagram(
    out_path: str | Path = "results/shhs1_healthy_n1738_ctxTrue_wsqrt_bs32_ep100_pat20/figures/methodology_pipeline_dual_eeg.png",
) -> Path:
    fig, ax = plt.subplots(figsize=(19.0, 11.4), dpi=250)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 64)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    ax.text(
        50, 61.8,
        "End-to-End Methodology & Training Architecture — Dual-EEG (C4-A1 + C3-A2) on SHHS1",
        ha="center", va="center", fontsize=16.0, fontweight="bold", color="#0F172A"
    )
    ax.text(
        50, 59.6,
        "Color Legend:   [Blue] Custom Thesis Pipeline (src/)     |     [Teal] TinySleepNet Spec (Supratak & Guo, 2020, PyTorch Re-Implementation)     |     [Purple] Hybrid / Adapted",
        ha="center", va="center", fontsize=10.3, fontweight="semibold", color="#334155"
    )

    def draw_box(x: float, y: float, w: float, h: float, title: str, body_lines: list[str], style: str = "custom", title_size: float = 10.3, body_size: float = 8.35) -> None:
        if style == "custom":
            fc, ec, tc, badge, badge_bg = "#EFF6FF", "#1D4ED8", "#1E3A8A", "CUSTOM PIPELINE", "#DBEAFE"
        elif style == "tsn":
            fc, ec, tc, badge, badge_bg = "#ECFDF5", "#047857", "#065F46", "TINYSLEEPNET SPEC", "#D1FAE5"
        elif style == "hybrid":
            fc, ec, tc, badge, badge_bg = "#F5F3FF", "#6D28D9", "#4C1D95", "HYBRID / ADAPTED", "#EDE9FE"
        else:
            fc, ec, tc, badge, badge_bg = "#FFFBEB", "#B45309", "#78350F", "RESULTS (10-FOLD CV)", "#FEF3C7"

        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.25,rounding_size=0.8",
            facecolor=fc, edgecolor=ec, linewidth=2.0
        )
        ax.add_patch(box)

        ax.text(
            x + w - 0.45, y + 0.45,
            badge,
            ha="right", va="bottom",
            fontsize=6.3, fontweight="bold", color=ec,
            bbox=dict(boxstyle="round,pad=0.18", facecolor=badge_bg, edgecolor=ec, lw=0.75)
        )
        ax.text(
            x + w / 2, y + h - 1.15,
            title,
            ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=tc
        )
        ax.plot([x + 0.6, x + w - 0.6], [y + h - 2.15, y + h - 2.15], color=ec, lw=0.9, alpha=0.5)
        ax.text(
            x + 0.75, y + h - 2.65,
            "\n".join(body_lines),
            ha="left", va="top",
            fontsize=body_size, color="#1E293B", linespacing=1.34
        )

    def draw_arrow(x1: float, y1: float, x2: float, y2: float, label: str = "") -> None:
        arr = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>,head_length=6.5,head_width=4.2",
            color="#334155", linewidth=2.0,
            connectionstyle="arc3,rad=0.0"
        )
        ax.add_patch(arr)
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(
                mx, my + 1.05, label,
                ha="center", va="center", fontsize=7.5, fontweight="bold", color="#0F172A",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="#F8FAFC", edgecolor="#94A3B8", lw=0.8)
            )

    # Row 1
    draw_box(
        1.2, 40.8, 21.8, 16.6,
        "1. SHHS1 Cohort Selection",
        [
            "• Source: SHHS Visit 1 (5,804 PSG nights)",
            "• Healthy Apnea Filter (AASM):",
            "    ahi_a0h3 < 5.0 & rdi3p < 5.0",
            "    -> 1,771 healthy sleepers",
            "• Channel Integrity Filter:",
            "    Require [C4-A1, C3-A2, EOG(L), EOG(R)]",
            "    (-5 corrupt EDFs, -28 missing C3-A2)",
            "• Final Cohort: 1,738 Subjects",
            "    1,761,352 epochs (30s) | ~14,678 hrs"
        ],
        style="custom"
    )
    draw_arrow(23.0, 49.1, 26.2, 49.1, "EDF+XML")
    draw_box(
        26.2, 40.8, 22.4, 16.6,
        "2. Standardized DSP Pipeline",
        [
            "• Protocol: van der Aar et al. (2024, 2025)",
            "• 60 Hz IIR Notch Filter (Q = 30)",
            "    Applied prior to resampling (fs=125 Hz)",
            "• Polyphase Resampling -> 100 Hz",
            "    3,000 samples per 30-second epoch",
            "• Zero-Phase Butterworth Bandpass:",
            "    5th-order, 0.2 - 49.0 Hz (sosfiltfilt)",
            "• R&K -> 5 AASM Classes:",
            "    Wake(0), N1(1), N2(2), N3(3+4), REM(4)",
            "• Saved in physical uV (float16 .npz)"
        ],
        style="custom"
    )
    draw_arrow(48.6, 49.1, 51.8, 49.1, "uV Signals")
    draw_box(
        51.8, 40.8, 22.0, 16.6,
        "3. Channel Sim & Z-Score",
        [
            "• Config: dual_eeg (C = 2 channels)",
            "    Ch 1: C4-A1 (Central Right - Mastoid)",
            "    Ch 2: C3-A2 (Central Left - Mastoid)",
            "• Deferred Physical Montage Engine:",
            "    Allows exact uV bipolar subtraction",
            "    before z-scoring (for ablation configs)",
            "• Per-Subject, Per-Channel Z-Score:",
            "    x_norm = (x - mean_subj) / std_subj",
            "• Epoch Tensor: (N_epochs, 2, 3000)"
        ],
        style="custom"
    )
    draw_arrow(73.8, 49.1, 77.0, 49.1, "1,738 Subj.")
    draw_box(
        77.0, 40.8, 21.8, 16.6,
        "4. 10-Fold CV Split & Aug.",
        [
            "• Subject-Independent 10-Fold CV:",
            "    80% Train (~1,390 subj | ~1.41M ep)",
            "    10% Val   (~174 subj   | ~176k ep)",
            "    10% Test  (~174 subj   | ~176k ep)",
            "• Sequence Windowing: M = 20 (10 min)",
            "    Batch = 32 seqs -> (32, 20, 2, 3000)",
            "• TinySleepNet GPU Augmentations:",
            "    1) Random time roll (+-10% = +-300 samp)",
            "    2) Random sign flip (p = 0.5, x -> -x)",
            "    3) Random seq start offset (0..5 epochs)"
        ],
        style="hybrid"
    )

    ax.plot([87.9, 87.9, 14.2], [40.55, 38.2, 38.2], color="#334155", linewidth=2.0)
    arr_down = FancyArrowPatch(
        (14.2, 38.2), (14.2, 35.65),
        arrowstyle="-|>,head_length=6.5,head_width=4.2",
        color="#334155", linewidth=2.0
    )
    ax.add_patch(arr_down)
    ax.text(
        51.0, 38.2,
        "Mini-Batch Input: 32 Sequences x 20 Contiguous Epochs = (640, 2, 3000) @ 100 Hz",
        ha="center", va="center", fontsize=8.4, fontweight="bold", color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#F8FAFC", edgecolor="#64748B", lw=0.9)
    )

    # Row 2
    draw_box(
        1.2, 18.8, 26.0, 16.6,
        "5. CNN Feature Extractor (Part 1)",
        [
            "• Input: (640, 2, 3000) [Early Channel Fusion]",
            "• Conv1d_1 (Frequency Filter Bank):",
            "    128 filters, kernel = 50 (0.5s = Fs/2),",
            "    stride = 6 (Fs/16), SAME pad + BN + ReLU",
            "    -> Output: (640, 128, 500)",
            "• MaxPool1d_1 (k=8, s=8) -> (640, 128, 62)",
            "• Dropout_1 (p = 0.5)",
            "• Captures delta, theta, alpha, spindle waves"
        ],
        style="tsn"
    )
    draw_arrow(27.2, 27.1, 30.6, 27.1, "(640,128,62)")
    draw_box(
        30.6, 18.8, 24.4, 16.6,
        "6. CNN Representation (Part 2)",
        [
            "• 3x Stacked Conv1D Blocks (Conv2..Conv4):",
            "    Conv1d_2: 128 filters, k=8, s=1 + BN + ReLU",
            "    Conv1d_3: 128 filters, k=8, s=1 + BN + ReLU",
            "    Conv1d_4: 128 filters, k=8, s=1 + BN + ReLU",
            "    -> Output: (640, 128, 62)",
            "• MaxPool1d_2 (k=4, s=4) -> (640, 128, 15)",
            "• Dropout_2 (p = 0.5)",
            "• Flatten -> Epoch Feature f_t: (640, 1920)",
            "• CNN Params: 407,040 (27.9% of model)"
        ],
        style="tsn"
    )
    draw_arrow(55.0, 27.1, 58.4, 27.1, "(32,20,1920)")
    draw_box(
        58.4, 18.8, 19.6, 16.6,
        "7. Context LSTM Head",
        [
            "• Reshape to Contiguous Seqs:",
            "    (32, 20, 1920) [10-min windows]",
            "• SequentialLSTMHead:",
            "    1-Layer Unidirectional LSTM",
            "    input = 1920 -> hidden = 128",
            "    -> Output: (32, 20, 128)",
            "• Dropout_3 (p = 0.5)",
            "• Linear Classifier (128 -> 5):",
            "    Logits (32, 20, 5) | 1.05M params"
        ],
        style="tsn"
    )
    draw_arrow(78.0, 27.1, 81.0, 27.1, "Logits")
    draw_box(
        81.0, 18.8, 17.8, 16.6,
        "8. Loss & Optimization",
        [
            "• Custom Sqrt Class Weights:",
            "    w_c ~ (N / (5 * N_c))^0.5",
            "    Solves N1 (3.4%) collapse!",
            "• Adam Optimizer:",
            "    lr = 1e-4, wd = 1e-3",
            "    Grad norm clip = 5.0",
            "• Early Stopping (Val Set):",
            "    Monitor: val_macro_f1",
            "    Patience = 20, MaxEp = 100",
            "• bfloat16 AMP on NVIDIA GB10"
        ],
        style="hybrid"
    )

    # Row 3
    draw_arrow(89.9, 18.8, 89.9, 15.8, "")
    draw_box(
        1.2, 1.2, 47.2, 14.6,
        "9. Two-Tier Standardized Evaluation Protocol",
        [
            "• Tier 1 — Epoch-Level Classification (Markov et al., 2025 Eq. 1-29 across 1,761,352 test epochs & 10 folds):",
            "    Overall Accuracy (ACC), Cohen's Kappa (k), Matthews Correlation Coefficient (MCC), Macro-F1, Weighted-F1,",
            "    and Per-Class Sensitivity (SE), Specificity (SP), Precision (PPV), NPV, and F1 for Wake, N1, N2, N3, REM.",
            "• Tier 2 — Night-Level Clinical Sleep Architecture & Bland-Altman Agreement (N = 1,738 Full Test Nights):",
            "    Full-night hypnogram reconstruction -> Total Sleep Time (TST), Sleep Efficiency (SE%), Sleep Onset Latency (SOL),",
            "    Wake After Sleep Onset (WASO), Stage % of TST -> Mean Bias, 95% Limits of Agreement (LoA), MAE, Pearson r."
        ],
        style="custom"
    )
    draw_arrow(48.4, 8.5, 51.8, 8.5, "10 Folds")
    draw_box(
        51.8, 1.2, 47.0, 14.6,
        "10. Achieved 10-Fold CV Benchmark Results — dual_eeg (N = 1,738)",
        [
            "• Epoch-Level (10-Fold Mean +- SD):",
            "    ACC = 84.8% +- 0.5%  |  Kappa = 0.792 +- 0.007  |  MCC = 0.793 +- 0.007  |  Macro-F1 = 78.5% +- 0.6%",
            "• Per-Stage F1-Scores:   Wake = 92.7%  |  REM = 85.8%  |  N2 = 84.9%  |  N3 = 79.3%  |  N1 = 50.1% (61.3% SE)",
            "• Clinical Sleep Agreement (N = 1,738 Test Nights):",
            "    - Total Sleep Time (TST): Bias = +5.6 min (MAE = 13.2m, r = 0.926)  |  SE%: Bias = +1.1% (MAE = 2.6%, r = 0.917)",
            "    - Deep Sleep (N3 % TST):  Bias = +1.5% (r = 0.699)                  |  REM (% TST): Bias = +1.4% (r = 0.668)"
        ],
        style="result"
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    saved = generate_methodology_diagram()
    print(f"Saved methodology diagram to {saved}")
