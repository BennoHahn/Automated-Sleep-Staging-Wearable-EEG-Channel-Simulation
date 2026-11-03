# Project Specification: Automated Sleep Staging & Wearable EEG Simulation Pipeline

## 1. Project Overview & Objectives
This project implements a modular, extensible deep learning pipeline in Python (PyTorch) for automated 5-stage sleep classification (`Wake`, `N1`, `N2`, `N3`, `REM`) according to AASM standards. 
The primary goal of the current development phase (**Phase 1 / MVP**) is to evaluate how sleep staging performance changes across a **Channel Ablation & Wearable Simulation study** using a clinical Polysomnography (PSG) dataset and the **TinySleepNet** architecture.

The architecture and pipeline must be designed modularly:
1. **Current Implementation Scope (Phase 1):**
   - Standardized Preprocessing of clinical PSG recordings.
   - Channel Ablation & Mathematical Simulation of 5 channel configurations (Full PSG, Single-Lead PSG, and 3 Simulated Wearable Headband setups).
   - Deep Learning Model: **TinySleepNet** running in **Model A (Context-Blind / One-to-One)** mode, but architected so **Model B (Context-Aware / LSTM)** can be toggled on via configuration.
   - Standardized Evaluation using 10-fold subject-independent cross-validation and balanced metrics ($\kappa$, MCC, Macro-F1, Confusion Matrices, and clinical sleep parameters).
2. **Future Extension Scope (Phase 2 - Do Not Implement Yet, Design Interfaces Only):**
   - Toggling **Model B (Context-Aware TinySleepNet with LSTM)** and optional secondary models (**DeepSleepNet**, **SeqSleepNet**).
   - Loading a secondary real-hardware wearable dataset (**Bitbrain Open Access Sleep Dataset `ds005555`**).
   - **Transfer Learning** strategies (Direct Transfer, Training from Scratch, Fine-Tuning [Full vs. Head Re-training], and Paired Feature Matching).

---

## 2. Reference Literature in `/papers`
You have access to a `/papers` directory containing the scientific literature governing this pipeline. Consult these PDFs whenever you need exact mathematical formulas, hyperparameter settings, or evaluation equations:

### Existing Papers in `/papers`:
1. **Heremans et al. (2022)** - *"Feature matching as improved transfer learning technique for wearable EEG"* (`arXiv:2201.00644v1`).
   - *Look up:* Section 3.2 & Fig. 2 (Feature Matching MSE loss, p. 4–6); Section 4 (ARNN & SeqSleepNet architectures, p. 5–7); Section 5.2 (Cross-validation and low-data subset splits, p. 6–7).
2. **Waters & Clifford (2022)** - *"Comparison of deep transfer learning algorithms and transferability measures for wearable sleep staging"* (`BioMedical Engineering OnLine`).
   - *Look up:* Page 3 ("Head re-training"); Page 18–21 (DeepSleepNet two-phase training and layer freezing strategies).
3. **van der Aar et al. (2025)** - *"Optimizing wearable single-channel electroencephalography sleep staging in a heterogeneous sleep-disordered population using transfer learning"* (`J Clin Sleep Med`).
   - *Look up:* Page 316 ("Classification model" - TinySleepNet description); Page 317 ("PSG" & "Training and validation strategies" - 5th-order Butterworth 0.2–49 Hz filter, 50 Hz notch, 100 Hz resampling, **F3 and F4 averaging to simulate Fz-M2**, 10-fold CV 80/10/10 split, early stopping, learning rates $10^{-4}$ vs $10^{-5}$, and clinical sleep parameter evaluation).
4. **He et al. (2023)** - *"Cross-scenario automatic sleep stage classification using transfer learning and single-channel EEG"* (`Biomedical Signal Processing and Control`).
   - *Look up:* Section 5.3.1.1 & Fig. 2 (Sleep stage transition probability matrices and adjacent epoch dependencies, p. 5); Section 6.3 (Strict subject-based splitting vs. epoch-based splitting to prevent data leakage, p. 7).
5. **van der Aar et al. (2024)** - *"Deep transfer learning for automated single-lead EEG sleep staging with channel and population mismatches"* (`Frontiers in Physiology`).
   - *Look up:* Section 2.1.2, 2.1.3 & Fig. 1 (p. 3–4, Preprocessing & dataset definition); Section 2.2 & 2.3 (TinySleepNet training, 200 iterations, early stopping after 50 iterations, p. 4); **Section 3.1, 3.3 & Section 4.4 (p. 5–7, 11: Simulating mastoid-free wearable via bipolar `F3-F4` vs. `F3-M2`, and why `F3-F4` subtracts shared N3 slow-wave activity)**.
6. **Li et al. (2025)** / **Huang et al. (2014)** - Forehead wearable EEG validation (`Fp1-M2`, `Fp2-M1`) and clinical sleep metrics (TST, SE, SOL, WASO, Bland-Altman analysis).
7. **Markov et al. (2025)** - *"Evaluating the performance of wearable EEG sleep monitoring devices: a meta-analysis approach"* (`npj Biomedical Innovations`).
   - *Look up:* **Section "Methods - Data analysis" (Pages 20–23, Equations 1 to 29)** for the exact mathematical formulas of Multiclass Cohen's Kappa ($\kappa$), Multiclass Matthews Correlation Coefficient (MCC), and Micro- vs. Macro-averaged Accuracy, Sensitivity (SE), Specificity (SP), PPV, NPV, and F1-score.
8. **Kosnacova et al. (2026)** - *"Smart Wearable EEG Devices: A Review..."* (`Biosensors`).
   - *Look up:* Section 2.1.2 & Table 2 (Wearable headband specifications, Bitbrain, Dreem, Sleep Profiler) and Section 3.4 (Typical wearable EEG processing pipelines, p. 25–26).
9. *(If added to `/papers`)* **Supratak & Guo (2020)** (*TinySleepNet*) & **Guillot et al. (2020)** (*Dreem Open Datasets*).

---

## 3. Project Directory & Architecture Setup
Set up a clean, modular Python project structure controlled by YAML configuration files so experiments can be reproduced easily:

```text
project_root/
├── papers/                      # PDF reference papers (DO NOT MODIFY)
├── configs/                     # YAML experiment configurations
│   ├── base_config.yaml         # Shared preprocessing & CV parameters
│   ├── ablation_channels.yaml   # Configs for the 5 channel setups
│   └── future_transfer.yaml     # Placeholder config for Phase 2 Transfer Learning
├── data/
│   ├── raw/                     # Raw PSG dataset files (and later Bitbrain dataset)
│   ├── processed/               # Standardized 100 Hz, filtered, 30s epoch tensors/HDF5
│   └── splits/                  # Saved 10-fold subject-independent split JSONs
├── Dataset_information/         # Information about the dataset 
├── src/
│   ├── data/
│   │   ├── preprocessing.py     # Filtering, notch, resampling, epoching
│   │   ├── channel_sim.py       # Channel extraction & mathematical wearable simulations
│   │   └── dataset.py           # PyTorch Dataset classes (supports 1-epoch & M-epoch sequences)
│   ├── models/
│   │   ├── tinysleepnet.py      # Modular TinySleepNet (CNN block + optional LSTM block)
│   │   └── future_models.py     # Stubs/interfaces for DeepSleepNet / SeqSleepNet
│   ├── training/
│   │   ├── trainer.py           # Training loop, validation, early stopping
│   │   └── transfer.py          # Stubs/interfaces for Phase 2 Transfer Learning
│   └── evaluation/
│       ├── metrics.py           # Kappa, MCC, Macro-F1, Confusion Matrix (Markov et al. 2025)
│       └── clinical_sleep.py    # TST, SE%, SOL, WASO, Bland-Altman plots
├── scripts/
│   ├── run_preprocessing.py
│   ├── run_ablation.py
│   └── evaluate_results.py
├── PROJECT_INFORMATION.md # current document, contains information about the project
├── PROJECT_SPEC.md # projects specs, for now only contains the exact TinySleepNet architectural numbers ( For exact 1:1 PyTorch replica of the original TinySleepNet repo)
├── requirements.txt
└── README.md

```
---

## 4. Detailed Component Specifications (Phase 1 - Implement Now)

### Step 1: Data Ingestion & Standardized Preprocessing (`src/data/preprocessing.py`)
*Reference: van der Aar et al. (2024), Section 2.1.3 (p. 4) & van der Aar et al. (2025), Section "PSG" (p. 317).*
1. **Label Standardization:** Map sleep stage annotations to the 5 AASM classes: `0: Wake`, `1: N1`, `2: N2`, `3: N3` (merge N3 and N4 if R&K scored), `4: REM`. Exclude `Movement` or `Unknown` epochs (or non-scored leading/trailing segments as standard).
2. **Filtering:** 
   - Apply a zero-phase **5th-order Butterworth bandpass filter between 0.2 Hz and 49 Hz** (or 0.3–40 Hz if original sampling rate is 100 Hz).
   - Apply a **50 Hz notch filter** (or 60 Hz depending on dataset origin) to remove mains power interference.
3. **Resampling:** Downsample all physiological signals to a common sampling frequency of **100 Hz**.
4. **Epoching:** Segment continuous recordings into non-overlapping **30-second epochs** ($30 \times 100\text{ Hz} = 3000$ data points per epoch per channel).
5. **Normalization:** Support configurable per-recording (subject-level) z-score normalization (zero mean, unit variance) or raw amplitude scaling as required by TinySleepNet.

---

### Step 2: Channel Ablation & Wearable Simulation (`src/data/channel_sim.py`)
*Reference: van der Aar et al. (2025) p. 317; van der Aar et al. (2024) p. 4 & 11; Guillot et al. (2020) DOD-H/O.*
The pipeline must dynamically extract or mathematically compute **5 distinct channel configurations** from the clinical PSG dataset so their classification performances can be systematically compared:

1. **Config 1 — `full_psg` (Multi-Channel Clinical Baseline):**
   - Multi-channel input combining Frontal (`F3-M2`), Central (`C3-M2`), and Occipital (`O1-M2`) EEG, plus `EOG` and `EMG` (if available in the dataset).
   - *Purpose:* Establishes the upper-bound clinical PSG performance.
2. **Config 2 — `single_lead_psg` (Standard Single-Lead PSG):**
   - Standard unilateral clinical derivation: **`C3-M2`** (or `C4-M1`) and **`F3-M2`** tested individually.
   - *Purpose:* Standard single-channel PSG benchmark.
3. **Config 3 — `sim_wearable_1_mastoid` (Simulated Wearable 1 — Forehead with Mastoid Reference):**
   - **Target Simulated Channel:** Midline Frontopolar-to-Mastoid (`Fpz-M2` / `Fz-M2`), mimicking headbands with an ear/mastoid reference clip (e.g., Philips SmartSleep, Bitbrain Ikon Sleep `AF7-M2`).
   - **Mathematical Derivation:** Average the left and right frontal mastoid-referenced channels:
     $$\text{Sim\_Fz\_M2} = \frac{\text{F3-M2} + \text{F4-M2}}{2}$$
     *(Note: If the dataset only provides `F4-M1` instead of `F4-M2`, re-reference using `M1-M2` if present, or fall back to unilateral frontal/frontopolar-mastoid `F3-M2` / `Fp1-M2`).*
   - *Reference:* van der Aar et al. (2025), p. 317.
4. **Config 4 — `sim_wearable_2_no_mastoid` (Simulated Wearable 2 — Local Forehead Bipolar Strip):**
   - **Target Simulated Channel:** Cross-forehead bipolar derivation (`Fp1-Fp2` / `AF7-AF8` / `F3-F4`), mimicking a self-contained forehead patch or strip without any behind-the-ear wire (e.g., Sleep Profiler, ZMax, Focusband).
   - **Mathematical Derivation:** Subtract the right frontal channel from the left frontal channel (when sharing a common reference, which cancels out):
     $$\text{Sim\_F3\_F4} = (\text{F3-M2}) - (\text{F4-M2})$$
     *(Or if raw unreferenced `F3` and `F4` are available, compute `F3 - F4` directly).*
   - *Reference:* van der Aar et al. (2024), Section 2.1.2 & Section 4.4 (p. 4, 11).
5. **Config 5 — `sim_wearable_3_headband` (Simulated Wearable 3 — Fronto-Occipital Headband):**
   - **Target Simulated Channel:** Front-to-back headband derivation (`Fp1-O1` / `Fpz-O1` / `F3-O1`), mimicking a full-circumference headband with forehead and back-strap electrodes and no mastoid wire (e.g., Dreem Headband).
   - **Mathematical Derivation:** Subtract the occipital channel from the ipsilateral frontal channel:
     $$\text{Sim\_F3\_O1} = (\text{F3-M2}) - (\text{O1-M2})$$
   - *Reference:* Dreem Open Datasets (DOD-H / DOD-O) & Arnal et al. (2020).

---

### Step 3: Deep Learning Architecture (`src/models/tinysleepnet.py`)
*Reference: Supratak & Guo (2020) "TinySleepNet"; van der Aar et al. (2024), Section 2.2 (p. 4).*

Design the `TinySleepNet` class in PyTorch so it separates **Representation Learning (CNN)** from **Sequential Context Learning (LSTM)** via a boolean flag `use_context: bool`:

1. **Part 1: CNN Feature Extractor (`RepresentationLearning` module):**
   - Accepts raw 30-second epoch tensors of shape `(Batch, Channels, 3000)`.
   - Note: While standard TinySleepNet takes 1 channel (`Channels = 1`), parameterize `in_channels` so it works for both single-channel (`C = 1`) and `full_psg` (`C = N`).
   - Consists of 4 consecutive 1D Convolutional blocks (`Conv1d` + `BatchNorm1d` + `ReLU`), interleaved with 2 `MaxPool1d` layers and `Dropout(0.5)` layers, following the exact TinySleepNet specification (first conv layer uses a larger filter size, e.g., $F_s / 2 = 50$ samples with stride $F_s / 16$ to capture frequency information, followed by 3 smaller conv layers of kernel size 8).
   - Outputs a 1D feature vector $f_t$ for each 30-second epoch.
2. **Part 2: Classification Head — Model A: Context-Blind (`use_context = False`):**
   - **Implement and run this mode first.**
   - Passes the extracted single-epoch CNN feature vector $f_t$ directly through a Dropout layer and a linear classification head (`Linear(feat_dim, 5)`) to predict the 5 sleep stages for each isolated 30-second epoch (One-to-One classification).
3. **Part 3: Sequential Module — Model B: Context-Aware (`use_context = True`):**
   - **Build the module into the class now so it is ready to toggle on in the next step.**
   - Takes a sequence of $M$ consecutive epoch feature vectors $(f_1, f_2, \dots, f_M)$ (default $M = 20$ epochs = 10 minutes), passes them through a **single unidirectional LSTM layer** followed by `Dropout(0.5)`, and classifies each step via the linear head (Many-to-Many causal sequence classification).

---

### Step 4: Training & Cross-Validation Protocol (`src/training/trainer.py`)
*Reference: van der Aar et al. (2024) Section 2.3 (p. 4); van der Aar et al. (2025) p. 317; He et al. (2023) Section 6.3 (p. 7).*
1. **Strict Subject-Independent 10-Fold Cross-Validation:**
   - Split all subjects/recordings into 10 folds: **80% Training, 10% Validation, 10% Test**.
   - **CRITICAL:** Splitting MUST be done at the **subject/recording level**, never at the epoch level (epochs from the same subject must never appear in more than one set simultaneously). Save the fold assignments to `data/splits/` so all 5 channel configurations are evaluated on the exact same subject splits.
2. **Optimization & Early Stopping:**
   - Optimizer: Adam (`lr = 1e-4`, weight decay as specified in TinySleepNet).
   - Loss Function: Cross-Entropy Loss (optionally class-weighted if needed).
   - Train for a maximum of **200 training iterations/epochs**, evaluating on the validation set after each epoch.
   - Apply **Early Stopping** if there is no improvement in validation weighted F1-score / accuracy for **50 consecutive iterations**, and retain the best-performing model checkpoint for test-set evaluation.

---

### Step 5: Standardized Evaluation Module (`src/evaluation/`)
*Reference: Markov et al. (2025) Pages 20–23 (Equations 1–29); van der Aar et al. (2025) Pages 317–321.*
1. **Epoch-by-Epoch Classification Metrics (`metrics.py`):**
   - Implement the exact equations from **Markov et al. (2025), pp. 20–23**:
     - Overall **Cohen's Kappa ($\kappa$)** (Eq. 1–3)
     - Overall **Multiclass Matthews Correlation Coefficient (MCC)** (Eq. 4–6)
     - **Micro- and Macro-averaged** Accuracy (ACC), Sensitivity/Recall (SE), Specificity (SP), Positive Predictive Value/Precision (PPV), Negative Predictive Value (NPV), and **F1-score** (plus Weighted F1-score).
     - **Per-Class Metrics** ($\kappa$, MCC, F1, SE, SP, PPV) for each of the 5 stages (`W`, `N1`, `N2`, `N3`, `REM`).
     - Normalized **Confusion Matrices** (mean ± SD across participants/folds, formatted like Fig. 2 in *van der Aar et al., 2025*).
2. **Whole-Night Clinical Sleep Parameters (`clinical_sleep.py`):**
   - For each test subject's full night hypnogram (predicted vs. manual PSG ground truth), compute:
     - **TST** (Total Sleep Time in minutes)
     - **SE (%)** (Sleep Efficiency)
     - **SOL** (Sleep Onset Latency in minutes)
     - **WASO** (Wake After Sleep Onset in minutes)
     - Duration (minutes) and Percentage (%) spent in `Wake`, `N1`, `N2`, `N3`, and `REM`.
   - Generate **Bland–Altman plots** and Pearson correlations ($r$) comparing predicted vs. true clinical parameters (*van der Aar et al., 2025*, Table 3 & Fig. 3).

### Rule:
Create a tests/ folder and a --dry-run flag in the training script. Before running a full 10-fold cross-validation, verify every module on a tiny 2-subject subset (or synthetic dummy PSG tensors of shape (Batch, Channels, 3000)) for 2 epochs to confirm that all 5 channel configurations, both use_context=False and use_context=True forward passes, and all evaluation metrics run end-to-end without shape or NaN errors.

---

## 5. Future Steps & Extensibility Roadmap (Phase 2 - Design Interfaces Only)
*Do NOT execute these experiments yet, but structure the codebase (e.g., base classes, config keys, model checkpoints) so they can be plugged in seamlessly in Phase 2:*

1. **Future Step A — Context Awareness Comparison (Model A vs. Model B):**
   - Once Model A (CNN-only, Context-Blind) is benchmarked across all 5 channel configurations, we will toggle `use_context = True` in `TinySleepNet` (enabling the unidirectional LSTM with sequence length $M=20$) to quantify how much temporal context compensates for reduced spatial channels (especially for `N1` and `REM` in frontal channels, and `N3` in bipolar `F3-F4`).
   - *Optional Secondary Architectures:* Keep `src/models/future_models.py` ready to add **DeepSleepNet** (*Waters & Clifford, 2022*) or **SeqSleepNet** (*Heremans et al., 2022*) for multi-model comparison.
2. **Future Step B — Secondary Real Wearable Dataset Integration (Bitbrain `ds005555`):**
   - The `Dataset` loader should be dataset-agnostic so we can later ingest the **Bitbrain Open Access Sleep Dataset (BOAS)**, which contains simultaneous clinical PSG (`F3`, `F4`, `C3`, `C4`, `O1`, `O2`) and real dry-textile forehead headband EEG (`AF7`, `AF8`).
3. **Future Step C — Transfer Learning Module (`src/training/transfer.py`):**
   - Ensure Phase 1 saves clean model checkpoints (`.pt` files) separating `feature_extractor` weights and `classifier` head weights.
   - In Phase 2, `transfer.py` will load the pre-trained PSG weights (from `F3-M2` or `Sim_Fz_M2`) and evaluate 4 transfer strategies on the Bitbrain wearable `AF7`/`AF8` channels:
     1. **Direct Transfer:** Evaluate pre-trained PSG model directly on `AF7`/`AF8` with zero retraining (*van der Aar et al., 2025*).
     2. **From Scratch:** Train TinySleepNet from random initialization only on Bitbrain `AF7`/`AF8`.
     3. **Fine-Tuning:** Load PSG weights, lower learning rate from $10^{-4}$ to $10^{-5}$, and compare **Full Fine-Tuning** (all layers trainable, *van der Aar et al., 2024*) vs. **Head Re-training** (freezing CNN layers, retraining only top layers, *Waters & Clifford, 2022*).
     4. **Feature Matching:** Concurrently train source (`F3-M2`) and target (`AF7`/`AF8`) networks on paired simultaneous recordings using the combined Cross-Entropy + MSE Feature Matching loss $L = L_C(\hat{y}_S, y_S) + L_C(\hat{y}_T, y_T) + \lambda_1 L_M(f_S, f_T) + \lambda_2 L_2$ (*Heremans et al., 2022*, Eq. 1, p. 5).
