# Automated Sleep Staging & Wearable EEG Channel Simulation Pipeline

Modular PyTorch pipeline for 5-stage AASM sleep staging (`Wake`, `N1`, `N2`, `N3`, `REM`) using **TinySleepNet**, comparing clinical Polysomnography (PSG) channels against location-close and mathematically simulated wearable EEG derivations.

See:
- [`PROJECT_INSTRUCTIONS.md`](PROJECT_INSTRUCTIONS.md) & [`PROJECT_SPEC.md`](PROJECT_SPEC.md) for architectural and experimental specifications.
- [`LITERATURE_AND_CHANNEL_SIMULATION_NOTES.md`](LITERATURE_AND_CHANNEL_SIMULATION_NOTES.md) for the complete literature synthesis, Master Table of Wearable Channel Simulations, and dataset mappings (**SHHS1** vs. **Bitbrain BOAS `ds005555`** & **Dreem DOD**).

---

## 1. Environment Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2. Main Pipeline Notebook (`phase1_pipeline.ipynb`)

Open [`phase1_pipeline.ipynb`](phase1_pipeline.ipynb) in VS Code / Jupyter using the `.venv` Python kernel to run the interactive 5-step pipeline:
1. **Step 1 — Standardized Preprocessing (`src/data/preprocessing.py`):** 60 Hz notch, 100 Hz polyphase resampling, 5th-order zero-phase Butterworth bandpass (`0.2–49 Hz`), 30s epoching, and AASM 5-stage mapping.
2. **Step 2 — Channel Ablation & Wearable Simulation (`src/data/channel_sim.py`):** Evaluates clinical baselines (`full_psg`, `eeg_eog`, `dual_eeg`), location-close proxies (`single_eeg_c4`, `single_eeg_c3`, `single_eog_l`, `dual_eog`), and mathematically simulated wearable derivations (`sim_bipolar_eog`, `sim_avg_eog`, `sim_bipolar_eeg`, `sim_avg_eeg`, `sim_eeg_minus_eog`), plus 10–20 PSG derivations (`Sim_Fz_M2`, `Sim_F3_F4`, `Sim_F3_O1`).
3. **Step 3 — TinySleepNet (`src/models/tinysleepnet.py`):** Toggles cleanly between **Model A (`use_context=False`, Context-Blind CNN)** and **Model B (`use_context=True`, Context-Aware CNN + Unidirectional LSTM)**.
4. **Step 4 — Subject-Independent 10-Fold Cross-Validation (`src/training/trainer.py`):** Strict subject-level 80/10/10 train/val/test splitting (`data/splits/shhs_10fold_splits.json`) with early stopping and separated `feature_extractor`/`classifier` checkpoints for Phase 2 Transfer Learning.
5. **Step 5 — Standardized Evaluation (`src/evaluation/metrics.py`, `src/evaluation/clinical_sleep.py`):** Exact **Markov et al. (2025)** Equations 1–29 ($\kappa$, MCC, Micro/Macro/Per-Stage metrics), Normalized Confusion Matrices (`mean ± SD`), Whole-Night Clinical Sleep Parameters (`TST`, `SE%`, `SOL`, `WASO`), Hypnograms, and **Bland–Altman** plots.

---

## 3. CLI Commands & Verification Tests

```powershell
# Run unit tests across all 5 steps
.venv\Scripts\python.exe -m unittest tests/test_phase1_pipeline.py -v

# Step 1: Preprocess all 10 SHHS1 EDF & Profusion XML files into data/processed/
.venv\Scripts\python.exe scripts/run_preprocessing.py

# Dry-run verification (3 subjects, 2 epochs, testing both use_context=False and use_context=True)
.venv\Scripts\python.exe scripts/run_ablation.py --dry-run

# Run ablation on specific channel configs
.venv\Scripts\python.exe scripts/run_ablation.py --channels full_psg single_eeg_c3 single_eog_l sim_avg_eog sim_bipolar_eog

# Evaluate and plot saved results
.venv\Scripts\python.exe scripts/evaluate_results.py
```
