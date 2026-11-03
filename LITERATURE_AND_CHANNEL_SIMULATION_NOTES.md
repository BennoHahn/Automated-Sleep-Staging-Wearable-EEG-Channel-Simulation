# Literature Review, Wearable Channel Simulation Methods, and Dataset Mapping

This document serves as the persistent reference for the project's scientific rationale, literature findings from `/papers`, mathematical channel simulation derivations, and dataset mappings across **Phase 1 (SHHS1 PSG Channel Ablation & Wearable Simulation)** and **Phase 2 (Simultaneous PSG + Real Wearable Validation & Transfer Learning)**.

---

## 1. Project Concept & Two-Phase Strategy

Clinical Polysomnography (PSG) is the gold standard for sleep staging, but its full scalp montage (`F3, F4, C3, C4, O1, O2` referenced to mastoids `M1/M2`, plus `EOG` and `EMG`) is impractical for longitudinal home monitoring. Consumer and clinical-grade wearable EEG devices (forehead patches, sleep headbands, eye-masks, around-ear cEEGrid, and in-ear buds) restrict electrodes to **hairless regions** ( Primarily the forehead `Fp1, Fp2, Fpz, AF7, AF8` and periorbital/temple regions, sometimes paired with mastoid `M1/M2` or back-of-head `O1/O2`).

### Phase 1: Simulating Reduced & Wearable Sensors on Clinical PSG (SHHS1)
Because large public PSG repositories (like **SHHS**) do not include dedicated wearable hardware channels, we first simulate wearable/reduced sensor setups from PSG electrodes using two complementary approaches:
1. **Option 1 — Location-Close PSG Sensors (Anatomical Proxies):**
   Selecting standard PSG channels that sit in or immediately adjacent to wearable electrode locations (e.g., **EOG electrodes** at the outer canthus of the eye near `AF7/AF8/Fp1/Fp2`, or single-lead mastoid-referenced central/frontal EEG).
2. **Option 2 — Mathematical Channel Simulations (Re-referencing & Linear Combinations):**
   Calculating how wearable channels look by linearly combining or re-referencing PSG channels:
   - **Bilateral averaging** to simulate a **midline forehead electrode with a mastoid reference** (e.g., $\text{Fz-M2} = \frac{(\text{F3-M2}) + (\text{F4-M2})}{2}$ or $\text{EOG}_{\text{avg}} = \frac{\text{EOG(L)} + \text{EOG(R)}}{2}$).
   - **Bipolar subtraction** to cancel out the shared mastoid reference and simulate a **self-contained, mastoid-free wearable strip/headband** (e.g., cross-forehead $\text{F3-F4} = (\text{F3-M2}) - (\text{F4-M2})$, cross-temple $\text{EOG(L)} - \text{EOG(R)}$, or front-to-back headband $\text{F3-O1} = (\text{F3-M2}) - (\text{O1-M2})$).

### Phase 2: Validation Against Real Wearable Data & Transfer Learning (BOAS & Dreem)
In Phase 2, we introduce datasets that contain **full 10–20 PSG alongside simultaneously recorded wearable EEG** (such as **Bitbrain Open Access Sleep [BOAS] `ds005555`** with PSG + wearable `AF7, AF8` headband, and **Dreem Open Datasets `DOD-H / DOD-O`**):
1. **Simulation Fidelity Validation:** Compare the mathematically simulated wearable channels (calculated from the PSG leads) directly against the **actual wearable channels** recorded simultaneously on the same subject and night.
2. **Transfer Learning Evaluation:** Evaluate how models trained on clinical PSG or simulated wearable channels transfer to real wearable hardware using four strategies:
   - *Direct Transfer* (zero-shot evaluation on target wearable)
   - *Training from Scratch* (target wearable only)
   - *Fine-Tuning* (full network vs. classifier head only)
   - *Feature Matching* (aligning intermediate latent representations for small target datasets $N \le 10$)

---

## 2. Master Table of Wearable Channel Simulations from Literature

The table below integrates the reference table with exact mathematical definitions and empirical findings verified from the papers in `/papers`:

| Simulated Wearable Montage | Used PSG Channels / Formula | Mastoid Required? | Reference Paper(s) in `/papers` | Key Biophysical & Empirical Findings |
| :--- | :--- | :--- | :--- | :--- |
| **`AF7-M2`, `Fp1-M2`**<br>*(Single-lead forehead + mastoid reference)* | `F3-M2` or `F4-M1`<br>*(or `C3-A2`/`C3-M2` when frontal leads are absent in SHHS)* | **Yes** | **Waters & Clifford (2022)**;<br>**van der Aar et al. (2024)** | Active frontal electrode paired with an electrically distant, quiet mastoid reference (`M2`). Preserves large-amplitude bilateral slow-wave (N3 delta) activity. Achieves $\kappa = 0.82\text{–}0.84$ in healthy subjects and $\kappa = 0.74\text{–}0.77$ in OSA/insomnia (van der Aar 2024). Note: For SHHS specifically, Waters & Clifford (2022, pp. 17–18) used `C3-A2` because SHHS lacks `F3/F4`. |
| **`Fpz-M2`**<br>*(Midline forehead + mastoid reference, e.g., Philips SmartSleep)* | `Fz-M2` calculated as:<br>$$\text{Sim\_Fz\_M2} = \frac{(\text{F3-M2}) + (\text{F4-M2})}{2}$$ | **Yes** | **van der Aar et al. (2025)** (pp. 316–317) | Target wearable is Philips SmartSleep (dry `Fpz` electrode + wet `M2` mastoid reference). Averaging `F3-M2` and `F4-M2` estimates midline `Fz-M2`. Direct transfer from simulated `Fz-M2` to real wearable `Fpz-M2` reaches $\kappa = 0.769$ (and $\kappa = 0.778$ after fine-tuning), outperforming training from scratch on small wearable cohorts ($N < 60$). |
| **`Fp1-Fp2` / `AF7-AF8`**<br>*(Self-contained cross-forehead patch/strip)* | `F3-F4` calculated as:<br>$$\text{Sim\_F3\_F4} = (\text{F3-M2}) - (\text{F4-M2})$$ | **No** | **van der Aar et al. (2024)** (pp. 4, 6–7, 11);<br>**Dreem / Guillot & Thorey (2021)** | Subtracting `(F3-M2) - (F4-M2)` cancels the shared mastoid `M2`, simulating a wireless forehead patch. **Crucial finding:** Because `F3` and `F4` are closely spaced and symmetrical across the hemispheres, synchronous bilateral slow waves (N3 delta) cancel out during subtraction! Direct transfer `F3-M2 -> F3-F4` drops from $\kappa = 0.84 \to 0.53$ (46–61% of N3 misclassified as N2). Training/fine-tuning directly on `F3-F4` recovers $\kappa$ to $0.66\text{–}0.73$, though N3 sensitivity remains lower than mastoid-referenced channels. |
| **`(Eye-mask / temple wearable) AF7-AF8`** | `EOG-L` (single lead),<br>`mean(EOG-L, EOG-R)` (mastoid ref), or<br>`EOG-L - EOG-R` (bipolar, no mastoid) | **Optional** | **Heremans et al. (2022)** (p. 3, Sec. 2.2.1);<br>**Huang et al. (2014)** (pp. 2, 10) | EOG electrodes sit on the outer canthus of the eyes (centimeters from `AF7/AF8/Fp1/Fp2` on hairless skin) and capture both frontal cortical EEG (delta slow waves, spindles) and ocular movements (REM rapid eye movements, N1 slow rolling eye movements). In Heremans et al. (2022, Table 2), training from scratch on EOG achieves $\kappa = 0.766$ (nearly identical to `C4-A1` EEG at $\kappa = 0.769$). |
| **`Fp1-O1`, `Fp2-O2`, `F3-O1`, `F4-O2`**<br>*(Front-to-back headband, e.g., Dreem 2/3)* | Fronto-occipital subtraction:<br>$$\text{Sim\_F3\_O1} = (\text{F3-M2}) - (\text{O1-M2})$$<br>$$\text{Sim\_F4\_O2} = (\text{F4-M1}) - (\text{O2-M1})$$ | **No, but Back-of-Head** | **Dreem / Guillot & Thorey (2021)**;<br>**Markov et al. (2025)**;<br>**Kosnacova et al. (2026)** | Simulates a rigid/elastic headband with dry forehead (`Fpz, F7, F8`) and occipital (`O1, O2`) sensors. Subtracting `(F3-M2) - (O1-M2)` eliminates the adhesive mastoid reference while maintaining a long front-to-back inter-electrode distance. This **avoids the bilateral N3 slow-wave cancellation** of `F3-F4` and captures both frontal slow waves/K-complexes and occipital alpha rhythms (Wake/N1). |
| **`F3-F4`, `Fp1-F3`, `Fp2-F4`**<br>*(Local frontal bipolar)* | $$\text{Sim\_F3\_F4} = (\text{F3-M2}) - (\text{F4-M2})$$ | **No** | **Dreem / Guillot & Thorey (2021)** (`DreemLearning_README.md`) | Included in DOD-H / DOD-O alongside fronto-occipital channels. Useful as an auxiliary local frontal channel for eye-blink/asymmetry detection, though inferior to fronto-occipital (`F3-O1`/`F4-O2`) or mastoid-referenced (`F3-M2`) channels for N3 staging on its own. |

---

## 3. Detailed Paper-by-Paper Synthesis (`/papers`)

### 3.1 Architecture & Baseline Sleep Staging Papers

1. **Supratak & Guo (2020) — *TinySleepNet: An Efficient Deep Learning Model for Sleep Stage Scoring Based on Raw Single-Channel EEG*** (`p08.txt`)
   - **Role in Project:** Primary deep learning architecture for Phase 1 & Phase 2 (`src/models/tinysleepnet.py`).
   - **Datasets & Channels Evaluated:** MASS (`F4-EOG(L)` and `C4-EOG(L)` derived via montage re-referencing), Sleep-EDF (`Fpz-Cz` and `Pz-Oz`).
   - **Architecture Details:**
     - **Representation Learning (`FeatureExtractor`):** 4 convolutional blocks (`Conv1D -> BatchNorm1d -> ReLU`) with two `MaxPool1d` layers and `Dropout(0.5)`.
       - `Conv1`: 128 filters, kernel size `Fs/2` (`50` at 100 Hz). *(Note: Paper Fig. 1 text mentions stride `Fs/4 = 25` while official TinySleepNet/DeepSleepNet code uses stride `Fs/16 = 6`; our implementation dynamically computes `feat_dim` after a dummy forward pass so both stride settings work seamlessly).*
       - `MaxPool1`: kernel size `8`, stride `8`, followed by `Dropout(0.5)`.
       - `Conv2`, `Conv3`, `Conv4`: 128 filters each, kernel size `8`, stride `1`, `same` padding.
       - `MaxPool2`: kernel size `4`, stride `4`, followed by `Dropout(0.5)`.
     - **Sequential Learning (`SequenceClassifier`):** Single-layer **unidirectional LSTM** (`hidden_size = 128`) followed by `Dropout(0.5)` and `Linear(128, 5)`.
     - **Context-Free Variant (`use_context: false`):** Replaces the LSTM with `Linear(feat_dim, 5)` to isolate pure spatial/channel signal quality within a single 30 s epoch from temporal sequence memory.
   - **Training Protocol & Data Augmentation:**
     - Adam optimizer (`lr = 1e-4`, `betas = (0.9, 0.999)`, `weight_decay = 1e-3`), gradient clipping (`max_norm = 5.0`).
     - Weighted Cross-Entropy loss: class weights `[1.0, 1.5, 1.0, 1.0, 1.0]` for `[W, N1, N2, N3, REM]` to boost `N1` sensitivity, or balanced class weights.
     - Two data augmentations during training: (1) random time-shift of each epoch by $\pm 10\%$ of epoch duration ($\pm 3\text{ s}$), and (2) random sequence start offset (`0..5` epochs) per epoch pass.
   - **Efficiency:** Only `~1.3M` parameters (vs. `~2.6M` in DeepSleepNet), achieving `85.4%` accuracy / $\kappa = 0.79$ on MASS and `83.1%` / $\kappa = 0.77$ on Sleep-EDF.

2. **Supratak et al. (2017) — *DeepSleepNet: A Model for Automatic Sleep Stage Scoring Based on Raw Single-Channel EEG*** (`p01.txt`)
   - **Role in Project:** Predecessor to TinySleepNet; establishes dual-branch (small-filter temporal + large-filter frequency) representation learning + two-step pre-training/fine-tuning with BiLSTM, and demonstrates bipolar EEG–EOG re-referencing (`F4-EOG(L)`, `C4-EOG(L)`).

3. **Huang et al. (2014) — *Knowledge-Based Identification of Sleep Stages Based on Two Forehead EEG Channels*** (`p00.txt`)
   - **Role in Project:** Foundational study demonstrating that two hairless forehead channels (`Fp1-A2` and `Fp2-A1`) capture both frontal EEG rhythms (delta, alpha, spindles) and ocular activity (REM/N1 eye movements), because `Fp1/Fp2` and `EOG-L/EOG-R` sit in the same frontopolar/periorbital region. Highlights that N1 and N3 are the stages most affected by moving from full scalp PSG to forehead electrodes.

---

### 3.2 Channel Mismatch, Wearable Simulation & Transfer Learning Papers

4. **van der Aar et al. (2024) — *Deep Transfer Learning for Automated Single-Lead EEG Sleep Staging with Channel and Population Mismatches*** (`p04.txt`)
   - **Role in Project:** Primary theoretical and empirical reference for **mastoid-referenced (`F3-M2`) vs. mastoid-free cross-forehead bipolar (`F3-F4 = (F3-M2) - (F4-M2)`)** wearable simulations.
   - **Key Findings:**
     - **Mastoid-referenced `F3-M2`** achieves $\kappa = 0.82\text{–}0.84$ in healthy subjects and $\kappa = 0.74\text{–}0.77$ in sleep-disordered (OSA/insomnia) populations.
     - **Cross-forehead bipolar `F3-F4`** suffers a dramatic drop under direct transfer from `F3-M2` ($\kappa = 0.84 \to 0.53$ in healthy; $0.76 \to 0.43$ in OSA/insomnia), with **46% to 61% of N3 epochs misclassified as N2**.
     - **Why N3 Cancels Out in `F3-F4`:** Slow-wave sleep (N3) is characterized by high-amplitude ($>75\,\mu\text{V}$), low-frequency ($0.5\text{–}4\text{ Hz}$) delta waves that are **highly synchronous and symmetrical across the left (`F3`) and right (`F4`) frontal hemispheres**. Subtracting `(F3-M2) - (F4-M2)` cancels out the common delta wave, leaving a low-amplitude residual that looks like N2 to a model trained on `F3-M2`.
     - **Recovery via Training/Fine-Tuning on `F3-F4`:** Training from scratch or fine-tuning on `F3-F4` teaches the CNN to adjust its amplitude thresholds, recovering $\kappa$ to $0.66\text{–}0.73$ (though still statistically lower than `F3-M2`, Cohen's $d = 0.96$).
     - **Channel Mismatch > Population Mismatch:** Channel montage mismatch (`F3-M2` vs. `F3-F4`) degrades accuracy far more severely than clinical population mismatch (healthy vs. OSA/insomnia).

5. **van der Aar et al. (2025) — *Optimizing Wearable Single-Channel EEG Sleep Staging in a Heterogeneous Sleep-Disordered Population Using Transfer Learning*** (`p05.txt`)
   - **Role in Project:** Primary reference for **simulating a midline forehead wearable channel (`Fpz-M2`) by averaging left and right PSG leads (`Fz-M2 = (F3-M2 + F4-M2)/2`)** and transferring to actual wearable hardware (Philips SmartSleep headband).
   - **Key Findings:**
     - Pre-training on 901 clinical PSGs using simulated `Fz-M2 = 0.5 * (F3-M2 + F4-M2)` and applying **direct transfer** to real wearable `Fpz-M2` already achieves $\kappa = 0.769$ (`82.9%` accuracy).
     - **Fine-tuning** with real wearable recordings improves performance to $\kappa = 0.778$ (`83.6%` accuracy), whereas **training from scratch** on the wearable dataset only reaches $\kappa = 0.751$ (and performs poorly when $N < 60$ recordings).
     - Demonstrates that matching the **reference scheme** (both simulated source `Fz-M2` and real wearable target `Fpz-M2` use a mastoid `M2` reference) makes zero-shot direct transfer remarkably effective.

6. **Heremans et al. (2022) — *Feature Matching as Improved Transfer Learning Technique for Wearable EEG*** (`p02.txt`)
   - **Role in Project:** Primary reference for **(a) using PSG EOG as a location-close wearable proxy (`EOG-L`, `mean(EOG-L, EOG-R)`, `EOG-L - EOG-R`)** and **(b) the Feature Matching transfer learning loss (`src/training/transfer.py`)**.
   - **Wearable setups studied:**
     - Clinical PSG source (`C4-A1` or `Fpz-Cz`) $\to$ Target wearable proxies: **EOG** (`MASS-EOG` = `mean(EOG-L, EOG-R)` and `Sleep-EDF-EOG` = bipolar `EOG horizontal`) and **around-the-ear cEEGrid** (`right-ear bipolar` vs. `cross-head right-to-left ear`).
   - **Feature Matching Algorithm (Equations 1–4, pp. 6–8):**
     - Standard fine-tuning overfits badly when the target wearable dataset is very small ($N = 2\text{ to }10$ subjects).
     - Feature Matching freezes the classifier head and updates the feature extractor ($W_{\text{feat}}$) to minimize:
       $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{class}} + \lambda_1 \mathcal{L}_{\text{FM}} + \lambda_2 \mathcal{L}_{\text{reg}}$$
       where $\mathcal{L}_{\text{FM}} = \text{MSE}(\bar{f}_{\text{target}}, \bar{f}_{\text{source}})$ aligns the class-conditional mean feature vectors of the target wearable domain to the source PSG domain, and $\mathcal{L}_{\text{reg}} = \|W_{\text{feat}} - W_{\text{feat, init}}\|_2^2$ prevents catastrophic forgetting.
   - **Key Empirical Results (Table 2, p. 14):**
     - On `MASS C4-A1 -> EOG`: Direct transfer gets $\kappa = 0.592$; fine-tuning on 10 subjects gets $\kappa = 0.680$; **Feature Matching** on 10 subjects reaches $\kappa = 0.715$; full training from scratch on all EOG data reaches $\kappa = 0.766$ (vs. $\kappa = 0.769$ on `C4-A1`).

7. **Waters & Clifford (2022) — *Comparison of Deep Transfer Learning Algorithms and Transferability Measures for Wearable Sleep Staging*** (`p07.txt`)
   - **Role in Project:** Benchmarks transfer learning from 6 clinical PSG datasets (**including SHHS**, $N = 6441$, using **`C3-A2` / `C3-M2`**) to the **X4 Sleep Profiler** forehead wearable (`AF7-Fpz`).
   - **Key Findings:**
     - Compares **Full Fine-Tuning** vs. **Last-Layer (Head-Only) Fine-Tuning**, plus Domain Adversarial Neural Networks (DANN) and subspace alignment.
     - Pre-training on **SHHS (`C3-A2`)** provides the highest transferability and lowest error when fine-tuning on the forehead wearable (`AF7-Fpz`), because of SHHS's large sample size and diversity.
     - **Full Fine-Tuning** consistently outperforms head-only fine-tuning once $\ge 10\text{–}20$ target recordings are available, because low-level CNN filters must adapt to the amplitude and spectral differences of forehead dry/hydrogel electrodes.

8. **He et al. (2022/2023) — *Cross-Scenario Automatic Sleep Stage Classification Using Transfer Learning and Single-Channel EEG*** (`p09.txt`)
   - **Role in Project:** Investigates cross-dataset and cross-channel transfer learning using **SHHS** (`C4-A1`, `C3-A2`) and Sleep-EDF (`Fpz-Cz`, `Pz-Oz`), confirming that `C4-A1` and `C3-A2` on SHHS provide strong single-channel representations.

---

### 3.3 Wearable Hardware Reviews & Clinical Evaluation Meta-Analysis

9. **Markov et al. (2025) — *Evaluating the Performance of Wearable EEG Sleep Monitoring Devices: A Meta-Analysis Approach*** (`p06.txt`)
   - **Role in Project:** Defines the **clinical sleep architecture evaluation framework** (`src/evaluation/clinical_sleep.py`) and compares headband/forehead/in-ear wearable devices (Dreem 2/3, Sleep Profiler, Prodigy, Zmachine, Philips SmartSleep, Somfit, IDUN Guardian, cEEGrid).
   - **Standardized Equations (Equations 1–19, pp. 18–20):**
     - **Total Recording Time (TRT / TIB):** Total duration from lights-off to lights-on (minutes).
     - **Total Sleep Time (TST):** Total duration of all non-Wake (`N1 + N2 + N3 + REM`) epochs in minutes: $\text{TST} = N_{\text{sleep}} \times 0.5\text{ min}$.
     - **Sleep Efficiency (SE):** $\text{SE (\%)} = \frac{\text{TST}}{\text{TRT}} \times 100$.
     - **Sleep Onset Latency (SOL):** Elapsed minutes from recording start to the **first epoch of any sleep stage** (`N1, N2, N3, or REM`).
     - **Wake After Sleep Onset (WASO):** Total minutes of `Wake` epochs occurring **after** sleep onset and **before** the final awakening.
     - **REM Latency (REML):** Elapsed minutes from **sleep onset** to the first `REM` epoch.
     - **Stage Durations & Percentages (`%W, %N1, %N2, %N3, %REM`):** Percentage of `TST` (for sleep stages) or `TRT` (for Wake).
     - **Method Agreement Metrics:** Mean bias ($\bar{d} = \frac{1}{n}\sum (\hat{y}_i - y_i)$), SD of bias ($s_d$), **95% Bland-Altman Limits of Agreement** ($\text{LoA} = \bar{d} \pm 1.96\,s_d$), Mean Absolute Error (MAE), Pearson's $r$, and Intraclass Correlation Coefficient (ICC).
   - **Meta-Analysis Benchmark Numbers:** Across 35 studies, wearable EEG devices show a pooled bias vs. PSG of $\approx -0.7\text{ min}$ for TST, $+2.3\text{ min}$ for SOL, $-2.1\text{ min}$ for WASO, and achieve their highest epoch accuracy on `W`, `N2`, and `REM`, with the largest discrepancies in `N1` and `N3`.

10. **Kosnacova et al. (2026) — *Smart Wearable EEG Devices: A Review*** (`p03.txt`)
    - **Role in Project:** Comprehensive hardware taxonomy of wearable EEG form factors:
      - **Headbands** (Dreem 2/3 `Fpz, F7, F8, O1, O2`; Muse S `AF7, AF8, TP9, TP10`; Bitbrain Diadem `AF7, AF8`; Philips SmartSleep `Fpz-M2`)
      - **Forehead Patches & Masks** (X4 Sleep Profiler `AF7-Fpz, AF8-Fpz`; Somfit; Zmachine `A1-A2/mastoid-forehead`; Lumos/Naptime eye masks)
      - **Around-Ear & In-Ear Devices** (cEEGrid, IDUN Guardian, NextSense, ear-EEG)

---

## 4. Connecting Literature to Our Datasets: Phase 1 (SHHS1) vs. Phase 2 (BOAS & Dreem)

### 4.1 Phase 1 Dataset on Disk: SHHS1 (`data/raw/edf/shhs1-200001.edf` .. `200010.edf`)
Inspection of the 10 EDF headers in `data/raw/edf/` confirmed the exact channels and sampling rates recorded in SHHS1:

| SHHS1 EDF Channel Label | Sampling Rate | Physical Units | Clinical / Anatomical Meaning | Used in Our Pipeline? |
| :--- | :---: | :---: | :--- | :---: |
| **`EEG`** | `125 Hz` | `uV` | **C4-A1** (Right central EEG referenced to left mastoid) | **Yes** (Resampled to `100 Hz`) |
| **`EEG(sec)`** | `125 Hz` | `uV` | **C3-A2** (Left central EEG referenced to right mastoid) | **Yes** (Resampled to `100 Hz`) |
| **`EOG(L)`** | `50 Hz` | `uV` | **Left Outer Canthus EOG** (near `AF7/Fp1`, mastoid/nasion ref) | **Yes** (Upsampled to `100 Hz`) |
| **`EOG(R)`** | `50 Hz` | `uV` | **Right Outer Canthus EOG** (near `AF8/Fp2`, mastoid/nasion ref) | **Yes** (Upsampled to `100 Hz`) |
| **`EMG`** | `125 Hz` | `uV` | **Bipolar Submental (Chin) EMG** | **Yes** (Resampled to `100 Hz`) |
| `ECG`, `SaO2`, `H.R.`, `THOR RES`, `ABDO RES`, `POSITION`, `LIGHT`, `NEW AIR`/`AIRFLOW`, `OX stat` | `1–125 Hz` | Various | Non-EEG cardiorespiratory & position sensors | Ignored |

#### Approved Phase 1 SHHS1 Channel Configurations (`configs/ablation_channels.yaml`)
Because SHHS1 records `C4-A1`, `C3-A2`, `EOG(L)`, `EOG(R)`, and `EMG` (without `F3/F4/O1/O2`), we evaluate **12 configurations** on SHHS1 grouped into three categories:

1. **Category A — Clinical Multi-Channel Baselines (Upper Bound):**
   - `full_psg` (`5ch`): `[C4-A1, C3-A2, EOG(L), EOG(R), EMG]` — Full neurological PSG montage.
   - `eeg_eog` (`4ch`): `[C4-A1, C3-A2, EOG(L), EOG(R)]` — PSG without chin EMG.
   - `dual_eeg` (`2ch`): `[C4-A1, C3-A2]` — Bilateral central EEG.

2. **Category B — Option 1: Location-Close Single/Dual Sensors:**
   - `single_eeg_c4` (`1ch`): `[C4-A1]` — Standard single-lead PSG (`EEG`).
   - `single_eeg_c3` (`1ch`): `[C3-A2]` — Standard single-lead PSG (`EEG(sec)`), exact SHHS source montage used by Waters & Clifford (2022).
   - `single_eog_l` (`1ch`): `[EOG(L)]` — Direct anatomical proxy for an eye-mask/temple/forehead wearable (`AF7/Fp1` neighborhood), following Heremans et al. (2022).
   - `dual_eog` (`2ch`): `[EOG(L), EOG(R)]` — Two-channel periorbital/forehead proxy, following Huang et al. (2014).

3. **Category C — Option 2: Mathematically Simulated Wearable Derivations:**
   - `sim_bipolar_eog` (`1ch`): `EOG(L) - EOG(R)` — **Simulated mastoid-free cross-forehead/temple strip (`AF7-AF8` / `Fp1-Fp2`)**. Cancels the common reference, testing a self-contained horizontal eye-mask/forehead strip (Heremans 2022; van der Aar 2024).
   - `sim_avg_eog` (`1ch`): `(EOG(L) + EOG(R)) / 2` — **Simulated mastoid-referenced midline forehead sensor (`Fpz-M2`)** using the bilateral averaging formula of Heremans (2022) (`MASS-EOG`) and van der Aar (2025).
   - `sim_bipolar_eeg` (`1ch`): `(C3-A2) - (C4-A1)` — **Simulated cross-hemisphere bipolar subtraction**, testing van der Aar et al. (2024)'s bilateral slow-wave (N3 delta) cancellation effect on central EEG.
   - `sim_avg_eeg` (`1ch`): `((C3-A2) + (C4-A1)) / 2` — **Simulated mastoid-referenced midline central sensor (`Cz-M`)**, testing van der Aar et al. (2025)'s bilateral averaging formula on central EEG.
   - `sim_eeg_minus_eog` (`1ch`): `(C4-A1) - EOG(L)` — **Simulated mastoid-free fronto-central differential (`C4-EOG(L)`)**, following Supratak et al. (2017, 2020).

---

### 4.2 Phase 2 Datasets: Full 10–20 PSG + Real Wearable Channels

Our `src/data/channel_sim.py` and `configs/ablation_channels.yaml` also include the **exact 10–20 PSG mathematical formulas from the Master Table** so they work out-of-the-box when loading Phase 2 datasets:

1. **Bitbrain Open Access Sleep (BOAS, OpenNeuro `ds005555`, $N = 128$ nights)** (`papers/The Bitbrain Open Access Sleep (BOA.txt`):
   - **Simultaneous Recordings on Every Night:**
     - **Clinical PSG (`Micromed`):** `F3, F4, C3, C4, O1, O2` (mastoid-referenced), `EOG`, `EMG`, respiratory/spo2/heart rate.
     - **Real Wearable EEG Headband (`Bitbrain`):** Dry forehead electrodes at **`AF7` and `AF8`** (`256 Hz`, downsampled to `100 Hz`).
   - **Triple Consensus Labels:** Scored independently by 3 expert clinicians + AI consensus into AASM 5 stages (`W=0, N1=1, N2=2, N3=3, REM=4`).
   - **What BOAS Enables in Phase 2:**
     - Compute every exact formula from the Master Table using BOAS's PSG leads (`F3-M2`, `Sim_Fz_M2 = (F3-M2 + F4-M2)/2`, `Sim_F3_F4 = (F3-M2) - (F4-M2)`, `Sim_F3_O1 = (F3-M2) - (O1-M2)`, `EOG-L`) and **directly compare them against the simultaneously recorded wearable `AF7`, `AF8`, and `AF7-AF8` channels** on the exact same nights!
     - Perform transfer learning from SHHS1 / BOAS-PSG $\to$ BOAS Real Wearable (`AF7, AF8`).

2. **Dreem Open Datasets (`DOD-H` $N=25$ Healthy, `DOD-O` $N=55$ OSA)** (`papers/DreemLearning_README.md`):
   - **Channels Provided (`250 Hz`):** `F3_M2`, `F4_M1`, `F3_F4`, `F3_O1`, `F4_O2`, `C3_M2`, `O1_M2`, `EOG1`, `EOG2`, `EMG`.
   - **What DOD Enables in Phase 2:** Direct comparison of pre-computed clinical `F3_M2` and `C3_M2` against mastoid-free cross-forehead `F3_F4` and fronto-occipital headband `F3_O1` / `F4_O2` in both healthy (`DOD-H`) and sleep-apnea (`DOD-O`) cohorts.
