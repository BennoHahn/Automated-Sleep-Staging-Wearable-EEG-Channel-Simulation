# Sleep Datasets Used Across Reference Papers (`papers/`)

This document summarizes all polysomnography (PSG) and wearable EEG sleep datasets used or documented across the reference literature in [`papers/`](../papers/), including subject counts, recording nights, exact EEG/EOG/EMG/wearable channels, scoring standards, and availability.

---

## 1. Master Comparison Table

| Dataset | Subjects ($N$) & Nights | EEG Channels Available | Frontal (`F3/F4/Fpz`)? | EOG / EMG / Wearable Channels | Scoring & $f_s$ | Used in Paper(s) in `papers/` | Access Status |
| :--- | :--- | :--- | :---: | :--- | :--- | :--- | :--- |
| **PhysioNet CinC 2018 (`CiCC` / `PN-set`, MGH)** | **$1,985$ subjects** ($994$ public train + $989$ test) | **6 AASM EEG:** `F3-M2`, `F4-M1`, `C3-M2`, `C4-M1`, `O1-M2`, `O2-M1` | Yes (`F3`, `F4`) | `1 EOG` (`E1-M2`), `Chin EMG`, ECG, Resp, $\text{SaO}_2$ | AASM, $200\text{ Hz}$ | Waters & Clifford (2022); He et al. (2023) | **Open** (PhysioNet) |
| **Wisconsin Sleep Cohort (`WSC`)** | **$1,123$ subjects** ($2,570$ nights) | **Frontal, Central, Occipital:** `F3`/`Fz`, `C3-M2` (`C3-A2`), `C4-M1`, `O1`/`Oz` | Yes (`F3`/`Fz`) | `2 EOG`, `Chin EMG`, ECG, Resp | R&K / AASM, $100\text{--}200\text{ Hz}$ | Waters & Clifford (2022) | **NSRR** (Credentialed) |
| **SOMNIA + HealthBed + PRISM** | **$901$ PSG subjects** + **$52$ wearable** | **Full 10–20 AASM:** `F3-M2`, `F4-M1`, `F4-M2`, `F3-F4`, `Fz-M2`, `C3-M2`, `C4-M1`, `O1-M2`, `O2-M1` | Yes (`F3`, `F4`, `Fpz`) | `2 EOG`, `Chin EMG` + **Philips SmartSleep (`Fpz-M2`)** in PRISM ($N=52$) | AASM, $256\text{--}512\text{ Hz}$ | van der Aar et al. (2024); van der Aar et al. (2025) | **Private** (Kempenhaeghe Hospital) |
| **MASS (Cohort 1, `SS1–SS5`)** | **$200$ subjects** (`SS1`: 53, `SS2`: 19, `SS3`: 62, `SS4`: 40, `SS5`: 26) | **Up to 20 scalp EEG** (`SS1, SS3, SS5` full 10–20: `Fp1, Fp2, F3, F4, Fz, F7, F8, C3, C4, Cz, P3, P4, Pz, T3–T6, O1, O2, Oz`; `SS4`: 4 EEG) | Yes (`Fp1`, `Fp2`, `F3`, `F4`, `Fz`) | `2 EOG` (`Left`, `Right`), `3 Chin EMG`, `1 ECG` | AASM (`SS1, SS3`) & R&K (`SS2, SS4, SS5`), $256\text{ Hz}$ | DeepSleepNet (2017); TinySleepNet (2020); Heremans et al. (2021); Waters & Clifford (2022) | **Restricted** (Academic request) |
| **ISRUC-Sleep (Coimbra)** | **$118\text{--}126$ subjects** (Subgroups 1–3) | **6 AASM EEG:** `F3-A2`, `C3-A2`, `O1-A2`, `F4-A1`, `C4-A1`, `O2-A1` | Yes (`F3`, `F4`) | `2 EOG` (`LOC-A2`, `ROC-A1`), `Chin EMG`, ECG | AASM, $200\text{ Hz}$ | Waters & Clifford (2022) | **Open** (ISRUC website) |
| **Bitbrain Open Access Sleep (`BOAS`, `ds005555`)** | **$100$ subjects** ($128$ nights) | **6 AASM PSG EEG:** `F3`, `F4`, `C3`, `C4`, `O1`, `O2` | Yes (`F3`, `F4` + `AF7`, `AF8`) | `2 EOG`, `Chin EMG` + **Simultaneous Forehead Headband (`AF7`, `AF8`)** | AASM (3 scorers), $256\text{ Hz}$ | Bitbrain `BOAS` Dataset Doc (2024) | **Open** (OpenNeuro `ds005555`) |
| **Dreem Open Datasets (`DOD-H` & `DOD-O`)** | **$80$ subjects** ($25$ healthy in `DOD-H` + $55$ OSA in `DOD-O`) | **8–12 PSG EEG:** `F3-M2`, `F4-M1`, `F3-F4`, `F4-O2`, `C3-M2`, `C4-M1`, `O1-M2`, `O2-M1` | Yes (`F3`, `F4` + `Fpz`, `F7`, `F8`) | `2 EOG`, `Chin EMG` + **Simultaneous Dreem Headband (`Fpz, F7, F8, O1, O2`)** | AASM (5 scorers), $250\text{ Hz}$ | `DreemLearning_README.md`; Markov et al. (2025) | **Open** (AWS S3 / Zenodo) |
| **X4 Sleep Profiler Target Cohort** | **$24$ subjects** ($75$ nights) | **Forehead Wearable EEG:** `AF7-Fpz` (from `AF7, AF8, Fpz`) | Yes (`AF7-Fpz`) | Forehead wearable headband only | AASM, $256\text{ Hz}$ | Waters & Clifford (2022) | **Private / Authors** |
| **NCTU Forehead + PSG Cohort** | **$10$ healthy subjects** ($10$ nights) | **6 AASM PSG EEG:** `F3-A2`, `F4-A1`, `C3-A2`, `C4-A1`, `O1-A2`, `O2-A1` + **Forehead `Fp1-A2`, `Fp2-A1`** | Yes (`Fp1`, `Fp2`, `F3`, `F4`) | `2 EOG`, `Chin EMG`, `Leg EMG`, ECG, Resp | AASM, $128\text{ Hz}$ | Huang et al. (2014) | **Private** |
| **SHHS (`SHHS1` + `SHHS2`)** | **$5,793$ (`SHHS1`)** / $6,441$ total ($1,738$ healthy 4-ch in our benchmark) | **2 Central EEG only:** `C4-A1`, `C3-A2` | **No** (Central only) | `2 EOG` (`EOG(L)`, `EOG(R)`), `Chin EMG`, ECG, Resp | R&K, $125\text{ Hz}$ | Waters & Clifford (2022); He et al. (2023); **This Thesis** | **NSRR** (In repository) |
| **MrOS Sleep Study** | **$2,900$ subjects** (older men) | **2 Central EEG only:** `C3-A2`, `C4-A1` | **No** (Central only) | `2 EOG`, `Chin EMG`, ECG, Resp | R&K / AASM, $256\text{ Hz}$ | Waters & Clifford (2022) | **NSRR** (Credentialed) |
| **Sleep-EDF Expanded (`SC` + `ST`)** | **$78$ healthy (`SC`, $153$ nights)** + $22$ `ST` ($44$ nights); v1 has $20$ subj ($39$ nights) | **2 Midline Bipolar EEG:** `Fpz-Cz`, `Pz-Oz` | **Partial** (`Fpz-Cz` uses frontopolar `Fpz`) | `1 Horizontal EOG`, `1 Chin EMG` | R&K, $100\text{ Hz}$ | DeepSleepNet (2017); TinySleepNet (2020); Heremans et al. (2021); He et al. (2023) | **Open** (PhysioNet) |

---

## 2. Grouping by Channel Architecture & Research Use Case

### A. Large Open/Credentialed PSG Cohorts WITH Full 6- to 20-Channel Montages (`F3/F4`, `C3/C4`, `O1/O2`)
1. **PhysioNet Computing in Cardiology Challenge 2018 (`CiCC` / `PN-set`, MGH) — $N = 1,985$ subjects ($994$ public training set)**
   - **Channels:** All 6 standard AASM derivations (`F3-M2`, `F4-M1`, `C3-M2`, `C4-M1`, `O1-M2`, `O2-M1`), `EOG (E1-M2)`, `Chin EMG` at $200\text{ Hz}$.
   - **Relevance:** Largest completely open-access dataset on PhysioNet containing modern AASM frontal (`F3/F4`), central (`C3/C4`), and occipital (`O1/O2`) channels.
2. **Wisconsin Sleep Cohort (`WSC`) — $N = 1,123$ subjects ($2,570$ nights)**
   - **Channels:** Laboratory PSG with frontal (`F3`/`Fz`), central (`C3`/`C4`), and occipital (`O1`/`Oz`) EEG, `2 EOG`, and `Chin EMG`. Available via NSRR.
3. **Montreal Archive of Sleep Studies (`MASS`, SS1–SS5) — $N = 200$ subjects**
   - **Channels:** Full 10–20 laboratory montage with up to 20 EEG channels (`Fp1, Fp2, F3, F4, Fz, C3, C4, Cz, P3, P4, Pz, O1, O2, Oz, T3–T6`), `2 EOG`, `3 EMG` at $256\text{ Hz}$.
   - **Relevance:** Used in TinySleepNet (2020) and DeepSleepNet (2017) to construct cross-modal frontal-to-EOG derivations (`F4-EOG(Left)`).
4. **ISRUC-Sleep (University of Coimbra) — $N = 118\text{--}126$ subjects**
   - **Channels:** 6 AASM EEG channels (`F3-A2`, `C3-A2`, `O1-A2`, `F4-A1`, `C4-A1`, `O2-A1`), `2 EOG` (`LOC-A2`, `ROC-A1`), `Chin EMG` at $200\text{ Hz}$.

### B. Open-Access Datasets with Simultaneous Full PSG AND Wearable Forehead Headband
1. **Bitbrain Open Access Sleep (`BOAS`, OpenNeuro `ds005555`, 2024) — $N = 100$ subjects ($128$ nights)**
   - **PSG Channels:** `F3`, `F4`, `C3`, `C4`, `O1`, `O2`, `EOG(L)`, `EOG(R)`, `Chin EMG`.
   - **Wearable Headband Channels:** Simultaneous 2-channel dry forehead EEG (`HB_1 ≈ AF7`, `HB_2 ≈ AF8`) at $256\text{ Hz}$, scored by 3 independent experts.
2. **Dreem Open Datasets (`DOD-H` & `DOD-O`, 2020) — $N = 80$ subjects ($25$ healthy, $55$ OSA)**
   - **PSG Channels:** `F3-M2`, `F4-M1`, `F3-F4`, `F4-O2`, `C3-M2`, `C4-M1`, `O1-M2`, `O2-M1`, `EOG1`, `EOG2`, `EMG`.
   - **Wearable Headband Channels:** Simultaneous 5-electrode dry EEG (`Fpz-O1`, `Fpz-O2`, `Fpz-F7`, `F8-F7`, `F7-O1`, `F8-O2`) at $250\text{ Hz}$, scored by 5 independent experts.

### C. Older R&K Ambulatory Cohorts Without `F3/F4`
1. **SHHS (`SHHS1`, 1995–1998) — $N = 5,793$ subjects ($1,738$ healthy 4-channel in our benchmark)**
   - **Channels:** Central EEG only (`C4-A1`, `C3-A2`), `2 EOG` (`EOG(L)`, `EOG(R)`), `Chin EMG` ($125\text{ Hz}$).
2. **MrOS Sleep Study — $N = 2,900$ subjects**
   - **Channels:** Central EEG only (`C3-A2`, `C4-A1`), `2 EOG`, `Chin EMG` ($256\text{ Hz}$).
3. **Sleep-EDF Expanded (1987–1994) — $N = 78$ healthy (`SC`) + $22$ `ST` subjects**
   - **Channels:** 2 midline bipolar EEG channels (`Fpz-Cz`, `Pz-Oz`), `1 horizontal EOG`, `1 Chin EMG` ($100\text{ Hz}$).
