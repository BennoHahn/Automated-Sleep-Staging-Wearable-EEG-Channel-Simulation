# End-to-End Methodology & Architecture Diagrams (`dual_eeg` on SHHS1)

This file contains the Mermaid source code (rendered natively on GitHub and VS Code) and links to the high-resolution PNG diagram for the **10-fold subject-independent CV training of `dual_eeg` (`C4-A1` + `C3-A2`)** on **1,738 healthy SHHS1 subjects** (`1,761,352` 30-second epochs).

---

## 1. End-to-End Pipeline & Training Architecture (Mermaid Flowchart)

- **Blue (`Custom Pipeline`)**: Custom-engineered modules written from scratch for this thesis (`src/data/preprocessing.py`, `src/data/channel_sim.py`, `src/evaluation/metrics.py`, `src/evaluation/clinical_sleep.py`).
- **Green (`TinySleepNet Spec`)**: Exact neural network architecture & signal augmentations from Supratak & Guo (EMBC 2020), custom re-implemented in modular PyTorch (`src/models/tinysleepnet.py`).
- **Purple (`Hybrid / Adapted`)**: TinySleepNet training protocol extended with subject-independent 10-fold CV, square-root inverse-frequency class weights (`w_c`), and `bfloat16` mixed precision (`src/data/dataset.py`, `src/training/trainer.py`).

```mermaid
flowchart TD
    classDef custom fill:#EFF6FF,stroke:#1D4ED8,stroke-width:2px,color:#1E3A8A
    classDef tsn fill:#ECFDF5,stroke:#047857,stroke-width:2px,color:#065F46
    classDef hybrid fill:#F5F3FF,stroke:#6D28D9,stroke-width:2px,color:#4C1D95
    classDef result fill:#FFFBEB,stroke:#B45309,stroke-width:2px,color:#78350F

    subgraph S1["Stage 1: Data Ingestion & Standardized Preprocessing (Custom Pipeline)"]
        A["1. SHHS1 Cohort Filter 5,804 PSG Nights -> 1,738 Subjects (1,761,352 30s epochs)"]:::custom
        B["2. Preprocessing <br/>• Resample to 100 Hz (3,000 samp/ep)<br/>• Bandpass Filter (0.2 - 49.0 Hz)<br/>• R&K -> 5 AASM Stages (W, N1, N2, N3=S3+S4, REM)<br/>• Stored in physical uV (float16 .npz)"]:::custom
        A --> B
    end

    subgraph S2["Stage 2: 10-Fold Subject Split & Augmentation (Hybrid)"]
        D["3. Subject-Independent 10-Fold Split<br/>• 80% Train | 10% Val | 10% Test<br/>"]:::hybrid
        E["4. Sequence Batching & GPU Augmentations <br/>• Contiguous Sequences (Context): M = 20 epochs (10 min), Batch = 32<br/>• Random time shift (+-10% = +-300 samples)<br/>• Random sign flip (50% prob, x -> -x) & seq offset (0..5 ep)"]:::hybrid
        B --> D --> E
    end

    subgraph S3["Stage 3: Multi-Channel TinySleepNet Architecture"]
        H["5. Tinysleep Arcitexture<br/>• Early Fusion for Multi-Channel Configuration"]:::tsn
        E --> H
    end

    subgraph S4["Stage 4: Optimization & Two-Tier Clinical Evaluation"]
        I["6. Loss & Optimization<br/>• Sqrt Inverse-Frequency Class Weights<br/>• Early Stopping on val_macro_f1 (patience=20, max_epochs=100)"]:::hybrid
        J["7. Result Evaluation"]:::result
        H --> I --> J
    end
```

---

## 2. Multi-Channel Early Fusion (`Conv1d_1`) vs. Separate CNN Branches (Mermaid Diagram)

```mermaid
flowchart LR
    classDef ch fill:#EFF6FF,stroke:#1D4ED8,stroke-width:2px,color:#1E3A8A
    classDef early fill:#ECFDF5,stroke:#047857,stroke-width:2px,color:#065F46
    classDef shared fill:#F8FAFC,stroke:#334155,stroke-width:2px,color:#0F172A

    C4["Channel 1: C4-A1<br/>Right Central EEG<br/>(640, 1, 3000) @ 100 Hz"]:::ch
    C3["Channel 2: C3-A2<br/>Left Central EEG<br/>(640, 1, 3000) @ 100 Hz"]:::ch

    STACK["Stacked Input Tensor<br/>(640, 2, 3000)"]:::ch
    C4 & C3 --> STACK

    CONV1["Conv1d_1 (Spatiotemporal Early Fusion)<br/>128 Filters x (2 Channels x 50 Samples)<br/>y_k(t) = Sum_tau [w_Right * C4 + w_Left * C3]<br/>Output: (640, 128, 500) | 12,800 params"]:::early

    POOL1["MaxPool1d(8,8) + Dropout(0.5)<br/>Output: (640, 128, 62)"]:::shared
    CONV24["Conv1d_2 -> Conv1d_3 -> Conv1d_4<br/>3x [128 filters, k=8, s=1, BN, ReLU]<br/>MaxPool1d(4,4) + Dropout(0.5) + Flatten<br/>Output: (640, 1920) | 394,240 params"]:::shared
    LSTM["SequentialLSTMHead (M = 20 Epochs)<br/>Reshape (32, 20, 1920) -> LSTM(128) -> FC(5)<br/>Output: (32, 20, 5) | 1,050,245 params"]:::shared

    STACK --> CONV1 --> POOL1 --> CONV24 --> LSTM
```

---

## 3. High-Resolution PNG Diagram

![Methodology Pipeline Diagram](figures/methodology_pipeline_dual_eeg.png)
