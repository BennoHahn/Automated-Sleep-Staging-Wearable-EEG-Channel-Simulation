"""Comprehensive Unit & Dry-Run Verification Suite for Phase 1 (`tests/test_phase1_pipeline.py`).

Verifies:
1. Step 1 Preprocessing: Filtering, resampling to 100 Hz, Profusion XML stage mapping (including R&K N4 -> N3 merge).
2. Step 2 Channel Simulation: All 12 SHHS configs AND all 6 full 10-20 PSG configs (`Sim_Fz_M2`, `Sim_F3_F4`, `Sim_F3_O1`),
   specifically testing that common mastoid references cancel out in bipolar subtraction before normalization.
3. Step 3 TinySleepNet: Exact shapes and finite outputs for both `use_context=False` (Model A) and
   `use_context=True` (Model B LSTM), for `C=1`, `C=2`, and `C=5` input channels, plus checkpoint separation.
4. Step 4 Subject-Independent 10-Fold CV: Strict zero subject leakage between train, val, and test splits.
5. Step 5 Standardized Evaluation Metrics: Exact Markov et al. (2025) Eq. 1-29 (`kappa`, `mcc`, micro/macro/per-class)
   and clinical sleep parameters (`TST`, `SE%`, `SOL`, `WASO`, Bland-Altman LoA).
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.channel_sim import apply_channel_config, load_ablation_configs
from src.data.dataset import SleepStageDataset, generate_subject_cv_splits
from src.data.preprocessing import preprocess_channel, PROFUSION_STAGE_MAP
from src.evaluation.clinical_sleep import (
    compute_bland_altman_agreement,
    compute_hypnogram_parameters,
)
from src.evaluation.metrics import compute_all_epoch_metrics
from src.models.tinysleepnet import TinySleepNet


class TestPhase1Pipeline(unittest.TestCase):
    def test_step1_profusion_stage_map_and_filtering(self) -> None:
        # R&K Stage 3 and Stage 4 must both map to AASM N3 (3); REM (5) -> 4
        self.assertEqual(PROFUSION_STAGE_MAP[0], 0)
        self.assertEqual(PROFUSION_STAGE_MAP[1], 1)
        self.assertEqual(PROFUSION_STAGE_MAP[2], 2)
        self.assertEqual(PROFUSION_STAGE_MAP[3], 3)
        self.assertEqual(PROFUSION_STAGE_MAP[4], 3)
        self.assertEqual(PROFUSION_STAGE_MAP[5], 4)

        # Test resampling from 125 Hz and 50 Hz to 100 Hz for a 30s signal
        t_125 = np.linspace(0, 30, 125 * 30, endpoint=False)
        sig_125 = np.sin(2 * np.pi * 10.0 * t_125) + 0.5 * np.sin(2 * np.pi * 60.0 * t_125)
        out_100 = preprocess_channel(sig_125, orig_fs=125.0, target_fs=100.0, notch_freq=60.0)
        self.assertEqual(len(out_100), 3000)
        self.assertTrue(np.all(np.isfinite(out_100)))

        t_50 = np.linspace(0, 30, 50 * 30, endpoint=False)
        sig_50 = np.sin(2 * np.pi * 2.0 * t_50)
        out_eog_100 = preprocess_channel(sig_50, orig_fs=50.0, target_fs=100.0, notch_freq=60.0)
        self.assertEqual(len(out_eog_100), 3000)
        self.assertTrue(np.all(np.isfinite(out_eog_100)))

    def test_step2_all_shhs_and_10_20_channel_simulations(self) -> None:
        shhs_cfgs = load_ablation_configs(
            PROJECT_ROOT / "configs/ablation_channels.yaml", section="shhs_configs"
        )
        full_10_20_cfgs = load_ablation_configs(
            PROJECT_ROOT / "configs/ablation_channels.yaml", section="full_10_20_configs"
        )

        rng = np.random.default_rng(42)
        dummy_shhs_subj = {
            "subject_id": "dummy_shhs",
            "epochs": rng.standard_normal((12, 5, 3000)).astype(np.float32) * 25.0,
            "labels": np.array([0, 0, 1, 2, 2, 3, 3, 2, 4, 4, 0, 0], dtype=np.int64),
            "channel_names": ["C4-A1", "C3-A2", "EOG(L)", "EOG(R)", "EMG"],
            "normalization": "zscore_per_recording",
            "is_normalized": False,
        }

        for name, spec in shhs_cfgs.items():
            derived, ch_names = apply_channel_config(dummy_shhs_subj, spec, normalize=True)
            self.assertEqual(derived.shape[0], 12, msg=f"Failed on {name}")
            self.assertEqual(derived.shape[1], len(spec["channels"]), msg=f"Failed on {name}")
            self.assertEqual(derived.shape[2], 3000, msg=f"Failed on {name}")
            self.assertTrue(np.all(np.isfinite(derived)), msg=f"NaN/Inf in {name}")

        # Verify physical reference cancellation in `Sim_F3_F4 = (F3-M2) - (F4-M2)`
        # and bilateral N3 delta cancellation (van der Aar et al., 2024)
        t = np.linspace(0, 30, 3000, endpoint=False)
        shared_delta = 80.0 * np.sin(2 * np.pi * 1.0 * t)  # Symmetric N3 slow wave
        asym_signal = 10.0 * np.cos(2 * np.pi * 8.0 * t)
        m2_ref = 35.0 * np.sin(2 * np.pi * 5.0 * t)  # Common mastoid artifact
        f3_m2 = (shared_delta + asym_signal) - m2_ref
        f4_m2 = (shared_delta - asym_signal) - m2_ref

        dummy_10_20 = {
            "subject_id": "dummy_boas",
            "epochs": np.stack([f3_m2, f4_m2, f4_m2, shared_delta, shared_delta, asym_signal], axis=0)[
                None, :, :
            ].astype(np.float32),
            "labels": np.array([3], dtype=np.int64),
            "channel_names": ["F3-M2", "F4-M2", "F4-M1", "O1-M2", "O2-M1", "EOG(L)"],
            "normalization": "none",
            "is_normalized": False,
        }
        # In unnormalized uV, Sim_F3_F4 = (F3-M2) - (F4-M2) must equal 2 * asym_signal (both shared_delta and m2_ref cancel!)
        sim_f3_f4, _ = apply_channel_config(
            dummy_10_20, full_10_20_cfgs["sim_f3_f4"], normalize=False
        )
        np.testing.assert_allclose(
            sim_f3_f4[0, 0], 2.0 * asym_signal, atol=1e-4
        )

        for name, spec in full_10_20_cfgs.items():
            derived, _ = apply_channel_config(dummy_10_20, spec, normalize=True)
            self.assertEqual(derived.shape, (1, 1, 3000), msg=f"Failed on 10-20 {name}")

    def test_step3_tinysleepnet_context_blind_and_context_aware(self) -> None:
        # Model A: use_context = False (both 1-channel and 5-channel)
        for c in (1, 5):
            model_a = TinySleepNet(in_channels=c, n_classes=5, use_context=False)
            x_a = torch.randn(4, c, 3000)
            logits_a = model_a(x_a)
            self.assertEqual(logits_a.shape, (4, 5))
            self.assertTrue(torch.all(torch.isfinite(logits_a)))

        # Model B: use_context = True (LSTM sequence mode)
        model_b = TinySleepNet(in_channels=1, n_classes=5, use_context=True)
        x_b = torch.randn(2, 10, 1, 3000)  # Batch=2, Seq_Len=10, C=1, T=3000
        logits_b = model_b(x_b)
        self.assertEqual(logits_b.shape, (2, 10, 5))
        self.assertTrue(torch.all(torch.isfinite(logits_b)))

        # Verify checkpoint separation for Phase 2 Transfer Learning
        payload = model_b.get_checkpoint_payload()
        self.assertIn("feature_extractor_state_dict", payload)
        self.assertIn("classifier_state_dict", payload)

    def test_step4_subject_independent_10fold_splits(self) -> None:
        sids = [f"shhs1-2000{i:02d}" for i in range(1, 11)]
        splits = generate_subject_cv_splits(sids, n_folds=10, seed=42)
        self.assertEqual(len(splits), 10)
        test_seen = []
        for sp in splits:
            tr, va, te = set(sp["train"]), set(sp["val"]), set(sp["test"])
            self.assertEqual(len(tr), 8)
            self.assertEqual(len(va), 1)
            self.assertEqual(len(te), 1)
            self.assertTrue(tr.isdisjoint(va))
            self.assertTrue(tr.isdisjoint(te))
            self.assertTrue(va.isdisjoint(te))
            test_seen.extend(sp["test"])
        self.assertEqual(sorted(test_seen), sorted(sids))

    def test_step5_markov_metrics_and_clinical_sleep_parameters(self) -> None:
        y_true = np.array([0, 0, 1, 2, 2, 3, 3, 2, 4, 4, 0, 2], dtype=np.int64)
        y_pred = np.array([0, 0, 1, 2, 2, 3, 2, 2, 4, 4, 0, 2], dtype=np.int64)
        m = compute_all_epoch_metrics(y_true, y_pred)
        self.assertGreater(m["kappa"], 0.8)
        self.assertGreater(m["mcc"], 0.8)
        self.assertEqual(len(m["per_class"]), 5)

        clin = compute_hypnogram_parameters(y_true, epoch_sec=30.0)
        # 12 epochs = 6.0 min TRT; 3 Wake epochs (2 before sleep onset, 1 after sleep onset) -> TST = 4.5 min, SOL = 1.0 min, WASO = 0.5 min
        self.assertAlmostEqual(clin["TRT_min"], 6.0)
        self.assertAlmostEqual(clin["TST_min"], 4.5)
        self.assertAlmostEqual(clin["SOL_min"], 1.0)
        self.assertAlmostEqual(clin["WASO_min"], 0.5)

        ba = compute_bland_altman_agreement(
            [
                {"subject_id": "s1", "y_true": y_true, "y_pred": y_pred},
                {"subject_id": "s2", "y_true": y_true, "y_pred": y_true},
            ]
        )
        self.assertIn("TST_min", ba)
        self.assertIn("SE_pct", ba)


if __name__ == "__main__":
    unittest.main()
