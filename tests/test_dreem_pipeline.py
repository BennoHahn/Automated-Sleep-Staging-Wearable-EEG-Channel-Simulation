"""Unit and integration tests for Dreem Open Datasets (DOD-H + DOD-O) pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import h5py
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_dreem_training import run_dreem_pipeline, select_dreem_channel_configs
from src.data.channel_sim import (
    apply_channel_config,
    is_channel_config_compatible,
    load_ablation_configs,
)
from src.data.dataset import load_subject_records_from_dir
from src.data.dreem import (
    DODH_ONLY_EEG_CHANNELS,
    DODO_ONLY_EEG_CHANNELS,
    DREEM_ALL_EEG_CHANNELS,
    DREEM_SHARED_EEG_CHANNELS,
    _resolve_zenodo_download_url,
    discover_dreem_h5_files,
    ensure_dreem_cv_splits,
    filter_splits_for_subjects,
    preprocess_all_dreem_recordings,
    preprocess_dreem_subject,
    read_dreem_h5_recording,
)
from src.training.trainer import get_experiment_results_dir


def _write_synthetic_h5(
    path: Path,
    channels: tuple[str, ...],
    n_epochs: int = 30,
    fs: float = 250.0,
    unit: str = "uV",
    scale: float = 1.0,
    include_hypnogram: bool = True,
    seed: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_samples = int(round(n_epochs * 30.0 * fs))
    t = np.arange(n_samples, dtype=np.float64) / fs

    with h5py.File(path, "w") as f:
        sig_grp = f.create_group("signals")
        eeg_grp = sig_grp.create_group("eeg")
        eeg_grp.attrs["fs"] = fs
        eeg_grp.attrs["unit"] = unit
        for idx, ch in enumerate(channels):
            key = ch.replace("-", "_")
            wave = 20.0 * np.sin(2.0 * np.pi * (6.0 + idx) * t) + rng.normal(0.0, 4.0, size=n_samples)
            ds = eeg_grp.create_dataset(key, data=(wave * scale).astype(np.float32))
            ds.attrs["fs"] = fs
            ds.attrs["unit"] = unit
        if include_hypnogram:
            hyp = np.tile(np.array([0, 1, 2, 3, 4], dtype=np.int64), n_epochs // 5 + 1)[:n_epochs]
            hyp[0] = -1
            hyp[-1] = -1
            f.create_dataset("hypnogram", data=hyp)


class TestDreemPipeline(unittest.TestCase):
    def setUp(self) -> None:
        with open(PROJECT_ROOT / "configs" / "dreem_config.yaml", "r", encoding="utf-8") as f:
            self.dreem_cfg = yaml.safe_load(f)
        with open(PROJECT_ROOT / "configs" / "base_config.yaml", "r", encoding="utf-8") as f:
            self.shhs_cfg = yaml.safe_load(f)

    def test_dreem_configs_cover_all_14_singular_eeg_channels(self) -> None:
        cfgs = load_ablation_configs(PROJECT_ROOT / "configs" / "ablation_channels.yaml", section="dreem_configs")
        self.assertEqual(len(cfgs), 14)

        shared = select_dreem_channel_configs(cfgs, "shared")
        dodh = select_dreem_channel_configs(cfgs, "dodh")
        dodo = select_dreem_channel_configs(cfgs, "dodo")
        self.assertEqual(len(shared), 6)
        self.assertEqual(len(dodh), 6)
        self.assertEqual(len(dodo), 2)

        extracted_names = {
            spec["channels"][0]["name"] for spec in cfgs.values()
        }
        self.assertEqual(extracted_names, set(DREEM_ALL_EEG_CHANNELS))

        by_channel = select_dreem_channel_configs(cfgs, "FP1-F3,C3_M2")
        self.assertEqual(set(by_channel.keys()), {"dodh_fp1_f3", "dod_c3_m2"})

    def test_discovery_excludes_unscored_uuid_and_unit_scaling(self) -> None:
        dodh_chs = (*DREEM_SHARED_EEG_CHANNELS, *DODH_ONLY_EEG_CHANNELS)
        dodo_chs = (*DREEM_SHARED_EEG_CHANNELS, *DODO_ONLY_EEG_CHANNELS)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            h5_uv = root / "dod-h" / "subj_uv.h5"
            h5_v = root / "dod-h" / "subj_v.h5"
            h5_o = root / "dod-o" / "subj_o1.h5"
            excluded_o = root / "dod-o" / "5d8e93a5-d26a-436d-bc2c-e9527dc92cbf.h5"
            no_hyp = root / "dod-o" / "no_hyp.h5"

            _write_synthetic_h5(h5_uv, dodh_chs, n_epochs=15, unit="uV", scale=1.0, seed=7)
            _write_synthetic_h5(h5_v, dodh_chs, n_epochs=15, unit="V", scale=1e-6, seed=7)
            _write_synthetic_h5(h5_o, dodo_chs, n_epochs=15, unit="uV", scale=1.0, seed=9)
            _write_synthetic_h5(excluded_o, dodo_chs, n_epochs=15, include_hypnogram=False, seed=10)
            _write_synthetic_h5(no_hyp, dodo_chs, n_epochs=15, include_hypnogram=False, seed=11)

            discovered = discover_dreem_h5_files(raw_dir=root)
            sids = [d["subject_id"] for d in discovered]
            self.assertNotIn("dodo_5d8e93a5-d26a-436d-bc2c-e9527dc92cbf", sids)
            self.assertEqual(set(sids), {"dodh_subj_uv", "dodh_subj_v", "dodo_no_hyp", "dodo_subj_o1"})

            rec_uv = preprocess_dreem_subject(h5_uv, "dodh_subj_uv", self.dreem_cfg, cohort="dod-h")
            rec_v = preprocess_dreem_subject(h5_v, "dodh_subj_v", self.dreem_cfg, cohort="dod-h")
            self.assertIsNotNone(rec_uv)
            self.assertIsNotNone(rec_v)
            assert rec_uv is not None and rec_v is not None
            # -1 epochs (index 0 and 14) removed -> 13 valid epochs, 12 channels, 3000 samples (100 Hz)
            self.assertEqual(rec_uv["epochs"].shape, (13, 12, 3000))
            self.assertEqual(rec_uv["fs"], 100)
            self.assertTrue(np.all((rec_uv["labels"] >= 0) & (rec_uv["labels"] <= 4)))
            self.assertTrue(np.allclose(rec_uv["epochs"], rec_v["epochs"], atol=1e-4))
            # Check missing hypnogram returns None cleanly
            self.assertIsNone(preprocess_dreem_subject(no_hyp, "dodo_no_hyp", self.dreem_cfg, cohort="dod-o"))

    def test_preprocessing_compatibility_and_stratified_80_subject_splits(self) -> None:
        dodh_chs = (*DREEM_SHARED_EEG_CHANNELS, *DODH_ONLY_EEG_CHANNELS)
        dodo_chs = (*DREEM_SHARED_EEG_CHANNELS, *DODO_ONLY_EEG_CHANNELS)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = {
                **self.dreem_cfg,
                "paths": {
                    **self.dreem_cfg["paths"],
                    "raw_dir": str(root / "raw"),
                    "dodh_dir": str(root / "raw" / "dod-h"),
                    "dodo_dir": str(root / "raw" / "dod-o"),
                    "processed_dir": str(root / "processed"),
                },
            }
            _write_synthetic_h5(root / "raw" / "dod-h" / "h01.h5", dodh_chs, n_epochs=10, seed=1)
            _write_synthetic_h5(root / "raw" / "dod-o" / "o01.h5", dodo_chs, n_epochs=10, seed=2)
            _write_synthetic_h5(
                root / "raw" / "dod-o" / "5d8e93a5-d26a-436d-bc2c-e9527dc92cbf.h5",
                dodo_chs,
                n_epochs=10,
                include_hypnogram=False,
                seed=3,
            )

            saved = preprocess_all_dreem_recordings(cfg, verbose=False)
            self.assertEqual(len(saved), 2)

            records = load_subject_records_from_dir(root / "processed", lazy=False, required_channels=None)
            self.assertEqual(len(records), 2)

            dreem_cfgs = load_ablation_configs(PROJECT_ROOT / "configs" / "ablation_channels.yaml", section="dreem_configs")
            for cfg_name, spec in dreem_cfgs.items():
                compat = [r for r in records if is_channel_config_compatible(r["channel_names"], spec)]
                scope = spec["cohort_scope"]
                if scope == "shared":
                    self.assertEqual(len(compat), 2, cfg_name)
                elif scope == "dodh_only":
                    self.assertEqual(len(compat), 1, cfg_name)
                else:
                    self.assertEqual(len(compat), 1, cfg_name)

                derived_x, ch_names = apply_channel_config(compat[0], spec)
                self.assertEqual(derived_x.shape[1], 1)
                self.assertEqual(len(ch_names), 1)

            # Verify stratified 80-subject (25 DOD-H + 55 DOD-O) 10-fold CV split balance
            mock_ids = [f"dodh_{i:02d}" for i in range(25)] + [f"dodo_{i:02d}" for i in range(55)]
            splits = ensure_dreem_cv_splits(mock_ids, root / "splits" / "dreem_dod_10fold_splits.json", n_folds=10, seed=42)
            self.assertEqual(len(splits), 10)
            for sp in splits:
                self.assertEqual(len(sp["test"]), 8)
                self.assertEqual(len(sp["val"]), 8)
                self.assertEqual(len(sp["train"]), 64)
                n_h_test = sum(1 for s in sp["test"] if s.startswith("dodh_"))
                self.assertIn(n_h_test, {2, 3})

            dodh_splits = filter_splits_for_subjects(splits, [s for s in mock_ids if s.startswith("dodh_")])
            dodo_splits = filter_splits_for_subjects(splits, [s for s in mock_ids if s.startswith("dodo_")])
            self.assertEqual(len(dodh_splits), 10)
            self.assertEqual(len(dodo_splits), 10)
            self.assertTrue(all(len(s["test"]) >= 2 for s in dodh_splits))
            self.assertTrue(all(len(s["test"]) >= 5 for s in dodo_splits))

    def test_results_dir_preserves_shhs1_and_tags_dreem(self) -> None:
        shhs_dir = get_experiment_results_dir(PROJECT_ROOT, self.shhs_cfg, n_subjects=1771)
        dreem_dir = get_experiment_results_dir(PROJECT_ROOT, self.dreem_cfg, n_subjects=80)
        self.assertTrue(shhs_dir.name.startswith("shhs1_healthy_n1771_"))
        self.assertTrue(dreem_dir.name.startswith("dreem_dod_all_n80_"))

    def test_end_to_end_dry_run_training_on_shared_and_cohort_channels(self) -> None:
        args = argparse.Namespace(
            config="configs/dreem_config.yaml",
            ablation_config="configs/ablation_channels.yaml",
            download=False,
            preprocess_only=False,
            overwrite_prep=False,
            channels="dod_f3_f4,dodh_fp1_f3,dodo_o1_m2",
            limit=None,
            dry_run=True,
            no_context=False,
            class_weights=None,
            batch_size=8,
            max_epochs=1,
            early_stopping_patience=1,
            max_folds=1,
            force_retrain=True,
            eval_only=False,
        )
        _, summary_df = run_dreem_pipeline(args)
        self.assertEqual(set(summary_df["Config"]), {"dod_f3_f4", "dodh_fp1_f3", "dodo_o1_m2"})

    def test_ignores_macos_sidecars_extracts_local_zip_and_explains_invalid_h5(self) -> None:
        dodh_chs = (*DREEM_SHARED_EEG_CHANNELS, *DODH_ONLY_EEG_CHANNELS)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir(parents=True, exist_ok=True)
            real_h5_src = root / "tmp_real.h5"
            _write_synthetic_h5(real_h5_src, dodh_chs, n_epochs=8, seed=21)

            # Package real .h5 + macOS __MACOSX / ._* AppleDouble junk inside dodh.zip
            zip_path = raw / "dodh.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.write(real_h5_src, arcname="dodh/subj01.h5")
                zf.writestr("dodh/._subj01.h5", b"\x00\x05\x16\x07mac-appledouble")
                zf.writestr("__MACOSX/dodh/._subj01.h5", b"\x00\x05\x16\x07mac-appledouble")

            cfg = {
                **self.dreem_cfg,
                "paths": {
                    **self.dreem_cfg["paths"],
                    "raw_dir": str(raw),
                    "dodh_dir": str(raw / "dod-h"),
                    "dodo_dir": str(raw / "dod-o"),
                    "processed_dir": str(root / "processed"),
                },
            }
            saved = preprocess_all_dreem_recordings(cfg, verbose=False)
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0].name, "dodh_subj01.npz")
            self.assertFalse((raw / "dod-h" / "._subj01.h5").exists())

            # Fake non-AppleDouble non-HDF5 .h5 file must raise a contextual error
            bad_h5 = raw / "dod-h" / "corrupt_json.h5"
            bad_h5.write_bytes(b'{"id": "15900394", "key": "dodh.zip"}')
            with self.assertRaises(OSError) as ctx:
                preprocess_all_dreem_recordings(cfg, verbose=False)
            self.assertIn("corrupt_json.h5", str(ctx.exception))
            self.assertIn("JSON API metadata", str(ctx.exception))

            with self.assertRaises(OSError) as read_ctx:
                read_dreem_h5_recording(bad_h5)
            self.assertIn("corrupt_json.h5", str(read_ctx.exception))

    def test_zenodo_url_prefers_content_over_bare_self_metadata(self) -> None:
        url1 = _resolve_zenodo_download_url(
            {"links": {"self": "https://zenodo.org/api/records/15900394/files/dodh.zip", "content": "https://zenodo.org/api/records/15900394/files/dodh.zip/content"}},
            "15900394",
            "dodh.zip",
        )
        self.assertEqual(url1, "https://zenodo.org/api/records/15900394/files/dodh.zip/content")

        url2 = _resolve_zenodo_download_url(
            {"links": {"self": "https://zenodo.org/api/records/15900394/files/dodh.zip"}},
            "15900394",
            "dodh.zip",
        )
        self.assertEqual(url2, "https://zenodo.org/api/records/15900394/files/dodh.zip/content")


if __name__ == "__main__":
    unittest.main()
