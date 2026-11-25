import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch
from zipfile import ZipFile

import numpy as np

from src import artifacts, evaluate, precision, report, reproduce


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.directory = Path(tmp.name)
        self.config, _ = artifacts.training_config(["--model", "multimodal", "--recipe", "v2"])
        self.dataset = {"file": "val.csv", "sha256": "fixture", "bytes": 50}
        history = self.directory / "history.csv"
        history.write_text("epoch,val_macro_f1,val_conflict_ap\n1,0.7,0.8\n")
        records = {}
        for key, metric, score in (("classification", "macro_f1", 0.7), ("warning", "conflict_ap", 0.8),
                                   ("last", None, None)):
            path = self.directory / f"{key}.pt"
            with ZipFile(path, "w") as stream:
                stream.writestr("fixture/data.pkl", key)
            records[key] = artifacts.checkpoint_record(path, 1, metric, score)
        self.checkpoint = records["classification"]
        self.manifest = {**self.config, "training_config": self.config, "checkpoints": records,
                         "config_sha256": artifacts.identity(self.config), "best_epoch": 1,
                         "history_sha256": artifacts.sha256(history), "dataset": {"val": self.dataset}}
        self.write_manifest()

    def write_manifest(self):
        artifacts.write_json(self.directory / "config.json", self.manifest)

    def test_missing_checkpoint_despite_config(self):
        self.assertEqual(artifacts.validate_training(self.directory, self.config), self.manifest)
        (self.directory / "warning.pt").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing trained checkpoint.*warning.pt"):
            artifacts.validate_training(self.directory, self.config)
        with contextlib.redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit):
            evaluate.main(["--run-dir", str(self.directory), "--split", "test"])
        self.assertIn("Missing trained checkpoint", stderr.getvalue())

    def test_corrupt_checkpoint_and_hash_mismatch(self):
        path = self.directory / "classification.pt"
        path.write_bytes(b"invalid")
        with self.assertRaisesRegex(ValueError, "Corrupt PyTorch checkpoint"):
            artifacts.validate_training(self.directory)
        with ZipFile(path, "w") as stream:
            stream.writestr("fixture/data.pkl", "different")
        with self.assertRaisesRegex(ValueError, "Checkpoint hash mismatch"):
            artifacts.validate_training(self.directory)

    def test_mismatched_configuration(self):
        other = {**self.config, "lr_text": 1e-5}
        self.assertNotEqual(artifacts.run_directory(self.config), artifacts.run_directory(other))
        with self.assertRaisesRegex(ValueError, "Stale or mismatched configuration"):
            artifacts.validate_training(self.directory, other)
        self.manifest["lr_text"] = other["lr_text"]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Stale or mismatched configuration"):
            artifacts.validate_training(self.directory)

    def evaluation_fixture(self):
        path = self.directory / "eval_val.json"
        label = {"macro_f1": 0.7, "accuracy": 0.8, "critical_recall": 0.6, "ece": 0.1, "per_class_f1": {}}
        warning = {"conflict_f1": 0.6, "conflict_precision": 0.5, "conflict_recall": 0.8, "conflict_ap": 0.8}
        metrics = {condition: {"all": {"text": label, "image": label, "warning": warning}}
                   for condition in artifacts.CONDITIONS}
        artifacts.write_json(path, metrics)
        hashes = {}
        arrays = {"text": np.zeros((2, 5)), "image": np.zeros((2, 5)), "agree": np.zeros(2),
                  "y_agree": np.array([0, 1]), "y_text_label": np.zeros(2), "y_image_label": np.zeros(2),
                  "tweet_id": np.array(["1", "2"]), "image_id": np.array(["a", "b"])}
        for condition in artifacts.CONDITIONS:
            archive = self.directory / f"pred_val_{condition}.npz"
            np.savez_compressed(archive, **arrays)
            hashes[condition] = artifacts.sha256(archive)
        meta = {"checkpoint": self.checkpoint, "dataset": self.dataset,
                "config_sha256": artifacts.identity(self.config), "metrics_sha256": artifacts.sha256(path),
                "predictions": hashes}
        artifacts.write_json(self.directory / "evaluation_val.json", meta)
        return meta

    def evaluation_complete(self):
        return artifacts.evaluation_complete(self.directory, "val", self.config, self.checkpoint, self.dataset)

    def test_incomplete_evaluation_artifacts(self):
        self.evaluation_fixture()
        self.assertTrue(self.evaluation_complete())
        (self.directory / "pred_val_image_missing.npz").unlink()
        self.assertFalse(self.evaluation_complete())

    def test_matching_hash_does_not_accept_invalid_archive(self):
        meta = self.evaluation_fixture()
        path = self.directory / "pred_val_clean.npz"
        np.savez_compressed(path, text=np.zeros((2, 5)))
        meta["predictions"]["clean"] = artifacts.sha256(path)
        artifacts.write_json(self.directory / "evaluation_val.json", meta)
        self.assertFalse(self.evaluation_complete())

    def test_stale_evaluation_checkpoint(self):
        self.evaluation_fixture()
        different_checkpoint = {**self.checkpoint, "epoch": 2}
        self.assertFalse(artifacts.evaluation_complete(
            self.directory, "val", self.config, different_checkpoint, self.dataset))

    def test_evaluation_refuses_to_overwrite_different_dataset(self):
        self.evaluation_fixture()
        with (patch.object(evaluate, "dataset_metadata", return_value={**self.dataset, "sha256": "new"}),
              patch.object(evaluate, "evaluate_run") as run,
              contextlib.redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit)):
            # Rename the validation fixture to simulate an existing frozen test evaluation.
            (self.directory / "evaluation_val.json").rename(self.directory / "evaluation_test.json")
            evaluate.main(["--run-dir", str(self.directory), "--split", "test"])
        run.assert_not_called()
        self.assertIn("Evaluation identity changed", stderr.getvalue())

    def test_evaluation_uses_independent_checkpoint_records(self):
        with (patch.object(evaluate, "dataset_metadata", return_value=self.dataset),
              patch.object(evaluate, "evaluate_run") as run):
            evaluate.main(["--run-dir", str(self.directory)])
        calls = [(call.args[2], call.args[4]["file"]) for call in run.call_args_list]
        self.assertEqual(calls, [("classification", "classification.pt"), ("warning", "warning.pt")])


class SelectionTests(unittest.TestCase):
    def test_independent_selection_and_patience(self):
        history = [(0.7, 0.5), (0.72, 0.55), (0.71, 0.6), (0.70, 0.62), (0.70, 0.62), (0.69, 0.61)]

        def replay():
            selected = {"classification": None, "warning": None}
            decisions = []
            for epoch, (classification, warning) in enumerate(history, 1):
                selected["classification"] = artifacts.update_selection(
                    selected["classification"], epoch, classification, 1e-4)
                selected["warning"] = artifacts.update_selection(selected["warning"], epoch, warning)
                decisions.append(artifacts.should_stop(epoch, selected, 2))
            return selected, decisions

        selection, decisions = replay()
        self.assertEqual((selection, decisions), replay())
        self.assertEqual(selection["classification"]["epoch"], 2)
        self.assertEqual(selection["warning"]["epoch"], 4)
        self.assertEqual(decisions, [False, False, False, False, False, True])

    def test_minimum_delta_and_undefined_ap(self):
        classification = {"epoch": 1, "score": 0.7}
        self.assertIs(artifacts.update_selection(classification, 2, 0.70009, 1e-4), classification)
        self.assertIsNone(artifacts.update_selection(None, 2, None))
        self.assertIsNone(artifacts.update_selection(None, 2, float("nan")))
        self.assertFalse(artifacts.should_stop(4, {"classification": classification, "warning": None}, 2))

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn unavailable")
    def test_warning_metrics_and_single_class(self):
        from src.metrics import agreement_metrics

        normal = agreement_metrics(np.array([0.9, 0.5, 0.4, 0.6]), np.array([0, 0, 1, 1]))
        self.assertEqual(normal["conflict_precision"], 0.5)
        self.assertEqual(normal["conflict_recall"], 0.5)
        self.assertEqual(normal["conflict_f1"], 0.5)
        self.assertAlmostEqual(normal["conflict_ap"], 5 / 6)
        self.assertAlmostEqual(normal["conflict_auroc"], 0.75)
        for targets in (np.zeros(2), np.ones(2)):
            undefined = agreement_metrics(np.array([0.9, 0.1]), targets)
            self.assertIsNone(undefined["conflict_ap"])
            self.assertNotIn("conflict_auroc", undefined)


class CommandTests(unittest.TestCase):
    def test_processed_split_rejects_a_misplaced_later_row(self):
        with tempfile.TemporaryDirectory() as output, patch.object(artifacts, "PROCESSED_DIR", Path(output)):
            path = Path(output) / "val.csv"
            path.write_text("split,event_name,tweet_id,image_id,image_path,text,text_label,image_label,agree\n"
                            "val,event,1,2,path,caption,a,a,1\n"
                            "test,event,3,4,path,caption,a,a,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unexpected split marker at CSV line 3"):
                artifacts.dataset_metadata("val")

    def test_explicit_recipes(self):
        v2, _ = artifacts.training_config(["--model", "multimodal", "--recipe", "v2"])
        default, _ = artifacts.training_config(["--model", "multimodal"])
        self.assertEqual((v2["weight_decay"], v2["encoder_dropout"], v2["fusion_dropout"]), (0.01, None, 0.0))
        self.assertEqual(default["recipe"], "v3")
        self.assertEqual((default["weight_decay"], default["encoder_dropout"], default["fusion_dropout"]),
                         (0.05, 0.2, 0.2))

    def test_test_only_missing_checkpoint_has_no_subprocess(self):
        with (patch.object(reproduce, "validate_training", side_effect=FileNotFoundError("Missing checkpoint")),
              patch.object(reproduce, "step") as step,
              contextlib.redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit)):
            reproduce.main(["--recipe", "v2", "--matrix", "fusion", "--split", "test"])
        step.assert_not_called()
        self.assertIn("Missing checkpoint", stderr.getvalue())

    def test_test_only_runs_evaluation_and_report(self):
        manifests = {"checkpoints": {"classification": {"epoch": 1}, "warning": {"epoch": 2}}}
        with (patch.object(reproduce, "validate_training", return_value=manifests),
              patch.object(reproduce, "dataset_metadata", return_value={}),
              patch.object(reproduce, "evaluation_complete", return_value=False),
              patch.object(reproduce, "step") as step):
            reproduce.main(["--recipe", "v2", "--matrix", "fusion", "--split", "test", "--seeds", "42"])
        self.assertEqual([call.args[0] for call in step.call_args_list],
                         ["src.evaluate", "src.evaluate", "src.report"])

    def test_full_v2_fusion_matrix_resolves_six_distinct_directories(self):
        manifest = {"checkpoints": {"classification": {"epoch": 1}, "warning": {"epoch": 2}}}
        with (patch.object(reproduce, "validate_training", return_value=manifest) as validate,
              patch.object(reproduce, "dataset_metadata", return_value={}),
              patch.object(reproduce, "evaluation_complete", return_value=True),
              patch.object(reproduce, "step") as step):
            reproduce.main(["--recipe", "v2", "--matrix", "fusion", "--seeds", "42", "43", "44",
                            "--tag", "track2", "--workers", "6", "--split", "test"])
        directories = {call.args[0] for call in validate.call_args_list}
        self.assertEqual(len(directories), 6)
        self.assertTrue(all(path.parent == artifacts.RUN_ROOT / "v2" for path in directories))
        step.assert_called_once()
        self.assertEqual(step.call_args.args[0], "src.report")

    def test_validation_uses_only_train_and_dev_metadata(self):
        manifest = {"checkpoints": {"classification": {"epoch": 1}, "warning": {"epoch": 2}}}
        with (patch.object(reproduce, "validate_training", return_value=manifest),
              patch.object(reproduce, "dataset_metadata", return_value={}) as dataset,
              patch.object(reproduce, "evaluation_complete", return_value=False),
              patch.object(reproduce, "step") as step):
            reproduce.main(["--recipe", "v2", "--matrix", "fusion", "--seeds", "42"])
        self.assertEqual({call.args[0] for call in dataset.call_args_list}, {"train", "val"})
        self.assertEqual([call.args[0] for call in step.call_args_list],
                         ["src.train", "src.train", "src.evaluate", "src.evaluate", "src.report"])

    def test_archived_report_uses_committed_results_without_checkpoints(self):
        with tempfile.TemporaryDirectory() as output:
            real_write = report.write_report
            with patch.object(report, "write_report", side_effect=lambda rows, split, _:
                              real_write(rows, split, Path(output))):
                report.main(["--archive", "v2", "--split", "val"])
            result = (Path(output) / "summary_val.md").read_text(encoding="utf-8")
            self.assertIn("text+image pair", result)
            self.assertIn("0.7821", result)
            self.assertIn("0.7572", result)
            self.assertIn("Humanitarian-label disagreement", result)


class PrecisionTests(unittest.TestCase):
    def test_precision_uses_native_bf16_or_fp16_fallback(self):
        torch = MagicMock()
        cpu, cuda = SimpleNamespace(type="cpu"), SimpleNamespace(type="cuda")
        with patch.dict(sys.modules, {"torch": torch}):
            self.assertIsNone(precision.autocast_dtype(cpu))
            torch.cuda.device.assert_not_called()
            torch.cuda.is_bf16_supported.return_value = True
            self.assertIs(precision.autocast_dtype(cuda), torch.bfloat16)
            torch.cuda.device.assert_called_once_with(cuda)
            torch.cuda.is_bf16_supported.assert_called_once_with(including_emulation=False)
            torch.cuda.is_bf16_supported.return_value = False
            self.assertIs(precision.autocast_dtype(cuda), torch.float16)

    def test_fp16_clips_unscaled_gradients_and_skips_scheduler_on_overflow(self):
        for next_scale in (128.0, 64.0):
            with self.subTest(next_scale=next_scale):
                calls = Mock()
                calls.scaler.get_scale.side_effect = [128.0, next_scale]
                with patch.dict(sys.modules, {"torch": calls.torch}):
                    precision.optimizer_step(calls.loss, calls.optimizer, calls.scheduler,
                                             calls.parameters, calls.scaler)
                methods = [call[0] for call in calls.mock_calls]
                self.assertLess(methods.index("scaler.unscale_"), methods.index("torch.nn.utils.clip_grad_norm_"))
                self.assertLess(methods.index("torch.nn.utils.clip_grad_norm_"), methods.index("scaler.step"))
                self.assertEqual(calls.scheduler.step.called, next_scale == 128.0)


if __name__ == "__main__":
    unittest.main()
