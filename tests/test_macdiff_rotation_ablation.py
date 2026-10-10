"""CPU tests for native rotation experiment orchestration and LP result integrity."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("rotation_ablation", REPO / "tools/run_macdiff_rotation_ablation.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def write_log(path, count=100, offset=0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"epoch": epoch, "test_acc1": 80 + epoch * .01 + offset,
             "train_loss": 0.5, "train_lr": 1e-5} for epoch in range(count)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return rows


class RotationAblationTests(unittest.TestCase):
    def test_exact_best_and_earliest_tie(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.txt"
            rows = write_log(path)
            rows[23]["test_acc1"] = rows[41]["test_acc1"] = 86.19602107350796
            path.write_text("".join(json.dumps(row) + "\n" for row in reversed(rows)), encoding="utf-8")
            stats = runner.lp_stats(path)
            self.assertEqual(stats["best_lp_acc"], 86.19602107350796)
            self.assertEqual(stats["best_epoch"], 23)
            self.assertEqual(stats["last_lp_acc"], 80.99)

    def test_incomplete_duplicate_and_nonfinite_logs_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.txt"
            write_log(path, 99)
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                runner.lp_stats(path)
            rows = write_log(path)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rows[0]) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                runner.lp_stats(path)
            for bad in (float("nan"), float("inf"), None, True, 101):
                with self.subTest(bad=bad):
                    rows[20]["test_acc1"] = bad
                    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        runner.lp_stats(path)

    def test_json_csv_and_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_log(root / "no_rot/lp/log.txt")
            write_log(root / "rot/lp/log.txt", offset=.25)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                summary = runner.write_summary(root)
            self.assertAlmostEqual(summary["rot_minus_no_rot_best_pp"], .25)
            self.assertIn("no_rot: 80.990000% @ epoch 99", output.getvalue())
            self.assertIn("rot: 81.240000% @ epoch 99", output.getvalue())
            self.assertEqual(json.loads((root / "rotation_comparison.json").read_text()), summary)
            self.assertEqual(len((root / "rotation_comparison.csv").read_text().splitlines()), 3)

    def test_other_pt_factor_rejected(self):
        configs = runner.make_configs("dataset.npz")
        configs["rot"]["model_args"]["uncond_ratio"] = .2
        with self.assertRaisesRegex(ValueError, "beyond"):
            runner.validate_pair(configs)

    def test_subprocess_failure_saved_and_propagated(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "pt/console.log"
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(subprocess.CalledProcessError) as raised:
                    runner.run_stage([sys.executable, "-c", "print('rank failure'); raise SystemExit(7)"], log, dict(runner.os.environ))
            self.assertEqual(raised.exception.returncode, 7)
            self.assertIn("rank failure", log.read_text())

    def test_pipeline_order_snapshots_and_final_results(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "new_run"
            def fake_stage(command, console_path, env):
                variant = console_path.parent.parent.name
                stage = console_path.parent.name
                calls.append((variant, stage))
                config = json.loads(Path(command[command.index("--config") + 1]).read_text())
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1")
                self.assertEqual(config["batch_size"], 64 if stage == "pt" else 128)
                if stage == "pt":
                    self.assertEqual(config["train_feeder_args"]["random_rot"], variant == "rot")
                    write_log(console_path.parent / "log.txt", 500)
                    (console_path.parent / "checkpoint-499.pth").write_text("mock")
                else:
                    self.assertTrue(Path(command[command.index("--finetune") + 1]).exists())
                    self.assertTrue(config["train_feeder_args"]["random_rot"])
                    self.assertFalse(config["val_feeder_args"]["random_rot"])
                    write_log(console_path.parent / "log.txt", offset=.25 if variant == "rot" else 0)
            with mock.patch.object(runner, "preflight", return_value={}), \
                 mock.patch.object(runner, "run_stage", side_effect=fake_stage), \
                 mock.patch.object(runner, "verify_checkpoint", return_value={"args_verified": True}), \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.main(["--gpus", "0,1", "--output-dir", str(root)])
            self.assertEqual(calls, [("no_rot", "pt"), ("no_rot", "lp"), ("rot", "pt"), ("rot", "lp")])
            manifest = json.loads((root / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(len(manifest["completed_stages"]), 4)
            summary = json.loads((root / "rotation_comparison.json").read_text())
            self.assertAlmostEqual(summary["rot_minus_no_rot_best_pp"], .25)

    def test_pipeline_failure_stops_and_keeps_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "failed_run"
            with mock.patch.object(runner, "preflight", return_value={}), \
                 mock.patch.object(runner, "run_stage", side_effect=subprocess.CalledProcessError(3, ["mock"])) as stage, \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(subprocess.CalledProcessError):
                    runner.main(["--gpus", "0,1", "--output-dir", str(root)])
            self.assertEqual(stage.call_count, 1)
            self.assertEqual(json.loads((root / "run_manifest.json").read_text())["status"], "failed")
            self.assertFalse((root / "rotation_comparison.json").exists())

    def test_existing_directory_and_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(runner, "preflight") as preflight, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(FileExistsError):
                    runner.main(["--gpus", "0,1", "--output-dir", str(root)])
                preflight.assert_not_called()
                destination = root / "dry"
                runner.main(["--gpus", "0,1", "--output-dir", str(destination), "--dry-run"])
                self.assertFalse(destination.exists())
                preflight.assert_not_called()


if __name__ == "__main__":
    unittest.main()
