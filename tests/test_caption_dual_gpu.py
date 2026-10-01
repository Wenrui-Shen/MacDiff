"""CPU tests for dual-GPU orchestration, sorted merging and interrupted resume."""
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PILOT = Path(__file__).resolve().parents[1] / "tools" / "vlm_pilot"
sys.path.insert(0, str(PILOT))
from caption_output import CaptionOutput, LOCAL_PARTS, repair_incomplete_jsonl_tail, sidecar_paths
import run_caption_dual_gpu as launcher


def caption():
    return {"persons": [{"person_index": 0, "global": "The arms rise and lower.",
                         "local": {part: "The %s remains still." % part for part in LOCAL_PARTS}}]}


def make_renders(count=6):
    return {i: {"sample_index": i, "render": {"visible_actor_count": 1}} for i in range(count)}


def save_shard(options, shard_id, indices):
    path = launcher.shard_path(options, shard_id)
    worker_args = SimpleNamespace(output_path=path, prompt_path=options.prompt_path,
                                 model=options.model, revision=None, caption_schema="global_local",
                                 resume=True, num_shards=len(options.gpu_ids), shard_id=shard_id,
                                 rendered_root=options.rendered_root)
    with CaptionOutput(worker_args, launcher.sha256_file(options.prompt_path)) as sink:
        for index in indices:
            sink.write_record({"sample_index": index, "status": "accepted", "caption": caption()})


class InterruptedTailTests(unittest.TestCase):
    def test_incomplete_final_line_backed_up_without_losing_successful_rows(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "shard0.jsonl"
            complete = b'{"sample_index": 0}\n'
            broken = b'{"sample_index": 2, "persons": ['
            path.write_bytes(complete + broken)
            backup = repair_incomplete_jsonl_tail(path)
            self.assertEqual(backup.read_bytes(), broken)
            self.assertEqual(path.read_bytes(), complete)
            self.assertIsNone(repair_incomplete_jsonl_tail(path))

    def test_large_unicode_tail_and_valid_last_line(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "shard0.jsonl"
            complete = b'{"sample_index": 0}\n'
            tail = ('{"text": "' + '动' * 10000).encode("utf-8")[:-1]
            path.write_bytes(complete + tail)
            backup = repair_incomplete_jsonl_tail(path)
            self.assertEqual(backup.read_bytes(), tail)
            self.assertEqual(path.read_bytes(), complete)
            valid = complete + b'{"sample_index": 1}'
            path.write_bytes(valid)
            self.assertIsNone(repair_incomplete_jsonl_tail(path))
            self.assertEqual(path.read_bytes(), valid)

    def test_completed_malformed_line_is_not_silently_removed(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "shard.jsonl"
            broken = b'{"broken":\n'
            path.write_bytes(broken)
            self.assertIsNone(repair_incomplete_jsonl_tail(path))
            self.assertEqual(path.read_bytes(), broken)


class DualGpuTests(unittest.TestCase):
    def options(self, root, *extra):
        return launcher.parse_args([
            "--rendered_root", str(root), "--output_dir", str(root / "output"),
            "--model", "test-model", "--expected_samples", "6", *extra])

    def test_sorted_merge_and_completeness(self):
        with tempfile.TemporaryDirectory() as folder:
            options = self.options(Path(folder))
            save_shard(options, 0, [4, 0, 2])
            save_shard(options, 1, [5, 1, 3])
            summary = launcher.merge_results(options, make_renders(), set(range(6)),
                                             launcher.sha256_file(options.prompt_path))
            path = options.output_dir / "captions.json"
            rows = json.loads(path.read_text())
            self.assertEqual([v["sample_index"] for v in rows], list(range(6)))
            self.assertTrue(summary["complete"])
            self.assertEqual(sum(v.startswith('{"sample_index"') for v in path.read_text().splitlines()), 6)
            self.assertNotIn('"model"', path.read_text())
            metadata = json.loads(sidecar_paths(path)[0].read_text())
            self.assertEqual(len(metadata["sources"]), 2)

    def test_resume_recovers_tail_skips_finished_gpu_and_merges(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            options = self.options(root)
            save_shard(options, 0, [0, 2, 4])
            save_shard(options, 1, [1])
            odd_path = launcher.shard_path(options, 1)
            with odd_path.open("ab") as sink:
                sink.write(b'{"sample_index":3')
            spawned = []
            def worker(command, **kw):
                def value(flag):
                    return command[command.index(flag) + 1]
                shard_id = int(value("--shard_id"))
                spawned.append((shard_id, kw["env"]["CUDA_VISIBLE_DEVICES"]))
                self.assertIn("--resume", command)
                save_shard(options, shard_id, [3, 5])
                return SimpleNamespace(poll=lambda: 0, returncode=0)
            argv = ["--rendered_root", str(root), "--output_dir", str(options.output_dir),
                    "--model", "test-model", "--expected_samples", "6"]
            with patch.object(launcher, "load_rendered_samples", return_value=make_renders()), \
                    patch.object(launcher.subprocess, "Popen", side_effect=worker), \
                    patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(launcher.main(argv), 0)
                self.assertEqual(launcher.main(argv), 0)
            self.assertEqual(spawned, [(1, "1")])
            self.assertEqual([v["sample_index"] for v in json.loads(
                (options.output_dir / "captions.json").read_text())], list(range(6)))
            self.assertEqual(len(list(options.output_dir.glob("*.interrupted-tail.*.bin"))), 1)

    def test_fresh_launch_uses_two_independent_gpu_workers(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            options = self.options(root, "--gpus", "2,3")
            spawned = []
            def worker(command, **kw):
                shard_id = int(command[command.index("--shard_id") + 1])
                spawned.append(kw["env"]["CUDA_VISIBLE_DEVICES"])
                save_shard(options, shard_id, [i for i in range(6) if i % 2 == shard_id])
                return SimpleNamespace(poll=lambda: 0, returncode=0)
            with patch.object(launcher, "load_rendered_samples", return_value=make_renders()), \
                    patch.object(launcher.subprocess, "Popen", side_effect=worker), \
                    patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(launcher.main([
                    "--rendered_root", str(root), "--output_dir", str(options.output_dir),
                    "--model", "test-model", "--gpus", "2,3", "--expected_samples", "6"]), 0)
            self.assertEqual(spawned, ["2", "3"])

    def test_partial_merge_reports_missing_and_wrong_shard_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            options = self.options(Path(folder))
            save_shard(options, 0, [0, 2])
            summary = launcher.merge_results(options, make_renders(), set(range(6)),
                                             launcher.sha256_file(options.prompt_path))
            self.assertEqual(summary["missing_indices"], [1, 3, 4, 5])
            self.assertFalse(summary["complete"])
            save_shard(options, 1, [4])
            with self.assertRaisesRegex(ValueError, "wrong shard"):
                launcher.merge_results(options, make_renders(), set(range(6)),
                                       launcher.sha256_file(options.prompt_path))

    def test_model_or_shard_count_change_and_render_missing_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            options = self.options(Path(folder))
            save_shard(options, 0, [0])
            options.model = "different"
            with self.assertRaisesRegex(ValueError, "differs"):
                launcher.saved_indices(options, 0, launcher.sha256_file(options.prompt_path), recover=True)
            options.model = "test-model"
            options.gpu_ids = ["0"]
            with self.assertRaisesRegex(ValueError, "sharding differs"):
                launcher.saved_indices(options, 0, launcher.sha256_file(options.prompt_path))
            with self.assertRaisesRegex(ValueError, "coverage"):
                launcher.expected_indices(options, make_renders(5))

    def test_lock_released_and_parallel_launcher_blocked(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            with launcher.run_lock(directory):
                with self.assertRaisesRegex(RuntimeError, "Another caption launcher"):
                    with launcher.run_lock(directory):
                        pass
            with launcher.run_lock(directory):
                pass

    def test_interrupt_stops_started_worker_and_saves_partial_sorted_result(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            options = self.options(root)
            process = SimpleNamespace(returncode=None, poll=lambda: None)
            def first_worker(command, **kw):
                save_shard(options, 0, [0])
                return process
            def stop(jobs):
                self.assertEqual(len(jobs), 1)
                process.returncode = -15
            calls = []
            def dispatch(command, **kw):
                calls.append(command)
                if len(calls) == 1:
                    return first_worker(command, **kw)
                raise KeyboardInterrupt
            with patch.object(launcher, "load_rendered_samples", return_value=make_renders()), \
                    patch.object(launcher.subprocess, "Popen", side_effect=dispatch), \
                    patch.object(launcher, "stop_workers", side_effect=stop), \
                    patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(launcher.main([
                    "--rendered_root", str(root), "--output_dir", str(options.output_dir),
                    "--model", "test-model", "--expected_samples", "6"]), 130)
            self.assertEqual(json.loads((options.output_dir / "captions.json").read_text())[0]["sample_index"], 0)
            with launcher.run_lock(options.output_dir):
                pass


if __name__ == "__main__":
    unittest.main()
