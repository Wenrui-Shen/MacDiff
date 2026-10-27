"""Dataset reuse, real CPU GIF rendering, and mocked GPU-worker integration."""
import io
import contextlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PILOT = Path(__file__).resolve().parents[1] / "tools" / "vlm_pilot"
sys.path.insert(0, str(PILOT))
import dataset_caption_pipeline as pipeline
import run_caption_ntu120 as ntu120
import run_caption_ntu120_xset as render_first_xset
import run_caption_ntu60_xview as xview
import run_caption_pku as pku
import check_text_pipeline as audit
from caption_output import CaptionOutput, LOCAL_PARTS, OUTPUT_SCHEMA, sidecar_paths

try:
    import numpy as np
    import PIL
except ImportError:
    np = None


def persons(count=1, text="The arms move up and down."):
    return [{"person_index": person, "global": text,
             "local": {part: "The %s remains upright." % part for part in LOCAL_PARTS}}
            for person in range(count)]


class StatisticsTests(unittest.TestCase):
    def test_real_train_id_overlap_and_subject_major_order(self):
        root = PILOT.parents[1] / "data"
        names60 = ntu120.train_names(root / "ntu/statistics", ntu120.NTU60_SUBJECTS)
        names120 = ntu120.train_names(root / "ntu120/statistics", ntu120.NTU120_SUBJECTS)
        self.assertEqual((len(names60), len(names120)), (40091, 63026))
        self.assertEqual(len(set(names60) & set(names120)), 40091)
        self.assertEqual(len(set(names120) - set(names60)), 22935)

    def test_xview_camera_major_split_and_overlap(self):
        root = PILOT.parents[1] / "data/ntu/statistics"
        names_sub = ntu120.train_names(root, ntu120.NTU60_SUBJECTS)
        names_view = ntu120.train_names(root, (2, 3), group="camera")
        self.assertEqual(len(names_view), 37646)
        self.assertEqual(len(set(names_view) & set(names_sub)), 26676)
        extra = set(names_view) - set(names_sub)
        self.assertEqual(len(extra), 10970)
        self.assertTrue(all(int(name[9:12]) not in ntu120.NTU60_SUBJECTS for name in extra))
        self.assertEqual([name[4:8] for name in names_view], sorted(name[4:8] for name in names_view))

    def test_xset_even_setup_major_split_and_union_reuse(self):
        root = PILOT.parents[1] / "data"
        names_set = ntu120.train_names(root / "ntu120/statistics", range(2, 33, 2), group="setup")
        names_sub = ntu120.train_names(root / "ntu120/statistics", ntu120.NTU120_SUBJECTS)
        names60 = ntu120.train_names(root / "ntu/statistics", ntu120.NTU60_SUBJECTS)
        names_view = ntu120.train_names(root / "ntu/statistics", (2, 3), group="camera")
        self.assertEqual(len(names_set), 54468)
        self.assertEqual(len(set(names_set) & set(names60)), 16087)
        self.assertEqual(len(set(names_set) & set(names_sub)), 25053)
        self.assertEqual(len(set(names_set) - (set(names_sub) | set(names_view))), 22503)
        setups = [int(name[1:4]) for name in names_set]
        self.assertTrue(all(setup % 2 == 0 for setup in setups))
        self.assertEqual(setups, sorted(setups))


@unittest.skipIf(np is None, "requires NumPy and Pillow")
class DatasetPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "output"
        self.prompt = PILOT / "skeleton_motion_prompt_v2.txt"
        self.spawned = []

    def save_npz(self, name, data):
        path = self.root / name
        # Object labels would fail allow_pickle=False if this pipeline tried to read them.
        np.savez(path, x_train=data, y_train=np.array([{"forbidden": True}], dtype=object))
        return path

    def save_statistics(self, name, names):
        directory = self.root / name
        directory.mkdir()
        (directory / "skes_available_name.txt").write_text("\n".join(names))
        (directory / "performer.txt").write_text("\n".join(str(int(item[9:12])) for item in names))
        (directory / "camera.txt").write_text("\n".join(str(int(item[5:8])) for item in names))
        (directory / "setup.txt").write_text("\n".join(str(int(item[1:4])) for item in names))
        return directory

    def sample(self, value, people=1, frames=10):
        # Moving, nonzero joint coordinates; different values distinguish source rows.
        data = np.zeros((frames, 2, 25, 3), dtype=np.float32)
        pose = np.arange(75, dtype=np.float32).reshape(25, 3) / 100 + value
        for frame in range(frames):
            data[frame, 0] = pose + frame / 100
            if people == 2:
                data[frame, 1] = pose + frame / 100 + 1
        return data.reshape(frames, 150)

    def ntu_fixture(self):
        first, second = self.sample(1), self.sample(2, people=2)
        data60 = self.save_npz("ntu60.npz", np.stack([first, second]))
        target = np.zeros((3, 12, 150), dtype=np.float32)
        target[0, :10] = self.sample(3)
        target[1, :10, :75] = first[:, :75]
        target[1, :10, 75:] = first[:, :75]  # NTU120 single-person clone.
        target[2, :10] = second
        data120 = self.save_npz("ntu120.npz", target)
        a, b, new = "S001C001P001R001A001", "S001C001P002R001A002", "S018C001P001R001A061"
        stat60 = self.save_statistics("stat60", [b, a])
        stat120 = self.save_statistics("stat120", [new, b, a])
        source = self.root / "source.json"
        source.write_text(json.dumps([{"sample_index": 0, "persons": persons(text="The right hand rises.")},
                                           {"sample_index": 1, "persons": persons(2)}]))
        sidecar_paths(source)[0].write_text(json.dumps({
            "schema_version": OUTPUT_SCHEMA, "model": "test-model", "requested_revision": None,
            "caption_schema": "global_local", "prompt": {"sha256": pipeline.sha256_file(self.prompt)}}))
        return ["--data_path", str(data120), "--ntu60_data_path", str(data60),
                "--ntu60_captions", str(source), "--ntu60_statistics_dir", str(stat60),
                "--ntu120_statistics_dir", str(stat120), "--model", "test-model",
                "--output_root", str(self.output)]

    def worker(self, command, **options):
        import caption_qwen3vl_rendered_train_transformers as rendered
        def value(flag):
            return command[command.index(flag) + 1]
        shard = int(value("--shard_id"))
        count = int(value("--num_shards"))
        render_root = Path(value("--rendered_root"))
        records = rendered.load_rendered_samples(render_root)
        start = int(value("--start_index"))
        stop = start + int(value("--max_samples")) if "--max_samples" in command else max(records) + 1
        args = SimpleNamespace(output_path=Path(value("--output_path")), model="test-model", revision=None,
                               prompt_path=self.prompt, caption_schema="global_local", resume=True,
                               num_shards=count, shard_id=shard, rendered_root=render_root)
        with CaptionOutput(args, pipeline.sha256_file(self.prompt)) as sink:
            for index in sorted(records):
                if index % count == shard and start <= index < stop and index not in sink.accepted:
                    self.spawned.append((str(render_root), index, options["env"]["CUDA_VISIBLE_DEVICES"]))
                    sink.write_record({"sample_index": index, "status": "accepted",
                                       "caption": {"persons": persons(records[index]["render"]["visible_actor_count"])}})
        return SimpleNamespace(poll=lambda: 0, returncode=0)

    def test_ntu120_reuse_remaps_rows_clones_slots_and_resumes_without_npz_reads(self):
        argv = self.ntu_fixture()
        source = Path(argv[argv.index("--ntu60_captions") + 1])
        original = source.read_bytes()
        with patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(ntu120.main(argv + ["--stage", "render"]), 0)
        directory = self.output / "ntu120_xsub"
        self.assertEqual([p.name for p in (directory / "rendered").iterdir()], ["train_000000"])
        import run_caption_dual_gpu as launcher
        with patch.object(launcher.subprocess, "Popen", side_effect=self.worker), \
                patch.object(pipeline, "load_train", side_effect=AssertionError("caption reopened NPZ")), \
                patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(ntu120.main(argv + ["--stage", "caption"]), 0)
            self.assertEqual(ntu120.main(argv + ["--stage", "caption"]), 0)
        self.assertEqual([entry[1] for entry in self.spawned], [0])
        rows = json.loads((directory / "captions.json").read_text())
        self.assertEqual([row["sample_index"] for row in rows], [0, 1, 2])
        self.assertEqual(rows[1]["persons"][0]["global"], "The right hand rises.")
        self.assertEqual(len(rows[1]["persons"]), 2)
        self.assertEqual(rows[1]["persons"][1]["person_index"], 1)
        self.assertEqual(rows[1]["persons"][0]["local"], rows[1]["persons"][1]["local"])
        self.assertEqual(source.read_bytes(), original)
        metadata = json.loads(sidecar_paths(directory / "captions.json")[0].read_text())
        self.assertTrue(metadata["summary"]["dataset_complete"])
        self.assertEqual(metadata["summary"]["reused"], 2)
        from util.structured_text_cache import read_captions, train_person_valid
        samples, _, _ = read_captions([directory / "captions.json"], 3)
        valid = train_person_valid(Path(argv[1]))
        self.assertEqual([len(row["persons"]) for row in samples], valid.sum(axis=1).tolist())

    def test_dry_run_has_no_outputs_and_changed_skeleton_is_rejected(self):
        argv = self.ntu_fixture()
        with patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(ntu120.main(argv + ["--dry_run"]), 0)
        self.assertFalse(self.output.exists())
        target_path = Path(argv[1])
        target = pipeline.load_train(target_path)
        target[1, 0, 0] += 0.5
        np.savez(target_path, x_train=target)
        with patch("sys.stdout", new_callable=io.StringIO), self.assertRaisesRegex(ValueError, "different coordinates"):
            ntu120.main(argv + ["--dry_run"])

    def test_new_ntu120_clone_is_captioned_once_and_expanded_for_raw_clip_slots(self):
        argv = self.ntu_fixture()
        target = Path(argv[1])
        data = pipeline.load_train(target)
        data[0, :, 75:] = data[0, :, :75]
        np.savez(target, x_train=data, y_train=np.array([{"forbidden": True}], dtype=object))
        import run_caption_dual_gpu as launcher
        with patch.object(launcher.subprocess, "Popen", side_effect=self.worker), \
                patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(ntu120.main(argv), 0)
        directory = self.output / "ntu120_xsub"
        generated = json.loads((directory / "generated/captions.json").read_text())
        self.assertEqual(len(generated[0]["persons"]), 1)
        rows = json.loads((directory / "captions.json").read_text())
        self.assertEqual(len(rows[0]["persons"]), 2)
        self.assertEqual(rows[0]["persons"][0]["global"], rows[0]["persons"][1]["global"])
        self.assertEqual(rows[0]["persons"][0]["local"], rows[0]["persons"][1]["local"])
        self.assertEqual(rows[0]["persons"][1]["person_index"], 1)
        import cache_clip_motion_text as cache
        _, valid, _, _ = cache.build_plan(target, [directory / "captions.json"])
        self.assertTrue(valid.all())

    def test_incomplete_source_captions_rejected_before_rendering(self):
        argv = self.ntu_fixture()
        source = Path(argv[argv.index("--ntu60_captions") + 1])
        source.write_text(json.dumps([{"sample_index": 0, "persons": persons()}]))
        with self.assertRaisesRegex(ValueError, "cover every"):
            ntu120.main(argv + ["--dry_run"])
        self.assertFalse(self.output.exists())

    def test_partial_merge_reports_missing_then_generation_completes(self):
        argv = self.ntu_fixture()
        import run_caption_dual_gpu as launcher
        with patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(ntu120.main(argv + ["--stage", "render"]), 0)
            self.assertEqual(ntu120.main(argv + ["--merge_only"]), 1)
        directory = self.output / "ntu120_xsub"
        metadata = json.loads(sidecar_paths(directory / "captions.json")[0].read_text())
        self.assertEqual(metadata["summary"]["missing_indices"], [0])
        with patch.object(launcher.subprocess, "Popen", side_effect=self.worker), \
                patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(ntu120.main(argv + ["--stage", "caption"]), 0)

    def test_identity_change_rejected_and_range_keeps_original_indices(self):
        argv = self.ntu_fixture()
        with patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(ntu120.main(argv + ["--stage", "render", "--max_samples", "1"]), 0)
            self.assertEqual(ntu120.main(argv + ["--merge_only", "--start_index", "1"]), 0)
        directory = self.output / "ntu120_xsub"
        rows = json.loads((directory / "captions.json").read_text())
        self.assertEqual([row["sample_index"] for row in rows], [1, 2])
        with self.assertRaisesRegex(ValueError, "changed"):
            ntu120.main(argv + ["--stage", "caption", "--num_frames", "16"])

    def test_pku_versions_have_independent_renders_workers_and_sorted_results(self):
        first = self.save_npz("v1.npz", np.stack([self.sample(1), self.sample(2, 2)]))
        second = self.save_npz("v2.npz", np.stack([self.sample(3)]))
        argv = ["--pkuv1_data_path", str(first), "--pkuv2_data_path", str(second),
                "--model", "test-model", "--output_root", str(self.output)]
        import run_caption_dual_gpu as launcher
        with patch.object(launcher.subprocess, "Popen", side_effect=self.worker), \
                patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(pku.main(argv), 0)
            self.assertEqual(pku.main(argv + ["--stage", "caption"]), 0)
        self.assertEqual(len(self.spawned), 3)
        for version, count in (("v1", 2), ("v2", 1)):
            directory = self.output / ("pku_%s_xsub" % version)
            rows = json.loads((directory / "captions.json").read_text())
            self.assertEqual([row["sample_index"] for row in rows], list(range(count)))
            self.assertEqual(len(list((directory / "rendered").glob("train_*"))), count)

    def test_xview_three_stages_and_noop_caption_resume_preserves_clip_cache(self):
        from tests.test_clip_text_cache import Encoder
        from tests.test_structured_text_cache import SentenceTokenizer
        import cache_clip_motion_text as cache
        import run_caption_dual_gpu as launcher
        # XSub uses subject order; XView uses camera order. Camera 1 is excluded.
        a, b, c, new = ("S001C001P001R001A001", "S001C003P001R001A001",
                        "S001C002P001R001A001", "S001C002P003R001A001")
        statistics = self.save_statistics("stats", [new, b, a, c])
        source_data = self.save_npz("sub.npz", np.stack([self.sample(2, 2), self.sample(1), self.sample(3)]))
        target_data = self.save_npz("view.npz", np.stack([self.sample(4), self.sample(3), self.sample(2, 2)]))
        source = self.root / "sub_captions.json"
        source.write_text(json.dumps([{"sample_index": i, "persons": persons(2 if i == 0 else 1,
                                       text="The person moves %d." % i)} for i in range(3)]))
        sidecar_paths(source)[0].write_text(json.dumps({
            "schema_version": OUTPUT_SCHEMA, "model": "test-model", "requested_revision": None,
            "caption_schema": "global_local", "prompt": {"sha256": pipeline.sha256_file(self.prompt)}}))
        argv = ["--data_path", str(target_data), "--ntu60_data_path", str(source_data),
                "--ntu60_captions", str(source), "--ntu60_statistics_dir", str(statistics),
                "--model", "test-model", "--output_root", str(self.output)]
        directory = self.output / "ntu60_xview"
        with patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(xview.main(argv + ["--stage", "render"]), 0)
        self.assertFalse((directory / "captions.json").exists())
        self.assertEqual(len(list((directory / "rendered").glob("train_*"))), 1)
        with patch.object(launcher.subprocess, "Popen", side_effect=self.worker), \
                patch.object(pipeline, "load_train", side_effect=AssertionError("caption must use GIFs")), \
                patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(xview.main(argv + ["--stage", "caption"]), 0)
        self.assertEqual([item[1] for item in self.spawned], [0])
        rows = json.loads((directory / "captions.json").read_text())
        self.assertEqual(rows[1]["persons"][0]["global"], "The person moves 2.")
        self.assertEqual(rows[2]["persons"][0]["global"], "The person moves 0.")
        clip = self.root / "clip"
        clip.mkdir()
        (clip / "config.json").write_text("{}")
        (clip / "pytorch_model.bin").write_bytes(b"mock CLIP weights")
        options = SimpleNamespace(data_path=str(target_data), captions=[str(directory / "captions.json")],
                                  clip_model=str(clip), output_dir=str(directory / "clip_sentence_cache_v3"),
                                  batch_size=2, device="cpu", resume=True)
        encoder = Encoder()
        fake_torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
        config = SimpleNamespace(projection_dim=3, max_position_embeddings=77)
        with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(cache, "load_encoder",
                return_value=(encoder, SentenceTokenizer(), config)), patch("sys.stdout", new_callable=io.StringIO):
            cache.run(options)
        manifest = json.loads((Path(options.output_dir) / "manifest.json").read_text())
        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["sample_count"], 3)
        self.assertEqual(np.load(Path(options.output_dir) / "token_features.npy").shape, (3, 2, 6, 3))
        before = [(path, path.read_bytes()) for path in
                  (directory / "captions.json", sidecar_paths(directory / "captions.json")[0])]
        with patch.object(launcher.subprocess, "Popen", side_effect=AssertionError("no pending captions")), \
                patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(xview.main(argv + ["--stage", "caption"]), 0)
        self.assertTrue(all(path.read_bytes() == content for path, content in before))
        with patch.object(cache, "load_encoder", side_effect=AssertionError("cache is complete")), \
                patch("sys.stdout", new_callable=io.StringIO):
            cache.run(options)
        spec = {"dataset": "ntu60_xview", "split": {}, "data_path": target_data,
                "rendered": directory / "rendered", "captions": directory / "captions.json",
                "cache": Path(options.output_dir), "plan": directory / "dataset_manifest.json"}
        audit_options = audit.parse_args(["--stages", "render", "caption", "cache", "--check_gifs"])
        with patch("sys.stderr", new_callable=io.StringIO):
            report = audit.inspect(spec, audit_options)
        self.assertTrue(report["complete"], report)
        self.assertEqual(report["stages"]["render"]["expected_new_renders"], 1)
        metadata_path = sidecar_paths(directory / "captions.json")[0]
        changed = json.loads(metadata_path.read_text())
        changed["changed_after_cache"] = True
        metadata_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "different captions/metadata"):
            audit.check_cache(spec, quick=False)

    def test_xset_reuses_three_sources_and_remaps_setup_major_rows(self):
        import cache_clip_motion_text as cache
        import run_caption_dual_gpu as launcher
        old, base, view, extension, new = ("S001C001P001R001A001", "S002C001P001R001A001",
                                          "S002C002P003R001A001", "S018C001P045R001A061",
                                          "S020C001P003R001A062")
        stat60 = self.save_statistics("stat60", [old, base, view])
        stat120 = self.save_statistics("stat120", [new, extension, old, base, view])
        def clone(sample):
            value = sample.copy()
            value[:, 75:] = value[:, :75]
            return value
        data60 = self.save_npz("sub60.npz", np.stack([self.sample(1), self.sample(2)]))
        data_view = self.save_npz("view60.npz", np.stack([self.sample(3)]))
        data_sub120 = self.save_npz("sub120.npz", np.stack([clone(self.sample(1)), clone(self.sample(2)), clone(self.sample(4))]))
        data_set = self.save_npz("set120.npz", np.stack([clone(self.sample(2)), clone(self.sample(3)),
                                                       clone(self.sample(4)), clone(self.sample(5))]))
        def bank(name, labels, people):
            path = self.root / (name + ".json")
            path.write_text(json.dumps([{"sample_index": i, "persons": persons(people, text=label)}
                                             for i, label in enumerate(labels)]))
            sidecar_paths(path)[0].write_text(json.dumps({
                "schema_version": OUTPUT_SCHEMA, "model": "test-model", "requested_revision": None,
                "caption_schema": "global_local", "prompt": {"sha256": pipeline.sha256_file(self.prompt)}}))
            return path
        captions60 = bank("sub60", ["The arms lower.", "The right hand rises."], 1)
        captions_view = bank("view60", ["The head turns."], 1)
        captions120 = bank("sub120", ["The arms lower.", "The right hand rises.", "The legs bend."], 2)
        argv = ["--benchmark", "XSet", "--data_path", str(data_set), "--ntu60_data_path", str(data60),
                "--ntu60_captions", str(captions60), "--ntu60_statistics_dir", str(stat60),
                "--ntu120_statistics_dir", str(stat120), "--ntu60_xview_data_path", str(data_view),
                "--ntu60_xview_captions", str(captions_view), "--ntu120_xsub_data_path", str(data_sub120),
                "--ntu120_xsub_captions", str(captions120), "--output_root", str(self.output), "--model", "test-model"]
        with patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(ntu120.main(argv + ["--dry_run", "--reuse_splits"]), 0)
            self.assertIn("reuse=1, generate=3", output.getvalue())
        with patch.object(launcher.subprocess, "Popen", side_effect=self.worker), \
                patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(ntu120.main(argv), 0)
            self.assertEqual(ntu120.main(argv + ["--stage", "caption"]), 0)
        self.assertEqual([item[1] for item in self.spawned], [3])
        directory = self.output / "ntu120_xset"
        plan = json.loads((directory / "dataset_manifest.json").read_text())
        self.assertEqual([item["source_dataset"] for item in plan["reuse"]],
                         ["ntu60_xsub", "ntu60_xview", "ntu120_xsub"])
        self.assertEqual(plan["generate_indices"], [3])
        rows, valid, _, _ = cache.build_plan(data_set, [directory / "captions.json"])
        self.assertEqual([row["persons"][0]["global"] for row in rows[:3]],
                         ["The right hand rises.", "The head turns.", "The legs bend."])
        self.assertTrue(valid.all())
        self.assertEqual(rows[1]["persons"][1]["global"], "The head turns.")

    def test_audit_reports_absent_data_and_uses_headers_without_labels(self):
        options = audit.parse_args(["--data_root", str(self.root / "absent"), "--datasets", "pku_v1_xsub",
                                    "--stages", "data"])
        spec = audit.dataset_specs(options)["pku_v1_xsub"]
        result = audit.inspect(spec, options)
        self.assertFalse(result["complete"])
        self.assertEqual(result["stages"]["data"]["status"], "missing")
        path = self.root / "pku.npz"
        np.savez(path, x_train=np.stack([self.sample(1), self.sample(2)]),
                 x_test=np.stack([self.sample(3)]), y_train=np.array([object()], dtype=object))
        spec["data_path"] = path
        result = audit.inspect(spec, options)
        self.assertTrue(result["complete"])
        self.assertEqual(result["stages"]["data"]["train"], 2)
        self.assertEqual(result["stages"]["data"]["test"], 1)
        self.assertFalse(result["stages"]["data"]["labels_read"])

    def test_unplanned_render_counts_are_unknown_with_separate_estimates(self):
        path = self.save_npz("data.npz", np.stack([self.sample(1), self.sample(2)]))
        spec = {"dataset": "ntu60_xview", "data_path": path,
                "split": {"new_after_previous_splits": 1},
                "rendered": self.output / "rendered", "plan": self.output / "dataset_manifest.json"}
        result = audit.check_render(spec, decode=False)
        self.assertEqual(result["status"], "not_planned")
        self.assertIsNone(result["expected_new_renders"])
        self.assertEqual(result["estimated_new_after_prerequisites"], 1)
        self.assertNotIn("missing_count", result)
        self.assertFalse(self.output.exists())
        spec["dataset"] = "pku_v1_xsub"
        spec["split"] = {}
        result = audit.check_render(spec, decode=False)
        self.assertEqual(result["status"], "not_planned")
        self.assertIsNone(result["expected_new_renders"])
        self.assertEqual(result["estimated_new_after_prerequisites"], 2)
        # Legacy NTU60 files have no new manifest; their coverage still gets checked.
        spec["dataset"] = "ntu60_xsub"
        result = audit.check_render(spec, decode=False)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["plan_kind"], "legacy_full_train")
        self.assertEqual(result["missing_count"], 2)

    def test_new_render_entry_matches_original_ntu60_xsub_gif_bytes(self):
        import render_qwen3vl_train as legacy
        from render_skeleton_sample import make_demo_sample
        pair = make_demo_sample(72)
        single = pair.copy()
        single[:, 1] = 0
        clone = single.copy()
        clone[:, 1] = clone[:, 0]
        data = np.stack([sample.reshape(72, 150) for sample in (single, clone, pair)])
        path = self.save_npz("parity.npz", data)
        args = pku.parse_args(["--output_root", str(self.output)])
        old_root, new_root = self.root / "old", self.root / "new"
        old_args = SimpleNamespace(**{**vars(args), "data_path": path, "output_root": old_root,
                                     "width": 960, "height": 360, "num_shards": 1, "shard_id": 0,
                                     "resume": True, "dry_run": False})
        with patch.object(legacy, "parse_args", return_value=old_args), \
                patch("sys.stdout", new_callable=io.StringIO):
            legacy.main()
            pipeline.render_selected(data, set(range(3)), args, path, new_root)
        for index in range(3):
            before = old_root / ("train_%06d" % index)
            after = new_root / ("train_%06d" % index)
            self.assertEqual((before / "preview.gif").read_bytes(), (after / "preview.gif").read_bytes())
            old = json.loads((before / "render_metadata.json").read_text())
            new = json.loads((after / "render_metadata.json").read_text())
            self.assertEqual(old["config"], new["config"])
            self.assertEqual(old["render"], new["render"])

    def xset_render_first_fixture(self):
        old, base, view, extension, new = ("S001C001P001R001A001", "S002C001P001R001A001",
                                          "S002C002P003R001A001", "S018C001P045R001A061",
                                          "S020C001P003R001A062")
        stat60 = self.save_statistics("stat60", [old, base, view])
        stat120 = self.save_statistics("stat120", [new, extension, old, base, view])
        def clone(value):
            result = self.sample(value)
            result[:, 75:] = result[:, :75]
            return result
        data60 = self.save_npz("sub60.npz", np.stack([self.sample(1), self.sample(2)]))
        data_view = self.save_npz("view60.npz", np.stack([self.sample(3)]))
        data_sub = self.save_npz("sub120.npz", np.stack([clone(1), clone(2), clone(4)]))
        target = self.save_npz("set120.npz", np.stack([clone(2), clone(3), clone(4), clone(5)]))
        banks = [(self.root / "captions60.json", ["The arms lower.", "The right hand rises."], 1),
                 (self.root / "captions_view.json", ["The head turns."], 1),
                 (self.root / "captions120.json", ["The arms lower.", "The right hand rises.", "The legs bend."], 2)]
        argv = ["--data_path", str(target), "--ntu60_data_path", str(data60),
                "--ntu60_xview_data_path", str(data_view), "--ntu120_xsub_data_path", str(data_sub),
                "--ntu60_captions", str(banks[0][0]), "--ntu60_xview_captions", str(banks[1][0]),
                "--ntu120_xsub_captions", str(banks[2][0]), "--ntu60_statistics_dir", str(stat60),
                "--ntu120_statistics_dir", str(stat120), "--model", "test-model", "--output_root", str(self.output)]
        return argv, banks, target

    def write_render_first_source_captions(self, banks):
        for path, texts, count in banks:
            path.write_text(json.dumps([{"sample_index": i, "persons": persons(count, text=text)}
                                             for i, text in enumerate(texts)]))
            sidecar_paths(path)[0].write_text(json.dumps({
                "schema_version": OUTPUT_SCHEMA, "model": "test-model", "requested_revision": None,
                "caption_schema": "global_local", "prompt": {"sha256": pipeline.sha256_file(self.prompt)}}))

    def test_xset_can_render_and_resume_before_source_captions_exist(self):
        argv, banks, _ = self.xset_render_first_fixture()
        with patch.object(render_first_xset, "load_captions", side_effect=AssertionError("render must not read captions")), \
                patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(render_first_xset.main(argv + ["--dry_run"]), 0)
            self.assertFalse(self.output.exists())
            self.assertEqual(render_first_xset.main(argv + ["--stage", "render"]), 0)
        directory = self.output / "ntu120_xset"
        plan = json.loads((directory / "dataset_manifest.json").read_text())
        self.assertEqual(plan["generate_indices"], [3])
        self.assertEqual([item["source_dataset"] for item in plan["reuse"]],
                         ["ntu60_xsub", "ntu60_xview", "ntu120_xsub"])
        gif = directory / "rendered/train_000003/preview.gif"
        original = (gif.read_bytes(), gif.stat().st_mtime_ns)
        with patch.object(render_first_xset, "load_captions", side_effect=AssertionError("render must not read captions")), \
                patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(render_first_xset.main(argv + ["--stage", "render"]), 0)
        self.assertEqual((gif.read_bytes(), gif.stat().st_mtime_ns), original)
        self.assertTrue(all(not path.exists() for path, _, _ in banks))
        self.assertFalse((directory / "reuse_caption_sources.json").exists())
        self.assertFalse((directory / "captions.json").exists())

    def test_xset_deferred_text_binding_and_caption_resume_without_npz_loading(self):
        argv, banks, target = self.xset_render_first_fixture()
        import run_caption_dual_gpu as launcher
        with patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(render_first_xset.main(argv + ["--stage", "render"]), 0)
        directory = self.output / "ntu120_xset"
        with patch.object(launcher.subprocess, "Popen", side_effect=AssertionError("captions are not ready")), \
                self.assertRaisesRegex(ValueError, "before --stage caption"):
            render_first_xset.main(argv + ["--stage", "caption"])
        self.assertFalse((directory / "reuse_caption_sources.json").exists())
        self.write_render_first_source_captions(banks)
        with patch.object(render_first_xset, "load_train", side_effect=AssertionError("caption reopened NPZ")), \
                patch.object(pipeline, "load_train", side_effect=AssertionError("caption reopened NPZ")), \
                patch.object(launcher.subprocess, "Popen", side_effect=self.worker), \
                patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(render_first_xset.main(argv + ["--stage", "caption"]), 0)
        self.assertEqual([item[1] for item in self.spawned], [3])
        rows = json.loads((directory / "captions.json").read_text())
        self.assertEqual([row["persons"][0]["global"] for row in rows[:3]],
                         ["The right hand rises.", "The head turns.", "The legs bend."])
        metadata_path = sidecar_paths(directory / "captions.json")[0]
        snapshot = json.loads((directory / "reuse_caption_sources.json").read_text())
        self.assertEqual(json.loads(metadata_path.read_text())["reuse_caption_sources"], snapshot)
        before = ((directory / "captions.json").read_bytes(), metadata_path.read_bytes())
        with patch.object(launcher.subprocess, "Popen", side_effect=AssertionError("no pending samples")), \
                patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(render_first_xset.main(argv + ["--stage", "caption"]), 0)
        self.assertEqual(((directory / "captions.json").read_bytes(), metadata_path.read_bytes()), before)
        import cache_clip_motion_text as cache
        _, valid, _, _ = cache.build_plan(target, [directory / "captions.json"])
        self.assertTrue(valid.all())
        source = banks[1][0]
        changed = json.loads(source.read_text())
        changed[0]["persons"][0]["global"] = "The head tilts."
        source.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "changed since XSet captioning began"):
            render_first_xset.main(argv + ["--stage", "caption"])

    def test_render_first_xset_rejects_coordinate_conflicts_without_reading_text(self):
        argv, _, target = self.xset_render_first_fixture()
        data = pipeline.load_train(target)
        data[1, 0, 0] += 1
        np.savez(target, x_train=data)
        with patch.object(render_first_xset, "load_captions", side_effect=AssertionError("no text needed")), \
                patch("sys.stdout", new_callable=io.StringIO), \
                self.assertRaisesRegex(ValueError, "different coordinates"):
            render_first_xset.main(argv + ["--stage", "render"])
        self.assertFalse((self.output / "ntu120_xset/dataset_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
