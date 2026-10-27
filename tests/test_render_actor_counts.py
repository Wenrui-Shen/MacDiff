"""Distinct-track rendering, raw-slot caption alignment, and legacy GIF repair."""
import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

PILOT = Path(__file__).resolve().parents[1] / "tools" / "vlm_pilot"
sys.path.insert(0, str(PILOT))
import render_skeleton_sample as renderer
import dataset_caption_pipeline as pipeline
import repair_render_actor_counts as repair
import run_caption_pku as pku
from caption_qwen3vl_rendered_train_transformers import render_settings
from caption_qwen3vl_sample import determine_actor_count


class ActorCountTests(unittest.TestCase):
    def setUp(self):
        self.pair = renderer.make_demo_sample(12)
        self.clone = self.pair.copy()
        self.clone[:, 1] = self.clone[:, 0]

    def render(self, sample):
        frames, metadata = renderer.render_sample_frames(sample, num_frames=2, temporal_smooth="none")
        try:
            colors = np.asarray(frames[0])
            red = int(np.all(colors == renderer.PERSON_COLORS[0], axis=-1).sum())
            blue = int(np.all(colors == renderer.PERSON_COLORS[1], axis=-1).sum())
            return metadata, red, blue
        finally:
            for frame in frames:
                frame.close()

    def test_clone_is_one_red_actor_without_mutating_raw_slots(self):
        before = self.clone.copy()
        info, red, blue = self.render(self.clone)
        self.assertEqual(info["visible_actor_count"], 1)
        self.assertEqual(info["raw_active_person_indices"], [0, 1])
        self.assertEqual(info["rendered_person_indices"], [0])
        self.assertTrue(info["duplicate_person1"])
        self.assertFalse(info["blue_skeleton_visible"])
        self.assertEqual(info["active_rendered_frames_per_person"], [2, 2])
        self.assertGreater(red, 0)
        self.assertEqual(blue, 0)
        np.testing.assert_array_equal(self.clone, before)

    def test_real_pair_remains_two_people(self):
        info, red, blue = self.render(self.pair)
        self.assertEqual(info["visible_actor_count"], 2)
        self.assertFalse(info["duplicate_person1"])
        self.assertGreater(red, 0)
        self.assertGreater(blue, 0)

    def test_compare_full_sequence_not_only_sampled_frames_or_matching_pose(self):
        sample = self.clone.copy()
        sample[5, 1, 7, 0] += .01  # Not among the two sampled endpoints.
        info, _, _ = self.render(sample)
        self.assertEqual(info["visible_actor_count"], 2)
        self.assertFalse(info["duplicate_person1"])
        sample = self.clone.copy()
        sample[5, 1] = 0  # A dropped track is also not an exact clone.
        self.assertEqual(renderer.actor_layout(sample)["visible_actor_count"], 2)

    def test_single_red_and_single_blue_are_one_person_without_reordering(self):
        for person in (0, 1):
            sample = self.pair.copy()
            sample[:, 1 - person] = 0
            info, red, blue = self.render(sample)
            self.assertEqual(info["visible_actor_count"], 1)
            self.assertEqual(info["raw_active_person_indices"], [person])
            self.assertEqual(info["rendered_person_indices"], [person])
            self.assertEqual(red > 0, person == 0)
            self.assertEqual(blue > 0, person == 1)
            if person == 1:
                with self.assertRaisesRegex(ValueError, "empty raw person0"):
                    render_settings({"sample_index": 4, "render": info, "config": {"sample_fps": 8}})

    def test_blue_pixels_alone_are_not_a_two_person_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = root / "frames"
            frames.mkdir()
            path = frames / "frame.png"
            Image.new("RGB", (12, 12), renderer.PERSON_COLORS[1]).save(path)
            with self.assertRaisesRegex(ValueError, "Blue-only"):
                determine_actor_count(frames, [path])


class LegacyRenderRepairTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.root = base / "ntu120_xsub"
        self.path = base / "data.npz"
        pair = renderer.make_demo_sample(12)
        clone = pair.copy()
        clone[:, 1] = clone[:, 0]
        self.data = np.stack([clone, pair]).reshape(2, 12, 150)
        np.savez(self.path, x_train=self.data, y_train=np.array([{"forbidden": True}], dtype=object))
        self.args = pku.parse_args(["--num_frames", "2", "--temporal_smooth", "none"])
        identity = pipeline.pipeline_identity(self.args, self.root.name, self.path)
        identity["implementation"] = {name: sorted(hashes)[0]
                                      for name, hashes in repair.LEGACY_IMPLEMENTATIONS.items()}
        self.plan = {"schema_version": pipeline.PLAN_SCHEMA, "identity": identity,
                     "sample_count": 2, "generate_indices": [0, 1], "reuse": []}
        pipeline.atomic_json(self.root / "dataset_manifest.json", self.plan)
        original = renderer.actor_layout
        def legacy_layout(sequence):
            layout = original(sequence)
            layout.update(actor_layout_version="legacy", rendered_person_indices=[0, 1],
                          visible_actor_count=2, duplicate_person1=False)
            return layout
        with patch.object(renderer, "actor_layout", side_effect=legacy_layout), \
                contextlib.redirect_stdout(io.StringIO()):
            pipeline.render_selected(self.data, {0, 1}, self.args, self.path, self.root / "rendered")
        for path in (self.root / "rendered").glob("*/render_metadata.json"):
            metadata = json.loads(path.read_text())
            for key in ("actor_layout_version", "raw_active_person_indices", "rendered_person_indices", "duplicate_person1"):
                metadata["render"].pop(key)
            pipeline.atomic_json(path, metadata)

    def snapshot(self):
        return {path.relative_to(self.root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
                for path in self.root.rglob("*") if path.is_file()}

    def test_repair_only_clone_then_noop_resume_preserves_other_files(self):
        before = self.snapshot()
        with contextlib.redirect_stdout(io.StringIO()):
            report = repair.repair_dataset(self.root, dry_run=True)
        self.assertEqual(report["rerendered"], 1)
        self.assertEqual(self.snapshot(), before)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(repair.repair_dataset(self.root)["rerendered"], 1)
        after = self.snapshot()
        for name in ("preview.gif", "render_metadata.json"):
            key = "rendered/train_000001/" + name
            self.assertEqual(after[key], before[key])
        self.assertNotEqual(after["rendered/train_000000/preview.gif"][0], before["rendered/train_000000/preview.gif"][0])
        self.assertEqual(json.loads((self.root / "dataset_manifest.before_actor_layout_v2.json").read_text()), self.plan)
        current = json.loads((self.root / "dataset_manifest.json").read_text())
        self.assertEqual(current["identity"]["implementation"], repair.current_implementation())
        current["identity"]["implementation"] = self.plan["identity"]["implementation"]
        self.assertEqual(current, self.plan)
        info = json.loads((self.root / "rendered/train_000000/render_metadata.json").read_text())["render"]
        self.assertTrue(info["duplicate_person1"])
        self.assertEqual(info["visible_actor_count"], 1)
        with patch.object(repair, "load_train", side_effect=AssertionError("no-op must not read NPZ")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(repair.repair_dataset(self.root)["already_current"])
        self.assertEqual(self.snapshot(), after)

    def test_refuse_existing_captions_unknown_producer_or_changed_data(self):
        captions = self.root / "captions.json"
        captions.write_text("[]")
        with self.assertRaisesRegex(ValueError, "render-only"):
            repair.repair_dataset(self.root)
        captions.unlink()
        changed = copy.deepcopy(self.plan)
        changed["identity"]["implementation"]["render_skeleton_sample.py"] = "unknown"
        pipeline.atomic_json(self.root / "dataset_manifest.json", changed)
        with self.assertRaisesRegex(ValueError, "Unknown previous"):
            repair.repair_dataset(self.root)
        pipeline.atomic_json(self.root / "dataset_manifest.json", self.plan)
        self.path.write_bytes(self.path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "Source NPZ identity changed"):
            repair.repair_dataset(self.root)

    def test_interrupted_repair_keeps_old_plan_and_can_resume(self):
        with patch.object(repair, "render_sample_frames", side_effect=RuntimeError("interrupted")), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "interrupted"):
            repair.repair_dataset(self.root)
        self.assertEqual(json.loads((self.root / "dataset_manifest.json").read_text()), self.plan)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(repair.repair_dataset(self.root)["rerendered"], 1)


if __name__ == "__main__":
    unittest.main()
