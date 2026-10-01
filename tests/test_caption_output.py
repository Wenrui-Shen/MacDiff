"""CPU-only output/resume and global/local caption checks."""
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
from caption_output import (CaptionOutput, LOCAL_PARTS, caption_schema,
                            load_accepted_indices, sidecar_paths)
from caption_qwen3vl_sample import render_prompt, validate_caption
from caption_qwen3vl_train_transformers import parse_response, write_jsonl_record
import caption_qwen3vl_rendered_train_transformers as rendered
import caption_qwen3vl_train_vllm as vllm_train
from inspect_random_captions import load_latest_accepted, format_record


def caption(people=1):
    return {"persons": [{"person_index": i, "global": "Both arms rise and lower.",
                         "local": {part: "%s remains upright." % part for part in LOCAL_PARTS}}
                        for i in range(people)]}


def args(path, resume=False):
    return SimpleNamespace(output_path=path, model="test-model", revision=None,
                           prompt_path=PILOT / "skeleton_motion_prompt_v2.txt",
                           caption_schema="global_local", resume=resume, shard_id=0)


def record(index, status="accepted", people=1):
    return {"sample_index": index, "status": status,
            "caption": caption(people) if status == "accepted" else None,
            "raw_response": "raw", "attempts": [{"status": status}],
            "model": "should not appear in sample output", "inference": {"device": "GPU"}}


class CaptionSchemaTests(unittest.TestCase):
    def test_new_prompt_and_one_two_people(self):
        prompt = (PILOT / "skeleton_motion_prompt_v2.txt").read_text(encoding="utf-8")
        self.assertEqual(caption_schema(prompt), "global_local")
        for count in (1, 2):
            self.assertNotIn("{ACTOR_", render_prompt(prompt, count))
            self.assertEqual(parse_response(json.dumps(caption(count)), count, "global_local")[1], [])

    def test_missing_part_person_mismatch_empty_and_wrong_schema(self):
        bad = caption()
        del bad["persons"][0]["local"]["left_leg"]
        self.assertTrue(validate_caption(bad, 1, "global_local"))
        self.assertTrue(validate_caption(caption(), 2, "global_local"))
        bad = caption()
        bad["persons"][0]["global"] = ""
        self.assertTrue(validate_caption(bad, 1, "global_local"))
        self.assertTrue(validate_caption({"actors": 1, "persons": []}, 1, "global_local"))
        self.assertTrue(validate_caption(caption(), 1, "legacy"))


class CaptionOutputTests(unittest.TestCase):
    def test_json_and_jsonl_samples_resume_and_separate_metadata(self):
        for extension in (".json", ".jsonl"):
            with self.subTest(extension=extension), tempfile.TemporaryDirectory() as root:
                path = Path(root) / ("captions" + extension)
                first = record(0, people=2)
                first["caption"]["persons"][0]["global"] = "The arms rise — then lower."
                with CaptionOutput(args(path), "hash") as sink:
                    write_jsonl_record(sink, first)
                    write_jsonl_record(sink, record(1, "invalid"))
                    write_jsonl_record(sink, record(2, "pipeline_error"))
                    write_jsonl_record(sink, record(3))
                saved = list(load_latest_accepted([path])[0].values())
                self.assertEqual([v["sample_index"] for v in saved], [0, 3])
                self.assertEqual(len(saved[0]["persons"]), 2)
                self.assertEqual(saved[0]["persons"][0]["global"], "The arms rise — then lower.")
                text = path.read_text(encoding="utf-8")
                self.assertNotIn('"model"', text)
                self.assertNotIn('"raw_response"', text)
                self.assertEqual(sum(line.startswith('{"sample_index"') for line in text.splitlines()), 2)
                if extension == ".json":
                    self.assertEqual(len(json.loads(text)), 2)
                metadata, diagnostics = sidecar_paths(path)
                self.assertEqual(json.loads(metadata.read_text())["model"], "test-model")
                audit = [json.loads(line) for line in diagnostics.read_text().splitlines()]
                self.assertEqual([v["status"] for v in audit],
                                 ["accepted", "invalid", "pipeline_error", "accepted"])
                self.assertNotIn("model", audit[0])
                with CaptionOutput(args(path, True), "hash") as sink:
                    sink.write_record(record(1))
                self.assertEqual(load_accepted_indices(path, expected_model="test-model",
                                                      expected_prompt_hash="hash"), {0, 1, 3})
                self.assertEqual(len(json.loads(metadata.read_text())["runs"]), 2)
                inspected, _ = load_latest_accepted([path])
                self.assertIn("left_arm:", format_record(inspected[0], 1, 3, False))

    def test_invalid_only_can_resume_empty_array(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "captions.json"
            with CaptionOutput(args(path), "hash") as sink:
                sink.write_record(record(0, "invalid"))
            self.assertEqual(json.loads(path.read_text()), [])
            with CaptionOutput(args(path, True), "hash") as sink:
                sink.write_record(record(0))
            self.assertEqual(json.loads(path.read_text())[0]["sample_index"], 0)

    def test_identity_missing_metadata_and_duplicate_protection(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "captions.jsonl"
            with CaptionOutput(args(path), "hash") as sink:
                sink.write_record(record(0))
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    sink.write_record(record(0))
            with self.assertRaisesRegex(ValueError, "differs"):
                load_accepted_indices(path, expected_model="other", expected_prompt_hash="hash")
            with self.assertRaisesRegex(ValueError, "differs"):
                load_accepted_indices(path, expected_model="test-model", expected_prompt_hash="other")
            metadata, _ = sidecar_paths(path)
            metadata.unlink()
            with self.assertRaisesRegex(ValueError, "Missing caption metadata"):
                load_accepted_indices(path, expected_model="test-model", expected_prompt_hash="hash")

    def test_jsonl_resume_accepts_valid_last_line_without_newline(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "captions.jsonl"
            with CaptionOutput(args(path), "hash") as sink:
                sink.write_record(record(0))
            path.write_bytes(path.read_bytes().rstrip(b"\n"))
            with CaptionOutput(args(path, True), "hash") as sink:
                sink.write_record(record(1))
            self.assertEqual(load_accepted_indices(path, expected_model="test-model",
                                                  expected_prompt_hash="hash"), {0, 1})

    def test_generated_records_support_global_local_without_legacy_fields(self):
        result = rendered.make_caption_record(
            render_record={"sample_index": 4, "_gif_path": Path("preview.gif"),
                           "_metadata_path": Path("render_metadata.json")},
            actor_count=1, raw_text="raw", caption=caption(), errors=[], attempts=[],
            args=SimpleNamespace(rendered_root=Path("rendered")), prompt_hash="hash",
            preparation_seconds=1, generation_seconds=2)
        self.assertEqual(result["status"], "accepted")
        self.assertNotIn("model", result)
        self.assertEqual(result["caption"]["persons"][0]["local"], caption()["persons"][0]["local"])

    def test_vllm_records_parse_and_write_new_schema(self):
        raw = json.dumps(caption())
        parsed, status, errors = vllm_train.parse_response(raw, 1, schema="global_local")
        state = {"sample_index": 2, "sample_id": "train_2", "status": status,
                 "caption": parsed, "errors": errors, "render_info": {}, "actor_count": 1,
                 "preparation_seconds": 0, "generation_seconds": 0,
                 "attempts": [{"raw_response": raw}]}
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "captions.jsonl"
            with CaptionOutput(args(path), "hash") as sink:
                sink.write_record(vllm_train.make_record(state, SimpleNamespace(), "hash"))
                vllm_train.write_pipeline_error(sink, 3, SimpleNamespace(), "hash", ValueError("bad"))
            self.assertEqual(json.loads(path.read_text())["sample_index"], 2)
            self.assertEqual(len(sidecar_paths(path)[1].read_text().splitlines()), 2)

    def test_rendered_main_handles_success_failure_and_resume(self):
        model = SimpleNamespace(eval=lambda: None, parameters=lambda: iter([
            SimpleNamespace(dtype="float32", device="cpu")]))
        transformers = SimpleNamespace(
            __version__="test", AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace()),
            AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **k: object()),
            AutoModelForImageTextToText=SimpleNamespace(from_pretrained=lambda *a, **k: model))
        torch = SimpleNamespace(__version__="test", version=SimpleNamespace(cuda=None),
                                float16="float16", bfloat16="bfloat16", float32="float32",
                                cuda=SimpleNamespace(is_available=lambda: False))
        qwen = SimpleNamespace(process_vision_info=lambda *a, **k: None)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "captions.json"
            options = args(path)
            options.__dict__.update(rendered_root=Path(root), start_index=0, max_samples=None,
                                    num_shards=1, min_pixels=1, max_pixels=1, total_pixels=1,
                                    max_new_tokens=1024, max_retries=1, dry_run=False, check_gifs=False,
                                    trust_remote_code=False, dtype="auto", attn_implementation=None)
            records = {i: {"sample_index": i, "config": {"sample_fps": 8},
                           "render": {"visible_actor_count": 1},
                           "_gif_path": Path("preview.gif"), "_metadata_path": Path("render_metadata.json")}
                       for i in range(3)}
            def media(**kw):
                index = kw["record"]["sample_index"]
                if index == 2 and not options.resume:
                    raise OSError("bad GIF")
                return [index], 1, "prompt", None, None, None, {}
            def generate(**kw):
                return "invalid JSON" if kw["frames"] == [1] and not options.resume else json.dumps(caption())
            with patch.dict(sys.modules, {"torch": torch, "transformers": transformers, "qwen_vl_utils": qwen}), \
                    patch.object(rendered, "parse_args", return_value=options), \
                    patch.object(rendered, "load_rendered_samples", return_value=records), \
                    patch.object(rendered, "prepare_rendered_media", side_effect=media), \
                    patch.object(rendered, "generate_once", side_effect=generate), \
                    patch("sys.stdout", new_callable=io.StringIO):
                rendered.main()
                self.assertEqual([v["sample_index"] for v in json.loads(path.read_text())], [0])
                _, diagnostics = sidecar_paths(path)
                audit = [json.loads(v) for v in diagnostics.read_text().splitlines()]
                self.assertEqual([v["status"] for v in audit], ["accepted", "invalid", "pipeline_error"])
                self.assertEqual(audit[1]["retry_count"], 1)
                options.resume = True
                rendered.main()
            self.assertEqual([v["sample_index"] for v in json.loads(path.read_text())], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
