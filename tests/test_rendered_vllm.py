"""CPU-only checks for the GIF-to-vLLM boundary; no model download required."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "vlm_pilot"))
from caption_qwen3vl_rendered_train_transformers import generate_vllm_once, select_records


class RenderedVllmTests(unittest.TestCase):
    def test_video_metadata_and_sampling_preserved(self):
        requests = []
        model = SimpleNamespace(generate=lambda inputs, params, **kw: (
            requests.extend(inputs) or [SimpleNamespace(outputs=[SimpleNamespace(text="result")])]
        ))
        processor = SimpleNamespace(apply_chat_template=lambda messages, **kw: "chat")
        args = SimpleNamespace(sample_fps=8, min_pixels=6144, max_pixels=368640,
                               total_pixels=12582912, sampling_params=object())
        metadata = {"fps": 8, "frames_indices": list(range(32)), "total_num_frames": 32}
        video = object()
        result = generate_vllm_once(
            model=model, processor=processor, frames=[object()] * 32, prompt="test",
            images=None, videos=[video], video_metadata=[metadata],
            video_kwargs={"do_sample_frames": True}, args=args, torch=None,
        )
        self.assertEqual(result, "result")
        self.assertIs(requests[0]["multi_modal_data"]["video"][0][0], video)
        self.assertEqual(requests[0]["multi_modal_data"]["video"][0][1], metadata)
        self.assertFalse(requests[0]["mm_processor_kwargs"]["do_sample_frames"])
        self.assertFalse(requests[0]["mm_processor_kwargs"]["do_resize"])

    def test_single_sample_then_full_resume(self):
        records = {i: {"sample_index": i} for i in range(6)}
        self.assertEqual(select_records(records, start_index=2, max_samples=1,
                         num_shards=1, shard_id=0, accepted=[]), [records[2]])
        self.assertEqual(select_records(records, start_index=0, max_samples=None,
                         num_shards=1, shard_id=0, accepted=[2]),
                         [records[i] for i in (0, 1, 3, 4, 5)])


if __name__ == "__main__":
    unittest.main()
