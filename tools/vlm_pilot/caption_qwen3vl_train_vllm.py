#!/usr/bin/env python3
"""Extract person-specific captions from every ``x_train`` skeleton sequence.

The script uses one persistent offline vLLM engine. Skeleton frames are rendered
in memory, so labels, sample paths, and class metadata are never sent to the VLM.
Accepted results are appended to a resumable JSONL manifest.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from caption_qwen3vl_sample import extract_json, render_prompt, validate_caption
from render_skeleton_sample import load_font, render_sample_frames


SCHEMA_VERSION = "macdiff.person_caption.v1"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run label-free Qwen video captioning over the complete x_train "
            "array with a persistent offline vLLM engine."
        )
    )
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--prompt_path",
        type=Path,
        default=Path(__file__).with_name("skeleton_motion_prompt_v1.txt"),
    )
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--quantization",
        default=None,
        help="Optional vLLM quantization override; checkpoint metadata is used by default.",
    )
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Process at most this many indices; 0 means through the end of x_train.",
    )
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--sample_fps", type=float, default=8.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--max_retries",
        type=int,
        default=2,
        help="Validation-guided correction attempts after the first generation.",
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument(
        "--dtype",
        choices=("auto", "half", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing manifest and skip sample indices already accepted.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Render and inspect one selected sample without importing vLLM or writing output.",
    )
    args = parser.parse_args()

    if args.start_index < 0:
        parser.error("--start_index must be non-negative")
    if args.max_samples < 0:
        parser.error("--max_samples must be non-negative")
    if args.num_frames < 2:
        parser.error("--num_frames must be at least 2")
    if args.sample_fps <= 0:
        parser.error("--sample_fps must be positive")
    if args.width <= 0 or args.width % 3:
        parser.error("--width must be positive and divisible by 3")
    if args.height <= 0:
        parser.error("--height must be positive")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if args.max_new_tokens <= 0:
        parser.error("--max_new_tokens must be positive")
    if args.max_retries < 0:
        parser.error("--max_retries must be non-negative")
    if args.tensor_parallel_size <= 0:
        parser.error("--tensor_parallel_size must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu_memory_utilization must be in (0, 1]")
    return args


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_accepted_indices(path):
    accepted = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "%s has malformed JSON on line %d; repair or remove the partial "
                    "line before resuming" % (path, line_number)
                ) from exc
            if record.get("status") == "accepted":
                accepted.add(int(record["sample_index"]))
    return accepted


def select_indices(total_samples, start_index, max_samples, accepted):
    if start_index >= total_samples:
        raise ValueError(
            "start_index %d is outside x_train with %d samples"
            % (start_index, total_samples)
        )
    stop_index = total_samples
    if max_samples:
        stop_index = min(total_samples, start_index + max_samples)
    return [index for index in range(start_index, stop_index) if index not in accepted]


def make_messages(frames, prompt_text, sample_fps):
    return [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": frames,
                "sample_fps": sample_fps,
                "raw_fps": sample_fps,
                "min_pixels": 6 * 32 * 32,
                "max_pixels": 360 * 32 * 32,
                "total_pixels": 12288 * 32 * 32,
            },
            {"type": "text", "text": prompt_text},
        ],
    }]


def make_repair_prompt(base_prompt, errors):
    error_lines = "\n".join("- %s" % error for error in errors)
    return (
        base_prompt
        + "\n\nYour previous response failed validation. Correct every violation "
          "below while preserving only visually supported motion:\n"
        + error_lines
        + "\nReturn the corrected JSON object only."
    )


def prepare_request(
    sample,
    sample_index,
    prompt_template,
    processor,
    process_vision_info,
    args,
    font,
):
    started = time.perf_counter()
    frames, render_info = render_sample_frames(
        sample,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        font=font,
    )
    try:
        actor_count = render_info["visible_actor_count"]
        base_prompt = render_prompt(prompt_template, actor_count)
        messages = make_messages(frames, base_prompt, args.sample_fps)
        chat_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=getattr(processor.image_processor, "patch_size", 16),
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        if "video" not in mm_data:
            raise ValueError("Qwen preprocessing returned no video input")
        video_kwargs["do_resize"] = False
        llm_input = {
            "prompt": chat_prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }
    except Exception:
        for frame in frames:
            frame.close()
        raise
    return {
        "sample_index": sample_index,
        "sample_id": "train_%d" % sample_index,
        "actor_count": actor_count,
        "base_prompt": base_prompt,
        "frames": frames,
        "render_info": render_info,
        "llm_input": llm_input,
        "attempts": [],
        "caption": None,
        "errors": [],
        "status": None,
        "preparation_seconds": time.perf_counter() - started,
        "generation_seconds": 0.0,
    }


def close_frames(state):
    for frame in state.get("frames", []):
        frame.close()
    state["frames"] = []


def parse_response(raw_response, actor_count):
    try:
        caption = extract_json(raw_response)
    except (ValueError, json.JSONDecodeError) as exc:
        return None, "invalid_json", [str(exc)]
    errors = validate_caption(caption, actor_count)
    if errors:
        return caption, "invalid_content", errors
    return caption, "accepted", []


def generate_resilient(llm, inputs, sampling_params):
    """Generate a batch, splitting failures so one bad request is isolated."""
    if not inputs:
        return []
    try:
        outputs = llm.generate(
            inputs, sampling_params=sampling_params, use_tqdm=False
        )
        return [output.outputs[0].text for output in outputs]
    except Exception as exc:  # vLLM errors vary across engine versions.
        if len(inputs) == 1:
            return [exc]
        middle = len(inputs) // 2
        return (
            generate_resilient(llm, inputs[:middle], sampling_params)
            + generate_resilient(llm, inputs[middle:], sampling_params)
        )


def run_caption_attempts(states, llm, processor, sampling_params, args):
    pending = list(states)
    for retry_count in range(args.max_retries + 1):
        if not pending:
            break
        inputs = []
        for state in pending:
            if retry_count == 0:
                inputs.append(state["llm_input"])
                continue
            repair_prompt = make_repair_prompt(state["base_prompt"], state["errors"])
            messages = make_messages(state["frames"], repair_prompt, args.sample_fps)
            chat_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            repaired_input = dict(state["llm_input"])
            repaired_input["prompt"] = chat_prompt
            inputs.append(repaired_input)

        started = time.perf_counter()
        responses = generate_resilient(llm, inputs, sampling_params)
        batch_seconds = time.perf_counter() - started
        per_sample_seconds = batch_seconds / max(len(pending), 1)
        next_pending = []
        for state, response in zip(pending, responses):
            state["generation_seconds"] += per_sample_seconds
            if isinstance(response, Exception):
                status = "engine_error"
                caption = None
                errors = ["%s: %s" % (type(response).__name__, response)]
                raw_response = ""
            else:
                raw_response = response
                caption, status, errors = parse_response(
                    raw_response, state["actor_count"]
                )
            state["attempts"].append({
                "retry_count": retry_count,
                "status": status,
                "errors": errors,
                "raw_response": raw_response,
            })
            state["caption"] = caption
            state["status"] = status
            state["errors"] = errors
            if status != "accepted" and retry_count < args.max_retries:
                next_pending.append(state)
        pending = next_pending


def make_record(state, args, prompt_hash):
    accepted = state["status"] == "accepted"
    caption = state["caption"]
    texts = []
    if accepted:
        texts = [
            {
                "person_index": person["person_index"],
                "color": person["color"],
                "text": person["text"],
            }
            for person in caption["persons"]
        ]
    last_response = state["attempts"][-1]["raw_response"]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_id": state["sample_id"],
        "sample_index": state["sample_index"],
        "source_split": "train",
        "labels_read": False,
        "model": args.model,
        "model_revision": args.resolved_model_revision,
        "quantization": args.resolved_quantization,
        "inference": {
            "engine": "vllm_offline",
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.0,
            "seed": args.seed,
            "sample_fps": args.sample_fps,
        },
        "prompt": {
            "path": str(args.prompt_path.resolve()),
            "sha256": prompt_hash,
        },
        "render": {
            "layout": [
                "front_xy_root_centered",
                "side_zy_root_centered",
                "top_xz_root_centered",
            ],
            "person_colors": {"0": "red", "1": "blue"},
            "width": args.width,
            "height": args.height,
            **state["render_info"],
        },
        "actor_count": state["actor_count"],
        "status": state["status"],
        "errors": state["errors"],
        "retry_count": len(state["attempts"]) - 1,
        "preparation_seconds": state["preparation_seconds"],
        "generation_seconds": state["generation_seconds"],
        "raw_response": last_response,
        "caption": caption,
        "texts": texts,
        "attempts": state["attempts"],
    }


def write_pipeline_error(sink, sample_index, args, prompt_hash, exc):
    record = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_id": "train_%d" % sample_index,
        "sample_index": sample_index,
        "source_split": "train",
        "labels_read": False,
        "model": args.model,
        "model_revision": args.resolved_model_revision,
        "quantization": args.resolved_quantization,
        "prompt": {
            "path": str(args.prompt_path.resolve()),
            "sha256": prompt_hash,
        },
        "status": "pipeline_error",
        "errors": ["%s: %s" % (type(exc).__name__, exc)],
        "caption": None,
        "texts": [],
    }
    sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    sink.flush()


def dry_run(data, sample_index, prompt_template, args):
    frames, render_info = render_sample_frames(
        data[sample_index],
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
    )
    try:
        actor_count = render_info["visible_actor_count"]
        summary = {
            "sample_id": "train_%d" % sample_index,
            "labels_read": False,
            "actor_count": actor_count,
            "render": render_info,
            "prompt": render_prompt(prompt_template, actor_count),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        for frame in frames:
            frame.close()


def main():
    args = parse_args()
    if not args.data_path.is_file():
        raise FileNotFoundError(args.data_path)
    if not args.prompt_path.is_file():
        raise FileNotFoundError(args.prompt_path)

    prompt_template = args.prompt_path.read_text(encoding="utf-8")
    prompt_hash = sha256_file(args.prompt_path)
    accepted_indices = set()
    if args.output_path.exists():
        if not args.resume and not args.dry_run:
            raise FileExistsError(
                "%s already exists; pass --resume to append and skip accepted samples"
                % args.output_path
            )
        if args.resume:
            accepted_indices = load_accepted_indices(args.output_path)

    archive = np.load(str(args.data_path), mmap_mode="r", allow_pickle=False)
    try:
        if "x_train" not in archive:
            raise KeyError("%s does not contain x_train" % args.data_path)
        data = archive["x_train"]
        total_samples = int(data.shape[0])
        indices = select_indices(
            total_samples,
            args.start_index,
            args.max_samples,
            accepted_indices,
        )
        if not indices:
            print("No samples remain in the selected range.")
            return
        if args.dry_run:
            dry_run(data, indices[0], prompt_template, args)
            return

        try:
            from qwen_vl_utils import process_vision_info
            from transformers import AutoConfig, AutoProcessor
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise SystemExit(
                "Missing batch-inference dependency. Install vllm>=0.11.0, "
                "transformers, qwen-vl-utils==0.0.14, Pillow, and a compatible "
                "CUDA PyTorch build."
            ) from exc

        print(
            "x_train samples=%d selected=%d already_accepted=%d"
            % (total_samples, len(indices), len(accepted_indices))
        )
        config = AutoConfig.from_pretrained(
            args.model,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
        )
        quantization_config = getattr(config, "quantization_config", None)
        checkpoint_quantization = None
        if isinstance(quantization_config, dict):
            checkpoint_quantization = quantization_config.get("quant_method")
        args.resolved_model_revision = (
            args.revision or getattr(config, "_commit_hash", None) or "local_or_unresolved"
        )
        args.resolved_quantization = (
            args.quantization or checkpoint_quantization or "none_or_checkpoint_default"
        )
        processor = AutoProcessor.from_pretrained(
            args.model,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
        )
        llm_kwargs = {
            "model": args.model,
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.dtype,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.batch_size,
            "limit_mm_per_prompt": {"video": 1},
            "enforce_eager": args.enforce_eager,
            "trust_remote_code": args.trust_remote_code,
            "seed": args.seed,
        }
        if args.revision is not None:
            llm_kwargs["revision"] = args.revision
        if args.quantization is not None:
            llm_kwargs["quantization"] = args.quantization
        llm = LLM(**llm_kwargs)
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=args.max_new_tokens,
            seed=args.seed,
        )
        font = load_font(14)
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        with args.output_path.open("a", encoding="utf-8") as sink:
            completed = 0
            for offset in range(0, len(indices), args.batch_size):
                batch_indices = indices[offset:offset + args.batch_size]
                states = []
                for sample_index in batch_indices:
                    try:
                        state = prepare_request(
                            data[sample_index],
                            sample_index,
                            prompt_template,
                            processor,
                            process_vision_info,
                            args,
                            font,
                        )
                        states.append(state)
                    except Exception as exc:
                        write_pipeline_error(
                            sink, sample_index, args, prompt_hash, exc
                        )
                        completed += 1
                        print(
                            "[%d/%d] train_%d pipeline_error: %s"
                            % (completed, len(indices), sample_index, exc),
                            file=sys.stderr,
                        )

                try:
                    run_caption_attempts(
                        states, llm, processor, sampling_params, args
                    )
                    for state in states:
                        record = make_record(state, args, prompt_hash)
                        sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                        sink.flush()
                        completed += 1
                        print(
                            "[%d/%d] %s %s actors=%d retries=%d"
                            % (
                                completed,
                                len(indices),
                                state["sample_id"],
                                state["status"],
                                state["actor_count"],
                                len(state["attempts"]) - 1,
                            )
                        )
                finally:
                    for state in states:
                        close_frames(state)
    finally:
        archive.close()


if __name__ == "__main__":
    main()
