#!/usr/bin/env python3
"""Caption the complete ``x_train`` split with Qwen3-VL and Transformers.

The model is loaded once and kept on the GPU for the lifetime of the process.
For multi-GPU inference, run one process per GPU and use ``--num_shards`` plus
``--shard_id``.  Each process must write to its own JSONL file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from caption_qwen3vl_sample import extract_json, render_prompt, validate_caption
from render_skeleton_sample import load_font, render_sample_frames


DEFAULT_PROMPT = SCRIPT_DIR / "skeleton_motion_prompt_v1.txt"
SCHEMA_VERSION = "macdiff.person_caption.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run persistent Qwen3-VL Transformers inference over x_train. "
            "Use one process per GPU for data-parallel sharding."
        )
    )
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--prompt_path", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--trust_remote_code", action="store_true")

    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit the global source range before it is divided into shards.",
    )
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)

    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--sample_fps", type=float, default=8.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--min_pixels", type=int, default=6 * 32 * 32)
    parser.add_argument("--max_pixels", type=int, default=360 * 32 * 32)
    parser.add_argument("--total_pixels", type=int, default=12288 * 32 * 32)

    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--attn_implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default=None,
    )

    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate data, rendering, prompt, and sharding without loading the model.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.start_index < 0:
        raise ValueError("--start_index must be non-negative")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max_samples must be positive")
    if args.num_shards <= 0:
        raise ValueError("--num_shards must be positive")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard_id must be in [0, --num_shards)")
    if args.num_frames <= 0:
        raise ValueError("--num_frames must be positive")
    if args.sample_fps <= 0:
        raise ValueError("--sample_fps must be positive")
    if args.width <= 0 or args.height <= 0 or args.width % 2:
        raise ValueError("--width/--height must be positive and --width divisible by 2")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive")
    if args.max_retries < 0:
        raise ValueError("--max_retries must be non-negative")
    if min(args.min_pixels, args.max_pixels, args.total_pixels) <= 0:
        raise ValueError("pixel limits must be positive")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_accepted_indices(
    path: Path,
    *,
    expected_model: str,
    expected_prompt_hash: str,
) -> Set[int]:
    accepted: Set[int] = set()
    if not path.exists():
        return accepted

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {path} at line {line_number}: {exc}"
                ) from exc
            if record.get("status") != "accepted":
                continue
            record_model = record.get("model")
            record_prompt_hash = (record.get("prompt") or {}).get("sha256")
            if record_model != expected_model or record_prompt_hash != expected_prompt_hash:
                raise ValueError(
                    f"Accepted records in {path} were produced with another model or "
                    "prompt. Use a new --output_path instead of mixing runs."
                )
            accepted.add(int(record["sample_index"]))
    return accepted


def select_indices(
    total_samples: int,
    *,
    start_index: int,
    max_samples: Optional[int],
    num_shards: int,
    shard_id: int,
    accepted: Set[int],
) -> List[int]:
    stop_index = total_samples
    if max_samples is not None:
        stop_index = min(total_samples, start_index + max_samples)
    if start_index >= stop_index:
        return []
    return [
        index
        for index in range(start_index, stop_index)
        if index % num_shards == shard_id and index not in accepted
    ]


def make_messages(
    frames: Sequence[Any],
    prompt: str,
    *,
    sample_fps: float,
    min_pixels: int,
    max_pixels: int,
    total_pixels: int,
) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": list(frames),
                    "sample_fps": sample_fps,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "total_pixels": total_pixels,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def make_repair_prompt(base_prompt: str, errors: Sequence[str]) -> str:
    summarized = "; ".join(errors[:8]) if errors else "invalid JSON output"
    return (
        f"{base_prompt}\n\n"
        "Your previous answer failed deterministic validation for these reasons: "
        f"{summarized}. Return a corrected object only. Do not add Markdown fences or "
        "explanations."
    )


def close_frames(frames: Sequence[Any]) -> None:
    for frame in frames:
        close = getattr(frame, "close", None)
        if callable(close):
            close()


def split_video_metadata(videos: Any) -> Tuple[Any, Any]:
    if videos is None or len(videos) == 0:
        return videos, None
    separated_videos, metadata = zip(*videos)
    return list(separated_videos), list(metadata)


def prepare_media(
    *,
    sample: np.ndarray,
    prompt: str,
    args: argparse.Namespace,
    font: Any,
    process_vision_info: Any,
) -> Tuple[List[Any], Dict[str, Any], Any, Any, Any, Dict[str, Any]]:
    frames: List[Any] = []
    try:
        frames, render_info = render_sample_frames(
            sample,
            num_frames=args.num_frames,
            width=args.width,
            height=args.height,
            font=font,
        )
        messages = make_messages(
            frames,
            prompt,
            sample_fps=args.sample_fps,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            total_pixels=args.total_pixels,
        )
        images, videos, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
            return_video_metadata=True,
            image_patch_size=16,
        )
        if videos is None:
            raise ValueError("Qwen preprocessing returned no video input")
        videos, video_metadata = split_video_metadata(videos)
        return frames, render_info, images, videos, video_metadata, video_kwargs
    except Exception:
        close_frames(frames)
        raise


def generate_once(
    *,
    model: Any,
    processor: Any,
    frames: Sequence[Any],
    prompt: str,
    images: Any,
    videos: Any,
    video_metadata: Any,
    video_kwargs: Dict[str, Any],
    args: argparse.Namespace,
    torch: Any,
) -> str:
    messages = make_messages(
        frames,
        prompt,
        sample_fps=args.sample_fps,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        total_pixels=args.total_pixels,
    )
    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    processor_kwargs: Dict[str, Any] = {
        "text": chat_text,
        "images": images,
        "videos": videos,
        "return_tensors": "pt",
        "do_resize": False,
        **video_kwargs,
    }
    if video_metadata is not None:
        processor_kwargs["video_metadata"] = video_metadata

    inputs = processor(**processor_kwargs)
    inputs = inputs.to(model.device)
    generated_ids = None
    try:
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        trimmed_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    finally:
        del inputs
        if generated_ids is not None:
            del generated_ids


def parse_response(raw_text: str, expected_actors: int) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    try:
        parsed = extract_json(raw_text)
    except (ValueError, json.JSONDecodeError) as exc:
        return None, [f"JSON parse failed: {exc}"]
    errors = validate_caption(parsed, expected_actors=expected_actors)
    return parsed, errors


def quantization_metadata(model_config: Any) -> Tuple[Optional[str], Any]:
    quant_config = getattr(model_config, "quantization_config", None)
    if quant_config is None:
        return None, None
    if hasattr(quant_config, "to_dict"):
        quant_dict = quant_config.to_dict()
    elif isinstance(quant_config, dict):
        quant_dict = quant_config
    else:
        quant_dict = {"value": str(quant_config)}
    return quant_dict.get("quant_method"), quant_dict


def make_record(
    *,
    sample_index: int,
    actor_count: int,
    render_info: Dict[str, Any],
    raw_text: str,
    caption: Optional[Dict[str, Any]],
    errors: Sequence[str],
    attempts: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    prompt_hash: str,
    render_seconds: float,
    generation_seconds: float,
) -> Dict[str, Any]:
    accepted = caption is not None and not errors
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": f"train_{sample_index:06d}",
        "sample_index": sample_index,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_split": "train",
        "labels_read": False,
        "model": args.model,
        "model_revision": args.resolved_revision,
        "quantization": args.quantization,
        "quantization_config": args.quantization_config,
        "inference": {
            "engine": "transformers",
            "transformers_version": args.transformers_version,
            "torch_version": args.torch_version,
            "dtype": args.resolved_dtype,
            "device": args.resolved_device,
            "attention": args.attn_implementation,
            "max_new_tokens": args.max_new_tokens,
            "sample_fps": args.sample_fps,
            "num_shards": args.num_shards,
            "shard_id": args.shard_id,
        },
        "prompt": {
            "path": str(args.prompt_path.resolve()),
            "sha256": prompt_hash,
        },
        "render": {
            "layout": [
                "front_xy_root_centered",
                "side_zy_root_centered",
            ],
            "person_colors": {"0": "red", "1": "blue"},
            "width": args.width,
            "height": args.height,
            **render_info,
        },
        "actor_count": actor_count,
        "status": "accepted" if accepted else "invalid",
        "errors": list(errors),
        "retry_count": max(0, len(attempts) - 1),
        "preparation_seconds": round(render_seconds, 4),
        "generation_seconds": round(generation_seconds, 4),
        "raw_response": raw_text,
        "caption": caption,
        "texts": (
            [
                {
                    "person_index": person["person_index"],
                    "color": person["color"],
                    "text": person["text"],
                }
                for person in caption["persons"]
            ]
            if accepted
            else []
        ),
        "attempts": list(attempts),
    }


def write_jsonl_record(handle: Any, record: Dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def write_pipeline_error(
    *,
    handle: Any,
    sample_index: int,
    error: Exception,
    args: argparse.Namespace,
    prompt_hash: str,
) -> None:
    write_jsonl_record(
        handle,
        {
            "schema_version": SCHEMA_VERSION,
            "sample_id": f"train_{sample_index:06d}",
            "sample_index": sample_index,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_split": "train",
            "labels_read": False,
            "model": args.model,
            "model_revision": args.resolved_revision,
            "inference": {
                "engine": "transformers",
                "num_shards": args.num_shards,
                "shard_id": args.shard_id,
            },
            "prompt": {
                "path": str(args.prompt_path.resolve()),
                "sha256": prompt_hash,
            },
            "status": "pipeline_error",
            "errors": [f"{type(error).__name__}: {error}"],
            "retry_count": 0,
            "raw_response": "",
            "caption": None,
            "texts": [],
            "attempts": [],
        },
    )


def run_dry_run(
    *,
    x_train: np.ndarray,
    selected_indices: Sequence[int],
    prompt_template: str,
    args: argparse.Namespace,
) -> None:
    print(
        json.dumps(
            {
                "mode": "dry_run",
                "total_train_samples": int(x_train.shape[0]),
                "selected_for_this_shard": len(selected_indices),
                "num_shards": args.num_shards,
                "shard_id": args.shard_id,
                "first_indices": list(selected_indices[:10]),
            },
            indent=2,
        )
    )
    if not selected_indices:
        return
    index = selected_indices[0]
    frames: List[Any] = []
    try:
        frames, info = render_sample_frames(
            x_train[index],
            num_frames=args.num_frames,
            width=args.width,
            height=args.height,
            font=load_font(18),
        )
        actor_count = int(info["visible_actor_count"])
        rendered_prompt = render_prompt(prompt_template, actor_count)
        print(
            json.dumps(
                {
                    "sample_index": index,
                    "actor_count": actor_count,
                    "render": info,
                    "rendered_frames": len(frames),
                    "rendered_prompt": rendered_prompt,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        close_frames(frames)


def main() -> None:
    args = parse_args()
    validate_args(args)

    if not args.data_path.is_file():
        raise FileNotFoundError(args.data_path)
    if not args.prompt_path.is_file():
        raise FileNotFoundError(args.prompt_path)

    prompt_template = args.prompt_path.read_text(encoding="utf-8").strip()
    prompt_hash = sha256_file(args.prompt_path)

    if args.output_path.exists() and not args.resume and not args.dry_run:
        raise FileExistsError(
            f"Output already exists: {args.output_path}. Pass --resume or use a new path."
        )
    accepted = (
        load_accepted_indices(
            args.output_path,
            expected_model=args.model,
            expected_prompt_hash=prompt_hash,
        )
        if args.resume
        else set()
    )

    with np.load(args.data_path, mmap_mode="r", allow_pickle=False) as archive:
        if "x_train" not in archive.files:
            raise KeyError(f"x_train not found in {args.data_path}; keys={archive.files}")
        x_train = archive["x_train"]
        selected_indices = select_indices(
            int(x_train.shape[0]),
            start_index=args.start_index,
            max_samples=args.max_samples,
            num_shards=args.num_shards,
            shard_id=args.shard_id,
            accepted=accepted,
        )

        if args.dry_run:
            run_dry_run(
                x_train=x_train,
                selected_indices=selected_indices,
                prompt_template=prompt_template,
                args=args,
            )
            return

        if not selected_indices:
            print("No samples remain for this shard.")
            return

        try:
            import torch
            import transformers
            from qwen_vl_utils import process_vision_info
            from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise SystemExit(
                "Missing Transformers inference dependency. Install transformers, "
                "qwen-vl-utils==0.0.14, Pillow, accelerate, and a compatible CUDA "
                "PyTorch build."
            ) from exc

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        load_kwargs: Dict[str, Any] = {
            "dtype": dtype_map[args.dtype],
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "trust_remote_code": args.trust_remote_code,
        }
        config_kwargs: Dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
        processor_kwargs: Dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
        if args.revision:
            load_kwargs["revision"] = args.revision
            config_kwargs["revision"] = args.revision
            processor_kwargs["revision"] = args.revision
        if args.attn_implementation:
            load_kwargs["attn_implementation"] = args.attn_implementation

        print(
            f"Loading {args.model} once for shard {args.shard_id}/{args.num_shards} "
            f"({len(selected_indices)} pending samples)...",
            flush=True,
        )
        model_config = AutoConfig.from_pretrained(args.model, **config_kwargs)
        processor = AutoProcessor.from_pretrained(args.model, **processor_kwargs)
        model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kwargs)
        model.eval()

        quantization, quantization_config = quantization_metadata(model_config)
        args.resolved_revision = args.revision or getattr(model_config, "_commit_hash", None)
        args.quantization = quantization
        args.quantization_config = quantization_config
        args.transformers_version = transformers.__version__
        args.torch_version = torch.__version__
        try:
            first_parameter = next(model.parameters())
            args.resolved_dtype = str(first_parameter.dtype).replace("torch.", "")
            args.resolved_device = str(first_parameter.device)
        except StopIteration:
            args.resolved_dtype = args.dtype
            args.resolved_device = str(model.device)

        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        output_mode = "a" if args.resume else "w"
        font = load_font(18)

        accepted_now = 0
        invalid_now = 0
        pipeline_errors = 0
        run_started = time.perf_counter()

        with args.output_path.open(output_mode, encoding="utf-8") as output_handle:
            for ordinal, sample_index in enumerate(selected_indices, start=1):
                frames: List[Any] = []
                try:
                    preprocess_started = time.perf_counter()
                    provisional_prompt = render_prompt(prompt_template, 1)
                    (
                        frames,
                        render_info,
                        images,
                        videos,
                        video_metadata,
                        video_kwargs,
                    ) = prepare_media(
                        sample=x_train[sample_index],
                        prompt=provisional_prompt,
                        args=args,
                        font=font,
                        process_vision_info=process_vision_info,
                    )
                    actor_count = int(render_info["visible_actor_count"])
                    base_prompt = render_prompt(prompt_template, actor_count)
                    render_seconds = time.perf_counter() - preprocess_started

                    raw_text = ""
                    caption: Optional[Dict[str, Any]] = None
                    errors: List[str] = []
                    attempts: List[Dict[str, Any]] = []
                    generation_seconds = 0.0

                    for attempt_index in range(args.max_retries + 1):
                        attempt_prompt = (
                            base_prompt
                            if attempt_index == 0
                            else make_repair_prompt(base_prompt, errors)
                        )
                        generation_started = time.perf_counter()
                        raw_text = ""
                        try:
                            raw_text = generate_once(
                                model=model,
                                processor=processor,
                                frames=frames,
                                prompt=attempt_prompt,
                                images=images,
                                videos=videos,
                                video_metadata=video_metadata,
                                video_kwargs=video_kwargs,
                                args=args,
                                torch=torch,
                            )
                            caption, errors = parse_response(raw_text, actor_count)
                        except Exception as exc:
                            caption = None
                            errors = [f"generation failed: {type(exc).__name__}: {exc}"]
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        generation_seconds += time.perf_counter() - generation_started
                        attempt_status = (
                            "accepted" if caption is not None and not errors else "invalid"
                        )
                        attempts.append(
                            {
                                "retry_count": attempt_index,
                                "status": attempt_status,
                                "raw_response": raw_text,
                                "errors": list(errors),
                            }
                        )
                        if caption is not None and not errors:
                            break

                    record = make_record(
                        sample_index=sample_index,
                        actor_count=actor_count,
                        render_info=render_info,
                        raw_text=raw_text,
                        caption=caption,
                        errors=errors,
                        attempts=attempts,
                        args=args,
                        prompt_hash=prompt_hash,
                        render_seconds=render_seconds,
                        generation_seconds=generation_seconds,
                    )
                    write_jsonl_record(output_handle, record)
                    if record["status"] == "accepted":
                        accepted_now += 1
                    else:
                        invalid_now += 1
                except Exception as exc:
                    pipeline_errors += 1
                    write_pipeline_error(
                        handle=output_handle,
                        sample_index=sample_index,
                        error=exc,
                        args=args,
                        prompt_hash=prompt_hash,
                    )
                finally:
                    close_frames(frames)

                elapsed = time.perf_counter() - run_started
                average = elapsed / ordinal
                remaining = average * (len(selected_indices) - ordinal)
                print(
                    f"[shard {args.shard_id}] {ordinal}/{len(selected_indices)} "
                    f"sample={sample_index} accepted={accepted_now} invalid={invalid_now} "
                    f"pipeline_error={pipeline_errors} eta={remaining / 60:.1f} min",
                    flush=True,
                )

        print(
            f"Finished shard {args.shard_id}: accepted={accepted_now}, "
            f"invalid={invalid_now}, pipeline_error={pipeline_errors}, "
            f"output={args.output_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
