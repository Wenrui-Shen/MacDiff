#!/usr/bin/env python3
"""Caption persistent skeleton GIFs without reopening the source NPZ.

This is stage 2 of the split rendering/captioning pipeline.  One model remains
resident per process.  Run one process per GPU with matching ``--num_shards``
and distinct JSONL output paths.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from caption_qwen3vl_sample import render_prompt
from caption_qwen3vl_train_transformers import (
    DEFAULT_PROMPT,
    SCHEMA_VERSION,
    close_frames,
    generate_once,
    load_accepted_indices,
    make_messages,
    make_repair_prompt,
    parse_response,
    quantization_metadata,
    sha256_file,
    split_video_metadata,
    write_jsonl_record,
)


RENDER_SCHEMA_VERSION = "macdiff.skeleton_render.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen3-VL over persistent rendered train GIFs."
    )
    parser.add_argument("--rendered_root", type=Path, required=True)
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
        help="Limit the global index range before it is divided into shards.",
    )
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
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
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.rendered_root.is_dir():
        raise FileNotFoundError(args.rendered_root)
    if not args.prompt_path.is_file():
        raise FileNotFoundError(args.prompt_path)
    if args.start_index < 0:
        raise ValueError("--start_index must be non-negative")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max_samples must be positive")
    if args.num_shards <= 0:
        raise ValueError("--num_shards must be positive")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard_id must be in [0, --num_shards)")
    if args.max_new_tokens <= 0 or args.max_retries < 0:
        raise ValueError("token and retry limits must be non-negative")
    if min(args.min_pixels, args.max_pixels, args.total_pixels) <= 0:
        raise ValueError("pixel limits must be positive")


def normalize_render_record(record: Dict[str, Any], rendered_root: Path) -> Dict[str, Any]:
    sample_index = int(record["sample_index"])
    sample_dir = rendered_root / f"train_{sample_index:06d}"
    gif_path = sample_dir / "preview.gif"
    metadata_path = sample_dir / "render_metadata.json"
    normalized = dict(record)
    normalized["sample_index"] = sample_index
    normalized["sample_id"] = f"train_{sample_index:06d}"
    normalized["_sample_dir"] = sample_dir.resolve()
    normalized["_gif_path"] = gif_path.resolve()
    normalized["_metadata_path"] = metadata_path.resolve()
    return normalized


def load_rendered_samples(rendered_root: Path) -> Dict[int, Dict[str, Any]]:
    records: Dict[int, Dict[str, Any]] = {}
    manifest_paths = sorted(rendered_root.glob("render_manifest_shard*.jsonl"))
    for manifest_path in manifest_paths:
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid render manifest {manifest_path}:{line_number}: {exc}"
                    ) from exc
                if record.get("status") != "rendered":
                    continue
                normalized = normalize_render_record(record, rendered_root)
                records[normalized["sample_index"]] = normalized

    if not records:
        for metadata_path in sorted(rendered_root.glob("train_*/render_metadata.json")):
            record = json.loads(metadata_path.read_text(encoding="utf-8"))
            if record.get("status") != "rendered":
                continue
            normalized = normalize_render_record(record, rendered_root)
            records[normalized["sample_index"]] = normalized

    valid_records: Dict[int, Dict[str, Any]] = {}
    for sample_index, record in records.items():
        if record.get("schema_version") != RENDER_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported render schema for sample {sample_index}: "
                f"{record.get('schema_version')}"
            )
        if not record["_gif_path"].is_file():
            raise FileNotFoundError(record["_gif_path"])
        valid_records[sample_index] = record
    return valid_records


def select_records(
    records: Dict[int, Dict[str, Any]],
    *,
    start_index: int,
    max_samples: Optional[int],
    num_shards: int,
    shard_id: int,
    accepted: Sequence[int],
) -> List[Dict[str, Any]]:
    stop_index = None if max_samples is None else start_index + max_samples
    accepted_set = set(accepted)
    return [
        records[index]
        for index in sorted(records)
        if index >= start_index
        and (stop_index is None or index < stop_index)
        and index % num_shards == shard_id
        and index not in accepted_set
    ]


def load_gif_frames(path: Path, expected_frames: Optional[int]) -> List[Any]:
    from PIL import Image, ImageSequence

    with Image.open(path) as animation:
        frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(animation)]
    if len(frames) < 2:
        close_frames(frames)
        raise ValueError(f"Expected at least two GIF frames in {path}")
    if expected_frames is not None and len(frames) != expected_frames:
        close_frames(frames)
        raise ValueError(
            f"GIF frame count mismatch for {path}: {len(frames)} vs {expected_frames}"
        )
    return frames


def render_settings(record: Dict[str, Any]) -> Tuple[int, float]:
    config = record.get("config") or {}
    render = record.get("render") or {}
    actor_count = int(render["visible_actor_count"])
    sample_fps = float(config["sample_fps"])
    if actor_count not in (1, 2):
        raise ValueError(
            f"Invalid visible_actor_count for sample {record['sample_index']}: {actor_count}"
        )
    if sample_fps <= 0:
        raise ValueError(f"Invalid sample_fps for sample {record['sample_index']}")
    return actor_count, sample_fps


def prepare_rendered_media(
    *,
    record: Dict[str, Any],
    prompt_template: str,
    args: argparse.Namespace,
    process_vision_info: Any,
) -> Tuple[List[Any], int, str, Any, Any, Any, Dict[str, Any]]:
    config = record.get("config") or {}
    expected_frames = config.get("num_frames")
    frames = load_gif_frames(record["_gif_path"], expected_frames)
    try:
        actor_count, sample_fps = render_settings(record)
        if sample_fps != args.sample_fps:
            raise ValueError(
                f"Mixed sample_fps in rendered inputs: {sample_fps} vs {args.sample_fps}"
            )
        base_prompt = render_prompt(prompt_template, actor_count)
        messages = make_messages(
            frames,
            base_prompt,
            sample_fps=sample_fps,
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
        return (
            frames,
            actor_count,
            base_prompt,
            images,
            videos,
            video_metadata,
            video_kwargs,
        )
    except Exception:
        close_frames(frames)
        raise


def make_caption_record(
    *,
    render_record: Dict[str, Any],
    actor_count: int,
    raw_text: str,
    caption: Optional[Dict[str, Any]],
    errors: Sequence[str],
    attempts: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    prompt_hash: str,
    preparation_seconds: float,
    generation_seconds: float,
) -> Dict[str, Any]:
    accepted = caption is not None and not errors
    sample_index = int(render_record["sample_index"])
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_id": f"train_{sample_index:06d}",
        "sample_index": sample_index,
        "source_split": "train",
        "labels_read": False,
        "model": args.model,
        "model_revision": args.resolved_revision,
        "quantization": args.quantization,
        "quantization_config": args.quantization_config,
        "inference": {
            "engine": "transformers_rendered_gif",
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
        "render_input": {
            "root": str(args.rendered_root.resolve()),
            "gif_path": str(render_record["_gif_path"]),
            "metadata_path": str(render_record["_metadata_path"]),
            "render_schema_version": render_record.get("schema_version"),
        },
        "render": render_record.get("render"),
        "actor_count": actor_count,
        "status": "accepted" if accepted else "invalid",
        "errors": list(errors),
        "retry_count": max(0, len(attempts) - 1),
        "preparation_seconds": round(preparation_seconds, 4),
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


def write_pipeline_error(
    *,
    handle: Any,
    render_record: Dict[str, Any],
    error: Exception,
    args: argparse.Namespace,
    prompt_hash: str,
) -> None:
    sample_index = int(render_record["sample_index"])
    write_jsonl_record(
        handle,
        {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "sample_id": f"train_{sample_index:06d}",
            "sample_index": sample_index,
            "source_split": "train",
            "labels_read": False,
            "model": args.model,
            "model_revision": args.resolved_revision,
            "inference": {
                "engine": "transformers_rendered_gif",
                "num_shards": args.num_shards,
                "shard_id": args.shard_id,
            },
            "prompt": {
                "path": str(args.prompt_path.resolve()),
                "sha256": prompt_hash,
            },
            "render_input": {"gif_path": str(render_record["_gif_path"])},
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
    record: Dict[str, Any],
    prompt_template: str,
    args: argparse.Namespace,
) -> None:
    frames: List[Any] = []
    try:
        actor_count, sample_fps = render_settings(record)
        frames = load_gif_frames(
            record["_gif_path"],
            (record.get("config") or {}).get("num_frames"),
        )
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "sample_index": record["sample_index"],
                    "gif_path": str(record["_gif_path"]),
                    "decoded_frames": len(frames),
                    "actor_count": actor_count,
                    "sample_fps": sample_fps,
                    "prompt": render_prompt(prompt_template, actor_count),
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

    print(f"Reading rendered manifests from {args.rendered_root}...", flush=True)
    rendered_samples = load_rendered_samples(args.rendered_root)
    selected = select_records(
        rendered_samples,
        start_index=args.start_index,
        max_samples=args.max_samples,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        accepted=accepted,
    )
    if not selected:
        print("No rendered samples remain for this shard.")
        return

    sample_fps_values = {render_settings(record)[1] for record in selected}
    if len(sample_fps_values) != 1:
        raise ValueError(f"Rendered inputs contain mixed sample_fps: {sample_fps_values}")
    args.sample_fps = sample_fps_values.pop()
    print(
        f"Shard {args.shard_id}/{args.num_shards}: {len(selected)} rendered samples, "
        f"sample_fps={args.sample_fps}.",
        flush=True,
    )
    if args.dry_run:
        run_dry_run(selected[0], prompt_template, args)
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
        f"Loading {args.model} once for rendered shard "
        f"{args.shard_id}/{args.num_shards}...",
        flush=True,
    )
    model_config = AutoConfig.from_pretrained(args.model, **config_kwargs)
    processor = AutoProcessor.from_pretrained(args.model, **processor_kwargs)
    model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kwargs)
    model.eval()
    args.quantization, args.quantization_config = quantization_metadata(model_config)
    args.resolved_revision = args.revision or getattr(model_config, "_commit_hash", None)
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
    accepted_now = 0
    invalid_now = 0
    pipeline_errors = 0
    run_started = time.perf_counter()

    with args.output_path.open(output_mode, encoding="utf-8") as output_handle:
        for ordinal, render_record in enumerate(selected, start=1):
            frames: List[Any] = []
            try:
                preparation_started = time.perf_counter()
                (
                    frames,
                    actor_count,
                    base_prompt,
                    images,
                    videos,
                    video_metadata,
                    video_kwargs,
                ) = prepare_rendered_media(
                    record=render_record,
                    prompt_template=prompt_template,
                    args=args,
                    process_vision_info=process_vision_info,
                )
                preparation_seconds = time.perf_counter() - preparation_started
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

                record = make_caption_record(
                    render_record=render_record,
                    actor_count=actor_count,
                    raw_text=raw_text,
                    caption=caption,
                    errors=errors,
                    attempts=attempts,
                    args=args,
                    prompt_hash=prompt_hash,
                    preparation_seconds=preparation_seconds,
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
                    render_record=render_record,
                    error=exc,
                    args=args,
                    prompt_hash=prompt_hash,
                )
            finally:
                close_frames(frames)

            elapsed = time.perf_counter() - run_started
            remaining = elapsed / ordinal * (len(selected) - ordinal)
            print(
                f"[caption shard {args.shard_id}] {ordinal}/{len(selected)} "
                f"sample={render_record['sample_index']} accepted={accepted_now} "
                f"invalid={invalid_now} pipeline_error={pipeline_errors} "
                f"eta={remaining / 60:.1f} min",
                flush=True,
            )

    print(
        f"Finished caption shard {args.shard_id}: accepted={accepted_now}, "
        f"invalid={invalid_now}, pipeline_error={pipeline_errors}, "
        f"output={args.output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
