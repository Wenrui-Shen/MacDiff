#!/usr/bin/env python3
"""Render ``x_train`` skeleton samples into persistent GIF inputs.

This is stage 1 of the split rendering/captioning pipeline.  It never reads
labels and never loads a VLM.  Each completed sample has one animated GIF and
one metadata JSON file; metadata is written last so ``--resume`` can identify
complete outputs safely.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from render_skeleton_sample import load_font, render_sample_frames


SCHEMA_VERSION = "macdiff.skeleton_render.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render persistent label-free GIFs for the complete x_train split."
    )
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
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
    parser.add_argument(
        "--temporal_smooth", choices=("none", "savgol"), default="savgol"
    )
    parser.add_argument("--median_window", type=int, default=3)
    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument("--smooth_polyorder", type=int, default=2)
    parser.add_argument("--max_interp_gap", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.data_path.is_file():
        raise FileNotFoundError(args.data_path)
    if args.start_index < 0:
        raise ValueError("--start_index must be non-negative")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max_samples must be positive")
    if args.num_shards <= 0:
        raise ValueError("--num_shards must be positive")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard_id must be in [0, --num_shards)")
    if args.num_frames < 2:
        raise ValueError("--num_frames must be at least 2")
    if args.sample_fps <= 0:
        raise ValueError("--sample_fps must be positive")
    if args.width <= 0 or args.height <= 0 or args.width % 2:
        raise ValueError("--width/--height must be positive and --width divisible by 2")
    for name in ("median_window", "smooth_window"):
        value = getattr(args, name)
        if value < 1 or value % 2 == 0:
            raise ValueError(f"--{name} must be a positive odd integer")
    if args.smooth_polyorder < 0 or args.smooth_polyorder >= args.smooth_window:
        raise ValueError("--smooth_polyorder must be in [0, --smooth_window)")
    if args.max_interp_gap < 0:
        raise ValueError("--max_interp_gap must be non-negative")


def select_indices(
    total_samples: int,
    *,
    start_index: int,
    max_samples: Optional[int],
    num_shards: int,
    shard_id: int,
) -> List[int]:
    stop_index = total_samples
    if max_samples is not None:
        stop_index = min(stop_index, start_index + max_samples)
    if start_index >= stop_index:
        return []
    return [
        index
        for index in range(start_index, stop_index)
        if index % num_shards == shard_id
    ]


def source_signature(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def expected_config(args: argparse.Namespace, signature: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source": signature,
        "split": "train",
        "layout": ["front_xy_root_centered", "side_zy_root_centered"],
        "num_frames": args.num_frames,
        "sample_fps": args.sample_fps,
        "width": args.width,
        "height": args.height,
        "temporal_smooth": args.temporal_smooth,
        "median_window": args.median_window,
        "smooth_window": args.smooth_window,
        "smooth_polyorder": args.smooth_polyorder,
        "max_interp_gap": args.max_interp_gap,
    }


def completed_sample(
    sample_dir: Path,
    *,
    sample_index: int,
    config: Dict[str, Any],
) -> bool:
    gif_path = sample_dir / "preview.gif"
    metadata_path = sample_dir / "render_metadata.json"
    if not gif_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("schema_version") == SCHEMA_VERSION
        and metadata.get("status") == "rendered"
        and metadata.get("sample_index") == sample_index
        and metadata.get("config") == config
        and gif_path.stat().st_size > 0
    )


def close_frames(frames: Sequence[Any]) -> None:
    for frame in frames:
        close = getattr(frame, "close", None)
        if callable(close):
            close()


def save_gif(frames: Sequence[Any], path: Path, sample_fps: float) -> None:
    if not frames:
        raise ValueError("Cannot save an empty animation")
    duration_ms = max(1, round(1000.0 / sample_fps))
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frames[0].save(
            temporary_path,
            format="GIF",
            save_all=True,
            append_images=list(frames[1:]),
            duration=duration_ms,
            loop=0,
            optimize=False,
            disposal=2,
        )
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_json_atomic(path: Path, value: Dict[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary_path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_manifest_line(handle: Any, record: Dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def make_metadata(
    *,
    sample_index: int,
    sample_dir: Path,
    config: Dict[str, Any],
    render_info: Dict[str, Any],
    render_seconds: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "rendered",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_id": f"train_{sample_index:06d}",
        "sample_index": sample_index,
        "source_split": "train",
        "labels_read": False,
        "config": config,
        "render": {
            "layout": [
                "front_xy_root_centered",
                "side_zy_root_centered",
            ],
            "root_centering": "primary actor NTU joint 2 at every frame",
            "global_translation_available": False,
            "person_colors": {"0": "red", "1": "blue"},
            **render_info,
        },
        "gif_path": str((sample_dir / "preview.gif").resolve()),
        "gif_duration_ms": max(1, round(1000.0 / args.sample_fps)),
        "render_seconds": round(render_seconds, 4),
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    signature = source_signature(args.data_path)
    config = expected_config(args, signature)

    print(f"Loading x_train from {args.data_path}...", flush=True)
    with np.load(args.data_path, mmap_mode="r", allow_pickle=False) as archive:
        if "x_train" not in archive.files:
            raise KeyError(f"x_train not found in {args.data_path}; keys={archive.files}")
        x_train = archive["x_train"]
        indices = select_indices(
            int(x_train.shape[0]),
            start_index=args.start_index,
            max_samples=args.max_samples,
            num_shards=args.num_shards,
            shard_id=args.shard_id,
        )
        print(
            f"Shard {args.shard_id}/{args.num_shards}: {len(indices)} selected from "
            f"{x_train.shape[0]} train samples.",
            flush=True,
        )
        if args.dry_run:
            print(json.dumps({"config": config, "first_indices": indices[:10]}, indent=2))
            return
        if not indices:
            return

        args.output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = args.output_root / f"render_manifest_shard{args.shard_id}.jsonl"
        if manifest_path.exists() and not args.resume:
            raise FileExistsError(
                f"Manifest already exists: {manifest_path}. Pass --resume or use a new root."
            )
        manifest_mode = "a" if args.resume else "w"
        font = load_font(18)
        rendered = 0
        skipped = 0
        failed = 0
        started = time.perf_counter()

        with manifest_path.open(manifest_mode, encoding="utf-8") as manifest:
            for ordinal, sample_index in enumerate(indices, start=1):
                sample_dir = args.output_root / f"train_{sample_index:06d}"
                if args.resume and completed_sample(
                    sample_dir,
                    sample_index=sample_index,
                    config=config,
                ):
                    skipped += 1
                    write_manifest_line(
                        manifest,
                        json.loads(
                            (sample_dir / "render_metadata.json").read_text(
                                encoding="utf-8"
                            )
                        ),
                    )
                else:
                    if sample_dir.exists() and not args.resume:
                        raise FileExistsError(
                            f"Sample output already exists: {sample_dir}. Pass --resume."
                        )
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    frames: List[Any] = []
                    sample_started = time.perf_counter()
                    try:
                        frames, render_info = render_sample_frames(
                            x_train[sample_index],
                            num_frames=args.num_frames,
                            width=args.width,
                            height=args.height,
                            font=font,
                            temporal_smooth=args.temporal_smooth,
                            median_window=args.median_window,
                            smooth_window=args.smooth_window,
                            smooth_polyorder=args.smooth_polyorder,
                            max_interp_gap=args.max_interp_gap,
                        )
                        save_gif(frames, sample_dir / "preview.gif", args.sample_fps)
                        metadata = make_metadata(
                            sample_index=sample_index,
                            sample_dir=sample_dir,
                            config=config,
                            render_info=render_info,
                            render_seconds=time.perf_counter() - sample_started,
                            args=args,
                        )
                        write_json_atomic(sample_dir / "render_metadata.json", metadata)
                        write_manifest_line(manifest, metadata)
                        rendered += 1
                    except Exception as exc:
                        failed += 1
                        write_manifest_line(
                            manifest,
                            {
                                "schema_version": SCHEMA_VERSION,
                                "status": "render_error",
                                "sample_id": f"train_{sample_index:06d}",
                                "sample_index": sample_index,
                                "labels_read": False,
                                "errors": [f"{type(exc).__name__}: {exc}"],
                            },
                        )
                    finally:
                        close_frames(frames)

                elapsed = time.perf_counter() - started
                remaining = elapsed / ordinal * (len(indices) - ordinal)
                print(
                    f"[render shard {args.shard_id}] {ordinal}/{len(indices)} "
                    f"sample={sample_index} rendered={rendered} skipped={skipped} "
                    f"failed={failed} eta={remaining / 60:.1f} min",
                    flush=True,
                )

    print(
        f"Finished render shard {args.shard_id}: rendered={rendered}, "
        f"skipped={skipped}, failed={failed}, root={args.output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
