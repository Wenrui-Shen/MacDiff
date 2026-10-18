#!/usr/bin/env python3
"""Shared dataset entry-point plumbing for the existing NTU60 GIF/VLM pipeline."""
import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from caption_output import (OUTPUT_SCHEMA, iter_caption_records, load_accepted_indices,
                            read_caption_metadata, sidecar_paths)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MODEL = "/home/user9/public3/swr/models/Qwen3-VL-8B-Instruct"
PLAN_SCHEMA = "macdiff.dataset_caption_plan.v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def add_common_arguments(parser):
    parser.add_argument("--output_root", type=Path, default=Path("vlm_pilot"))
    parser.add_argument("--stage", choices=("all", "render", "caption"), default="all",
                        help="all runs render then caption; CLIP encoding is a separate cache_clip_motion_text.py step.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt_path", type=Path, default=SCRIPT_DIR / "skeleton_motion_prompt_v2.txt")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit original dataset rows, before NTU60 reuse and GPU sharding.")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--attn_implementation", choices=("eager", "sdpa", "flash_attention_2"))
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--merge_only", action="store_true",
                        help="Merge saved/reused captions; implies --stage caption.")
    parser.add_argument("--dry_run", action="store_true", help="Validate inputs and show counts without writing or loading a VLM.")
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--sample_fps", type=float, default=8.0)
    parser.add_argument("--temporal_smooth", choices=("none", "savgol"), default="savgol")
    parser.add_argument("--median_window", type=int, default=3)
    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument("--smooth_polyorder", type=int, default=2)
    parser.add_argument("--max_interp_gap", type=int, default=2)


def validate_common_arguments(parser, args):
    gpu_ids = args.gpus.split(",")
    if (any(not item.strip().isdigit() for item in gpu_ids)
            or len(set(item.strip() for item in gpu_ids)) != len(gpu_ids)):
        parser.error("--gpus must contain distinct non-negative GPU indices")
    if args.start_index < 0 or (args.max_samples is not None and args.max_samples <= 0):
        parser.error("--start_index must be non-negative and --max_samples positive")
    if args.max_new_tokens <= 0 or args.max_retries < 0:
        parser.error("Invalid token/retry limit")
    if args.num_frames < 2 or args.sample_fps <= 0:
        parser.error("--num_frames must be >= 2 and --sample_fps positive")
    for name in ("median_window", "smooth_window"):
        if getattr(args, name) < 1 or getattr(args, name) % 2 == 0:
            parser.error("--%s must be a positive odd integer" % name)
    if not 0 <= args.smooth_polyorder < args.smooth_window or args.max_interp_gap < 0:
        parser.error("Invalid smoothing settings")
    if not args.prompt_path.is_file():
        parser.error("Prompt file does not exist: %s" % args.prompt_path)
    if args.merge_only:
        args.stage = "caption"


def file_signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def render_options(args, data_path):
    return SimpleNamespace(data_path=Path(data_path), width=960, height=360, num_shards=1, shard_id=0,
                           **{key: getattr(args, key) for key in
                              ("num_frames", "sample_fps", "temporal_smooth", "median_window",
                               "smooth_window", "smooth_polyorder", "max_interp_gap")})


def pipeline_identity(args, dataset, data_path):
    from caption_output import caption_schema
    if caption_schema(args.prompt_path.read_text(encoding="utf-8")) != "global_local":
        raise ValueError("Use the NTU60 global/local prompt (skeleton_motion_prompt_v2.txt)")
    return {"dataset": dataset, "source": file_signature(data_path), "split": "train",
            "labels_read": False, "model": args.model, "revision": args.revision,
            "prompt_sha256": sha256_file(args.prompt_path),
            "render": {key: value for key, value in vars(render_options(args, data_path)).items()
                       if key not in ("data_path", "num_shards", "shard_id")},
            "implementation": {name: sha256_file(SCRIPT_DIR / name) for name in
                               ("dataset_caption_pipeline.py", "render_skeleton_sample.py",
                                "render_qwen3vl_train.py")}}


def load_train(path):
    import numpy as np
    print("Loading x_train only: %s" % path, flush=True)
    with np.load(path, allow_pickle=False) as archive:
        data = archive["x_train"]
    if data.ndim != 3 or data.shape[2] != 150 or not len(data):
        raise ValueError("Expected nonempty x_train [N,T,150]: %s" % path)
    return data


def load_captions(path, args):
    metadata = read_caption_metadata(path, expected_model=args.model,
                                     expected_prompt_hash=sha256_file(args.prompt_path),
                                     expected_revision=args.revision)
    if metadata.get("caption_schema") != "global_local":
        raise ValueError("NTU60 source must contain global/local captions")
    load_accepted_indices(path, expected_model=args.model,
                          expected_prompt_hash=sha256_file(args.prompt_path), expected_revision=args.revision)
    return {row["sample_index"]: row for row in iter_caption_records(path)}, metadata


def selected_indices(count, args):
    stop = count if args.max_samples is None else min(count, args.start_index + args.max_samples)
    indices = set(range(args.start_index, stop))
    if not indices:
        raise ValueError("No training rows in the requested range")
    return indices


def render_selected(data, indices, args, data_path, root):
    from render_qwen3vl_train import (close_frames, completed_sample, expected_config,
                                     make_metadata, save_gif, source_signature, write_json_atomic)
    from render_skeleton_sample import load_font, render_sample_frames
    options = render_options(args, data_path)
    config = expected_config(options, source_signature(Path(data_path)))
    font = None
    for ordinal, index in enumerate(sorted(indices), 1):
        sample_dir = root / ("train_%06d" % index)
        if completed_sample(sample_dir, sample_index=index, config=config):
            continue
        metadata_path = sample_dir / "render_metadata.json"
        if metadata_path.exists():
            old = json.loads(metadata_path.read_text(encoding="utf-8"))
            if old.get("config") != config or old.get("sample_index") != index:
                raise ValueError("Existing render identity differs; use a new --output_root")
        sample_dir.mkdir(parents=True, exist_ok=True)
        font = load_font(18) if font is None else font
        frames = []
        started = time.perf_counter()
        try:
            frames, info = render_sample_frames(data[index], font=font,
                                                **{key: value for key, value in vars(options).items()
                                                   if key not in ("data_path", "num_shards", "shard_id", "sample_fps")})
            save_gif(frames, sample_dir / "preview.gif", args.sample_fps)
            metadata = make_metadata(sample_index=index, sample_dir=sample_dir, config=config,
                                     render_info=info, render_seconds=time.perf_counter() - started, args=options)
            write_json_atomic(metadata_path, metadata)
        finally:
            close_frames(frames)
        if ordinal == 1 or ordinal % 100 == 0 or ordinal == len(indices):
            print("Rendered %d/%d requested new samples; index=%d" % (ordinal, len(indices), index), flush=True)


def launcher_arguments(args, root):
    argv = ["--rendered_root", str(root / "rendered"), "--output_dir", str(root / "generated"),
            "--model", args.model, "--prompt_path", str(args.prompt_path), "--gpus", args.gpus,
            "--start_index", str(args.start_index), "--max_new_tokens", str(args.max_new_tokens),
            "--max_retries", str(args.max_retries), "--dtype", args.dtype]
    for name in ("revision", "max_samples", "attn_implementation"):
        if getattr(args, name) is not None:
            argv.extend(["--" + name, str(getattr(args, name))])
    for name in ("trust_remote_code", "merge_only"):
        if getattr(args, name):
            argv.append("--" + name)
    return argv


def merge_dataset(args, root, plan, wanted):
    from copy import deepcopy
    rows = {}
    reuse = plan.get("reuse", [])
    source_metadata = {}
    if reuse:
        banks = dict(plan["identity"].get("reuse_sources", {}), ntu60_xsub=plan["identity"]["ntu60"])
        source_rows = {}
        for source in {item.get("source_dataset", "ntu60_xsub") for item in reuse if item["sample_index"] in wanted}:
            source_rows[source], source_metadata[source] = load_captions(Path(banks[source]["captions"]["path"]), args)
        for item in reuse:
            index = item["sample_index"]
            if index not in wanted:
                continue
            source = item.get("source_dataset", "ntu60_xsub")
            source_index = item["source_sample_index"] if "source_sample_index" in item else item["ntu60_sample_index"]
            persons = deepcopy(source_rows[source][source_index]["persons"])
            if item["duplicate_person1"]:
                persons.append(dict(deepcopy(persons[0]), person_index=1))
            rows[index] = {"sample_index": index, "persons": persons}
    generated_path = root / "generated" / "captions.json"
    generated_metadata = None
    if generated_path.exists():
        generated, generated_metadata = load_captions(generated_path, args)
        allowed = set(plan["generate_indices"])
        if set(generated) - allowed:
            raise ValueError("Generated captions contain indices outside this dataset's generation plan")
        rows.update({index: row for index, row in generated.items() if index in wanted})
    missing = sorted(wanted - set(rows))
    output = root / "captions.json"
    temporary = output.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        sink.write("[\n")
        ordered = sorted(rows)
        for ordinal, index in enumerate(ordered):
            sink.write(json.dumps(rows[index], ensure_ascii=False))
            sink.write(",\n" if ordinal + 1 < len(ordered) else "\n")
        sink.write("]\n")
    summary = {"expected": len(wanted), "accepted": len(rows), "missing_count": len(missing),
               "missing_indices": missing, "complete": not missing,
               "dataset_samples": plan["sample_count"],
               "dataset_complete": not missing and len(wanted) == plan["sample_count"],
               "reused": sum(item["sample_index"] in wanted for item in reuse)}
    metadata_path = sidecar_paths(output)[0]
    metadata = {
        "schema_version": OUTPUT_SCHEMA, "output_format": "json_array", "caption_schema": "global_local",
        "model": args.model, "requested_revision": args.revision,
        "prompt": {"path": str(args.prompt_path.resolve()), "sha256": sha256_file(args.prompt_path)},
        "merged_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary, "dataset_plan": str((root / "dataset_manifest.json").resolve()),
        "identity": plan["identity"], "sources": {"reused": source_metadata, "generated": generated_metadata}}
    # CLIP caches fingerprint both text and metadata. A no-op caption resume must
    # leave both untouched, even when the lower-level merge refreshed its timestamp.
    unchanged = False
    if output.is_file() and metadata_path.is_file():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        unchanged = (sha256_file(output) == sha256_file(temporary)
                     and all(previous.get(key) == metadata[key] for key in
                             ("schema_version", "caption_schema", "model", "requested_revision",
                              "prompt", "identity", "summary")))
    if unchanged:
        temporary.unlink()
    else:
        temporary.replace(output)
        atomic_json(metadata_path, metadata)
    print("Merged %d/%d requested rows (%d reused): %s" %
          (len(rows), len(wanted), summary["reused"], output), flush=True)
    if missing:
        print("Missing indices (first 20): %s; rerun the same command." % missing[:20], flush=True)
    return 1 if missing else 0


def run_dataset(args, dataset, data_path, identity, build_plan):
    """build_plan returns (plan, x_train), with no writes or model loading."""
    root = (args.output_root / dataset).resolve()
    manifest_path = root / "dataset_manifest.json"
    if args.dry_run:
        return _run_dataset(args, root, manifest_path, data_path, identity, build_plan)
    from run_caption_dual_gpu import run_lock
    with run_lock(root):
        return _run_dataset(args, root, manifest_path, data_path, identity, build_plan)


def _run_dataset(args, root, manifest_path, data_path, identity, build_plan):
    data = None
    if manifest_path.exists():
        plan = json.loads(manifest_path.read_text(encoding="utf-8"))
        if plan.get("schema_version") != PLAN_SCHEMA or plan.get("identity") != identity:
            raise ValueError("Dataset, source captions, model, prompt or render settings changed; use a new --output_root")
    elif args.stage == "caption":
        raise ValueError("Run --stage render first; dataset_manifest.json is missing")
    else:
        plan, data = build_plan()
        plan.update(schema_version=PLAN_SCHEMA, identity=identity)
        if not args.dry_run:
            # Refuse to adopt old output files without the dataset plan that identifies them.
            if root.exists() and any(path.name != ".caption.lock" for path in root.iterdir()):
                raise ValueError("Output directory is nonempty without dataset_manifest.json")
            atomic_json(manifest_path, plan)
    wanted = selected_indices(plan["sample_count"], args)
    new_indices = wanted & set(plan["generate_indices"])
    print("%s: total=%d, requested=%d, reuse=%d, generate=%d" %
          (identity["dataset"], plan["sample_count"], len(wanted), len(wanted) - len(new_indices), len(new_indices)), flush=True)
    if args.dry_run:
        return 0
    if args.stage in ("all", "render") and new_indices:
        if data is None:
            data = load_train(data_path)
        if len(data) != plan["sample_count"]:
            raise ValueError("Training row count changed")
        render_selected(data, new_indices, args, data_path, root / "rendered")
    del data
    gc.collect()
    if args.stage == "render":
        return 0
    code = 0
    if new_indices:
        from caption_qwen3vl_rendered_train_transformers import load_rendered_samples
        from run_caption_dual_gpu import main as launch
        renders = load_rendered_samples(root / "rendered")
        if new_indices - set(renders):
            raise ValueError("Missing requested rendered samples; run --stage render for this range first")
        code = launch(launcher_arguments(args, root))
    merged = merge_dataset(args, root, plan, wanted)
    return code or merged


def cli_main(main):
    try:
        return main()
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print("Dataset caption pipeline failed: %s" % exc, file=sys.stderr)
        return 1
