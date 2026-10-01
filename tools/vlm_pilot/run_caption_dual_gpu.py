#!/usr/bin/env python3
"""Run independent GPU caption workers, resume automatically, then merge by index."""
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from caption_output import (OUTPUT_SCHEMA, caption_schema, iter_caption_records,
                            load_accepted_indices, read_caption_metadata,
                            repair_incomplete_jsonl_tail, sidecar_paths)
from caption_qwen3vl_rendered_train_transformers import load_rendered_samples, select_records
from caption_qwen3vl_train_transformers import DEFAULT_PROMPT, sha256_file

SCRIPT_DIR = Path(__file__).resolve().parent
WORKER = SCRIPT_DIR / "caption_qwen3vl_rendered_train_transformers.py"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rendered_root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--prompt_path", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--gpus", default="0,1", help="One independent worker per GPU; default 0,1.")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--expected_samples", type=int, default=None,
                        help="Check complete render coverage; NTU60 XSub train has 40091 samples.")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--attn_implementation", choices=("eager", "sdpa", "flash_attention_2"), default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--merge_only", action="store_true", help="Merge saved results without loading models.")
    parser.add_argument("--dry_run", action="store_true", help="Show pending counts and worker commands only.")
    args = parser.parse_args(argv)
    args.gpu_ids = [value.strip() for value in args.gpus.split(",")]
    if (not all(args.gpu_ids) or len(set(args.gpu_ids)) != len(args.gpu_ids)
            or any(not value.isdigit() for value in args.gpu_ids)):
        parser.error("--gpus must contain distinct non-negative GPU indices")
    if args.start_index < 0 or (args.max_samples is not None and args.max_samples <= 0):
        parser.error("--start_index must be non-negative and --max_samples positive")
    if args.expected_samples is not None and args.expected_samples <= 0:
        parser.error("--expected_samples must be positive")
    if args.max_new_tokens <= 0 or args.max_retries < 0:
        parser.error("--max_new_tokens must be positive and --max_retries non-negative")
    if not args.rendered_root.is_dir() or not args.prompt_path.is_file():
        parser.error("--rendered_root must be a directory and --prompt_path an existing file")
    args.rendered_root = args.rendered_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.prompt_path = args.prompt_path.resolve()
    return args


@contextmanager
def run_lock(directory):
    """OS-released lock: a stopped run never leaves a stale lock that blocks resume."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".caption.lock"
    with path.open("a+b") as handle:
        if not path.stat().st_size:
            handle.write(b" ")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another caption launcher is using %s" % directory) from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def shard_path(args, shard_id):
    return args.output_dir / ("shard%d.jsonl" % shard_id)


def worker_command(args, shard_id):
    command = [sys.executable, "-u", str(WORKER),
               "--rendered_root", str(args.rendered_root), "--model", args.model,
               "--prompt_path", str(args.prompt_path), "--output_path", str(shard_path(args, shard_id)),
               "--num_shards", str(len(args.gpu_ids)), "--shard_id", str(shard_id),
               "--start_index", str(args.start_index), "--max_new_tokens", str(args.max_new_tokens),
               "--max_retries", str(args.max_retries), "--dtype", args.dtype, "--resume"]
    for flag in ("revision", "max_samples", "attn_implementation"):
        value = getattr(args, flag)
        if value is not None:
            command.extend(["--" + flag, str(value)])
    if args.trust_remote_code:
        command.append("--trust_remote_code")
    return command


def expected_indices(args, renders):
    if args.expected_samples is not None:
        wanted = set(range(args.expected_samples))
        missing = sorted(wanted - set(renders))
        extra = sorted(set(renders) - wanted)
        if missing or extra:
            raise ValueError("Render coverage differs from --expected_samples: missing=%s, extra=%s" %
                             (missing[:20], extra[:20]))
    selected = select_records(renders, start_index=args.start_index, max_samples=args.max_samples,
                              num_shards=1, shard_id=0, accepted=[])
    if not selected:
        raise ValueError("No rendered samples in the requested range")
    return {record["sample_index"] for record in selected}


def saved_indices(args, shard_id, prompt_hash, recover=False):
    path = shard_path(args, shard_id)
    if not path.exists():
        # Metadata can have been saved just before interruption, before the output opened.
        if sidecar_paths(path)[0].exists():
            read_caption_metadata(path, expected_model=args.model, expected_prompt_hash=prompt_hash,
                                  expected_revision=args.revision)
        return set()
    metadata = read_caption_metadata(path, expected_model=args.model, expected_prompt_hash=prompt_hash,
                                     expected_revision=args.revision)
    if metadata.get("caption_schema") != caption_schema(args.prompt_path.read_text(encoding="utf-8")):
        raise ValueError("Saved caption schema differs; use a new --output_dir")
    for session in metadata.get("runs", []):
        settings = session.get("settings", {})
        if (settings.get("num_shards") != len(args.gpu_ids) or settings.get("shard_id") != shard_id
                or settings.get("rendered_root") != str(args.rendered_root)):
            raise ValueError("Saved render root or sharding differs; use the original settings or a new --output_dir")
    if recover:
        for candidate in (path, sidecar_paths(path)[1]):
            backup = repair_incomplete_jsonl_tail(candidate)
            if backup:
                print("Recovered interrupted final line of %s; backup=%s" % (candidate, backup), flush=True)
    indices = load_accepted_indices(path, expected_model=args.model, expected_prompt_hash=prompt_hash,
                                    expected_revision=args.revision)
    if any(index % len(args.gpu_ids) != shard_id for index in indices):
        raise ValueError("Sample assigned to the wrong shard: %s" % path)
    return indices


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def merge_results(args, renders, wanted, prompt_hash):
    rows = {}
    sources = []
    for shard_id in range(len(args.gpu_ids)):
        path = shard_path(args, shard_id)
        saved_indices(args, shard_id, prompt_hash)
        if not path.exists():
            continue
        metadata_path, diagnostics = sidecar_paths(path)
        sources.append({"shard_id": shard_id, "result": str(path),
                        "metadata": json.loads(metadata_path.read_text(encoding="utf-8")),
                        "diagnostics": str(diagnostics)})
        for row in iter_caption_records(path):
            index = row["sample_index"]
            if index not in renders:
                raise ValueError("Caption sample_index not found in rendered input: %d" % index)
            if index not in wanted:
                continue
            if index in rows:
                raise ValueError("Duplicate sample_index across shards: %d" % index)
            expected_actors = (renders[index].get("render") or {}).get("visible_actor_count")
            if expected_actors is not None and len(row["persons"]) != expected_actors:
                raise ValueError("Caption actor count differs from render metadata for sample %d" % index)
            rows[index] = row
    output = args.output_dir / "captions.json"
    temporary = output.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        sink.write("[\n")
        ordered = sorted(rows)
        for ordinal, index in enumerate(ordered):
            sink.write(json.dumps(rows[index], ensure_ascii=False))
            sink.write(",\n" if ordinal + 1 < len(ordered) else "\n")
        sink.write("]\n")
        sink.flush()
        os.fsync(sink.fileno())
    temporary.replace(output)
    missing = sorted(wanted - set(rows))
    summary = {"expected": len(wanted), "accepted": len(rows), "missing_count": len(missing),
               "missing_indices": missing, "complete": not missing}
    atomic_json(sidecar_paths(output)[0], {
        "schema_version": OUTPUT_SCHEMA, "output_format": "json_array", "model": args.model,
        "requested_revision": args.revision,
        "caption_schema": caption_schema(args.prompt_path.read_text(encoding="utf-8")),
        "prompt": {"path": str(args.prompt_path), "sha256": prompt_hash},
        "merged_at_utc": datetime.now(timezone.utc).isoformat(),
        "rendered_root": str(args.rendered_root), "summary": summary, "sources": sources,
    })
    print("Merged %d/%d samples in index order: %s" % (len(rows), len(wanted), output), flush=True)
    if missing:
        print("Missing %d samples (first 20: %s). Run the same command again to retry them." %
              (len(missing), missing[:20]), flush=True)
    return summary


def stop_workers(jobs):
    for process, log, shard_id in jobs:
        if process.poll() is None:
            try:
                if os.name == "nt":
                    process.terminate()
                else:
                    os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process, log, shard_id in jobs:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                process.kill()
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()


def last_log_line(path):
    if not path.exists():
        return ""
    with path.open("rb") as source:
        source.seek(max(0, path.stat().st_size - 4096))
        lines = source.read().decode("utf-8", errors="replace").splitlines()
    return lines[-1] if lines else ""


@contextmanager
def termination_signals():
    """Route termination/hangup through the same child cleanup as Ctrl+C."""
    handlers = {}
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    try:
        for name in ("SIGTERM", "SIGHUP"):
            sig = getattr(signal, name, None)
            if sig is not None:
                handlers[sig] = signal.signal(sig, interrupted)
        yield
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def run_workers(args, pending):
    with termination_signals():
        return _run_workers(args, pending)


def _run_workers(args, pending):
    jobs = []
    interrupted = False
    last_lines = {}
    try:
        for shard_id, count in enumerate(pending):
            if not count:
                continue
            gpu = args.gpu_ids[shard_id]
            log_path = args.output_dir / ("shard%d.log" % shard_id)
            log = log_path.open("a", encoding="utf-8")
            log.write("\n=== Run %s; CUDA_VISIBLE_DEVICES=%s ===\n" %
                      (datetime.now(timezone.utc).isoformat(), gpu))
            log.flush()
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment.setdefault("OMP_NUM_THREADS", "1")
            environment["PYTHONUNBUFFERED"] = "1"
            options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
            try:
                process = subprocess.Popen(worker_command(args, shard_id), env=environment,
                                           stdout=log, stderr=subprocess.STDOUT, **options)
            except BaseException:
                log.close()
                raise
            jobs.append((process, log, shard_id))
            print("GPU %s: shard %d, %d pending samples; log=%s" % (gpu, shard_id, count, log_path), flush=True)
        while any(process.poll() is None for process, _, _ in jobs):
            for process, log, shard_id in jobs:
                line = last_log_line(args.output_dir / ("shard%d.log" % shard_id))
                if line and line != last_lines.get(shard_id):
                    print("[GPU %s] %s" % (args.gpu_ids[shard_id], line), flush=True)
                    last_lines[shard_id] = line
            time.sleep(1)
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted. Stopping workers and saving a sorted partial result...", flush=True)
        stop_workers(jobs)
    except BaseException:
        stop_workers(jobs)
        raise
    finally:
        for _, log, _ in jobs:
            log.close()
    failures = {shard_id: process.returncode for process, _, shard_id in jobs if process.returncode}
    for shard_id, code in failures.items():
        print("Shard %d exited with code %s; see %s" %
              (shard_id, code, args.output_dir / ("shard%d.log" % shard_id)), flush=True)
    return interrupted, failures


def main(argv=None):
    args = parse_args(argv)
    prompt_hash = sha256_file(args.prompt_path)
    renders = load_rendered_samples(args.rendered_root)
    wanted = expected_indices(args, renders)
    if args.dry_run:
        for shard_id, gpu in enumerate(args.gpu_ids):
            saved = saved_indices(args, shard_id, prompt_hash)
            pending = sum(index % len(args.gpu_ids) == shard_id and index not in saved for index in wanted)
            print("GPU %s: %d pending; CUDA_VISIBLE_DEVICES=%s %s" %
                  (gpu, pending, gpu, shlex.join(worker_command(args, shard_id))))
        return 0
    with run_lock(args.output_dir):
        pending = []
        for shard_id in range(len(args.gpu_ids)):
            saved = saved_indices(args, shard_id, prompt_hash, recover=True)
            if saved - set(renders):
                raise ValueError("Saved sample indices are absent from the rendered input; use the original input.")
            pending.append(sum(index % len(args.gpu_ids) == shard_id and index not in saved for index in wanted))
        print("Automatic resume: %d/%d requested samples already saved." %
              (len(wanted) - sum(pending), len(wanted)), flush=True)
        interrupted, failures = (False, {}) if args.merge_only else run_workers(args, pending)
        # A worker can be killed mid-write: recover its incomplete tail before merging.
        for shard_id in range(len(args.gpu_ids)):
            saved_indices(args, shard_id, prompt_hash, recover=True)
        summary = merge_results(args, renders, wanted, prompt_hash)
        return 130 if interrupted else (1 if failures or not summary["complete"] else 0)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as exc:
        print("Caption launcher failed: %s" % exc, file=sys.stderr)
        raise SystemExit(1)
