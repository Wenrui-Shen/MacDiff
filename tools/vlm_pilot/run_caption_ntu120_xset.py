#!/usr/bin/env python3
"""Render NTU120 XSet before captions exist; bind reusable text at caption time."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path

from caption_output import sidecar_paths
from dataset_caption_pipeline import (PLAN_SCHEMA, PROJECT_ROOT, _run_dataset, add_common_arguments,
                                      atomic_json, cli_main, file_signature, load_captions, load_train,
                                      pipeline_identity, sha256_file, validate_common_arguments)
from run_caption_ntu120 import NTU60_SUBJECTS, NTU120_SUBJECTS, skeleton_identity, train_names

DATASET = "ntu120_xset"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.set_defaults(stage="render")
    parser.add_argument("--data_path", type=Path, default=Path("../data/MAMP/ntu120/NTU120_XSet.npz"))
    parser.add_argument("--ntu60_data_path", type=Path, default=Path("../data/MAMP/ntu/NTU60_XSub.npz"))
    parser.add_argument("--ntu60_xview_data_path", type=Path, default=Path("../data/MAMP/ntu/NTU60_XView.npz"))
    parser.add_argument("--ntu120_xsub_data_path", type=Path, default=Path("../data/MAMP/ntu120/NTU120_XSub.npz"))
    parser.add_argument("--ntu60_captions", type=Path, default=Path("vlm_pilot/ntu60_xsub_global_local_v2/captions.json"))
    parser.add_argument("--ntu60_xview_captions", type=Path, default=None)
    parser.add_argument("--ntu120_xsub_captions", type=Path, default=None)
    parser.add_argument("--ntu60_statistics_dir", type=Path, default=PROJECT_ROOT / "data/ntu/statistics")
    parser.add_argument("--ntu120_statistics_dir", type=Path, default=PROJECT_ROOT / "data/ntu120/statistics")
    args = parser.parse_args(argv)
    validate_common_arguments(parser, args)
    return args


def source_banks(args):
    return [
        {"dataset": "ntu60_xsub", "data_path": args.ntu60_data_path, "captions": args.ntu60_captions,
         "names": train_names(args.ntu60_statistics_dir, NTU60_SUBJECTS)},
        {"dataset": "ntu60_xview", "data_path": args.ntu60_xview_data_path,
         "captions": args.ntu60_xview_captions or args.output_root / "ntu60_xview/captions.json",
         "names": train_names(args.ntu60_statistics_dir, (2, 3), group="camera")},
        {"dataset": "ntu120_xsub", "data_path": args.ntu120_xsub_data_path,
         "captions": args.ntu120_xsub_captions or args.output_root / "ntu120_xsub/captions.json",
         "names": train_names(args.ntu120_statistics_dir, NTU120_SUBJECTS)},
    ]


def rendering_identity(args, banks, target_names):
    identity = pipeline_identity(args, DATASET, args.data_path)
    sources = {bank["dataset"]: {
        "data": file_signature(bank["data_path"]),
        # The location is fixed now; content is bound after the text exists.
        "captions": {"path": str(bank["captions"].resolve())},
        "statistics_sha256": hashlib.sha256("\n".join(bank["names"]).encode()).hexdigest(),
    } for bank in banks}
    identity["ntu60"] = sources.pop("ntu60_xsub")
    identity["reuse_sources"] = sources
    identity["caption_binding_policy"] = "deferred_until_caption_v1"
    identity["target_statistics_sha256"] = hashlib.sha256("\n".join(target_names).encode()).hexdigest()
    identity["entrypoint_sha256"] = sha256_file(Path(__file__))
    identity["reuse_implementation_sha256"] = sha256_file(Path(__file__).with_name("run_caption_ntu120.py"))
    return identity


def build_render_plan(args, banks, target_names):
    """Inspect skeletons only, including person slots needed for later text checks."""
    signatures, source_slots = {}, {}
    for bank in banks:
        data = load_train(bank["data_path"])
        if len(data) != len(bank["names"]):
            raise ValueError("%s x_train count differs from statistics" % bank["dataset"])
        print("Checking %s skeletons (captions not required)..." % bank["dataset"], flush=True)
        slots_by_row = []
        for index, name in enumerate(bank["names"]):
            digest, slots, _ = skeleton_identity(data[index])
            slots_by_row.append(slots)
            if name in signatures and signatures[name][1] != digest:
                raise ValueError("Conflicting coordinates across source datasets: %s" % name)
            signatures.setdefault(name, (index, digest, slots, bank["dataset"]))
        source_slots[bank["dataset"]] = slots_by_row
        del data
        gc.collect()
    target = load_train(args.data_path)
    if len(target) != len(target_names):
        raise ValueError("NTU120 XSet x_train count differs from statistics")
    reuse, generate = [], []
    for index, name in enumerate(target_names):
        if name not in signatures:
            generate.append(index)
            continue
        source_index, source_hash, source_people, source = signatures[name]
        digest, people, clone = skeleton_identity(target[index])
        if digest != source_hash:
            raise ValueError("Same skeleton ID has different coordinates: %s" % name)
        duplicate = source_people == 1 and people == 2 and clone
        if people != source_people and not duplicate:
            raise ValueError("Unsupported person-slot change: %s" % name)
        reuse.append({"sample_index": index, "source_dataset": source, "source_sample_index": source_index,
                      "skeleton_id": name, "skeleton_sha256": digest, "duplicate_person1": duplicate})
    return {"sample_count": len(target), "generate_indices": generate, "reuse": reuse,
            "source_person_slots": source_slots}, target


def read_plan(root, identity):
    path = root / "dataset_manifest.json"
    if not path.is_file():
        raise ValueError("Run this XSet entry with --stage render first")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("identity") != identity:
        raise ValueError("XSet plan identity differs; resume with its original entry and settings")
    return plan


def bind_source_captions(args, banks, plan, root, write=True):
    """Require complete aligned captions, then freeze their hashes before inference."""
    snapshot = {"schema_version": "macdiff.xset_caption_sources.v1", "model": args.model,
                "revision": args.revision, "prompt_sha256": sha256_file(args.prompt_path), "sources": {}}
    for bank in banks:
        path = bank["captions"]
        metadata_path = sidecar_paths(path)[0]
        if not path.is_file() or not metadata_path.is_file():
            raise ValueError("Complete %s captions before --stage caption: %s; saved XSet renders can be kept" %
                             (bank["dataset"], path))
        rows, _ = load_captions(path, args)
        slots = plan["source_person_slots"][bank["dataset"]]
        if set(rows) != set(range(len(slots))):
            raise ValueError("%s captions must cover every training row" % bank["dataset"])
        for index, people in enumerate(slots):
            if len(rows[index]["persons"]) != people:
                raise ValueError("%s captions/person slots differ at row %d" % (bank["dataset"], index))
        snapshot["sources"][bank["dataset"]] = {
            "captions": dict(file_signature(path), sha256=sha256_file(path)),
            "metadata": dict(file_signature(metadata_path), sha256=sha256_file(metadata_path)),
        }
    binding = root / "reuse_caption_sources.json"
    if binding.is_file():
        if json.loads(binding.read_text(encoding="utf-8")) != snapshot:
            raise ValueError("Reuse captions changed since XSet captioning began; restore the original source files")
    elif write:
        atomic_json(binding, snapshot)
    return snapshot


def publish_caption_binding(root, snapshot):
    path = root / "captions.metadata.json"
    if not path.is_file():
        return
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("reuse_caption_sources") != snapshot:
        metadata["reuse_caption_sources"] = snapshot
        atomic_json(path, metadata)


def main(argv=None):
    args = parse_args(argv)
    banks = source_banks(args)
    target_names = train_names(args.ntu120_statistics_dir, range(2, 33, 2), group="setup")
    identity = rendering_identity(args, banks, target_names)
    root = (args.output_root / DATASET).resolve()
    manifest = root / "dataset_manifest.json"
    factory = lambda: build_render_plan(args, banks, target_names)
    if args.dry_run:
        if args.stage == "caption":
            bind_source_captions(args, banks, read_plan(root, identity), root, write=False)
        return _run_dataset(args, root, manifest, args.data_path, identity, factory)
    from run_caption_dual_gpu import run_lock
    with run_lock(root):
        stages = ("render", "caption") if args.stage == "all" else (args.stage,)
        for stage in stages:
            phase = copy.copy(args)
            phase.stage = stage
            snapshot = None
            if stage == "caption":
                snapshot = bind_source_captions(args, banks, read_plan(root, identity), root)
            code = _run_dataset(phase, root, manifest, args.data_path, identity, factory)
            if snapshot is not None:
                # Generation can take hours; recheck the source files before
                # publishing a result bound to the original text snapshot.
                bind_source_captions(args, banks, read_plan(root, identity), root)
                publish_caption_binding(root, snapshot)
            if code:
                return code
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main(main))
