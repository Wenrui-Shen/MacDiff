#!/usr/bin/env python3
"""Generate NTU120 XSub/XSet train captions, reusing completed NTU descriptions."""
import argparse
import gc
import hashlib
import re
from pathlib import Path

from dataset_caption_pipeline import (PROJECT_ROOT, add_common_arguments, cli_main, file_signature,
                                      load_captions, load_train, pipeline_identity, run_dataset,
                                      sha256_file, validate_common_arguments)
from caption_output import sidecar_paths

NTU60_SUBJECTS = (1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38)
NTU120_SUBJECTS = NTU60_SUBJECTS + (45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59,
                                 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93,
                                 94, 95, 97, 98, 100, 103)


def train_names(statistics, subjects, *, group="performer"):
    """Reproduce subject-, camera-, or setup-major ordering from get_indices."""
    if group not in ("performer", "camera", "setup"):
        raise ValueError("Unsupported NTU split grouping: %s" % group)
    names = (statistics / "skes_available_name.txt").read_text(encoding="utf-8").split()
    values = [int(value) for value in (statistics / (group + ".txt")).read_text().split()]
    if len(names) != len(values) or len(set(names)) != len(names):
        raise ValueError("Statistics names/%s differ in length or contain duplicate names" % group)
    groups = {subject: [] for subject in subjects}
    for name, value in zip(names, values):
        match = re.fullmatch(r"S(\d{3})C(\d{3})P(\d{3})R\d{3}A\d{3}", name)
        if match is None or int(match.group({"setup": 1, "camera": 2, "performer": 3}[group])) != value:
            raise ValueError("Skeleton ID and %s statistics disagree: %s" % (group, name))
        if value in groups:
            groups[value].append(name)
    return [name for subject in subjects for name in groups[subject]]


def skeleton_identity(sample):
    """Compare full coordinates, ignoring only zero padding / an exact actor clone."""
    import numpy as np
    from render_skeleton_sample import normalize_sample
    sequence = normalize_sample(sample)
    if not np.isfinite(sequence).all():
        raise ValueError("Non-finite skeleton coordinates")
    second_present = bool(np.any(np.abs(sequence[:, 1]) > 1e-8))
    clone = second_present and np.array_equal(sequence[:, 0], sequence[:, 1])
    canonical = sequence[:, :1] if not second_present or clone else sequence
    canonical = np.array(canonical, dtype="<f4", order="C", copy=True)
    canonical[canonical == 0] = 0  # Treat IEEE -0 and +0 as identical.
    digest = hashlib.sha256(str(canonical.shape).encode("ascii") + canonical.tobytes()).hexdigest()
    return digest, 2 if second_present else 1, clone


def build_plan(args, names60, names_target, target_label="NTU120", extra_sources=()):
    sources = [{"dataset": "ntu60_xsub", "data_path": args.ntu60_data_path,
                "captions": args.ntu60_captions, "names": names60}] + list(extra_sources)
    signatures = {}
    for bank in sources:
        captions, _ = load_captions(bank["captions"], args)
        if set(captions) != set(range(len(bank["names"]))):
            raise ValueError("%s captions must cover every train row exactly once" % bank["dataset"])
        source = load_train(bank["data_path"])
        if len(source) != len(bank["names"]):
            raise ValueError("%s x_train count differs from the statistics" % bank["dataset"])
        print("Checking %s skeleton identities..." % bank["dataset"], flush=True)
        for index, name in enumerate(bank["names"]):
            digest, slots, _ = skeleton_identity(source[index])
            if len(captions[index]["persons"]) != slots:
                raise ValueError("%s caption/person slots disagree at row %d" % (bank["dataset"], index))
            if name in signatures and signatures[name][1] != digest:
                raise ValueError("Conflicting coordinates across source datasets: %s" % name)
            # Source priority: existing NTU60 XSub, then XView, then NTU120 XSub.
            signatures.setdefault(name, (index, digest, slots, bank["dataset"]))
        del source, captions
        gc.collect()  # Do not keep complete NPZ arrays resident together.
    target = load_train(args.data_path)
    if len(target) != len(names_target):
        raise ValueError("%s x_train count differs from the statistics; provide matching statistics" % target_label)
    reuse, generate = [], []
    for index, name in enumerate(names_target):
        if name not in signatures:
            generate.append(index)
            continue
        old_index, old_digest, old_slots, source_dataset = signatures[name]
        digest, slots, clone = skeleton_identity(target[index])
        if digest != old_digest:
            raise ValueError("Same skeleton ID has different coordinates: %s (%s=%d, %s=%d); "
                             "check NPZ/statistics provenance before reuse" %
                             (name, source_dataset, old_index, target_label, index))
        duplicate = old_slots == 1 and slots == 2 and clone
        if slots != old_slots and not duplicate:
            raise ValueError("Unsupported person-slot change for %s" % name)
        item = {"sample_index": index, "source_dataset": source_dataset, "source_sample_index": old_index,
                "skeleton_id": name, "skeleton_sha256": digest, "duplicate_person1": duplicate}
        if source_dataset == "ntu60_xsub":
            item["ntu60_sample_index"] = old_index  # Retain the original plan field for inspection.
        reuse.append(item)
    return {"sample_count": len(target), "generate_indices": generate, "reuse": reuse}, target


def source_identity(args, names60, names_target, target_key):
    return {"data": file_signature(args.ntu60_data_path),
            "captions": dict(file_signature(args.ntu60_captions), sha256=sha256_file(args.ntu60_captions)),
            "metadata_sha256": sha256_file(sidecar_paths(args.ntu60_captions)[0]),
            "statistics": {"ntu60": hashlib.sha256("\n".join(names60).encode()).hexdigest(),
                           target_key: hashlib.sha256("\n".join(names_target).encode()).hexdigest()}}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--benchmark", choices=("XSub", "XSet"), default="XSub",
                        help="XSet is the cross-setup (XSetup) protocol, with even setups for training.")
    parser.add_argument("--data_path", type=Path, default=None)
    parser.add_argument("--ntu60_data_path", type=Path, default=Path("../data/MAMP/ntu/NTU60_XSub.npz"))
    parser.add_argument("--ntu60_captions", type=Path,
                        default=Path("vlm_pilot/ntu60_xsub_global_local_v2/captions.json"))
    parser.add_argument("--ntu60_statistics_dir", type=Path, default=PROJECT_ROOT / "data/ntu/statistics")
    parser.add_argument("--ntu120_statistics_dir", type=Path, default=PROJECT_ROOT / "data/ntu120/statistics")
    parser.add_argument("--reuse_splits", choices=("ntu60_xview", "ntu120_xsub"), nargs="*", default=None,
                        help="XSet defaults to both completed banks; an empty list uses NTU60 XSub only.")
    parser.add_argument("--ntu60_xview_data_path", type=Path, default=Path("../data/MAMP/ntu/NTU60_XView.npz"))
    parser.add_argument("--ntu120_xsub_data_path", type=Path, default=Path("../data/MAMP/ntu120/NTU120_XSub.npz"))
    parser.add_argument("--ntu60_xview_captions", type=Path, default=None)
    parser.add_argument("--ntu120_xsub_captions", type=Path, default=None)
    args = parser.parse_args(argv)
    validate_common_arguments(parser, args)
    if args.data_path is None:
        args.data_path = Path("../data/MAMP/ntu120/NTU120_%s.npz" % args.benchmark)
    if args.reuse_splits is None:
        args.reuse_splits = ["ntu60_xview", "ntu120_xsub"] if args.benchmark == "XSet" else []
    if len(set(args.reuse_splits)) != len(args.reuse_splits):
        parser.error("--reuse_splits must not contain duplicates")
    if args.benchmark != "XSet" and args.reuse_splits:
        parser.error("Additional --reuse_splits are supported for XSet only")
    return args


def main(argv=None):
    args = parse_args(argv)
    names60 = train_names(args.ntu60_statistics_dir, NTU60_SUBJECTS)
    names120 = (train_names(args.ntu120_statistics_dir, NTU120_SUBJECTS) if args.benchmark == "XSub"
                else train_names(args.ntu120_statistics_dir, range(2, 33, 2), group="setup"))
    dataset = "ntu120_" + args.benchmark.lower()
    identity = pipeline_identity(args, dataset, args.data_path)
    identity["ntu60"] = source_identity(args, names60, names120, "ntu120")
    extra_sources = []
    for name in args.reuse_splits:
        names = (train_names(args.ntu60_statistics_dir, (2, 3), group="camera") if name == "ntu60_xview"
                 else train_names(args.ntu120_statistics_dir, NTU120_SUBJECTS))
        path = getattr(args, name + "_data_path")
        captions = getattr(args, name + "_captions") or args.output_root / name / "captions.json"
        if not captions.is_file() or not sidecar_paths(captions)[0].is_file():
            raise ValueError("Complete %s captions first: %s" % (name, captions))
        extra_sources.append({"dataset": name, "data_path": path, "captions": captions, "names": names})
    if extra_sources:
        identity["reuse_sources"] = {
            bank["dataset"]: {"data": file_signature(bank["data_path"]),
                              "captions": dict(file_signature(bank["captions"]), sha256=sha256_file(bank["captions"])),
                              "metadata_sha256": sha256_file(sidecar_paths(bank["captions"])[0]),
                              "statistics_sha256": hashlib.sha256("\n".join(bank["names"]).encode()).hexdigest()}
            for bank in extra_sources}
    identity["entrypoint_sha256"] = sha256_file(Path(__file__))
    return run_dataset(args, dataset, args.data_path, identity,
                       lambda: build_plan(args, names60, names120, "NTU120 " + args.benchmark, extra_sources))


if __name__ == "__main__":
    raise SystemExit(cli_main(main))
