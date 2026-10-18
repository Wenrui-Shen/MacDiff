#!/usr/bin/env python3
"""Read-only audit of NTU60 XSub/XView, NTU120 XSub/XSet and PKU v1/v2 XSub."""
import argparse
import contextlib
import json
import sys
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_caption_pipeline import file_signature
from run_caption_ntu120 import NTU60_SUBJECTS, NTU120_SUBJECTS, train_names

DATASETS = ("ntu60_xsub", "ntu60_xview", "ntu120_xsub", "ntu120_xset", "pku_v1_xsub", "pku_v2_xsub")


def dataset_specs(args):
    ntu, large = args.ntu60_statistics_dir, args.ntu120_statistics_dir
    names = {
        "ntu60_xsub": train_names(ntu, NTU60_SUBJECTS),
        "ntu60_xview": train_names(ntu, (2, 3), group="camera"),
        "ntu120_xsub": train_names(large, NTU120_SUBJECTS),
        "ntu120_xset": train_names(large, range(2, 33, 2), group="setup"),
    }
    totals = {"ntu60": len((ntu / "skes_available_name.txt").read_text().split()),
              "ntu120": len((large / "skes_available_name.txt").read_text().split())}
    paths = {"ntu60_xsub": "ntu/NTU60_XSub.npz", "ntu60_xview": "ntu/NTU60_XView.npz",
             "ntu120_xsub": "ntu120/NTU120_XSub.npz", "ntu120_xset": "ntu120/NTU120_XSet.npz",
             "pku_v1_xsub": "pku_v1/PKUv1_XSub.npz", "pku_v2_xsub": "pku_v2/PKUv2_XSub.npz"}
    result, covered = {}, set()
    for name in DATASETS:
        root = args.output_root / name
        spec = {"dataset": name, "data_path": args.data_root / paths[name],
                "rendered": root / "rendered", "captions": root / "captions.json",
                "cache": root / "clip_sentence_cache_v3", "plan": root / "dataset_manifest.json"}
        if name == "ntu60_xsub":
            spec.update(rendered=args.ntu60_xsub_rendered, captions=args.ntu60_xsub_captions,
                        cache=args.ntu60_xsub_cache)
        if name in names:
            sample_names = set(names[name])
            spec["split"] = {"train_expected": len(sample_names),
                             "test_expected": totals[name.split("_")[0]] - len(sample_names),
                             "overlap_ntu60_xsub": len(sample_names & set(names["ntu60_xsub"])),
                             "reuse_previous_splits": len(sample_names & covered),
                             "new_after_previous_splits": len(sample_names - covered)}
            covered.update(sample_names)
        else:
            spec["split"] = {"train_expected": None, "test_expected": None,
                             "note": "PKU counts require the actual NPZ; raw split files are not bundled."}
        result[name] = spec
    return result


def array_shape(path, key):
    import numpy as np
    with zipfile.ZipFile(path) as archive, archive.open(key + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version not in ((1, 0), (2, 0)):
            raise ValueError("Unsupported NPY header for " + key)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(stream)
    if len(shape) != 3 or shape[-1] != 150 or fortran or dtype.kind not in "fiu" or shape[0] < 1:
        raise ValueError("Expected numeric C-order [N,T,150]: " + key)
    return shape


def check_data(spec):
    train, test = array_shape(spec["data_path"], "x_train"), array_shape(spec["data_path"], "x_test")
    for split, count in (("train", train[0]), ("test", test[0])):
        expected = spec["split"][split + "_expected"]
        if expected is not None and count != expected:
            raise ValueError("%s count %d differs from statistics %d" % (split, count, expected))
    return {"status": "ok", "train": train[0], "test": test[0], "labels_read": False}


def check_render(spec, decode):
    count = array_shape(spec["data_path"], "x_train")[0]
    expected = set(range(count))
    render_config = None
    plan_kind = "legacy_full_train" if spec["dataset"] == "ntu60_xsub" else "reuse_plan"
    if not spec["plan"].is_file() and spec["dataset"] != "ntu60_xsub":
        # NTU reuse indices are not known until the renderer has checked IDs and
        # coordinates. Do not label the full training split as "new renders".
        estimated = spec.get("split", {}).get("new_after_previous_splits")
        is_ntu = spec["dataset"].startswith("ntu")
        return {"status": "not_planned", "dataset_train_samples": count,
                "expected_new_renders": None,
                "estimated_new_after_prerequisites": estimated if is_ntu else count,
                "estimate_basis": ("Statistics ID overlap after prerequisite caption banks are complete; "
                                   "coordinates not yet checked" if is_ntu
                                   else "All PKU training rows; caption banks are separate by version"),
                "detail": "Missing %s; run --stage render to build and verify the generation plan" % spec["plan"]}
    if spec["plan"].is_file():
        plan_kind = "reuse_plan"
        plan = json.loads(spec["plan"].read_text(encoding="utf-8"))
        expected = set(plan["generate_indices"])
        reused = [item["sample_index"] for item in plan["reuse"]]
        if (plan["sample_count"] != count or len(expected) != len(plan["generate_indices"])
                or expected & set(reused) or len(set(reused)) != len(reused)
                or expected | set(reused) != set(range(count))):
            raise ValueError("Render plan does not partition x_train exactly")
        if plan["identity"]["source"] != file_signature(spec["data_path"]):
            raise ValueError("Render plan was built for different source data")
        render_config = plan["identity"]["render"]
    source = file_signature(spec["data_path"])
    missing = []
    for index in sorted(expected):
        directory = spec["rendered"] / ("train_%06d" % index)
        gif, metadata_path = directory / "preview.gif", directory / "render_metadata.json"
        if not gif.is_file() or not gif.stat().st_size or not metadata_path.is_file():
            missing.append(index)
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (metadata.get("schema_version") != "macdiff.skeleton_render.v1"
                or metadata.get("sample_index") != index or metadata.get("status") != "rendered"
                or metadata.get("config", {}).get("source") != source):
            raise ValueError("Render source/index/status mismatch at row %d" % index)
        if render_config and any(metadata["config"].get(key) != value for key, value in render_config.items()):
            raise ValueError("Render settings differ from the saved plan at row %d" % index)
        if decode:
            from caption_qwen3vl_rendered_train_transformers import close_frames, load_gif_frames
            config = metadata["config"]
            with contextlib.redirect_stdout(sys.stderr):
                frames = load_gif_frames(gif, config["num_frames"], config["sample_fps"])
            close_frames(frames)
    return {"status": "incomplete" if missing else "ok", "expected_new_renders": len(expected),
            "missing_count": len(missing), "missing_first20": missing[:20], "gif_frames_decoded": decode,
            "plan_kind": plan_kind}


def check_captions(spec):
    from cache_clip_motion_text import build_plan
    with contextlib.redirect_stdout(sys.stderr):
        samples, valid, version, _ = build_plan(spec["data_path"], [spec["captions"]])
    metadata = json.loads(spec["captions"].with_suffix(".metadata.json").read_text(encoding="utf-8"))
    source = metadata.get("identity", {}).get("source")
    if source is not None and source != file_signature(spec["data_path"]):
        raise ValueError("Caption dataset identity differs from this NPZ")
    return {"status": "ok", "samples": len(samples), "persons": int(valid.sum()), "caption_version": version}


def check_cache(spec, quick):
    from util.clip_text_cache import file_identity
    from util.structured_text_cache import validate_cache
    count = array_shape(spec["data_path"], "x_train")[0]
    manifest = validate_cache(spec["cache"], spec["data_path"], count, skip_full_validation=quick)
    identity = manifest["identity"]
    if (identity["captions"] != [file_identity(spec["captions"])]
            or identity["caption_metadata"] != [file_identity(spec["captions"].with_suffix(".metadata.json"))]):
        raise ValueError("Cache was built from different captions/metadata")
    return {"status": "ok", "samples": manifest["sample_count"], "feature_dim": manifest["feature_dim"],
            "context_length": manifest["context_length"], "full_npz_and_feature_hashes": not quick}


def inspect(spec, args):
    result = {"dataset": spec["dataset"], "split": spec["split"], "stages": {}}
    checks = {"data": lambda: check_data(spec), "render": lambda: check_render(spec, args.check_gifs),
              "caption": lambda: check_captions(spec), "cache": lambda: check_cache(spec, args.quick)}
    for name in args.stages:
        try:
            result["stages"][name] = checks[name]()
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            result["stages"][name] = {"status": "missing" if isinstance(exc, FileNotFoundError) else "error",
                                      "detail": str(exc)}
    result["complete"] = all(stage["status"] == "ok" for stage in result["stages"].values())
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=Path("../data/MAMP"))
    parser.add_argument("--output_root", type=Path, default=Path("vlm_pilot"))
    parser.add_argument("--datasets", choices=DATASETS, nargs="+", default=list(DATASETS))
    parser.add_argument("--stages", choices=("data", "render", "caption", "cache"), nargs="+",
                        default=["data", "render", "caption", "cache"])
    parser.add_argument("--quick", action="store_true", help="Cache size/header check instead of full NPZ/feature hashes.")
    parser.add_argument("--check_gifs", action="store_true", help="Decode every expected new GIF and verify its frame count.")
    parser.add_argument("--ntu60_statistics_dir", type=Path, default=PROJECT_ROOT / "data/ntu/statistics")
    parser.add_argument("--ntu120_statistics_dir", type=Path, default=PROJECT_ROOT / "data/ntu120/statistics")
    parser.add_argument("--ntu60_xsub_rendered", type=Path, default=Path("vlm_pilot/ntu60_xsub_train_rendered_v3_2view_smooth_w5"))
    parser.add_argument("--ntu60_xsub_captions", type=Path, default=Path("vlm_pilot/ntu60_xsub_global_local_v2/captions.json"))
    parser.add_argument("--ntu60_xsub_cache", type=Path, default=Path("vlm_pilot/ntu60_xsub_clip_sentence_cache_v3"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    specs = dataset_specs(args)
    results = [inspect(specs[name], args) for name in args.datasets]
    print(json.dumps({"read_only": True, "results": results}, ensure_ascii=False, indent=2))
    return 0 if all(result["complete"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
