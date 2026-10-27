#!/usr/bin/env python3
"""Upgrade render-only dataset plans, rerendering only changed actor layouts.

No VLM, CLIP, labels, or source caption banks are loaded. Existing completed
captions/caches are deliberately not adopted under a different render identity.
"""
import argparse
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

from dataset_caption_pipeline import (PLAN_SCHEMA, SCRIPT_DIR, atomic_json, cli_main,
                                      file_signature, load_train, sha256_file)
from render_qwen3vl_train import close_frames, make_metadata, save_gif
from render_skeleton_sample import (ACTOR_LAYOUT_VERSION, actor_layout, load_font,
                                    normalize_sample, render_sample_frames)
from run_caption_dual_gpu import run_lock


# Known producer versions immediately before this fix; allow LF/CRLF checkouts.
LEGACY_IMPLEMENTATIONS = {
    "dataset_caption_pipeline.py": {
        "fe7ca9df242b366c1e7548a257b119972a812f98e1d4667bccd5097193bd95dc",
        "b3484c188a8440ff725060eeb317bb1c3a2246eff73e35810f31cb01bf994534"},
    "render_skeleton_sample.py": {
        "bf11f430bdcf8bae02d6b7877f8a51370e5f9c32cf3516edfe927515982dddca",
        "a6791f13595232426fafa322cfd995b505b6f917e69836ec4ac9413e4b5aa597"},
    "render_qwen3vl_train.py": {
        "5a8905f3efe9b67099582eb23f388ef82f641ad8e6750d437d2fff976013cf06",
        "e0162369881cd5fe612894dadc02059918259e40e589f161a967a9983f7148b8"},
}
DEFAULT_DATASETS = ("ntu60_xview", "ntu120_xsub", "ntu120_xset", "pku_v1_xsub", "pku_v2_xsub")


def current_implementation():
    return {name: sha256_file(SCRIPT_DIR / name) for name in LEGACY_IMPLEMENTATIONS}


def repair_dataset(root, dry_run=False):
    root = Path(root).resolve()
    manifest = root / "dataset_manifest.json"
    if not manifest.is_file():
        raise ValueError("Missing dataset plan: %s" % manifest)
    plan = json.loads(manifest.read_text(encoding="utf-8"))
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("Unsupported dataset plan: %s" % manifest)
    previous = plan["identity"]["implementation"]
    current = current_implementation()
    if previous == current:
        print("%s: actor-layout implementation already current; unchanged" % root.name, flush=True)
        return {"rerendered": 0, "already_current": True}
    if (set(previous) != set(LEGACY_IMPLEMENTATIONS)
            or any(previous[name] not in hashes for name, hashes in LEGACY_IMPLEMENTATIONS.items())):
        raise ValueError("Unknown previous producer hashes; refusing to adopt %s" % root)
    artifacts = [root / "captions.json", root / "captions.metadata.json"]
    artifacts += list((root / "generated").glob("*.json*"))
    artifacts += list(root.glob("*cache*/manifest.json"))
    if any(path.exists() for path in artifacts):
        raise ValueError("Only render-only outputs can be upgraded; captions/cache already exist in %s. "
                         "Use a new --output_root for corrected captions." % root)
    source = plan["identity"]["source"]
    data_path = Path(source["path"])
    if source != file_signature(data_path):
        raise ValueError("Source NPZ identity changed: %s" % data_path)
    data = load_train(data_path)
    if len(data) != plan["sample_count"]:
        raise ValueError("Training row count differs from the saved plan")
    indices = plan["generate_indices"]
    reused = [item["sample_index"] for item in plan["reuse"]]
    if (len(set(indices)) != len(indices) or len(set(reused)) != len(reused)
            or set(indices) & set(reused)
            or set(indices) | set(reused) != set(range(len(data)))):
        raise ValueError("Dataset plan does not partition x_train exactly")

    # Validate every available input before replacing any GIF or plan identity.
    pending, missing, blue_only = [], [], []
    for ordinal, index in enumerate(sorted(indices), 1):
        directory = root / "rendered" / ("train_%06d" % index)
        metadata_path = directory / "render_metadata.json"
        gif = directory / "preview.gif"
        if not metadata_path.is_file() or not gif.is_file() or not gif.stat().st_size:
            missing.append(index)
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        config = metadata.get("config", {})
        if (metadata.get("sample_index") != index or metadata.get("status") != "rendered"
                or metadata.get("schema_version") != "macdiff.skeleton_render.v1"
                or config.get("source") != source
                or any(config.get(key) != value for key, value in plan["identity"]["render"].items())):
            raise ValueError("Render identity/settings mismatch at row %d" % index)
        layout = actor_layout(normalize_sample(data[index]))
        if layout["raw_active_person_indices"] == [1]:
            blue_only.append(index)
        old = metadata["render"]
        if old.get("actor_layout_version") == ACTOR_LAYOUT_VERSION:
            if any(old.get(key) != value for key, value in layout.items()):
                raise ValueError("Corrected actor metadata differs from raw coordinates at row %d" % index)
            continue  # Resume after an interrupted repair.
        if (layout["duplicate_person1"] or layout["raw_active_person_indices"] == [1]
                or layout["visible_actor_count"] != old["visible_actor_count"]):
            pending.append((index, metadata))
        if ordinal == 1 or ordinal % 1000 == 0:
            print("%s: checked %d/%d saved renders" % (root.name, ordinal, len(indices)), flush=True)
    print("%s: rerender=%d, missing=%d, true_person1_only=%d%s" %
          (root.name, len(pending), len(missing), len(blue_only), " (dry run)" if dry_run else ""), flush=True)
    result = {"rerendered": len(pending), "missing_indices": missing, "person1_only_indices": blue_only}
    if dry_run:
        return result

    backup = root / "dataset_manifest.before_actor_layout_v2.json"
    if backup.exists():
        if json.loads(backup.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing plan backup differs; refusing to overwrite it")
    else:
        atomic_json(backup, plan)
    font = load_font(18)
    for ordinal, (index, metadata) in enumerate(pending, 1):
        directory = root / "rendered" / ("train_%06d" % index)
        config = metadata["config"]
        options = {key: config[key] for key in (
            "num_frames", "width", "height", "temporal_smooth", "median_window",
            "smooth_window", "smooth_polyorder", "max_interp_gap")}
        started = time.perf_counter()
        frames = []
        try:
            frames, info = render_sample_frames(data[index], font=font, **options)
            save_gif(frames, directory / "preview.gif", config["sample_fps"])
            repaired = make_metadata(
                sample_index=index, sample_dir=directory, config=config, render_info=info,
                render_seconds=time.perf_counter() - started,
                args=SimpleNamespace(sample_fps=config["sample_fps"],
                                     num_shards=metadata.get("num_shards", 1),
                                     shard_id=metadata.get("shard_id", 0)))
            atomic_json(directory / "render_metadata.json", repaired)
        finally:
            close_frames(frames)
        if ordinal == 1 or ordinal % 100 == 0 or ordinal == len(pending):
            print("%s: repaired %d/%d" % (root.name, ordinal, len(pending)), flush=True)
    report = {"actor_layout_version": ACTOR_LAYOUT_VERSION,
              "previous_implementation": previous, "implementation": current,
              "rerendered_indices": [index for index, _ in pending], **result}
    atomic_json(root / "actor_layout_upgrade.json", report)
    upgraded = copy.deepcopy(plan)
    upgraded["identity"]["implementation"] = current
    atomic_json(manifest, upgraded)  # Publish the new identity only after successful repairs.
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_roots", nargs="*", type=Path,
                        help="Dataset folders containing dataset_manifest.json; defaults to existing new splits.")
    parser.add_argument("--output_root", type=Path, default=Path("vlm_pilot"))
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)
    roots = args.dataset_roots or [args.output_root / name for name in DEFAULT_DATASETS
                                  if (args.output_root / name / "dataset_manifest.json").is_file()]
    if not roots:
        raise ValueError("No dataset plans found under %s" % args.output_root)
    for root in roots:
        if args.dry_run:
            repair_dataset(root, dry_run=True)
        else:
            with run_lock(root.resolve()):
                repair_dataset(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main(main))
