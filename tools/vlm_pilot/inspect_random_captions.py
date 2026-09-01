#!/usr/bin/env python3
"""Randomly inspect accepted person captions from one or more JSONL shards."""

from __future__ import annotations

import argparse
import glob
import html
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


DETAIL_FIELDS = (
    "main_part",
    "motion",
    "beginning",
    "middle",
    "end",
    "interaction",
    "text",
)

# Official NTU RGB+D A1-A60 names:
# https://rose1.ntu.edu.sg/dataset/actionRecognition/
NTU60_ACTION_NAMES = (
    "drink water",
    "eat meal",
    "brush teeth",
    "brush hair",
    "drop",
    "pick up",
    "throw",
    "sit down",
    "stand up",
    "clapping",
    "reading",
    "writing",
    "tear up paper",
    "put on jacket",
    "take off jacket",
    "put on a shoe",
    "take off a shoe",
    "put on glasses",
    "take off glasses",
    "put on a hat/cap",
    "take off a hat/cap",
    "cheer up",
    "hand waving",
    "kicking something",
    "reach into pocket",
    "hopping",
    "jump up",
    "phone call",
    "play with phone/tablet",
    "type on a keyboard",
    "point to something",
    "taking a selfie",
    "check time (from watch)",
    "rub two hands",
    "nod head/bow",
    "shake head",
    "wipe face",
    "salute",
    "put palms together",
    "cross hands in front",
    "sneeze/cough",
    "staggering",
    "falling down",
    "headache",
    "chest pain",
    "back pain",
    "neck pain",
    "nausea/vomiting",
    "fan self",
    "punch/slap",
    "kicking",
    "pushing",
    "pat on back",
    "point finger",
    "hugging",
    "giving object",
    "touch pocket",
    "shaking hands",
    "walking towards",
    "walking apart",
)
NTU60_ACTION_SOURCE = "https://rose1.ntu.edu.sg/dataset/actionRecognition/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read caption JSONL shards, keep accepted records, deduplicate by "
            "sample_index, and print a random selection."
        )
    )
    parser.add_argument(
        "--input_paths",
        nargs="+",
        required=True,
        help="One or more JSONL paths or quoted glob patterns.",
    )
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument(
        "--sample_indices",
        nargs="+",
        type=int,
        default=None,
        help="Inspect exact train indices instead of taking a random selection.",
    )
    parser.add_argument(
        "--visualization_dirs",
        nargs="+",
        type=Path,
        default=None,
        help=(
            "Existing renderer output directories; sample indices are read from "
            "their render_metadata.json files."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional deterministic random seed; omit for a new selection each run.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print only each person's final text instead of all structured fields.",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=None,
        help="NTU60 NPZ containing aligned x_train and y_train for review export.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="New standalone review directory for labels, GIFs, JSON, and HTML.",
    )
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--gif_duration_ms", type=int, default=180)
    return parser.parse_args()


def expand_paths(patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    seen = set()
    for pattern in patterns:
        matches = [Path(match) for match in glob.glob(pattern)]
        if not matches:
            candidate = Path(pattern)
            if candidate.is_file():
                matches = [candidate]
            else:
                raise FileNotFoundError(f"No JSONL file matched: {pattern}")
        for path in matches:
            resolved = path.resolve()
            if not resolved.is_file():
                continue
            if resolved not in seen:
                paths.append(resolved)
                seen.add(resolved)
    if not paths:
        raise FileNotFoundError("No input JSONL files were found")
    return sorted(paths)


def load_latest_accepted(
    paths: Iterable[Path],
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, int]]:
    records: Dict[int, Dict[str, Any]] = {}
    stats = {
        "lines": 0,
        "accepted_lines": 0,
        "non_accepted_lines": 0,
        "malformed_lines": 0,
    }
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                stats["lines"] += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    stats["malformed_lines"] += 1
                    continue
                if record.get("status") != "accepted":
                    stats["non_accepted_lines"] += 1
                    continue
                try:
                    sample_index = int(record["sample_index"])
                except (KeyError, TypeError, ValueError):
                    stats["malformed_lines"] += 1
                    continue
                record["_inspection_source"] = f"{path}:{line_number}"
                records[sample_index] = record
                stats["accepted_lines"] += 1
    return records, stats


def indices_from_visualizations(paths: Sequence[Path]) -> Dict[int, Path]:
    result: Dict[int, Path] = {}
    for path in paths:
        metadata_path = path if path.name == "render_metadata.json" else path / "render_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Visualization metadata not found: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        sample_index = metadata.get("sample_index")
        if sample_index is None:
            raise ValueError(f"Visualization has no dataset sample_index: {metadata_path}")
        result[int(sample_index)] = metadata_path.parent.resolve()
    return result


def person_lines(person: Dict[str, Any], compact: bool) -> List[str]:
    index = person.get("person_index", "?")
    color = person.get("color", "unknown")
    lines = [f"  person {index} ({color})"]
    fields = ("text",) if compact else DETAIL_FIELDS
    for field in fields:
        value = person.get(field)
        if value not in (None, ""):
            lines.append(f"    {field}: {value}")
    return lines


def format_record(record: Dict[str, Any], ordinal: int, total: int, compact: bool) -> str:
    sample_index = record.get("sample_index", "?")
    sample_id = record.get("sample_id", f"train_{sample_index}")
    caption = record.get("caption") or {}
    persons = caption.get("persons") or []
    actor_count = record.get("actor_count", caption.get("actors", len(persons)))

    header = f"[{ordinal}/{total}] {sample_id} | sample_index={sample_index}"
    label = record.get("_label")
    if label:
        header += f" | label={label['action_code']} {label['action_name']}"
    header += f" | actors={actor_count}"
    lines = [header]
    if persons:
        for person in sorted(persons, key=lambda item: item.get("person_index", 999)):
            lines.extend(person_lines(person, compact))
    else:
        texts = record.get("texts") or []
        if not texts:
            lines.append("  (accepted record has no printable caption content)")
        for text_record in texts:
            if isinstance(text_record, dict):
                lines.extend(person_lines(text_record, True))
            else:
                lines.append(f"  text: {text_record}")
    if record.get("_visualization_dir"):
        lines.append(f"  visualization: {record['_visualization_dir']}")
    lines.append(f"  source: {record.get('_inspection_source', 'unknown')}")
    return "\n".join(lines)


def decode_ntu60_label(label_vector: Any, sample_index: int) -> Dict[str, Any]:
    import numpy as np

    flat = np.asarray(label_vector).reshape(-1)
    if flat.size != len(NTU60_ACTION_NAMES):
        raise ValueError(
            f"y_train[{sample_index}] has {flat.size} values; expected a 60-class "
            "one-hot vector"
        )
    if not np.all(np.isfinite(flat)):
        raise ValueError(f"y_train[{sample_index}] contains non-finite values")
    class_index = int(np.argmax(flat))
    action_id = class_index + 1
    return {
        "class_index": class_index,
        "action_id": action_id,
        "action_code": f"A{action_id:03d}",
        "action_name": NTU60_ACTION_NAMES[class_index],
        "one_hot_value": float(flat[class_index]),
    }


def clean_caption_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def caption_texts(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    caption = record.get("caption") or {}
    persons = caption.get("persons") or []
    if persons:
        return [
            {
                "person_index": person.get("person_index"),
                "color": person.get("color"),
                "text": person.get("text", ""),
            }
            for person in sorted(
                persons,
                key=lambda item: item.get("person_index", 999),
            )
        ]
    normalized = []
    for position, value in enumerate(record.get("texts") or []):
        if isinstance(value, dict):
            normalized.append(
                {
                    "person_index": value.get("person_index", position),
                    "color": value.get("color", "unknown"),
                    "text": value.get("text", ""),
                }
            )
        else:
            normalized.append(
                {"person_index": position, "color": "unknown", "text": str(value)}
            )
    return normalized


def save_animation(frames: Sequence[Any], path: Path, duration_ms: int) -> None:
    if not frames:
        raise ValueError("Cannot save an empty animation")
    frames[0].save(
        path,
        save_all=True,
        append_images=list(frames[1:]),
        duration=duration_ms,
        loop=0,
        disposal=2,
    )


def close_frames(frames: Sequence[Any]) -> None:
    for frame in frames:
        close = getattr(frame, "close", None)
        if callable(close):
            close()


def make_summary_text(
    record: Dict[str, Any],
    label: Dict[str, Any],
    render_info: Dict[str, Any],
) -> str:
    lines = [
        f"sample_id: {record.get('sample_id')}",
        f"sample_index: {record.get('sample_index')}",
        f"label: {label['action_code']} | {label['action_name']}",
        f"class_index: {label['class_index']}",
        f"visible_actor_count: {render_info['visible_actor_count']}",
        "",
        "captions:",
    ]
    for person in caption_texts(record):
        lines.append(
            f"- person {person['person_index']} ({person['color']}): {person['text']}"
        )
    return "\n".join(lines) + "\n"


def make_index_html(entries: Sequence[Dict[str, Any]]) -> str:
    cards = []
    for entry in entries:
        label = entry["label"]
        text_items = "".join(
            "<li><strong>person {index} ({color}):</strong> {text}</li>".format(
                index=html.escape(str(person["person_index"])),
                color=html.escape(str(person["color"])),
                text=html.escape(str(person["text"])),
            )
            for person in entry["texts"]
        )
        cards.append(
            """
            <article class="card">
              <h2>{sample_id}</h2>
              <p class="label">{action_code} · {action_name}</p>
              <img src="{gif_path}" alt="Skeleton animation for {sample_id}">
              <ul>{text_items}</ul>
              <p class="links"><a href="{json_path}">review.json</a> ·
                 <a href="{png_path}">middle frame</a></p>
            </article>
            """.format(
                sample_id=html.escape(entry["sample_id"]),
                action_code=html.escape(label["action_code"]),
                action_name=html.escape(label["action_name"]),
                gif_path=html.escape(entry["gif_path"]),
                text_items=text_items,
                json_path=html.escape(entry["json_path"]),
                png_path=html.escape(entry["png_path"]),
            )
        )
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Random caption review</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 24px; background: #f5f6f8; color: #172033; }
    main { display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 20px; }
    .card { background: white; border: 1px solid #dce1ea; border-radius: 12px; padding: 16px; }
    h2 { margin: 0 0 4px; font-size: 20px; }
    .label { margin: 0 0 12px; color: #8a3100; font-weight: 700; }
    img { display: block; width: 100%; height: auto; border: 1px solid #dce1ea; }
    li { margin: 8px 0; line-height: 1.4; }
    .links { margin-bottom: 0; }
  </style>
</head>
<body>
  <h1>Random NTU60 caption review</h1>
  <main>
    __REVIEW_CARDS__
  </main>
</body>
</html>
"""
    return template.replace("__REVIEW_CARDS__", "\n".join(cards))


def export_review(
    *,
    selected: Sequence[Dict[str, Any]],
    input_paths: Sequence[Path],
    data_path: Path,
    output_dir: Path,
    seed: Any,
    num_frames: int,
    width: int,
    height: int,
    gif_duration_ms: int,
) -> None:
    import numpy as np

    from render_skeleton_sample import load_font, render_sample_frames

    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if output_dir.exists():
        raise FileExistsError(
            f"Review output already exists: {output_dir}. Use a new directory."
        )

    with np.load(data_path, mmap_mode="r", allow_pickle=False) as archive:
        missing = [key for key in ("x_train", "y_train") if key not in archive.files]
        if missing:
            raise KeyError(f"Missing {missing} in {data_path}; keys={archive.files}")
        x_train = archive["x_train"]
        y_train = archive["y_train"]
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"x_train/y_train length mismatch: {x_train.shape[0]} vs {y_train.shape[0]}"
            )
        for record in selected:
            sample_index = int(record["sample_index"])
            if not 0 <= sample_index < x_train.shape[0]:
                raise IndexError(
                    f"sample_index {sample_index} is outside x_train with "
                    f"{x_train.shape[0]} samples"
                )

        output_dir.mkdir(parents=True, exist_ok=False)
        font = load_font(18)
        entries: List[Dict[str, Any]] = []
        for ordinal, record in enumerate(selected, start=1):
            sample_index = int(record["sample_index"])
            sample_id = str(record.get("sample_id", f"train_{sample_index:06d}"))
            label = decode_ntu60_label(y_train[sample_index], sample_index)
            record["_label"] = label
            folder_name = f"train_{sample_index:06d}_{label['action_code']}"
            sample_dir = output_dir / folder_name
            sample_dir.mkdir()

            frames: List[Any] = []
            try:
                frames, render_info = render_sample_frames(
                    x_train[sample_index],
                    num_frames=num_frames,
                    width=width,
                    height=height,
                    font=font,
                )
                gif_path = sample_dir / "preview.gif"
                png_path = sample_dir / "preview_middle.png"
                save_animation(frames, gif_path, gif_duration_ms)
                frames[len(frames) // 2].save(png_path)
            finally:
                close_frames(frames)

            review = {
                "sample_id": sample_id,
                "sample_index": sample_index,
                "label_source": "y_train one-hot vector",
                "label": label,
                "render": render_info,
                "caption_record": clean_caption_record(record),
            }
            (sample_dir / "review.json").write_text(
                json.dumps(review, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (sample_dir / "summary.txt").write_text(
                make_summary_text(record, label, render_info),
                encoding="utf-8",
            )
            entry = {
                "ordinal": ordinal,
                "sample_id": sample_id,
                "sample_index": sample_index,
                "label": label,
                "texts": caption_texts(record),
                "directory": folder_name,
                "gif_path": f"{folder_name}/preview.gif",
                "png_path": f"{folder_name}/preview_middle.png",
                "json_path": f"{folder_name}/review.json",
            }
            entries.append(entry)
            print(
                f"Exported {ordinal}/{len(selected)}: {sample_id} -> "
                f"{label['action_code']} {label['action_name']}"
            )

    manifest = {
        "data_path": str(data_path.resolve()),
        "caption_paths": [str(path) for path in input_paths],
        "seed": seed,
        "num_samples": len(entries),
        "num_frames": num_frames,
        "gif_duration_ms": gif_duration_ms,
        "label_source": "y_train one-hot vector",
        "action_name_source": NTU60_ACTION_SOURCE,
        "samples": entries,
    }
    (output_dir / "review_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "index.html").write_text(
        make_index_html(entries),
        encoding="utf-8",
    )
    print(f"Review gallery: {output_dir / 'index.html'}")


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive")
    if (args.data_path is None) != (args.output_dir is None):
        raise ValueError("--data_path and --output_dir must be provided together")
    if args.num_frames < 2:
        raise ValueError("--num_frames must be at least 2")
    if args.width <= 0 or args.height <= 0 or args.width % 3:
        raise ValueError("--width/--height must be positive and --width divisible by 3")
    if args.gif_duration_ms <= 0:
        raise ValueError("--gif_duration_ms must be positive")

    paths = expand_paths(args.input_paths)
    records_by_index, stats = load_latest_accepted(paths)
    records = list(records_by_index.values())
    if not records:
        raise SystemExit("No accepted caption records were found.")

    visualization_map = indices_from_visualizations(args.visualization_dirs or [])
    requested_indices: List[int] = []
    for sample_index in (args.sample_indices or []) + list(visualization_map):
        if sample_index < 0:
            raise ValueError("sample indices must be non-negative")
        if sample_index not in requested_indices:
            requested_indices.append(sample_index)

    if requested_indices:
        missing_indices = [
            sample_index
            for sample_index in requested_indices
            if sample_index not in records_by_index
        ]
        if missing_indices:
            raise SystemExit(
                "No accepted caption found for sample_index: "
                + ", ".join(str(index) for index in missing_indices)
            )
        selected = [records_by_index[index] for index in requested_indices]
        for record in selected:
            sample_index = int(record["sample_index"])
            if sample_index in visualization_map:
                record["_visualization_dir"] = str(visualization_map[sample_index])
        count = len(selected)
    else:
        count = min(args.num_samples, len(records))
        selected = random.Random(args.seed).sample(records, count)
        selected.sort(key=lambda record: int(record["sample_index"]))

    if args.data_path is not None:
        export_review(
            selected=selected,
            input_paths=paths,
            data_path=args.data_path,
            output_dir=args.output_dir,
            seed=args.seed,
            num_frames=args.num_frames,
            width=args.width,
            height=args.height,
            gif_duration_ms=args.gif_duration_ms,
        )

    print(
        f"Loaded {len(paths)} file(s): {stats['lines']} JSONL records, "
        f"{len(records)} unique accepted samples, "
        f"{stats['non_accepted_lines']} non-accepted, "
        f"{stats['malformed_lines']} malformed."
    )
    if requested_indices:
        print(f"Showing {count} requested sample(s).\n")
    else:
        seed_text = "random" if args.seed is None else str(args.seed)
        print(f"Showing {count} sample(s), seed={seed_text}.\n")
    for ordinal, record in enumerate(selected, start=1):
        print(format_record(record, ordinal, count, args.compact))
        if ordinal != count:
            print()


if __name__ == "__main__":
    main()
