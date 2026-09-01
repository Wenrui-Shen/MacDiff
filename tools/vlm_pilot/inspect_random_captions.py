#!/usr/bin/env python3
"""Randomly inspect accepted person captions from one or more JSONL shards."""

from __future__ import annotations

import argparse
import glob
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

    lines = [
        f"[{ordinal}/{total}] {sample_id} | sample_index={sample_index} | actors={actor_count}"
    ]
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
    lines.append(f"  source: {record.get('_inspection_source', 'unknown')}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive")

    paths = expand_paths(args.input_paths)
    records_by_index, stats = load_latest_accepted(paths)
    records = list(records_by_index.values())
    if not records:
        raise SystemExit("No accepted caption records were found.")

    count = min(args.num_samples, len(records))
    selected = random.Random(args.seed).sample(records, count)
    selected.sort(key=lambda record: int(record["sample_index"]))

    print(
        f"Loaded {len(paths)} file(s): {stats['lines']} JSONL records, "
        f"{len(records)} unique accepted samples, "
        f"{stats['non_accepted_lines']} non-accepted, "
        f"{stats['malformed_lines']} malformed."
    )
    seed_text = "random" if args.seed is None else str(args.seed)
    print(f"Showing {count} sample(s), seed={seed_text}.\n")
    for ordinal, record in enumerate(selected, start=1):
        print(format_record(record, ordinal, count, args.compact))
        if ordinal != count:
            print()


if __name__ == "__main__":
    main()
