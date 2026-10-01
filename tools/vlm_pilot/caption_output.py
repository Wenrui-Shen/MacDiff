"""Text-only caption results, with run metadata and diagnostics in sidecars."""
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path


LOCAL_PARTS = ("head", "torso", "left_arm", "right_arm", "left_leg", "right_leg")
OUTPUT_SCHEMA = "macdiff.caption_text.v2"


def caption_schema(prompt):
    return "global_local" if '"global"' in prompt and '"local"' in prompt else "legacy"


def caption_persons(caption):
    """Keep the generated text once, in a stable person/body-region order."""
    persons = []
    for person in caption["persons"]:
        item = {"person_index": person["person_index"]}
        if "global" in person:
            item["global"] = person["global"].strip()
            item["local"] = {part: person["local"][part].strip() for part in LOCAL_PARTS}
        else:
            item["text"] = person["text"].strip()
        persons.append(item)
    return persons


def sidecar_paths(path):
    path = Path(path)
    return path.with_suffix(".metadata.json"), path.with_suffix(".diagnostics.jsonl")


def iter_caption_records(path):
    """Read a JSON array (.json) or one sample per line (other extensions)."""
    path = Path(path)
    with path.open(encoding="utf-8") as source:
        if path.suffix.lower() == ".json":
            records = json.load(source)
            if not isinstance(records, list):
                raise ValueError("Batch .json output must be a JSON array: %s" % path)
            yield from records
        else:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError("Invalid JSONL in %s at line %d: %s" % (path, number, exc)) from exc


def read_caption_metadata(path, *, expected_model, expected_prompt_hash, expected_revision=None):
    metadata_path, _ = sidecar_paths(path)
    if not metadata_path.is_file():
        raise ValueError("Missing caption metadata %s; use a new --output_path for legacy output." % metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (metadata.get("schema_version") != OUTPUT_SCHEMA
            or metadata.get("model") != expected_model
            or metadata.get("prompt", {}).get("sha256") != expected_prompt_hash
            or metadata.get("requested_revision") != expected_revision):
        raise ValueError("Caption model, revision, or prompt differs; use a new --output_path.")
    return metadata


def repair_incomplete_jsonl_tail(path):
    """Back up and remove only an invalid, unterminated final JSONL record."""
    path = Path(path)
    if not path.exists() or not path.stat().st_size:
        return None
    with path.open("rb") as source:
        source.seek(-1, 2)
        if source.read(1) == b"\n":
            return None
        end = source.tell()
        position = end
        pieces = []
        while position:
            size = min(position, 8192)
            position -= size
            source.seek(position)
            chunk = source.read(size)
            separator = chunk.rfind(b"\n")
            if separator >= 0:
                pieces.append(chunk[separator + 1:])
                position += separator + 1
                break
            pieces.append(chunk)
        tail = b"".join(reversed(pieces))
    if not tail.strip():
        return None
    try:
        json.loads(tail.decode("utf-8"))
        return None
    except (UnicodeDecodeError, json.JSONDecodeError):
        backup = path.with_name(path.name + ".interrupted-tail.%d.bin" % time.time_ns())
        with backup.open("xb") as saved:
            saved.write(tail)
            saved.flush()
            os.fsync(saved.fileno())
        with path.open("r+b") as output:
            output.truncate(position)
            output.flush()
            os.fsync(output.fileno())
        return backup


def load_accepted_indices(path, *, expected_model, expected_prompt_hash, expected_revision=None):
    path = Path(path)
    if not path.exists():
        return set()
    metadata = read_caption_metadata(path, expected_model=expected_model,
                                     expected_prompt_hash=expected_prompt_hash,
                                     expected_revision=expected_revision)
    accepted = set()
    for record in iter_caption_records(path):
        if not isinstance(record, dict) or set(record) != {"sample_index", "persons"}:
            raise ValueError("Expected text-only caption records; use a new --output_path for legacy output.")
        index = record["sample_index"]
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("Invalid caption sample_index: %r" % index)
        persons = record["persons"]
        if not isinstance(persons, list) or len(persons) not in (1, 2):
            raise ValueError("Invalid caption persons for sample %d" % index)
        # Import lazily: the validator also imports this module.
        from caption_qwen3vl_sample import validate_caption
        schema = metadata.get("caption_schema")
        if schema == "legacy":
            errors = []
            for position, person in enumerate(persons):
                if (not isinstance(person, dict) or set(person) != {"person_index", "text"}
                        or type(person["person_index"]) is not int
                        or person["person_index"] != position
                        or not isinstance(person["text"], str) or not person["text"].strip()):
                    errors.append("invalid legacy text")
        else:
            errors = validate_caption({"persons": persons}, len(persons), schema="global_local")
        if errors:
            raise ValueError("Invalid saved caption for sample %d: %s" % (index, "; ".join(errors)))
        if index in accepted:
            raise ValueError("Duplicate saved sample_index: %d" % index)
        accepted.add(index)
    return accepted


def _jsonable(value):
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_run_metadata(args, prompt_hash, torch=None):
    """Write model/device/settings once per run, never once per sample."""
    path = Path(args.output_path)
    metadata_path, _ = sidecar_paths(path)
    runtime = {"python": platform.python_version()}
    if torch is not None:
        runtime["torch"] = torch.__version__
        runtime["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            runtime["devices"] = [
                {"index": i, "name": torch.cuda.get_device_name(i)}
                for i in range(torch.cuda.device_count())
            ]
    session = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": _jsonable(vars(args)),
        "runtime": runtime,
    }
    if getattr(args, "resume", False) and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (metadata.get("schema_version") != OUTPUT_SCHEMA
                or metadata.get("model") != args.model
                or metadata.get("prompt", {}).get("sha256") != prompt_hash
                or metadata.get("requested_revision") != getattr(args, "revision", None)):
            raise ValueError("Caption model, revision, or prompt differs; use a new --output_path.")
        metadata["runs"].append(session)
    else:
        metadata = {
            "schema_version": OUTPUT_SCHEMA,
            "output_format": getattr(args, "output_format",
                                     "json_array" if path.suffix.lower() == ".json" else "jsonl"),
            "model": args.model,
            "requested_revision": getattr(args, "revision", None),
            "caption_schema": args.caption_schema,
            "prompt": {
                "path": str(args.prompt_path.resolve()),
                "sha256": prompt_hash,
                "text": args.prompt_path.read_text(encoding="utf-8"),
            },
            "runs": [session],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(metadata_path)


def diagnostic_record(record):
    # These are sample-specific. Model, hardware and run settings live in metadata.
    keys = ("sample_index", "sample_id", "created_at_utc", "status", "errors", "error",
            "retry_count", "actor_count", "preparation_seconds", "generation_seconds",
            "raw_response", "attempts", "render_input", "render")
    return {key: record[key] for key in keys if key in record}


class CaptionOutput:
    """Flush accepted text per sample; keep each .json or .jsonl sample on one line."""
    def __init__(self, args, prompt_hash, torch=None):
        self.args = args
        self.prompt_hash = prompt_hash
        self.torch = torch
        self.path = Path(args.output_path)
        self.array = self.path.suffix.lower() == ".json"
        self.accepted = set()

    def __enter__(self):
        resume = getattr(self.args, "resume", False)
        if self.path.exists():
            if not resume:
                raise FileExistsError("Output exists; use --resume or a new --output_path: %s" % self.path)
            self.accepted = load_accepted_indices(
                self.path, expected_model=self.args.model, expected_prompt_hash=self.prompt_hash,
                expected_revision=getattr(self.args, "revision", None))
        write_run_metadata(self.args, self.prompt_hash, self.torch)
        _, diagnostics = sidecar_paths(self.path)
        self.audit = diagnostics.open("a" if resume else "w", encoding="utf-8")
        try:
            if self.array:
                self.handle = self.path.open("r+b" if self.path.exists() else "w+b")
                if self.accepted:
                    # The validated array ends in ]; append before its closing whitespace.
                    self.handle.seek(0, 2)
                    position = self.handle.tell() - 1
                    while position >= 0:
                        self.handle.seek(position)
                        if self.handle.read(1) not in b" \r\n\t]":
                            break
                        position -= 1
                    self.offset = position + 1
                else:
                    self.handle.seek(0)
                    self.handle.write(b"[\n]\n")
                    self.handle.truncate()
                    self.handle.flush()
                    self.offset = 2
            else:
                self.handle = self.path.open("ab" if resume else "wb")
                if resume and self.path.stat().st_size:
                    with self.path.open("rb") as source:
                        source.seek(-1, 2)
                        if source.read(1) != b"\n":
                            self.handle.write(b"\n")
                            self.handle.flush()
        except Exception:
            self.audit.close()
            raise
        return self

    def write_record(self, record):
        if record.get("status") == "accepted":
            index = record["sample_index"]
            if index in self.accepted:
                raise ValueError("Duplicate caption sample_index: %d" % index)
            result = {"sample_index": index, "persons": caption_persons(record["caption"])}
            encoded = json.dumps(result, ensure_ascii=False).encode("utf-8")
            if self.array:
                self.handle.seek(self.offset)
                self.handle.write((b",\n" if self.accepted else b"") + encoded)
                self.offset = self.handle.tell()
                self.handle.write(b"\n]\n")
                self.handle.truncate()
            else:
                self.handle.write(encoded + b"\n")
            self.handle.flush()
            self.accepted.add(index)
        self.audit.write(json.dumps(diagnostic_record(record), ensure_ascii=False) + "\n")
        self.audit.flush()

    def __exit__(self, *unused):
        try:
            self.handle.close()
        finally:
            self.audit.close()
