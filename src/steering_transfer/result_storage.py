"""Immutable run settings and resumable, identity-checked records."""

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path

import pandas as pd


def content_hash(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_records(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    records = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid record at {path}:{line_number}") from error
    return pd.DataFrame(records)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def condition_settings(condition):
    settings = {
        "name": condition.name,
        "method": condition.method,
        "layers": list(condition.layers),
        "application_locations": list(condition.application_locations),
        "strength": condition.strength,
        "token_positions": condition.token_positions,
    }
    if condition.direction is not None:
        values = condition.direction.values.detach().float().cpu().tolist()
        settings["direction_hash"] = content_hash(values)
        settings["direction_provenance"] = condition.direction.provenance
    return settings


def prepare_run(directory, settings, record_filename, expected_ids):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    expected_ids = list(expected_ids)
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Expected record identities are not unique")
    manifest = {
        "schema_version": 1,
        "settings": settings,
        "settings_hash": content_hash(settings),
        "expected_records": len(expected_ids),
        "expected_ids_hash": content_hash(sorted(expected_ids)),
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        for key in ["settings_hash", "expected_records", "expected_ids_hash"]:
            if existing[key] != manifest[key]:
                raise ValueError(
                    f"Run settings changed: {key}; choose a new output directory"
                )
    else:
        write_json(manifest_path, manifest)
    record_path = directory / record_filename
    records = read_records(record_path) if record_path.exists() else pd.DataFrame()
    if not records.empty:
        if records.record_id.duplicated().any():
            raise ValueError("Duplicate saved record IDs")
        if not set(records.record_id).issubset(expected_ids):
            raise ValueError("Saved records do not belong to this run")
    return record_path, records


def append_records(path, records):
    with Path(path).open("a") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def finish_run(directory, records):
    path = Path(directory) / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["completed_records"] = len(records)
    successful = len(records)
    if "status" in records:
        successful = int(records.status.eq("success").sum())
    manifest["successful_records"] = successful
    manifest["complete"] = successful == manifest["expected_records"]
    write_json(path, manifest)


def settings_dict(settings):
    return asdict(settings) if is_dataclass(settings) else dict(settings)


def load_experiment(root, name):
    path = Path(root) / "configs/experiments" / f"{name}.json"
    return json.loads(path.read_text())


def merge_worker_records(paths, output_path, expected_ids):
    frames = [read_records(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    if combined.record_id.duplicated().any():
        raise ValueError("Workers produced duplicate record IDs")
    missing = set(expected_ids) - set(combined.record_id)
    extra = set(combined.record_id) - set(expected_ids)
    if missing or extra:
        raise ValueError(
            f"Worker merge mismatch: {len(missing)} missing, {len(extra)} extra"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    append_records(output_path, combined.sort_values("record_id").to_dict("records"))
    return combined


def find_project_root(start=None):
    start = Path.cwd() if start is None else Path(start).resolve()
    for directory in [start, *start.parents]:
        if (directory / "pyproject.toml").exists() and (
            directory / "src/steering_transfer"
        ).is_dir():
            return directory
    raise FileNotFoundError("Cannot find the steering-transfer project root")


@contextmanager
def exclusive_run(directory):
    """One writer per output directory; parallel workers need separate folders."""
    import fcntl

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another worker owns this output directory: {directory}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
