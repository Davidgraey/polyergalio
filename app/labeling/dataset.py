"""Text dataset loading.

Accepted formats (UTF-8):
* CSV  with a ``text`` column and an optional ``id`` (or ``sample_id``) column.
* JSONL with one object per line containing ``text`` and optional ``id``.

Integrity rules: ids must be unique, malformed JSONL lines are a hard error
(never silently dropped), and the file's SHA-256 is recorded so the app can
warn when a dataset changes underneath existing labels. Text is kept exactly
as stored — label offsets depend on it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List

from util import slugify

csv.field_size_limit(10 * 1024 * 1024)

SUPPORTED_EXT = (".csv", ".jsonl")


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class Sample:
    """One dataset row; ``extra`` holds every other column, lower-cased (e.g. ``{task}_labels``)."""

    id: str
    text: str
    extra: Dict[str, object] = field(default_factory=dict, compare=False)


@dataclass
class Dataset:
    name: str
    path: str
    sha256: str
    samples: List[Sample]
    warnings: List[str] = field(default_factory=list)
    ids_generated: bool = False

    def __post_init__(self) -> None:
        self.by_id: Dict[str, Sample] = {s.id: s for s in self.samples}
        self.index: Dict[str, int] = {s.id: i for i, s in enumerate(self.samples)}

    @property
    def project_id(self) -> str:
        return slugify(os.path.splitext(self.name)[0], "dataset")


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _has_text_column(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            first = fh.readline()
        if path.lower().endswith(".csv"):
            header = next(csv.reader([first]), [])
            return "text" in [h.strip().lower() for h in header]
        obj = json.loads(first)
        return isinstance(obj, dict) and "text" in obj
    except Exception:
        return False


def list_datasets(directory: str) -> List[str]:
    """Dataset files in ``directory`` that look like text datasets."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        f for f in os.listdir(directory)
        if f.lower().endswith(SUPPORTED_EXT) and _has_text_column(os.path.join(directory, f))
    )


def _rows_csv(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise DatasetError("CSV has no header row.")
        norm = {h: (h or "").strip().lower() for h in reader.fieldnames}
        return [{norm[k]: v for k, v in row.items() if k is not None} for row in reader]


def _rows_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8-sig") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetError(f"Line {lineno}: invalid JSON ({e.msg}).") from e
            if not isinstance(obj, dict):
                raise DatasetError(f"Line {lineno}: expected an object.")
            rows.append({str(k).lower(): v for k, v in obj.items()})
    return rows


def load_dataset(path: str) -> Dataset:
    if not os.path.exists(path):
        raise DatasetError(f"File not found: {path}")
    rows = _rows_csv(path) if path.lower().endswith(".csv") else _rows_jsonl(path)
    if rows and "text" not in rows[0]:
        raise DatasetError("Dataset needs a 'text' column.")

    id_key = "id" if rows and "id" in rows[0] else ("sample_id" if rows and "sample_id" in rows[0] else None)
    samples: List[Sample] = []
    seen: Dict[str, int] = {}
    skipped = 0
    for n, row in enumerate(rows):
        text = row.get("text")
        text = "" if text is None else str(text)
        if not text.strip():
            skipped += 1
            continue
        sid = str(row.get(id_key)).strip() if id_key and row.get(id_key) not in (None, "") else f"row{n:06d}"
        if sid in seen:
            raise DatasetError(f"Duplicate sample id {sid!r} (rows {seen[sid] + 1} and {n + 1}).")
        seen[sid] = n
        extra = {k: v for k, v in row.items() if k not in ("text", id_key) and v not in (None, "")}
        samples.append(Sample(sid, text, extra))

    warnings = []
    if skipped:
        warnings.append(f"Skipped {skipped} row(s) with empty text.")
    if not id_key:
        warnings.append(
            "No 'id' column: ids are derived from row order. Re-ordering or "
            "inserting rows will misalign existing labels."
        )
    if not samples:
        raise DatasetError("Dataset contains no non-empty text samples.")
    return Dataset(
        name=os.path.basename(path), path=path, sha256=file_sha256(path),
        samples=samples, warnings=warnings, ids_generated=id_key is None,
    )


def save_upload(directory: str, filename: str, data: bytes) -> str:
    """Atomically save an uploaded dataset without overwriting existing files."""
    from durable_io import atomic_write_bytes

    os.makedirs(directory, exist_ok=True)
    stem, ext = os.path.splitext(os.path.basename(filename))
    ext = ext.lower()
    if ext not in SUPPORTED_EXT:
        raise DatasetError("Upload a .csv or .jsonl file.")
    stem = slugify(stem, "dataset")
    path = os.path.join(directory, stem + ext)
    n = 1
    while os.path.exists(path):
        if hashlib.sha256(data).hexdigest() == file_sha256(path):
            return path
        n += 1
        path = os.path.join(directory, f"{stem}_{n}{ext}")
    atomic_write_bytes(path, data)
    try:
        load_dataset(path)
    except DatasetError:
        os.remove(path)
        raise
    return path


__all__ = ["Sample", "Dataset", "DatasetError", "list_datasets", "load_dataset",
           "save_upload", "file_sha256"]
