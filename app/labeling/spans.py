"""Pure helpers for token tags and span selections (no UI)."""

from __future__ import annotations

from typing import Any, List, Optional

from task_defs import OUTSIDE_TAG


def tags_to_spans(tokens: List[dict]) -> List[dict]:
    """Group contiguous non-O token tags (BIO-aware) into character spans."""
    spans: List[dict] = []
    for tk in tokens:
        tag = tk.get("tag") or OUTSIDE_TAG
        if tag == OUTSIDE_TAG:
            continue
        base = tag[2:] if tag[:2] in ("B-", "I-") else tag
        prev = spans[-1] if spans else None
        if prev and prev["label"] == base and prev["_last"] == tk["i"] - 1 and not tag.startswith("B-"):
            prev["end"], prev["_last"] = tk["end"], tk["i"]
        else:
            spans.append({"start": tk["start"], "end": tk["end"], "label": base, "_last": tk["i"]})
    return spans


def normalize_range(value: Any, size: int) -> tuple:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        a, b = sorted(int(v) for v in value)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        a = b = int(value)
    else:
        a = b = 0
    last = max(size - 1, 0)
    return min(max(a, 0), last), min(max(b, 0), last)


def add_span(existing: List[dict], span: dict, skip: Optional[int] = None) -> List[dict]:
    """Insert ``span`` into ``existing`` (optionally replacing index ``skip``), ordered by position.

    Overlapping and nested spans are allowed; only an exact repeat (same offsets and label) is rejected.
    """
    kept = [sp for i, sp in enumerate(existing) if i != skip]
    if any((sp["start"], sp["end"], sp["label"]) == (span["start"], span["end"], span["label"]) for sp in kept):
        raise ValueError("That exact span and label already exists.")
    return sorted([*kept, span], key=lambda sp: (sp["start"], sp["end"]))
