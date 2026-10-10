"""Dataset ingestion for the column contract.

Imported ``{task}_labels`` cells become ordinary events in the append-only log
(annotator ``import:<file>``); nothing is imported twice. ``{task}_pred`` cells
are checked here and read by the ``precomputed`` model backend.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from label_events import LabelEvent, validate_label
from prediction import PredFormatError, parse_pred_cell
from task_defs import Mode, TaskDef, TaskType


_VALUE_KEYS = ("label", "labels", "tokens", "spans", "pairs")


def parse_label_cell(task: TaskDef, raw: Any) -> Tuple[Any, float]:
    """Decode a ``{task}_labels`` cell.

    Returns
    -------
    tuple
        ``(value, confidence)``.

    Raises
    ------
    ValueError
        If the cell is unreadable.
    """
    obj = raw
    if isinstance(raw, str):
        text = raw.strip()
        if task.type is TaskType.CLASSIFICATION and not text.startswith(("{", "[")):
            if task.multi_label:
                return {"labels": [x.strip() for x in text.split("|") if x.strip()]}, 1.0
            return {"label": text}, 1.0
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"not valid JSON ({e.msg})") from e
    if task.type is TaskType.CLASSIFICATION and isinstance(obj, list):
        return {"labels": obj}, 1.0
    if not isinstance(obj, dict):
        raise ValueError("expected an object")
    if "value" in obj:
        return obj["value"], obj.get("confidence", 1.0)
    if any(k in obj for k in _VALUE_KEYS):
        return obj, 1.0
    raise ValueError("object has neither 'value' nor a label field")


def import_prelabels(store: Any, ds: Any, tasks: List[TaskDef], annotator: str) -> Dict[str, dict]:
    """Write validated ``{task}_labels`` cells from the dataset into the label log.

    Returns
    -------
    dict
        ``{task_id: {"imported", "existing", "invalid": [(sample_id, why)]}}``.
        Invalid cells are reported and skipped, never written.
    """
    report: Dict[str, dict] = {}
    who = f"import:{ds.name}"
    for t in tasks:
        col = t.labels_column.lower()
        r = {"imported": 0, "existing": 0, "invalid": []}
        for s in ds.samples:
            raw = s.extra.get(col)
            if raw is None:
                continue
            if store.current(s.id, t.id) is not None:
                r["existing"] += 1
                continue
            try:
                value, conf = parse_label_cell(t, raw)
            except ValueError as e:
                r["invalid"].append((s.id, str(e)))
                continue
            errs = validate_label(t, s.text, value, conf)
            if t.type is TaskType.CONTRASTIVE and isinstance(value, dict):
                errs += [f"candidate {p.get('sample_id')!r} not in dataset" for p in value.get("pairs", [])
                         if p.get("sample_id") not in ds.by_id]
            if errs:
                r["invalid"].append((s.id, " ".join(errs[:3])))
                continue
            store.append(LabelEvent(sample_id=s.id, task_id=t.id, task_type=t.type.value, value=value,
                                    confidence=float(conf), annotator=who, mode=Mode.INITIAL.value,
                                    note=f"imported from column {col} (annotator {annotator or 'unknown'})"))
            r["imported"] += 1
        report[t.id] = r
    return report


def pred_column_report(ds: Any, tasks: List[TaskDef]) -> Dict[str, dict]:
    """Check every ``{task}_pred`` cell present in the dataset. Contrastive cells aren't shape-checked."""
    out: Dict[str, dict] = {}
    for t in tasks:
        col = t.pred_column.lower()
        ok, bad = 0, []
        for s in ds.samples:
            raw = s.extra.get(col)
            if raw is None:
                continue
            try:
                parse_pred_cell(t, raw, s, [])
                ok += 1
            except PredFormatError as e:
                bad.append((s.id, str(e)))
        out[t.id] = {"ok": ok, "bad": bad, "missing": len(ds.samples) - ok - len(bad)}
    return out
