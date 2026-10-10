"""Exports: the wide per-sample frame (``{task}_labels`` / ``{task}_pred``) and the provenance JSONL."""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Dict, List, Optional

import pandas as pd

from columns import column_conflicts
from label_events import summarize_value
from label_store import LabelStore
from task_defs import TaskDef


def export_current_jsonl(store: LabelStore, samples: List[Any], tasks: List[TaskDef]) -> str:
    lines = []
    for s in samples:
        row: Dict[str, Any] = {"sample_id": s.id, "text": s.text, "labels": {}}
        for t in tasks:
            rec = store.current(s.id, t.id)
            if rec:
                row["labels"][t.id] = {
                    "task_type": t.type.value, "value": rec["value"],
                    "confidence": rec["confidence"], "annotator": rec["annotator"],
                    "ts": rec["ts"], "event_id": rec["event_id"],
                }
        if row["labels"]:
            lines.append(json.dumps(row, ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else "")


def export_current_csv(store: LabelStore, samples: List[Any], tasks: List[TaskDef]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    header = ["sample_id", "text"]
    for t in tasks:
        header += [f"{t.id}__value", f"{t.id}__confidence", f"{t.id}__summary"]
    w.writerow(header)
    for s in samples:
        recs = [store.current(s.id, t.id) for t in tasks]
        if not any(recs):
            continue
        row = [s.id, s.text]
        for t, rec in zip(tasks, recs):
            if rec:
                row += [json.dumps(rec["value"], ensure_ascii=False), rec["confidence"],
                        summarize_value(t.type.value, rec["value"])]
            else:
                row += ["", "", ""]
        w.writerow(row)
    return buf.getvalue()


def format_label_cell(value: Any, confidence: float) -> str:
    return json.dumps({"value": value, "confidence": float(confidence)}, ensure_ascii=False)


def _pred_cell(pred: Any) -> Optional[str]:
    if pred is None:
        return None
    probs = [[round(float(x), 6) for x in row] for row in pred.probs] if pred.probs.ndim == 2 \
        else [round(float(x), 6) for x in pred.probs]
    if pred.candidate_ids is not None:
        probs = {"candidate_ids": list(pred.candidate_ids), "probs": probs}
    return json.dumps(probs)


def build_frame(store: Any, ds: Any, tasks: List[TaskDef], preds: Optional[Dict[str, dict]] = None) -> pd.DataFrame:
    """One row per sample: ``id``, ``text``, then ``{task}_labels`` and (if ``preds``) ``{task}_pred``.

    ``preds`` maps task id -> {sample_id: Prediction}; predictions are exported as probabilities.
    """
    conflicts = column_conflicts(tasks)
    if conflicts:
        raise ValueError("; ".join(conflicts))
    rows = []
    for s in ds.samples:
        row: Dict[str, Any] = {"id": s.id, "text": s.text}
        for t in tasks:
            rec = store.current(s.id, t.id)
            row[t.labels_column] = format_label_cell(rec["value"], rec["confidence"]) if rec else None
            if preds is not None:
                row[t.pred_column] = _pred_cell(preds.get(t.id, {}).get(s.id))
        rows.append(row)
    return pd.DataFrame(rows)
