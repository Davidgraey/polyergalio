"""The task/column contract: reserved names, conflict checks and the per-task definition table."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from task_defs import TaskDef


RESERVED = {"id", "sample_id", "text"}


def column_conflicts(tasks: List[TaskDef]) -> List[str]:
    """Problems that would make two tasks share a column or shadow id/text."""
    seen: Dict[str, str] = {}
    problems = []
    for t in tasks:
        for col in (t.labels_column.lower(), t.pred_column.lower()):
            if col in RESERVED:
                problems.append(f"{t.name}: column {col!r} is reserved")
            elif col in seen and seen[col] != t.id:
                problems.append(f"{t.name}: column {col!r} is already used by task {seen[col]!r}")
            seen[col] = t.id
    return problems


def definitions_frame(tasks: List[TaskDef], ds: Any = None) -> pd.DataFrame:
    """One row per task: type, labels, column names and how many dataset rows carry each column."""
    rows = []
    for t in tasks:
        row = {"task": t.name, "id": t.id, "type": t.type.value, "labels": ", ".join(t.labels),
               "defined": f"{len(t.labels) - len(t.missing_definitions())}/{len(t.labels)}",
               "rubric": bool(t.rubric.strip()),
               "labels_column": t.labels_column, "pred_column": t.pred_column}
        if ds is not None:
            row["labels in dataset"] = sum(1 for s in ds.samples if t.labels_column.lower() in s.extra)
            row["preds in dataset"] = sum(1 for s in ds.samples if t.pred_column.lower() in s.extra)
        rows.append(row)
    return pd.DataFrame(rows)
