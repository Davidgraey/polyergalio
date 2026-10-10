"""Aggregate statistics over saved labels."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from task_defs import OUTSIDE_TAG, TaskDef, TaskType


def distribution(task: TaskDef, labeled: dict) -> pd.DataFrame:
    counts: Counter = Counter()
    for rec in labeled.values():
        v = rec["value"]
        if task.type is TaskType.CLASSIFICATION:
            counts.update(v.get("labels") if "labels" in v else [v.get("label")])
        elif task.type is TaskType.TOKEN:
            counts.update(t["tag"] for t in v["tokens"] if t["tag"] != OUTSIDE_TAG)
        elif task.type is TaskType.SPAN:
            counts.update(s["label"] for s in v["spans"])
        else:
            counts.update(p["relation"] for p in v["pairs"])
    names = list(dict.fromkeys([*(l for l in task.labels if l != OUTSIDE_TAG), *counts]))
    return pd.DataFrame({"name": names, "count": [counts[n] for n in names]})
