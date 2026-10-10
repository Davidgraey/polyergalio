"""Re-validate every current label against the dataset and the task definitions."""

from __future__ import annotations

from typing import Any, Dict, List

from label_events import validate_label
from label_store import LabelStore
from task_defs import TaskDef, TaskType


def integrity_check(store: LabelStore, samples_by_id: Dict[str, Any],
                    tasks_by_id: Dict[str, TaskDef]) -> List[Dict[str, str]]:
    """Re-validate every current label against the dataset and task defs."""
    issues = []
    for rec in store.all_current():
        sid, tid = rec["sample_id"], rec["task_id"]
        sample = samples_by_id.get(sid)
        task = tasks_by_id.get(tid)
        if sample is None:
            issues.append({"sample_id": sid, "task_id": tid, "issue": "sample not in dataset"})
            continue
        if task is None:
            issues.append({"sample_id": sid, "task_id": tid, "issue": "unknown task"})
            continue
        for e in validate_label(task, sample.text, rec["value"], rec["confidence"]):
            issues.append({"sample_id": sid, "task_id": tid, "issue": e})
        if task.type is TaskType.CONTRASTIVE and isinstance(rec["value"], dict):
            for p in rec["value"].get("pairs", []):
                if p.get("sample_id") not in samples_by_id:
                    issues.append({"sample_id": sid, "task_id": tid,
                                   "issue": f"candidate {p.get('sample_id')!r} not in dataset"})
    return issues
