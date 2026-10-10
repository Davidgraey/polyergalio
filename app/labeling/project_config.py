"""Per-project config files (tasks and manifest), written atomically."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from durable_io import atomic_write_json, read_json
from task_defs import DEFAULT_TASKS, TaskDef


def tasks_path(root: str) -> str:
    return os.path.join(root, "tasks.json")


def load_tasks(root: str) -> List[TaskDef]:
    raw = read_json(tasks_path(root), None)
    if not raw:
        tasks = [TaskDef.from_dict(t.to_dict()) for t in DEFAULT_TASKS]
        save_tasks(root, tasks)
        return tasks
    return [TaskDef.from_dict(t) for t in raw]


def save_tasks(root: str, tasks: List[TaskDef]) -> None:
    atomic_write_json(tasks_path(root), [t.to_dict() for t in tasks])


def manifest_path(root: str) -> str:
    return os.path.join(root, "manifest.json")


def load_manifest(root: str) -> Dict[str, Any]:
    return read_json(manifest_path(root), {})


def save_manifest(root: str, manifest: Dict[str, Any]) -> None:
    atomic_write_json(manifest_path(root), manifest)
