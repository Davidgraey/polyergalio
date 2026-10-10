"""Everything one script run needs, built once per run in ``main.py`` and passed explicitly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from dataset import Dataset
from label_store import LabelStore
from models import BaseModel
from task_defs import Mode, TaskDef


@dataclass
class AppContext:
    """Per-run state shared by every section of the page.

    Attributes
    ----------
    pred_cache : dict
        Model predictions, kept per session.
    notify : callable
        ``notify(task_id, level, text)`` posts a notice on a task tab.
    bump : callable
        ``bump(task_id)`` re-creates a task's widgets from its draft.
    """

    ds: Dataset
    store: LabelStore
    root: str
    tasks: List[TaskDef]
    annotator: str
    mode: Mode
    model_key: str
    model_version: int
    prefill: bool
    model: BaseModel
    issues: List[dict]
    pred_cache: Dict[tuple, Any]
    notify: Callable[[str, str, str], None]
    bump: Callable[[str], None]
