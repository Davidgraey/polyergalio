"""Creating and editing task definitions (validation only; persistence and UI live elsewhere)."""

from __future__ import annotations

from typing import List, Optional, Tuple

from columns import column_conflicts
from task_defs import OUTSIDE_TAG, TaskDef, TaskType
from util import slugify


DEFAULT_DEFINITIONS = {OUTSIDE_TAG: "Not part of any labeled item.",
                       "similar": "Describes the same underlying issue or topic as the anchor.",
                       "dissimilar": "Describes a different issue or topic from the anchor."}


def parse_definitions(text: str) -> dict:
    """``label: definition`` lines -> {label: definition}. Later lines for a label win."""
    out = {}
    for line in (text or "").splitlines():
        label, sep, meaning = line.partition(":")
        if sep and label.strip() and meaning.strip():
            out[label.strip()] = meaning.strip()
    return out


def build_task(existing, name, ttype, labels_str, multi, n_cand, rubric, label_col="", pred_col="",
               definitions_str=""):
    """Validate the add-task form.

    Returns
    -------
    tuple
        ``(task, message)``; ``task`` is None when the form is invalid.
    """
    name = (name or "").strip()
    if not name:
        return None, "Task name is required."
    tt = TaskType(ttype)
    labels = []
    for x in (labels_str or "").split(","):
        x = x.strip()
        if x and x not in labels:
            labels.append(x)
    if tt is TaskType.TOKEN and OUTSIDE_TAG not in labels:
        labels.insert(0, OUTSIDE_TAG)
    if tt is TaskType.SPAN:
        labels = [x for x in labels if x != OUTSIDE_TAG]
    if tt is TaskType.CONTRASTIVE and not labels:
        labels = ["similar", "dissimilar"]
    need = 1 if tt is TaskType.SPAN else 2
    if len(labels) < need:
        return None, f"Provide at least {need} label(s)."
    given = parse_definitions(definitions_str)
    definitions = {x: given.get(x) or DEFAULT_DEFINITIONS.get(x, "") for x in labels}
    missing = [x for x in labels if not definitions[x]]
    if missing:
        return None, ("Every label needs a definition. Add one line per label as `label: definition`. "
                      "Missing: " + ", ".join(missing))
    ids = {t.id for t in existing}
    base = slugify(name, "task")
    tid, n = base, 1
    while tid in ids:
        n += 1
        tid = f"{base}_{n}"
    task = TaskDef(tid, name, tt, labels, multi_label=bool(multi) and tt is TaskType.CLASSIFICATION,
                   n_candidates=int(n_cand or 4), rubric=(rubric or "").strip(), label_definitions=definitions,
                   label_col=(label_col or "").strip(), pred_col=(pred_col or "").strip())
    problems = column_conflicts([*existing, task])
    if problems:
        return None, "Column name problem: " + "; ".join(problems)
    return task, f"Added task {name}."


def apply_task_edit(task: TaskDef, tasks: List[TaskDef], *, name: str, rubric: str, add_labels: str,
                    definitions_str: str, label_col: str, pred_col: str,
                    n_candidates) -> Tuple[Optional[TaskDef], str, int]:
    """Validate an edit of ``task``.

    Existing labels can't be renamed or removed (saved labels refer to them); new labels can be added.

    Returns
    -------
    tuple
        ``(updated_task, error, n_new_labels)``; ``updated_task`` is None when the edit is invalid.
    """
    name = (name or "").strip()
    if not name:
        return None, "Task name is required.", 0
    new_labels: List[str] = []
    for x in (add_labels or "").split(","):
        x = x.strip()
        if x and x not in task.labels and x not in new_labels:
            new_labels.append(x)
    if task.type is TaskType.SPAN and OUTSIDE_TAG in new_labels:
        return None, f"{OUTSIDE_TAG} is reserved for token tasks.", 0
    labels = [*task.labels, *new_labels]
    given = parse_definitions(definitions_str)
    definitions = {x: given.get(x) or task.label_definitions.get(x) or DEFAULT_DEFINITIONS.get(x, "")
                   for x in labels}
    missing = [x for x in labels if not definitions[x]]
    if missing:
        return None, "Every label needs a definition (`label: definition`). Missing: " + ", ".join(missing), 0
    updated = TaskDef.from_dict({**task.to_dict(), "name": name, "labels": labels,
                                 "label_definitions": definitions, "rubric": (rubric or "").strip(),
                                 "label_col": (label_col or "").strip(), "pred_col": (pred_col or "").strip(),
                                 "n_candidates": int(n_candidates or task.n_candidates)})
    problems = column_conflicts([updated if t.id == task.id else t for t in tasks])
    if problems:
        return None, "Column name problem: " + "; ".join(problems), 0
    return updated, "", len(new_labels)
