"""Draft, status and save logic for one task on one sample (no UI; notifications go through app hooks)."""

from __future__ import annotations

import copy

import numpy as np

import uncertainty as unc
from label_events import LabelEvent, validate_label
from review_queue import fixed_candidates
from task_defs import OUTSIDE_TAG, TaskDef, TaskType
from tokenizing import TOKENIZER_ID, tokenize


def seed_draft(task: TaskDef, sample, cur, pred, prefill: bool, candidates) -> dict:
    """Starting draft: the saved label, a model pre-fill, or a blank label."""
    if cur:
        return {"value": copy.deepcopy(cur["value"]), "confidence": float(cur["confidence"])}
    t = task.type
    if t is TaskType.CLASSIFICATION:
        if task.multi_label:
            labels = [l for l, p in zip(pred.labels, pred.probs) if p >= 0.5] if prefill else []
            return {"value": {"labels": labels}, "confidence": 1.0}
        label = pred.labels[int(np.argmax(pred.probs))] if prefill else None
        return {"value": {"label": label}, "confidence": 1.0}
    if t is TaskType.TOKEN:
        default = OUTSIDE_TAG if OUTSIDE_TAG in task.labels else task.labels[0]
        tags = pred.argmax_labels() if prefill else []
        return {"value": {"tokenizer": TOKENIZER_ID, "tokens": [
            {"i": i, "text": w, "start": s, "end": e,
             "tag": tags[i] if i < len(tags) else default, "confidence": 1.0}
            for i, (w, s, e) in enumerate(tokenize(sample.text))]}, "confidence": 1.0}
    if t is TaskType.SPAN:
        spans = [{k: v for k, v in sp.items() if k != "model_prob"} | {"confidence": 1.0}
                 for sp in pred.suggested_spans(sample.text)] if prefill else []
        return {"value": {"tokenizer": TOKENIZER_ID, "spans": spans}, "confidence": 1.0}
    pairs = []
    for j, c in enumerate(candidates or []):
        rel = None
        if prefill and j < len(pred.probs):
            rel = task.labels[0] if pred.probs[j] >= 0.5 else task.labels[min(1, len(task.labels) - 1)]
        pairs.append({"sample_id": c.id, "relation": rel, "confidence": 1.0})
    return {"value": {"anchor_id": sample.id, "pairs": pairs}, "confidence": 1.0}


def finalize(task: TaskDef, draft: dict):
    """Label value and event confidence (mean of item confidences where items exist)."""
    value = copy.deepcopy(draft["value"])
    items = {
        TaskType.TOKEN: lambda: [t["confidence"] for t in value["tokens"]],
        TaskType.SPAN: lambda: [s["confidence"] for s in value["spans"]],
        TaskType.CONTRASTIVE: lambda: [p["confidence"] for p in value["pairs"]],
    }.get(task.type, lambda: [])()
    conf = float(np.mean(items)) if items else float(draft.get("confidence", 1.0))
    return value, round(conf, 4)


class TaskSession:
    """Draft, status and save logic for one task on the current sample."""

    def __init__(self, app, task: TaskDef, sample):
        self.app, self.task, self.sample = app, task, sample
        store = app.store
        self.cur = store.current(sample.id, task.id)
        self.candidates = fixed_candidates(app, task, sample)
        self.pred = app.model.predict(task, sample, candidates=self.candidates)
        self.scores = unc.all_scores(self.pred)
        self.draft_key = f"{task.id}::{sample.id}"
        self.stored = store.get_draft(self.draft_key)
        self.draft = self.stored if self.stored is not None else self.seed()
        if self.stored is None:
            self.status = "saved" if self.cur else "unsaved"
        elif self.cur and finalize(task, self.stored) == (self.cur["value"], self.cur["confidence"]):
            self.status = "saved"
        else:
            self.status = "edited" if self.cur else "draft"

    def seed(self) -> dict:
        return seed_draft(self.task, self.sample, self.cur, self.pred, self.app.prefill, self.candidates)

    def has_draft(self) -> bool:
        return self.app.store.get_draft(self.draft_key) is not None

    def edit(self, fn, bump: bool = False) -> None:
        """Apply ``fn`` to a copy of the draft and persist it; a ValueError from ``fn`` leaves the draft untouched."""
        d = copy.deepcopy(self.app.store.get_draft(self.draft_key) or self.seed())
        fn(d)
        try:
            self.app.store.save_draft(self.draft_key, d)
        except OSError as error:
            self.app.notify(self.task.id, "error", f"Draft could not be written: {error}")
            return
        if bump:
            self.app.bump(self.task.id)

    def commit(self, strategy=None) -> list:
        """Validate and durably append this task's draft.

        Returns
        -------
        list of str
            Validation or write errors; empty on success.
        """
        app, task, sample = self.app, self.task, self.sample
        value, conf = finalize(task, app.store.get_draft(self.draft_key) or self.seed())
        errs = validate_label(task, sample.text, value, conf)
        if task.type is TaskType.CONTRASTIVE:
            errs += [f"Unknown candidate {p['sample_id']}." for p in value.get("pairs", [])
                     if p["sample_id"] not in app.ds.by_id]
        if not app.annotator:
            errs.insert(0, "Set an annotator name in the sidebar.")
        if errs:
            return list(dict.fromkeys(errs))
        strat = strategy if strategy in unc.MODEL_STRATEGIES else "Entropy"
        cur = app.store.current(sample.id, task.id)
        try:
            app.store.append(LabelEvent(
                sample_id=sample.id, task_id=task.id, task_type=task.type.value, value=value,
                confidence=conf, annotator=app.annotator, mode=app.mode.value, model_key=app.model_key,
                model_version=app.model_version, model_score=round(self.scores[strat], 4),
                supersedes=cur["event_id"] if cur else None,
            ))
        except OSError as error:
            return [f"Write failed, this label was not saved: {error}"]
        try:
            app.store.clear_draft(self.draft_key)
        except OSError:
            pass
        return []

    def discard(self) -> None:
        self.app.store.clear_draft(self.draft_key)
        self.app.bump(self.task.id)

    def remove(self) -> None:
        self.app.store.retract(self.sample.id, self.task.id, self.task.type.value,
                               self.app.annotator, self.app.mode.value)
        self.app.store.clear_draft(self.draft_key)
        self.app.bump(self.task.id)
