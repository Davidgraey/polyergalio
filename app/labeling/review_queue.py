"""Which samples to show next: candidates, cached model predictions, uncertainty scoring and queue ranking."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

import uncertainty as unc
from task_defs import Mode, TaskDef, TaskType


RANK_MAX, RANK_MEAN = "Most uncertain task", "Average across tasks"


def stable_seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def labeled_for(app, task: TaskDef) -> dict:
    return {sid: r for sid, r in app.store.current_for_task(task.id).items() if sid in app.ds.by_id}


def candidates_for(app, task: TaskDef, anchor):
    """Stable candidate set for a contrastive anchor."""
    if task.type is not TaskType.CONTRASTIVE:
        return None
    others = [s for s in app.ds.samples if s.id != anchor.id]
    if not others:
        return []
    k = min(task.n_candidates, len(others))
    pool_n = min(len(others), k * (5 if app.mode is Mode.RELABEL else 1))
    rng = np.random.default_rng(stable_seed(anchor.id, task.id))
    pool = [others[i] for i in rng.choice(len(others), size=pool_n, replace=False)]
    if app.mode is Mode.RELABEL and pool_n > k:
        p = app.model.predict(task, anchor, candidates=pool).probs
        pool = [pool[i] for i in np.argsort(np.abs(p - 0.5))]
    return pool[:k]


def fixed_candidates(app, task: TaskDef, sample):
    """Candidates from a draft or saved label win, so pairs never drift."""
    if task.type is not TaskType.CONTRASTIVE:
        return None
    source = app.store.get_draft(f"{task.id}::{sample.id}") or app.store.current(sample.id, task.id)
    if source:
        return [app.ds.by_id[p["sample_id"]] for p in source["value"]["pairs"] if p["sample_id"] in app.ds.by_id]
    return candidates_for(app, task, sample)


def task_predictions(app, task: TaskDef) -> dict:
    ck = (app.ds.sha256, task.id, app.model_key, app.model_version, app.mode.value)
    cache = app.pred_cache
    if ck not in cache:
        cache[ck] = {s.id: app.model.predict(task, s, candidates=candidates_for(app, task, s))
                     for s in app.ds.samples}
        for old in list(cache)[:-8]:
            cache.pop(old)
    return cache[ck]


def score_frame(app, tasks) -> pd.DataFrame:
    """One row per (sample, task): model scores plus human-signal scores."""
    rows = []
    for task in tasks:
        preds, labeled = task_predictions(app, task), labeled_for(app, task)
        for s in app.ds.samples:
            pred, rec = preds[s.id], labeled.get(s.id)
            row = {"sample_id": s.id, "task_id": task.id, "task": task.name, "labeled": rec is not None,
                   "confidence": rec["confidence"] if rec else None, **unc.all_scores(pred)}
            for h in (unc.LOW_CONF, unc.DISAGREE):
                row[h] = unc.human_score(h, pred, rec) if h in unc.human_strategies(task) else None
            rows.append(row)
    return pd.DataFrame(rows)


def sample_queue(app, tasks, opts: dict):
    """Rank samples by aggregating their per-task scores.

    Returns
    -------
    tuple
        ``(order, scores, top_task)``: sample ids, their scores, and the task driving each score.
    """
    df = score_frame(app, tasks)
    strategy, rank = opts["strategy"], opts["rank"]
    if rank not in (RANK_MAX, RANK_MEAN):
        df = df[df["task_id"] == rank]
    if opts["scope"] == "Labeled":
        df = df[df["labeled"]]
    elif opts["scope"] == "Unlabeled":
        df = df[~df["labeled"]]
    df = df.dropna(subset=[strategy])
    if df.empty:
        return [], {}, {}
    df = df.astype({strategy: float})
    grouped = df.groupby("sample_id", sort=False)[strategy]
    scores = grouped.mean() if rank == RANK_MEAN else grouped.max()
    top = df.loc[grouped.idxmax()].set_index("sample_id")["task_id"].to_dict()
    scores = scores[scores >= opts["min_score"]].sort_values(ascending=False, kind="stable").head(opts["size"])
    return list(scores.index), scores.to_dict(), top


def needs_work(app, task: TaskDef, sid: str) -> bool:
    """True while a task has no saved label, or has unsaved edits, for a sample."""
    return app.store.current(sid, task.id) is None or app.store.get_draft(f"{task.id}::{sid}") is not None
