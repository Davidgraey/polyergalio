"""Model output: the ``Prediction`` container, and parsing of raw ``{task}_pred`` cells.

``probs`` shapes by task type:
    classification (single) : (k,)        distribution over task.labels
    classification (multi)  : (k,)        independent P(label present)
    token                   : (n_tok, k)  distribution over task.labels per token
    span                    : (n_tok, k+1) per-token distribution over ["O"] + labels
    contrastive             : (n_cand,)   P(candidate is `labels[0]`, e.g. similar)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np

from task_defs import OUTSIDE_TAG, TaskDef, TaskType
from tokenizing import tokenize


@dataclass
class Prediction:
    """A model's output for one sample; ``probs`` shapes are listed in the module docstring.

    Attributes
    ----------
    labels : list of str
        Column names of ``probs``.
    tokens : list, optional
        ``(text, start, end)`` per token, for token and span tasks.
    """

    task_type: TaskType
    labels: List[str]
    probs: np.ndarray
    multi_label: bool = False
    tokens: Optional[list] = None
    candidate_ids: Optional[List[str]] = None

    def distributions(self) -> np.ndarray:
        """Rows of probability distributions, for uncertainty scoring."""
        p = np.asarray(self.probs, dtype=float)
        if self.task_type is TaskType.CLASSIFICATION:
            if self.multi_label:
                return np.stack([1.0 - p, p], axis=1)
            return p[None, :]
        if self.task_type is TaskType.CONTRASTIVE:
            return np.stack([1.0 - p, p], axis=1) if p.size else np.zeros((0, 2))
        return p if p.ndim == 2 else np.zeros((0, max(1, len(self.labels))))

    def argmax_labels(self) -> List[str]:
        if self.probs.ndim != 2:
            return []
        return [self.labels[int(i)] for i in np.argmax(self.probs, axis=1)]

    def suggested_spans(self, text: str) -> List[dict]:
        """Contiguous tokens whose top label isn't 'O' (span tasks)."""
        if self.task_type is not TaskType.SPAN or not self.tokens:
            return []
        tags = self.argmax_labels()
        spans, i = [], 0
        while i < len(tags):
            if tags[i] == OUTSIDE_TAG:
                i += 1
                continue
            j = i
            while j + 1 < len(tags) and tags[j + 1] == tags[i]:
                j += 1
            s, e = self.tokens[i][1], self.tokens[j][2]
            p = float(np.mean([self.probs[t].max() for t in range(i, j + 1)]))
            spans.append({"start": s, "end": e, "text": text[s:e], "label": tags[i],
                          "token_start": i, "token_end": j, "model_prob": round(p, 3)})
            i = j + 1
        return spans


class PredFormatError(ValueError):
    pass


def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


def _as_probs(x: np.ndarray, multi: bool = False) -> np.ndarray:
    """Accept probabilities or logits. Rows already in [0, 1] (and summing to 1 unless multi) pass through."""
    if not np.all(np.isfinite(x)):
        raise PredFormatError("prediction contains NaN or infinite values")
    in_unit = bool(x.size) and x.min() >= 0.0 and x.max() <= 1.0
    if multi:
        return x if in_unit else 1.0 / (1.0 + np.exp(-x))
    if in_unit and np.allclose(x.sum(axis=-1), 1.0, atol=1e-3):
        return x
    return _softmax(x)


def parse_pred_cell(task: TaskDef, raw: Any, sample: Any, candidates: Optional[Sequence[Any]] = None) -> Prediction:
    """Turn a stored raw model output into a ``Prediction`` (shapes as in the module docstring).

    ``raw`` is a JSON string or an already-decoded list/dict. Contrastive output is
    ``{"candidate_ids": [...], "probs": [...]}``; candidates it doesn't cover get 0.5.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as e:
            raise PredFormatError(f"not valid JSON ({e.msg})") from e
    k = len(task.labels)
    try:
        if task.type is TaskType.CLASSIFICATION:
            a = np.asarray(raw, dtype=float)
            if a.shape != (k,):
                raise PredFormatError(f"expected {k} values (one per label), got shape {a.shape}")
            return Prediction(task.type, list(task.labels), _as_probs(a, task.multi_label),
                              multi_label=task.multi_label)
        if task.type in (TaskType.TOKEN, TaskType.SPAN):
            toks = tokenize(sample.text)
            cols = list(task.labels) if task.type is TaskType.TOKEN else [OUTSIDE_TAG] + list(task.labels)
            a = np.asarray(raw, dtype=float).reshape(len(toks), len(cols)) if toks else np.zeros((0, len(cols)))
            if np.asarray(raw).size != a.size:
                raise PredFormatError(f"expected {len(toks)} tokens x {len(cols)} columns")
            return Prediction(task.type, cols, _as_probs(a) if toks else a, tokens=toks)
        if task.type is TaskType.CONTRASTIVE:
            ids, probs = list(raw["candidate_ids"]), np.asarray(raw["probs"], dtype=float)
            if probs.shape != (len(ids),):
                raise PredFormatError("candidate_ids and probs must have the same length")
            probs = _as_probs(probs, multi=True)
            by_id = dict(zip(ids, probs))
            cands = list(candidates or [])
            return Prediction(task.type, list(task.labels), np.array([float(by_id.get(c.id, 0.5)) for c in cands]),
                              candidate_ids=[c.id for c in cands])
    except PredFormatError:
        raise
    except (TypeError, ValueError, KeyError) as e:
        raise PredFormatError(f"cannot read prediction: {e}") from e
    raise PredFormatError(f"unsupported task type {task.type}")


def uniform_prediction(task: TaskDef, sample: Any, candidates: Optional[Sequence[Any]] = None) -> Prediction:
    """Maximum-uncertainty stand-in used when a stored prediction is missing or unreadable."""
    k = len(task.labels)
    if task.type is TaskType.CLASSIFICATION:
        p = np.full(k, 0.5 if task.multi_label else 1.0 / k)
        return Prediction(task.type, list(task.labels), p, multi_label=task.multi_label)
    if task.type in (TaskType.TOKEN, TaskType.SPAN):
        toks = tokenize(sample.text)
        cols = list(task.labels) if task.type is TaskType.TOKEN else [OUTSIDE_TAG] + list(task.labels)
        return Prediction(task.type, cols, np.full((len(toks), len(cols)), 1.0 / len(cols)), tokens=toks)
    cands = list(candidates or [])
    return Prediction(task.type, list(task.labels), np.full(len(cands), 0.5), candidate_ids=[c.id for c in cands])
