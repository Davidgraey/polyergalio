"""Pluggable model interface + a mock backend.

The app only talks to ``BaseModel.predict``, which returns a ``Prediction``
whose ``probs`` shape depends on the task type:

    classification (single) : (k,)        distribution over task.labels
    classification (multi)  : (k,)        independent P(label present)
    token                   : (n_tok, k)  distribution over task.labels per token
    span                    : (n_tok, k+1) per-token distribution over ["O"] + labels
    contrastive             : (n_cand,)   P(candidate is `labels[0]`, e.g. similar)

To add a real model, subclass ``BaseModel``, implement ``predict`` (and
optionally ``fit``), and decorate with ``@register_model("key")``.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Type

import numpy as np

from prediction import PredFormatError, Prediction, parse_pred_cell, uniform_prediction
from task_defs import OUTSIDE_TAG, TaskDef, TaskType
from tokenizing import tokenize


class BaseModel(ABC):
    key = "base"
    display_name = "Base"

    def __init__(self, version: int = 0, **kwargs: Any):
        self.version = int(version)
        self.params = kwargs

    def fit(self, task: TaskDef, samples: Sequence[Any], labels: Dict[str, dict]) -> None:
        """Optional training hook. ``labels`` maps sample_id -> current label record."""

    @abstractmethod
    def predict(self, task: TaskDef, sample: Any,
                candidates: Optional[Sequence[Any]] = None) -> Prediction:
        ...


_REGISTRY: Dict[str, Type[BaseModel]] = {}


def register_model(key: str) -> Callable[[Type[BaseModel]], Type[BaseModel]]:
    def deco(cls: Type[BaseModel]) -> Type[BaseModel]:
        if key in _REGISTRY:
            raise ValueError(f"Model key {key!r} already registered")
        cls.key = key
        _REGISTRY[key] = cls
        return cls
    return deco


def available_models() -> List[str]:
    return list(_REGISTRY)


def create_model(key: str, version: int = 0, **kwargs: Any) -> BaseModel:
    if key not in _REGISTRY:
        raise KeyError(f"Unknown model {key!r}; registered: {available_models()}")
    return _REGISTRY[key](version=version, **kwargs)


@register_model("mock")
class MockModel(BaseModel):
    """Returns random floats shaped like real model outputs.

    Values are seeded from (model version, task, sample) so they're stable
    across Streamlit reruns; bumping the version ("retrain") re-rolls them.
    Raising uniform floats to a power spreads rows between confident and
    uncertain so the uncertainty queue has something to rank.
    """

    display_name = "Mock (random probabilities)"
    SHARPNESS = 3.0

    def _rng(self, *parts: Any) -> np.random.Generator:
        h = hashlib.sha256("|".join(map(str, (self.version,) + parts)).encode()).digest()
        return np.random.default_rng(int.from_bytes(h[:8], "little"))

    def _dist(self, rng: np.random.Generator, shape, boost_col: Optional[int] = None) -> np.ndarray:
        x = rng.random(shape) ** self.SHARPNESS + 1e-6
        if boost_col is not None:
            x[..., boost_col] *= 4.0
        return x / x.sum(axis=-1, keepdims=True)

    def predict(self, task: TaskDef, sample: Any,
                candidates: Optional[Sequence[Any]] = None) -> Prediction:
        rng = self._rng(task.id, sample.id)
        k = len(task.labels)
        if task.type is TaskType.CLASSIFICATION:
            probs = rng.random(k) if task.multi_label else self._dist(rng, k)
            return Prediction(task.type, list(task.labels), probs, multi_label=task.multi_label)

        if task.type is TaskType.TOKEN:
            toks = tokenize(sample.text)
            boost = task.labels.index(OUTSIDE_TAG) if OUTSIDE_TAG in task.labels else None
            probs = self._dist(rng, (len(toks), k), boost) if toks else np.zeros((0, k))
            return Prediction(task.type, list(task.labels), probs, tokens=toks)

        if task.type is TaskType.SPAN:
            toks = tokenize(sample.text)
            cols = [OUTSIDE_TAG] + list(task.labels)
            probs = self._dist(rng, (len(toks), len(cols)), 0) if toks else np.zeros((0, len(cols)))
            return Prediction(task.type, cols, probs, tokens=toks)

        if task.type is TaskType.CONTRASTIVE:
            cands = list(candidates or [])
            probs = np.array([
                float(self._rng(task.id, *sorted((sample.id, c.id))).random()) for c in cands
            ])
            return Prediction(task.type, list(task.labels), probs,
                              candidate_ids=[c.id for c in cands])
        raise ValueError(f"Unsupported task type {task.type}")


@register_model("precomputed")
class PrecomputedModel(BaseModel):
    """Reads each task's ``{task}_pred`` column from the dataset instead of running a model.

    Missing or unreadable cells fall back to a uniform (maximally uncertain) prediction
    and are recorded in ``problems`` so the UI can report them.
    """

    display_name = "Precomputed (dataset {task}_pred columns)"

    def __init__(self, version: int = 0, **kwargs: Any):
        super().__init__(version, **kwargs)
        self.problems: Dict[tuple, str] = {}

    def predict(self, task: TaskDef, sample: Any, candidates: Optional[Sequence[Any]] = None) -> Prediction:
        raw = getattr(sample, "extra", {}).get(task.pred_column.lower())
        if raw is None:
            self.problems[(task.id, sample.id)] = "no value"
            return uniform_prediction(task, sample, candidates)
        try:
            return parse_pred_cell(task, raw, sample, candidates)
        except PredFormatError as e:
            self.problems[(task.id, sample.id)] = str(e)
            return uniform_prediction(task, sample, candidates)


__all__ = ["BaseModel", "MockModel", "PrecomputedModel", "register_model", "available_models", "create_model"]
