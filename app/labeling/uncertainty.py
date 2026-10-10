"""Uncertainty scoring for HITL relabeling.

Every ``Prediction`` is reduced to rows of probability distributions
(``Prediction.distributions``); a per-row score in [0, 1] is computed and
aggregated (mean by default) into one score per sample. Higher = more
uncertain = more worth a human's attention.

Model-based strategies
    Entropy                     normalized Shannon entropy
    Least confidence            1 - p(top), normalized to [0, 1]
    Margin (decision boundary)  1 - (p(top1) - p(top2)); high near the boundary

Human-based strategies (need an existing label)
    Low annotator confidence    1 - annotator confidence
    Model–human disagreement    1 - p(human's label)  (single-label classification)
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from prediction import Prediction
from task_defs import TaskType


def _entropy(P: np.ndarray) -> np.ndarray:
    k = P.shape[1]
    if k < 2:
        return np.zeros(len(P))
    Q = np.clip(P, 1e-12, 1.0)
    return -(Q * np.log(Q)).sum(axis=1) / np.log(k)


def _least_confidence(P: np.ndarray) -> np.ndarray:
    k = P.shape[1]
    if k < 2:
        return np.zeros(len(P))
    return (1.0 - P.max(axis=1)) / (1.0 - 1.0 / k)


def _margin(P: np.ndarray) -> np.ndarray:
    if P.shape[1] < 2:
        return np.zeros(len(P))
    S = np.sort(P, axis=1)
    return 1.0 - (S[:, -1] - S[:, -2])


MODEL_STRATEGIES: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Entropy": _entropy,
    "Least confidence": _least_confidence,
    "Margin (decision boundary)": _margin,
}
LOW_CONF = "Low annotator confidence"
DISAGREE = "Model–human disagreement"


def score(pred: Prediction, strategy: str, agg: str = "mean") -> float:
    P = pred.distributions()
    if P.size == 0:
        return 0.0
    s = MODEL_STRATEGIES[strategy](P)
    s = np.clip(s, 0.0, 1.0)
    return float(s.max() if agg == "max" else s.mean())


def all_scores(pred: Prediction) -> Dict[str, float]:
    return {name: score(pred, name) for name in MODEL_STRATEGIES}


def human_strategies(task) -> list:
    out = [LOW_CONF]
    if task.type is TaskType.CLASSIFICATION and not task.multi_label:
        out.append(DISAGREE)
    return out


def human_score(strategy: str, pred: Prediction, rec: Optional[dict]) -> Optional[float]:
    """Score from an existing human label; None when not applicable."""
    if rec is None:
        return None
    if strategy == LOW_CONF:
        return 1.0 - float(rec["confidence"])
    if strategy == DISAGREE:
        lab = (rec.get("value") or {}).get("label")
        if lab in pred.labels:
            return 1.0 - float(pred.probs[pred.labels.index(lab)])
    return None


__all__ = ["MODEL_STRATEGIES", "LOW_CONF", "DISAGREE", "score", "all_scores",
           "human_strategies", "human_score"]
