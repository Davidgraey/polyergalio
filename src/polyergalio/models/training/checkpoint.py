import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import NDArray

from polyergalio.types import BasalModel

log = logging.getLogger(__name__)

DEFAULT_TOLERANCE = 1e-4
LOG_NAME = "collect.log"


@dataclass
class CollectedModel:
    """
    The orchestrator's canonical model and how far the healthy node weights are from this baseline

    Parameters
    ----------
    model : model rebuilt from the canonical node's weights
    canonical : node id the weights were taken from
    deviations : per node, largest absolute weight difference from the canonical node
    tolerance : deviation above which the nodes count as diverged
    timestamp : when the weights were collected, UTC
    """
    model: BasalModel
    canonical: str
    deviations: dict[str, float]
    tolerance: float
    timestamp: datetime

    @property
    def max_deviation(self) -> float:
        return max(self.deviations.values())

    @property
    def diverged(self) -> bool:
        return self.max_deviation > self.tolerance


def weight_vector(value) -> NDArray:
    """Every array and number inside nested dicts, lists and tuples as one flat vector."""
    parts = []
    if isinstance(value, dict):
        parts = [weight_vector(value[key]) for key in sorted(value, key=str)]
    elif isinstance(value, (list, tuple)):
        parts = [weight_vector(item) for item in value]
    elif isinstance(value, np.ndarray) and value.dtype.kind in "biufc":
        parts = [value.ravel()]
    elif isinstance(value, (int, float, complex)) and not isinstance(value, bool):
        parts = [np.array([value])]
    return np.concatenate(parts) if parts else np.zeros(0)


def deviation(reference: NDArray, other: NDArray) -> float:
    """Largest absolute difference between two weight vectors; infinite if their sizes differ."""
    if reference.shape != other.shape:
        return float("inf")
    return float(np.max(np.abs(reference - other))) if reference.size else 0.0


def collect_weights(states: dict[str, dict], model_type: type, tolerance: float = DEFAULT_TOLERANCE) -> CollectedModel:
    """
    Build the canonical model from nodes' serialized models and measure their diff from them.

    Parameters
    ----------
    states : serialized model of each healthy node, keyed by node id
    model_type : class whose deserialize rebuilds the canonical model
    tolerance : deviation above which the nodes count as diverged
    """
    canonical = sorted(states)[0]
    reference = weight_vector(states[canonical]["weights"])
    deviations = {node_id: deviation(reference, weight_vector(state["weights"])) for node_id, state in states.items()}
    return CollectedModel(model_type.deserialize(states[canonical]), canonical, deviations, tolerance, datetime.now(timezone.utc))


def persist(collected: CollectedModel, directory: Union[str, Path], locations: dict[str, tuple[str, int]]) -> Path:
    """
    Serialize the canonical model to a timestamped file and append a line to the collect log.

    Parameters
    ----------
    collected : result of collect_weights
    directory : folder for the model file and collect.log, created if missing
    locations : (host, port) of every node the orchestrator knows, healthy or not

    Returns
    -------
    path of the model file
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = collected.timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"model-{stamp}.ergalio"
    collected.model.serialize(path)
    healthy = set(collected.deviations)
    nodes = " ".join(
        f"{node_id}@{host}:{port}{'' if node_id in healthy else '(unavailable)'}"
        for node_id, (host, port) in sorted(locations.items())
    )
    line = (
        f"{collected.timestamp.isoformat()} collected from {len(healthy)} of {len(locations)} nodes: {nodes} "
        f"canonical={collected.canonical} max_deviation={collected.max_deviation:.3g} "
        f"diverged={collected.diverged} file={path.name}"
    )
    with open(directory / LOG_NAME, "a") as handle:
        handle.write(line + "\n")
    if collected.diverged:
        log.warning("node weights deviate by %.3g, above %.3g", collected.max_deviation, collected.tolerance)
    log.info(line)
    return path
