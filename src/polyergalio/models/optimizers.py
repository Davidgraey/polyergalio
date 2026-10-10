"""
Optimizers step an estimator, or a container of estimators, from its gradient dict.

The gradient dict mirrors the model: `gradient_<name>` for every parameter, with a nested dict wherever the parameter
is itself an estimator. A NeuralNetwork is keyed by node name, a layer by parameter name. The optimizer walks that tree
against the model, scales each leaf, and applies the whole scaled tree once with `update_weights`.

Along the way an estimator that is not training is skipped, and one that is not adaptive (the clustering layers) takes
a plain step at its own learning_rate, whatever the optimizer. Optimizer state is keyed by the path of gradient names,
so it needs no layer ordering and survives a graph being rewired.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from polyergalio.composite_model import Composite
from polyergalio.base_model import BasalEstimator

GRADIENT_PREFIX = "gradient_"


def parameter_members(model: BasalEstimator | Composite) -> dict:
    """Each parameter of an estimator, or each node's component of a container, by name."""
    if isinstance(model, Composite):
        return {node.name: node.component for node in model.nodes if not node.is_source}
    return {name: getattr(model, name) for name in model.parameter_names}


class Optimizer(ABC):
    """Base class for optimizers; a subclass says how one gradient array becomes a step."""

    def step(self, model: BasalEstimator | Composite, gradients: Optional[dict] = None, only=None) -> None:
        """
        Take one step on a model.

        Parameters
        ----------
        model : a layer, or a Network built on BasalEstimator or Composite
        gradients : a get_gradients() dict, such as one averaged across nodes; the model's own by default
        only : names (nodes, or parameters) to step; everything else is left alone
        """
        gradients = model.get_gradients() if gradients is None else gradients
        if only is not None:
            gradients = {key: value for key, value in gradients.items() if key.removeprefix(GRADIENT_PREFIX) in only}
        scaled = self.scale(model, gradients)
        if scaled:
            model.update_weights(**scaled)

    def scale(self, owner: BasalEstimator | Composite, gradients: dict, path: tuple = ()) -> dict:
        """
        The gradient tree with every leaf scaled into a step. An owner that is not training contributes nothing.

        Parameters
        ----------
        owner : the estimator or container the gradients belong to
        gradients : its `gradient_<name>` dict
        path : gradient names leading to owner, the key for optimizer state
        """
        if not getattr(owner, "training", True):
            return {}
        members = parameter_members(owner)
        unexpected = set(gradients) - {f"{GRADIENT_PREFIX}{name}" for name in members}
        if unexpected:
            raise ValueError(f"{owner.__class__.__name__} has no gradients named {sorted(unexpected)}")

        scaled = {}
        for name, member in members.items():
            key = f"{GRADIENT_PREFIX}{name}"
            gradient = gradients.get(key)
            if gradient is None:
                continue
            if isinstance(member, (BasalEstimator, Composite)):
                nested = self.scale(member, gradient, path + (key,))
                if nested:
                    scaled[key] = nested
            else:
                scaled[key] = self.scale_leaf(owner, gradient, path + (key,))
        return scaled

    @abstractmethod
    def scale_leaf(self, owner: BasalEstimator, gradient: NDArray, path: tuple) -> NDArray:
        """Turn one gradient array into the step to subtract."""

    def get_state(self) -> dict:
        """Optimizer state as plain values keyed by gradient-name paths, so it can cross processes."""
        return {}

    def set_state(self, state: dict) -> None:
        """Restore the output of get_state()."""

    def zero_gradients(self, model: BasalEstimator | Composite) -> None:
        """Reset the model's gradients, useful when they are accumulated."""
        model.zero_gradients()


class SGD(Optimizer):
    """Stochastic gradient descent: every leaf steps at the same learning rate."""

    def __init__(self, learning_rate: float = 0.001, clip_gradients: bool = False):
        """
        Parameters
        ----------
        learning_rate : step size, or alpha
        clip_gradients : accepted for clipping, not applied yet
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.max_norm = 1.0
        self.do_clipping = clip_gradients

    def scale_leaf(self, owner: BasalEstimator, gradient: NDArray, path: tuple) -> NDArray:
        return self.learning_rate * gradient


class Adam(Optimizer):
    """
    Adaptive moments, kept per gradient path, so a layer's weights and bias each have their own.
    An estimator that is not adaptive steps at its own learning_rate instead and keeps no state.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        momentum: float = 0.9,
        ridge_momentum: float = 0.999,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_decay = momentum
        self.ridge_decay = ridge_momentum
        self.eps = eps
        self.timestep = 0
        self.momenta: dict[tuple, NDArray] = {}
        self.ridges: dict[tuple, NDArray] = {}

    def step(self, model: BasalEstimator | Composite, gradients: Optional[dict] = None, only=None) -> None:
        self.timestep += 1
        super().step(model, gradients, only)

    def scale_leaf(self, owner: BasalEstimator, gradient: NDArray, path: tuple) -> NDArray:
        if not owner.adaptive:
            return owner.learning_rate * gradient

        if path not in self.momenta:
            self.momenta[path] = np.zeros_like(gradient)
            self.ridges[path] = np.zeros_like(np.abs(gradient))
        momentum = self.momentum_decay * self.momenta[path] + (1 - self.momentum_decay) * gradient
        ridge = self.ridge_decay * self.ridges[path] + (1 - self.ridge_decay) * np.abs(gradient) ** 2
        self.momenta[path] = momentum
        self.ridges[path] = ridge

        momentum_hat = momentum / (1 - self.momentum_decay ** self.timestep)
        ridge_hat = ridge / (1 - self.ridge_decay ** self.timestep)
        return self.learning_rate * momentum_hat / (np.sqrt(ridge_hat) + self.eps)

    def get_state(self) -> dict:
        return {
            "timestep": self.timestep,
            "momenta": {path: value.copy() for path, value in self.momenta.items()},
            "ridges": {path: value.copy() for path, value in self.ridges.items()},
        }

    def set_state(self, state: dict) -> None:
        self.timestep = state["timestep"]
        self.momenta = {path: value.copy() for path, value in state["momenta"].items()}
        self.ridges = {path: value.copy() for path, value in state["ridges"].items()}
