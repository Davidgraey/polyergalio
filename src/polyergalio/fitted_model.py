"""
Models that learn through fit() and answer through predict(), rather than through backpropagation.

FittedModel is what supervised and clustering models share: a fitted flag, a seeded generator, the fit / predict
contract, running standardization of the inputs and a loss.

Concerns
--------
    concern      declared by                 notes
    config       constructor arguments       includes seed when a subclass takes it
    state        state_names                 fitted, and the standardization statistics on a FittedModel
    parameters   parameter_names             the learned arrays, counted by num_parameters
    generator    seed                        rebuilt from seed, never persisted: a restored model predicts
                                             identically, and a new fit restarts from the seed

backward() is not defined for a fitted model. A model that has a useful gradient (a linear model's
coefficients) sets gradient_<name> itself, so parameters, get_gradients and `-=` behave as they do on a layer.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from polyergalio.base_model import BasalEstimator
from polyergalio.models.constants import EPSILON

log = logging.getLogger(__name__)


def count_arrays(value) -> int:
    """Total size of every array nested in a dict, list or tuple; scalars do not count."""
    if isinstance(value, np.ndarray):
        return value.size
    if isinstance(value, dict):
        return sum(count_arrays(member) for member in value.values())
    if isinstance(value, (list, tuple)):
        return sum(count_arrays(member) for member in value)
    return 0


class FittedModel(BasalEstimator, ABC):
    """A model with a fitted flag, a fit / predict contract, a loss and running standardization of its inputs."""
    state_names = ("fitted", "x_means", "x_stds", "num_seen_samples")

    def __init__(self, seed: int = 42, input_dimension: int = 1, output_dimension: int = 1):
        """
        Parameters
        ----------
        seed : seed of the generator the model draws from while fitting
        input_dimension : width of the inputs
        output_dimension : width of the outputs
        """
        super().__init__()
        self.seed = seed
        self.RNG = np.random.default_rng(seed)
        self.fitted = False
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.x_means = None
        self.x_stds = None
        self.num_seen_samples: int = 0

    @property
    def is_fitted(self) -> bool:
        return self.fitted

    @abstractmethod
    def fit(self, x_data: NDArray, **kwargs):
        """Learn from data."""

    @abstractmethod
    def predict(self, x_data: NDArray, **kwargs):
        """Apply what was learned."""

    def fit_predict(self, x_data: NDArray, *args, **kwargs):
        """Fit, then predict on the same data."""
        self.fit(x_data, *args, **kwargs)
        return self.predict(x_data)

    def forward(self, x_data: NDArray, **kwargs):
        return self.predict(x_data, **kwargs)

    def backward(self, incoming_gradient: NDArray) -> NDArray:
        raise NotImplementedError(f"{self.__class__.__name__} is fitted, not backpropagated")

    @property
    def num_parameters(self) -> int:
        """Elements of every learned array; scalars do not count."""
        return count_arrays(self.get_weights())

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def __repr__(self):
        return self.__class__.__name__

    @abstractmethod
    def calculate_loss(self, **kwargs) -> NDArray:
        """The loss on the current parameters."""

    def update_running_standardize(self, new_data_mean: NDArray, new_data_std: NDArray, new_data_count: int) -> None:
        """
        Fold a new batch into the running mean and standard deviation, weighting each group by its count.

        Parameters
        ----------
        new_data_mean : mean of the new observations
        new_data_std : standard deviation of the new observations
        new_data_count : number of new observations
        """
        full_count = self.num_seen_samples + new_data_count
        full_mean = (self.num_seen_samples * self.x_means + new_data_count * new_data_mean) / full_count
        sum_square_errors = self.x_stds ** 2 * (self.num_seen_samples - 1) + new_data_std ** 2 * (new_data_count - 1)
        sum_squares = (
            (self.x_means - full_mean) ** 2 * self.num_seen_samples
            + (new_data_mean - full_mean) ** 2 * new_data_count
        )
        self.x_means = full_mean
        self.x_stds = np.sqrt((sum_square_errors + sum_squares) / (full_count - 1))

    def standardize(self, data_array: NDArray, mean: NDArray = None, stds: NDArray = None) -> NDArray:
        """Subtract the mean and divide by the standard deviation; the running input statistics by default."""
        mean = self.x_means if mean is None else mean
        stds = self.x_stds if stds is None else stds
        if mean is None or data_array.shape[-1] != mean.shape[0]:
            log.error("initialize process hasn't been done yet!")
        return (data_array - mean) / (stds + EPSILON)

    def unstandardize(self, data_array: NDArray, mean: NDArray = None, stds: NDArray = None) -> NDArray:
        """Return standardized data to its original units; the running input statistics by default."""
        mean = self.x_means if mean is None else mean
        stds = self.x_stds if stds is None else stds
        assert data_array.shape[-1] == mean.shape[0]
        return data_array * (stds + EPSILON) + mean

    def init_standardize(self, x_data: NDArray) -> None:
        """Start the running statistics from x_data, or fold x_data into them when already fitted."""
        if self.x_means is None or self.num_seen_samples == 0:
            self.num_seen_samples = x_data.shape[0]
            self.x_means = np.mean(x_data, axis=0)
            self.x_stds = np.std(x_data, axis=0)
        else:
            self.update_running_standardize(np.mean(x_data, axis=0), np.std(x_data, axis=0), x_data.shape[0])
