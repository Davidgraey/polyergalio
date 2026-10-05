"""
------------------------ ABC for our encoder / processors ------------------------------
"""
from abc import ABC, abstractmethod

from numpy.typing import NDArray

from polyergalio.base_model import BasalEstimator


class Processor(BasalEstimator, ABC):
    """An estimator that learns a column mapping with fit and applies it with encode."""
    state_names = ("fitted", "obs_min_max")

    def __init__(self, target: (str | int)):
        super().__init__()
        self.target = target
        self.fitted = False
        self.obs_min_max: tuple = None

    @property
    def is_fitted(self) -> bool:
        return self.fitted

    @abstractmethod
    def inverse(self, values):
        """inverse of the encoding function"""

    @abstractmethod
    def encode(self, *columns):
        """encoding of the column values this processor is connected to"""

    @abstractmethod
    def fit_encode(self, *columns):
        """fit and encoding of the column values"""

    @abstractmethod
    def fit(self, *columns):
        """fit the encoder obj using the column values"""

    @property
    @abstractmethod
    def metadata(self):
        """gets the metadata of the encoder to be passed to backbone network"""

    def predict(self, *columns):
        return self.encode(*columns)

    def fit_predict(self, *columns):
        return self.fit_encode(*columns)

    def forward(self, *columns):
        return self.encode(*columns)

    def backward(self, incoming_gradient: NDArray) -> NDArray:
        raise NotImplementedError(f"{self.__class__.__name__} is fitted, not backpropagated")

    @property
    def info(self):
        info = {"self": self.__class__}
        info.update(self.__dict__)
        return info

    def __repr__(self):
        return f"{self.__class__.__name__} for {self.target}"
