"""
------------------------ ABC for our encoder / processors ------------------------------
"""
from abc import ABC, abstractmethod

from polyergalio.types import Serializable


class Processor(Serializable, ABC):
    _structural_state_keys: tuple[str, ...] = ("obs_min_max", "_fitted")

    def __init__(self, target: (str | int)):
        self.target = target
        self._fitted: bool = False
        self.obs_min_max: tuple = None

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

    @property
    def is_fitted(self):
        """simple bool - check if the encoder has been fitted to data"""
        return self._fitted

    @property
    def info(self):
        info = {"self": self.__class__}
        info.update(self.__dict__)
        return info

    def __repr__(self):
        return f"{self.__class__.__name__} for {self.target}"
