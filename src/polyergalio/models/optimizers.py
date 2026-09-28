from abc import ABC, abstractmethod
import numpy as np
from polyergalio.models.layers import Layer


def scale_gradients(value, factor: float):
    """Scale a gradient, or a nested dict of gradients, by factor."""
    if isinstance(value, dict):
        return {key: scale_gradients(sub, factor) for key, sub in value.items()}
    return factor * value


class Optimizer(ABC):
    """
    Abstract base class for all optimizers.

    Layers with adaptive = False (the clustering layers) are never rescaled or
    given state by adaptive optimizers (Adam), which apply a plain gradient
    step at the layer's own learning_rate instead. SGD treats every layer alike.
    """

    def __init__(self):
        pass

    @abstractmethod
    def step(self, layers: list[Layer]) -> None:
        """
        Take one step of the optimizer function

        Parameters
        ----------
        layers : layers (List[Layer]): the ORDERED LIST of model structure

        """
        pass


    def fixed_step(self, layer: Layer, gradients: dict) -> None:
        """Plain gradient step at the layer's own learning_rate, for non-adaptive layers under adaptive optimizers."""
        layer.update_weights(
            **{key: scale_gradients(sub, layer.learning_rate) for key, sub in gradients.items()}
        )

    def zero_gradients(self, layers: list[Layer]):
        """
        We'll have to set all of our gradients to zero
        Useful if gradients are accumulated.
        """
        for layer in layers:
            layer.zero_gradients()


class SGD(Optimizer):
    """
    STOCHASTIC GRADIENT DESCENT - as vanilla as we can get
    """
    def __init__(self, learning_rate: float = 0.001, clip_gradients: bool = False):
        """

        Parameters
        ----------
        learning_rate : our learning rate, or alpha
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.max_norm = 1.0
        self.do_clipping = clip_gradients  # TODO: fix this

    def step(self, layers: list[Layer]) -> None:

        for layer in layers:
            if layer.training != True:
                continue

            delta_grads = layer.get_gradients()
            if not delta_grads:
                continue

            layer.update_weights(
                **{key: scale_gradients(sub, self.learning_rate) for key, sub in delta_grads.items()}
            )


class Adam(Optimizer):
    """
    Momentum estimates are kept per (layer, parameter name), since a layer can
    expose several independently-shaped parameters -- FullyConnectedLayer's
    weights and bias, for instance, need separate moments.
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
        self._momenta: dict = {}
        self._ridges: dict = {}

    def update(self,
               value,
               path: tuple,
               momentum_update: float,
               ridge_update: float):
        # careful -- recursive
        if isinstance(value, dict):
            return {
                key: self.update(sub, path + (key,), momentum_update, ridge_update)
                for key, sub in value.items()
            }
        if value is None:
            return None

        if path not in self._momenta:
            self._momenta[path] = np.zeros_like(value)
            self._ridges[path] = np.zeros_like(value)
        m = self._momenta[path]
        v = self._ridges[path]

        m = self.momentum_decay * m + (1 - self.momentum_decay) * value
        v = self.ridge_decay * v + (1 - self.ridge_decay) * value ** 2
        self._momenta[path] = m
        self._ridges[path] = v

        m_hat = m / momentum_update
        v_hat = v / ridge_update
        return self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

    def step(self, layers: list[Layer]) -> None:
        self.timestep += 1
        momentum_update = 1 - self.momentum_decay ** self.timestep
        ridge_update = 1 - self.ridge_decay ** self.timestep

        for layer in layers:
            if layer.training != True:
                continue
            else:
                delta_grads = layer.get_gradients()
                if not delta_grads:
                    continue
                if not layer.adaptive:
                    self.fixed_step(layer, delta_grads)
                    continue

                layer.update_weights(**{key: self.update(sub, (layer, key), momentum_update, ridge_update)
                                        for key, sub in delta_grads.items()
                                        }
                                     )


# Scaled Conjugate Gradient
