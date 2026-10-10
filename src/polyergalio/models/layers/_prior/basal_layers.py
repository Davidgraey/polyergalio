from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

import polyergalio.models.activations as activations
from polyergalio.models.constants import (
    ANY_SHAPE,
    GLOBAL_DTYPE,
)
from polyergalio.models.weight_initialization import get_weight_init
from polyergalio.types import Serializable


def shape_conflict(produced: tuple, expected: tuple) -> Optional[str]:
    """
    Compare two declared shapes, right-aligned, and describe the first axis
    where they disagree.

    Shapes are in trailing-axis form, so they are matched from the last axis
    backwards and only the overlap is checked.
    none on either side is a wildcard "*" / any_value

    Returns
    -------
    a description of the offending axis, or None when the two are compatible
    """
    if (produced is None) or (expected is None):
        return None

    for offset, (made, wanted) in enumerate(
        zip(reversed(produced), reversed(expected)), start=1
    ):
        if (made is None) or (wanted is None):
            continue
        if made != wanted:
            return (
                f"{produced} cannot feed {expected}, axis -{offset} "
                f"is {made} against {wanted}"
            )
    return None


# ------------------------------------------------------------------
class Layer(Serializable, ABC):
    preserves_shape: bool = False
    adaptive: bool = True
    training: bool = True

    def __init__(self):
        super().__init__()
        self.RNG = np.random.RandomState()
        self.declare_shapes()

    def sublayers(self) -> list["Layer"]:
        """Layers held as attributes, directly or inside a tuple, list or dict."""
        found = []
        for value in vars(self).values():
            members = value.values() if isinstance(value, dict) else value if isinstance(value, (tuple, list)) else (value,)
            found.extend(member for member in members if isinstance(member, Layer) and member is not self)
        return found

    def train(self, mode: bool = True) -> "Layer":
        """
        Switch this layer, and every sublayer it holds, between training and
        inference. A forward pass given no training_now follows this mode.

        Returns
        -------
        self, so calls chain: layer.eval().forward(x)
        """
        self.training = mode
        for sublayer in self.sublayers():
            sublayer.train(mode)
        return self

    def eval(self) -> "Layer":
        """Switch to inference, see train()."""
        return self.train(False)

    def declare_shapes(self, inputs=(ANY_SHAPE,), outputs=(ANY_SHAPE,)) -> None:
        """
        record what this layer accepts and emits; via trailing-axis
        """
        self._in_shapes = tuple(inputs)
        self._out_shapes = tuple(outputs)

    @property
    def shapes(self) -> dict[str, tuple[tuple, ...]]:
        """
        The layer's declared (NOT OBSERVED) input and output shapes.
        """
        return {"input": self._in_shapes, "output": self._out_shapes}

    def infer_output_shapes(self, input_shapes: tuple[tuple, ...]) -> tuple[tuple, ...]:
        """
        resolve the output shapes given what actually arrives
        """
        if self.preserves_shape:
            return (input_shapes[0],)
        return self._out_shapes

    @abstractmethod
    def forward(self, x: NDArray) -> NDArray:
        pass

    @abstractmethod
    def backward(self, x: NDArray) -> NDArray:
        pass

    @abstractmethod
    def get_weights(self, for_serialize: bool) -> NDArray | dict[str, dict]:
        pass

    def set_weights(self, weights: dict) -> None:
        if not isinstance(weights, dict):
            raise NotImplementedError(
                f"{self.__class__.__name__} has weights but hasn't "
                "implemented set_weights processes restore them"
            )

    def get_gradients(self) -> dict[str, NDArray]:
        """Parameter gradients, reduced over the batch axis; None if there are none to report."""
        pass

    @abstractmethod
    def update_weights(self, **kwargs) -> None:
        pass

    @abstractmethod
    def purge(self) -> None:
        pass

    @abstractmethod
    def zero_gradients(self) -> None:
        pass

    @property
    def num_parameters(self) -> int:
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# TODO: build ENUMS for activations
class FullyConnectedLayer(Layer):
    def __init__(self,
                 ni: int,
                 no: int,
                 activation_type: str,
                 is_output: bool = False,
                 initialization_override: Optional[str] = None,
                 initialization_kwargs: Optional[dict] = None,
                 ):
        """
        **********ARGUMENTS**********
        :param ni: number of input units
        :param no: number of output units
        :param activation_type: string isdentifying activation type, 'linear', 'sigmoid', 'tanh', etc.
        :param is_output: boolean flag designating if this is an output layer or hidden layer
        :param initialization_override: any WEIGHT_INIT_DISPATCHER name, used instead of the activation's default
        :param initialization_kwargs: keyword arguments bound to the initializer, e.g. {"std": 0.01}
        """
        super().__init__()
        self.ni: int = ni
        self.no: int = no
        self.activation: str = activation_type
        self._func_activation: Callable = activations.activation_dictionary[
            self.activation
        ]
        self._func_derivative: Callable = activations.derivative_dictionary[
            self.activation
        ]

        self.is_output: bool = is_output

        self.initialization_override = initialization_override
        self.initialization_kwargs = dict(initialization_kwargs or {})
        initializer = get_weight_init(
            initialization_override or activation_type, **self.initialization_kwargs
        )
        self.weights: NDArray = initializer(self.RNG, ni=ni, no=no)
        self.shape: tuple = self.weights.shape
        self.bias: NDArray = np.zeros((1, no), dtype=GLOBAL_DTYPE)
        self.declare_shapes(inputs=((ni,),), outputs=((no,),))

        # these values will be rewritten or updated on each pass
        self.input = np.empty(shape=(ni, no))
        self.output = np.empty(shape=(ni, no))
        self.z = np.empty(shape=(ni, no))

        self.zero_gradients()

    def forward(self,
                incoming_x: NDArray,
                forced_activation: Optional[str] = None,
                mask: Optional[NDArray] = None,
                ) -> NDArray:
        """

        Parameters
        ----------
        incoming_x : input data that is already standardized, if called for
        forced_activation :
        mask : unused -- every row is projected independently of every other,
            so padding elsewhere in the sequence can't affect this layer's
            output. Accepted for pass-through compatibility with the rest of
            the graph.

        Returns
        -------
        Dot Product of forward pass of incoming values
        """
        self.in_shape = incoming_x.shape

        assert self.in_shape[-1] == self.weights.shape[0], (
            f"weights and xs don't match -- x:{incoming_x.shape} "
            f"weights: {self.weights.shape}"
        )
        self._used_activation = forced_activation or self.activation
        if forced_activation is None:  # standard layer activation
            this_activation: Callable = self._func_activation
        else:  # we call out the specific "forced" activation
            this_activation: Callable = activations.activation_dictionary[
                forced_activation
            ]

        # if self.training:
        self.input = incoming_x.reshape(-1, self.in_shape[-1])
        self.z = self.input @ self.weights + self.bias
        self.output = this_activation(self.z)
        return self.output.reshape(*self.in_shape[:-1], -1)


    def backward(self,
                 incoming_grad: NDArray,
                 forced_activation: Optional[str] = None
                 ) -> NDArray:
        """
        Backward pass through this layer and it's activation
        Parameters
        ----------
        incoming_grad : the backward passed gradient from the "previous" step

        Returns
        -------
        returns this layer's gradient contribution
        """
        if forced_activation is None:
            this_derivative: Callable = self._func_derivative
        else:
            this_derivative: Callable = activations.derivative_dictionary[
                forced_activation
            ]

        delta = incoming_grad.reshape(-1, incoming_grad.shape[-1])
        delta = this_derivative(self.output, self.z, delta)

        self.gradient_weights = self.input.T @ delta
        self.gradient_bias = delta.sum(axis=0, keepdims=True)

        final_grad = delta @ self.weights.T

        return final_grad.reshape(*self.in_shape[:-1], -1)

    def update_weights(self,
                       gradient_bias: NDArray,
                       gradient_weights: NDArray
                       ) -> None:
        """
        values passed in are the update to apply to this this layer's weights - this will already have learning rate,
        depreication or momentum / other calculations addressed in the upper level.
        Parameters
        ----------
        bias_delta : Bias gradient
        weights_delta : Weights gradient

        Returns
        -------
        None - inplace operation
        """
        self.bias -= gradient_bias
        self.weights -= gradient_weights

    def purge(self) -> None:
        """
        resets values tracked during training to zero - an inplace func
        """
        self.input = None
        self.output = None
        self.z = None

    def get_weights(self, for_serialize: bool = False):
        if for_serialize:
            return {"bias": self.bias, "weights": self.weights}
        return np.concatenate([self.bias.ravel(), self.weights.ravel()])

    def set_weights(self, weights: dict) -> None:
        if weights is not None:
            self.bias = np.asarray(weights["bias"], dtype=GLOBAL_DTYPE)
            self.weights = np.asarray(weights["weights"], dtype=GLOBAL_DTYPE)

    def get_gradients(self) -> dict[str, NDArray]:
        return {
            "gradient_bias": self.gradient_bias,
            "gradient_weights": self.gradient_weights,
        }

    def zero_gradients(self) -> None:
        self.gradient_weights = np.zeros_like(self.weights)
        self.gradient_bias = np.zeros_like(self.bias)

    @property
    def num_parameters(self) -> int:
        return self.bias.size + self.weights.size

    def get_config(self) -> dict:
        """activation_type is stored as self.activation, so the base
        introspection (which matches by name) can't find it on its own"""
        config = super().get_config()
        config["activation_type"] = self.activation
        return config

    def __str__(self):
        return f"Layer of {self.activation}, shaped {self.shape} -- output is {self.is_output}"

    def __repr__(self):
        return f"{self.shape}, using {self.activation}"


class DropoutLayer(Layer):
    """Hinton style or simple bool mask Dropout layer with scaling outputs"""

    preserves_shape = True

    def __init__(self,
                 dropout_prob=0.5,
                 use_rescale: bool = False
                 ):
        super().__init__()
        assert (dropout_prob > 0.0) and (dropout_prob < 1.0)
        self.dropout_prob: float = dropout_prob
        self.keep_prob: float = 1 - dropout_prob
        self.use_rescale = use_rescale

        self.mask = None
        self.input = None
        self.output = None

    def forward(self,
                incoming_x: NDArray,
                mask: Optional[NDArray] = None,
                ):
        """mask : unused -- dropout is applied per element regardless of
        padding; accepted for pass-through compatibility with the graph.
        training_now : None follows the layer's train() / eval() mode"""
        self.input = incoming_x
        if self.training:
            if self.use_rescale:
                self.mask = (
                    self.RNG.binomial(1, self.keep_prob, size=incoming_x.shape)
                    / self.keep_prob
                )
            else:
                self.mask = self.RNG.binomial(1, self.keep_prob, size=incoming_x.shape)

            return incoming_x * self.mask

        else:
            self.output = self.input

        return self.output

    def backward(self, incoming_grad: NDArray) -> NDArray:
        if self.mask is None:
            return incoming_grad
        return incoming_grad * self.mask

    def update_weights(self) -> None:
        pass

    def set_weights(self, weights: dict) -> None:
        pass

    def purge(self) -> None:
        """
        resets values tracked during training to zero - an inplace func
        """
        self.input = None
        self.mask = None
        self.gradient = np.empty(shape=(1, 1))

    def get_weights(self, for_serialize: bool = False):
        if for_serialize:
            return {}
        return None

    def get_gradients(self) -> dict[str, NDArray]:
        return {}

    @property
    def num_parameters(self) -> int:
        return 0

    def zero_gradients(self) -> None:
        pass

    def __str__(self):
        if self.use_rescale:
            method = "Hinton / rescaling dropout"
        else:
            method = "bool dropout"
        return f"Layer of {method} currently at {self.dropout_prob}"

    def __repr__(self):
        return self.__str__()


class NormalizeLayer(Layer):
    def __init__(self, ni: int, shift_scale: bool = True, eps: float = 1e-6):
        super().__init__()
        self.ni = ni
        self.eps = eps
        self.shift_scale = shift_scale
        self.declare_shapes(inputs=((ni,),), outputs=((ni,),))

        if shift_scale:
            self.scale_gamma = np.ones((1, ni), dtype=GLOBAL_DTYPE)
            self.shift_beta = np.zeros((1, ni), dtype=GLOBAL_DTYPE)
        else:
            self.scale_gamma = None
            self.shift_beta = None

        # cached for backward
        self.x_norm = None
        self.std = None

        self.zero_gradients()

    def forward(self, incoming_x: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """mask : unused -- normalization is over each position's own feature
        axis, independent of every other position; accepted for pass-through
        compatibility with the graph."""
        self.in_shape = incoming_x.shape

        # reshape to 2D in case (batch, sequence, hidden)
        # if self.training:
        self.input = incoming_x.reshape(-1, self.in_shape[-1])

        _mean = np.mean(self.input, axis=-1, keepdims=True)
        self.std = np.sqrt(np.var(self.input, axis=-1, keepdims=True) + self.eps)

        self.x_norm = (self.input - _mean) / self.std
        output = self.x_norm

        if self.shift_scale == True:
            output = self.scale_gamma * (self.x_norm) + self.shift_beta

        return output.reshape(self.in_shape)


    def update_weights(self,
                       gradient_beta: Optional[NDArray] = None,
                       gradient_gamma: Optional[NDArray] = None,
                       ) -> None:
        # update the shift & scale values based on gradient contributions
        if self.shift_scale == True:
            self.shift_beta -= gradient_beta
            self.scale_gamma -= gradient_gamma
        else:
            pass

    def backward(self, incoming_grad: NDArray) -> NDArray:
        original_shape = incoming_grad.shape
        incoming_grad = incoming_grad.reshape(-1, original_shape[-1])
        if self.shift_scale == True:
            self.gradient_beta = np.sum(incoming_grad, axis=0, keepdims=True)
            self.gradient_gamma = np.sum(
                incoming_grad * self.x_norm, axis=0, keepdims=True
            )
            z = incoming_grad * self.scale_gamma
        else:
            self.gradient_beta = None
            self.gradient_gamma = None
            z = incoming_grad

        gradient = (1.0 / self.std) * (
            z
            - np.mean(z, axis=-1, keepdims=True)
            - self.x_norm * np.mean(z * self.x_norm, axis=-1, keepdims=True)
        )
        return gradient.reshape(original_shape)

    def purge(self):
        self.input = None
        self.x_norm = None
        self.std = None

    def get_weights(self, for_serialize: bool = False):
        if self.shift_scale is True:
            if for_serialize:
                return {"shift_beta": self.shift_beta, "scale_gamma": self.scale_gamma}
            return (self.shift_beta, self.scale_gamma)
        if for_serialize:
            return {}
        return (None, None)

    def get_gradients(self) -> dict[str, NDArray]:
        if self.shift_scale is True:
            return {
                "gradient_beta": self.gradient_beta,
                "gradient_gamma": self.gradient_gamma,
            }
        return {}

    def set_weights(self, weights: dict) -> None:
        if weights is not None:
            self.shift_beta = np.asarray(weights["shift_beta"], dtype=GLOBAL_DTYPE)
            self.scale_gamma = np.asarray(weights["scale_gamma"], dtype=GLOBAL_DTYPE)

    @property
    def num_parameters(self) -> int:
        if not self.shift_scale:
            return 0
        return self.shift_beta.size + self.scale_gamma.size

    def zero_gradients(self) -> None:
        if not self.shift_scale:
            return
        self.gradient_beta = np.zeros_like(self.shift_beta)
        self.gradient_gamma = np.zeros_like(self.scale_gamma)

    def __str__(self):
        affine = "affine" if self.shift_scale else "no affine"
        return f"LayerNorm over {self.ni}, {affine}"

    def __repr__(self):
        return self.__str__()


class RMSNormLayer(Layer):
    """
    Root-mean-square norm over the last axis. No mean subtraction and no
    shift, so the only parameter is a learnable per-feature scale.
    """

    def __init__(self, ni: int, eps: float = 1e-6):
        super().__init__()
        self.ni = ni
        self.eps = eps
        self.scale_gamma = np.ones((1, ni), dtype=GLOBAL_DTYPE)
        self.declare_shapes(inputs=((ni,),), outputs=((ni,),))

        self.x_norm = None
        self.rms = None

        self.zero_gradients()

    def forward(self, incoming_x: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """mask : unused -- same reasoning as NormalizeLayer: each position
        is normalized against only its own features."""
        in_shape = incoming_x.shape
        xs = incoming_x.reshape(-1, in_shape[-1])

        # if self.training:
        self.input = xs
        self.rms = np.sqrt(np.mean(self.input**2, axis=-1, keepdims=True) + self.eps)
        self.x_norm = self.input / self.rms
        return (self.x_norm * self.scale_gamma).reshape(in_shape)


    def backward(self, incoming_grad: NDArray) -> NDArray:
        original_shape = incoming_grad.shape

        grad = incoming_grad.reshape(-1, original_shape[-1])
        self.gradient_gamma = np.sum(grad * self.x_norm, axis=0, keepdims=True)

        z = grad * self.scale_gamma
        gradient = (
            z - self.x_norm * np.mean(z * self.x_norm, axis=-1, keepdims=True)
        ) / self.rms
        return gradient.reshape(original_shape)

    def update_weights(self, gradient_gamma: NDArray) -> None:
        self.scale_gamma -= gradient_gamma

    def purge(self) -> None:
        self.input = None
        self.x_norm = None
        self.rms = None

    def get_weights(self, for_serialize: bool = False) -> NDArray:
        if for_serialize:
            return {"scale_gamma": self.scale_gamma}
        return self.scale_gamma

    def set_weights(self, weights: dict) -> None:
        if weights is not None:
            self.scale_gamma = np.asarray(weights["scale_gamma"], dtype=GLOBAL_DTYPE)

    def get_gradients(self) -> dict[str, NDArray]:
        return {"gradient_gamma": self.gradient_gamma}

    def zero_gradients(self) -> None:
        self.gradient_gamma = np.zeros_like(self.scale_gamma)

    @property
    def num_parameters(self) -> int:
        return self.scale_gamma.size

    def __str__(self):
        return f"RMSNorm over {self.ni}"

    def __repr__(self):
        return self.__str__()
