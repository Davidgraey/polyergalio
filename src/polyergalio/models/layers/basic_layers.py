"""
Exercise: the contents of basal_layers rebuilt on BasalEstimator.

RNG is a module-level stand-in for the global generator.
"""
from __future__ import annotations

from abc import ABC
from typing import Optional

import numpy as np
from numpy.typing import NDArray

import polyergalio.models.activations as activations
from polyergalio.base_model import BasalEstimator
from polyergalio.models.constants import EPSILON
from polyergalio.models.weight_initialization import get_weight_init
from polyergalio.utilities import init_standardize, standardize, unstandardize, update_running_standardize

RNG = np.random.default_rng()


class Layer(BasalEstimator, ABC):
    """
    An estimator that maps arrays to arrays and keeps forward-pass values for backward().

    Caches are named in cache_names; purge() clears them, and every sublayer's. Caches start as None.
    """
    cache_names: tuple[str, ...] = ()
    declarations: tuple[str, ...] = BasalEstimator.declarations + ("cache_names",)

    def __init__(self):
        super().__init__()
        for name in self.cache_names:
            setattr(self, name, None)

    def purge(self) -> None:
        """Clear the forward-pass caches and those of every sublayer."""
        for name in self.cache_names:
            setattr(self, name, None)
        for sublayer in self.sublayers():
            sublayer.purge()


class FullyConnectedLayer(Layer):
    """ activation(x @ weights + bias) """
    parameter_names = ("weights", "bias")
    cache_names = ("input", "output", "z", "in_shape")

    def __init__(self,
                 input_dimension: int,
                 output_dimension: int,
                 activation_type: str,
                 is_output: bool = False,
                 initialization_override: Optional[str] = None,
                 initialization_kwargs: Optional[dict] = None,
                 ):
        """
        Parameters
        ----------
        input_dimension : number of input units
        output_dimension : number of output units
        activation_type : name of an activation, 'linear', 'sigmoid', 'tanh', etc.
        is_output : marks the final layer of a network
        initialization_override : weight initializer name, used instead of the activation's default
        initialization_kwargs : keyword arguments bound to the initializer, e.g. {"std": 0.01}
        """
        super().__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.activation_type = activation_type
        self.is_output = is_output
        self.initialization_override = initialization_override
        self.initialization_kwargs = dict(initialization_kwargs or {})

        initializer = get_weight_init(
            initialization_override or activation_type, **self.initialization_kwargs
        )
        self.weights: NDArray = initializer(RNG, ni=input_dimension, no=output_dimension)
        self.bias: NDArray = np.zeros((1, output_dimension))
        self.declare_shapes(inputs=((input_dimension,),), outputs=((output_dimension,),))
        self.zero_gradients()

    @property
    def shape(self) -> tuple:
        return self.weights.shape

    def forward(self,
                incoming_x: NDArray,
                forced_activation: Optional[str] = None,
                mask: Optional[NDArray] = None,
                ) -> NDArray:
        """
        Parameters
        ----------
        incoming_x : (..., ni) input, already standardized if called for
        forced_activation : activation name used instead of activation_type
        mask : unused; every row is projected independently, kept for pass-through compatibility with the graph

        Returns
        -------
        (..., no) activated projection of the input
        """
        activation = activations.activation_dictionary[forced_activation or self.activation_type]
        self.in_shape = incoming_x.shape
        if not self.training:
            return activation(
                incoming_x.reshape(-1, self.in_shape[-1]) @ self.weights + self.bias
            ).reshape(*self.in_shape[:-1], -1)
        self.input = incoming_x.reshape(-1, self.in_shape[-1])
        self.z = self.input @ self.weights + self.bias
        self.output = activation(self.z)
        return self.output.reshape(*self.in_shape[:-1], -1)

    def backward(self,
                 incoming_grad: NDArray,
                 forced_activation: Optional[str] = None
                 ) -> NDArray:
        """
        Parameters
        ----------
        incoming_grad : gradient with respect to this layer's output
        forced_activation : activation name used instead of activation_type; match the one given to forward

        Returns
        -------
        gradient with respect to this layer's input
        """
        derivative = activations.derivative_dictionary[forced_activation or self.activation_type]
        delta = incoming_grad.reshape(-1, incoming_grad.shape[-1])
        delta = derivative(self.output, self.z, delta)

        self.gradient_weights = self.input.T @ delta
        self.gradient_bias = delta.sum(axis=0, keepdims=True)

        final_grad = delta @ self.weights.T
        return final_grad.reshape(*self.in_shape[:-1], -1)

    def __str__(self):
        return f"Layer of {self.activation_type}, shaped {self.shape} -- output is {self.is_output}"

    def __repr__(self):
        return f"{self.shape}, using {self.activation_type}"


class DropoutLayer(Layer):
    """Hinton style or simple bool mask dropout, active only in training mode."""
    preserves_shape = True
    cache_names = ("input", "mask", "output")

    def __init__(self,
                 dropout_prob: float = 0.5,
                 use_rescale: bool = False
                 ):
        """
        Parameters
        ----------
        dropout_prob : chance each element is zeroed, strictly between 0 and 1
        use_rescale : scale kept elements by 1 / keep_prob, so the expected activation is unchanged
        """
        super().__init__()
        assert (dropout_prob > 0.0) and (dropout_prob < 1.0)
        self.dropout_prob = dropout_prob
        self.use_rescale = use_rescale

    @property
    def keep_prob(self) -> float:
        return 1 - self.dropout_prob

    def forward(self,
                incoming_x: NDArray,
                mask: Optional[NDArray] = None,
                ) -> NDArray:
        """
        Parameters
        ----------
        incoming_x : any shape
        mask : unused; dropout is applied per element regardless of padding, kept for pass-through compatibility
            with the graph

        Returns
        -------
        the input with elements dropped in training mode, unchanged in inference
        """
        if not self.training:
            # pass through -- no modification
            return incoming_x

        self.mask = RNG.binomial(1, self.keep_prob, size=incoming_x.shape)
        if self.use_rescale:
            self.mask = self.mask / self.keep_prob
        self.output = incoming_x * self.mask
        return self.output

    def backward(self, incoming_grad: NDArray) -> NDArray:
        if self.mask is None:
            return incoming_grad
        return incoming_grad * self.mask

    def __str__(self):
        method = "Hinton / rescaling dropout" if self.use_rescale else "bool dropout"
        return f"Layer of {method} currently at {self.dropout_prob}"

    def __repr__(self):
        return self.__str__()


class NormalizeLayer(Layer):
    """Layer norm over the last axis, with an optional learnable shift and scale."""
    parameter_names = ("shift_beta", "scale_gamma")
    cache_names = ("in_shape", "input", "x_norm", "std")

    def __init__(self, input_dimension: int, shift_scale: bool = True, eps: float = 1e-6):
        """
        Parameters
        ----------
        input_dimension : size of the last axis
        shift_scale : learn a per-feature shift_beta and scale_gamma; without them the layer has no parameters
        eps : added to the variance for numerical stability
        """
        super().__init__()
        self.input_dimension = input_dimension
        self.eps = eps
        self.shift_scale = shift_scale
        self.declare_shapes(inputs=((input_dimension,),), outputs=((input_dimension,),))

        if shift_scale:
            self.scale_gamma = np.ones((1, input_dimension))
            self.shift_beta = np.zeros((1, input_dimension))
        else:
            self.parameter_names = ()
        self.zero_gradients()

    def forward(self, incoming_x: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        incoming_x : (..., ni)
        mask : unused; each position is normalized against only its own features, kept for pass-through
            compatibility with the graph
        """
        self.in_shape = incoming_x.shape
        input_reshaped = incoming_x.reshape(-1, self.in_shape[-1])

        mean = np.mean(input_reshaped, axis=-1, keepdims=True)

        if not self.training:
            centered = (input_reshaped - mean) / np.sqrt(np.var(input_reshaped, axis=-1, keepdims=True) + self.eps)
            if self.shift_scale:
                return (self.scale_gamma * centered + self.shift_beta).reshape(self.in_shape)

        self.std = np.sqrt(np.var(input_reshaped, axis=-1, keepdims=True) + self.eps)
        self.x_norm = (input_reshaped - mean) / self.std

        output = self.x_norm
        if self.shift_scale:
            output = self.scale_gamma * self.x_norm + self.shift_beta
        return output.reshape(self.in_shape)

    def backward(self, incoming_grad: NDArray) -> NDArray:
        original_shape = incoming_grad.shape
        incoming_grad = incoming_grad.reshape(-1, original_shape[-1])
        z = incoming_grad
        if self.shift_scale:
            self.gradient_shift_beta = np.sum(incoming_grad, axis=0, keepdims=True)
            self.gradient_scale_gamma = np.sum(incoming_grad * self.x_norm, axis=0, keepdims=True)
            z = incoming_grad * self.scale_gamma

        gradient = (1.0 / self.std) * (
            z
            - np.mean(z, axis=-1, keepdims=True)
            - self.x_norm * np.mean(z * self.x_norm, axis=-1, keepdims=True)
        )
        return gradient.reshape(original_shape)

    def __str__(self):
        affine = "affine" if self.shift_scale else "no affine"
        return f"LayerNorm over {self.input_dimension}, {affine}"

    def __repr__(self):
        return self.__str__()


class RMSNormLayer(Layer):
    """
    Root-mean-square norm over the last axis. No mean subtraction and no shift, so the only parameter is a
    learnable per-feature scale.
    """
    parameter_names = ("scale_gamma",)
    cache_names = ("input", "x_norm", "rms")

    def __init__(self, input_dimension: int, padding_epsilon: float = 1e-10):
        """
        Parameters
        ----------
        input_dimension : size of the last axis
        padding_epsilon : added to the mean square for numerical stability (avoid zero divide
        """
        super().__init__()
        self.input_dimension = input_dimension
        self.padding_epsilon = padding_epsilon
        self.scale_gamma = np.ones((1, input_dimension))
        self.declare_shapes(inputs=((input_dimension,),), outputs=((input_dimension,),))
        self.zero_gradients()

    def forward(self, incoming_x: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        incoming_x : (..., ni)
        mask : unused; each position is normalized against only its own features, kept for pass-through
            compatibility with the graph
        """
        in_shape = incoming_x.shape
        input_reshaped = incoming_x.reshape(-1, in_shape[-1])
        if not self.training:
            return (self.scale_gamma
                    * (input_reshaped
                       / (np.sqrt(np.mean(input_reshaped ** 2, axis=-1, keepdims=True) + self.padding_epsilon))
                       )).reshape(in_shape)

        self.rms = np.sqrt(np.mean(input_reshaped ** 2, axis=-1, keepdims=True) + self.padding_epsilon)
        self.x_norm = input_reshaped / self.rms
        return (self.x_norm * self.scale_gamma).reshape(in_shape)

    def backward(self, incoming_grad: NDArray) -> NDArray:
        original_shape = incoming_grad.shape
        grad = incoming_grad.reshape(-1, original_shape[-1])
        self.gradient_scale_gamma = np.sum(grad * self.x_norm, axis=0, keepdims=True)

        z = grad * self.scale_gamma
        gradient = (z - self.x_norm * np.mean(z * self.x_norm, axis=-1, keepdims=True)) / self.rms
        return gradient.reshape(original_shape)

    def __str__(self):
        return f"RMSNorm over {self.input_dimension}"

    def __repr__(self):
        return self.__str__()


class StandardizeLayer(Layer):
    """
    Feature-wise standardization with a moving mean and standard deviation, for layers that expect normalized rows.

    Training mode folds every batch into the statistics before standardizing it; eval mode uses them as they stand.
    The statistics are not trained, so optimizers skip the layer.
    """
    preserves_shape = True
    state_names = ("x_means", "x_stds", "num_seen_samples")

    def __init__(self, input_dimension: int):
        """
        Parameters
        ----------
        input_dimension : size of the last axis
        """
        super().__init__()
        self.input_dimension = input_dimension
        self.x_means = np.zeros(input_dimension)
        self.x_stds = np.ones(input_dimension)
        self.num_seen_samples = 0.0
        self.declare_shapes(inputs=((input_dimension,),), outputs=((input_dimension,),))
        self.zero_gradients()

    def forward(self, incoming_x: NDArray, training_now: Optional[bool] = None) -> NDArray:
        """
        Parameters
        ----------
        incoming_x : (..., input_dimension)
        training_now : whether the batch updates the statistics; None follows train() / eval()

        Returns
        -------
        (..., input_dimension) standardized rows
        """
        training_now = self.training if training_now is None else training_now
        incoming_x = np.asarray(incoming_x)
        flat = incoming_x.reshape(-1, self.input_dimension)
        if training_now and self.num_seen_samples == 0:
            init_standardize(self, flat)
        elif training_now:
            batch_std = flat.std(axis=0, ddof=min(1, len(flat) - 1))
            update_running_standardize(self, flat.mean(axis=0), batch_std, len(flat))
        elif self.num_seen_samples == 0:
            raise ValueError("nothing has been seen yet; run a training pass first")
        return standardize(self, incoming_x)

    def backward(self, incoming_grad: NDArray) -> NDArray:
        return incoming_grad / (self.x_stds + EPSILON)

    def inverse(self, standardized_x: NDArray) -> NDArray:
        """Return standardized rows to their original units."""
        return unstandardize(self, standardized_x)

    def __str__(self):
        return f"Standardize over {self.input_dimension}, seen {self.num_seen_samples:g}"

    def __repr__(self):
        return self.__str__()
