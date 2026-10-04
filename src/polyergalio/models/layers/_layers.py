"""
Exercise: Layer and FullyConnectedLayer rebuilt on BasalEstimator.

RNG is a module-level stand-in for the global generator.
"""
from __future__ import annotations

from abc import ABC
from typing import Optional

import numpy as np
from numpy.typing import NDArray

import polyergalio.models.activations as activations
from polyergalio.base_model import BasalEstimator
from polyergalio.models.constants import GLOBAL_DTYPE
from polyergalio.models.weight_initialization import get_weight_init

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
        self.purge()

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
                 ni: int,
                 no: int,
                 activation_type: str,
                 is_output: bool = False,
                 initialization_override: Optional[str] = None,
                 initialization_kwargs: Optional[dict] = None,
                 ):
        """
        Parameters
        ----------
        ni : number of input units
        no : number of output units
        activation_type : name of an activation, 'linear', 'sigmoid', 'tanh', etc.
        is_output : marks the final layer of a network
        initialization_override : weight initializer name, used instead of the activation's default
        initialization_kwargs : keyword arguments bound to the initializer, e.g. {"std": 0.01}
        """
        super().__init__()
        self.ni = ni
        self.no = no
        self.activation_type = activation_type
        self.is_output = is_output
        self.initialization_override = initialization_override
        self.initialization_kwargs = dict(initialization_kwargs or {})

        initializer = get_weight_init(
            initialization_override or activation_type, **self.initialization_kwargs
        )
        self.weights: NDArray = initializer(RNG, ni=ni, no=no)
        self.bias: NDArray = np.zeros((1, no), dtype=GLOBAL_DTYPE)
        self.declare_shapes(inputs=((ni,),), outputs=((no,),))
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
        assert self.in_shape[-1] == self.weights.shape[0], (
            f"weights and xs don't match -- x:{incoming_x.shape} weights: {self.weights.shape}"
        )
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
