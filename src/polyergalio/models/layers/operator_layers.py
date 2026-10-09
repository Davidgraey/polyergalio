"""
Operations -- track the incoming streams and backpropagate appropriately.
I started calling them latentXYZ because I was using them in latent spaces. No other reason.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from polyergalio.models.constants import ANY_SHAPE
from polyergalio.models.layers.basic_layers import Layer


def reduce_to_shape(grad: NDArray, target_shape: tuple) -> NDArray:
    """Sum a gradient down to target_shape wherever the forward pass broadcast."""
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(target_shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class LatentStack(Layer):
    """Stack two arrays along the latent dimension (-1)."""
    cache_names = ("split_dims", "a_shape", "b_shape")

    def __init__(self):
        super().__init__()
        self.declare_shapes(inputs=(ANY_SHAPE, ANY_SHAPE), outputs=(ANY_SHAPE,))

    def infer_output_shapes(self, input_shapes: tuple[tuple, ...]) -> tuple[tuple, ...]:
        """The output width is the sum of the incoming widths, so it is known only once they enter the layer."""
        widths = [shape[-1] for shape in input_shapes]
        if any(width is None for width in widths):
            return (ANY_SHAPE,)
        return ((sum(widths),),)

    def forward(self, array_a: NDArray, array_b: NDArray = None) -> NDArray:
        """
        Returns
        -------
        [array_a, array_b] joined along the hidden dimension
        """
        self.a_shape = array_a.shape
        self.b_shape = array_b.shape
        assert self.a_shape[0] == self.b_shape[0]

        self.split_dims = (self.a_shape[-1], self.b_shape[-1])

        if array_a.ndim == 1:
            return np.hstack((array_a.ravel(), array_b.ravel()))
        elif array_a.ndim == 2:
            return np.hstack((array_a, array_b))
        elif array_a.ndim == 3:
            return np.dstack((array_a, array_b))

    def backward(self, incoming_gradient: NDArray) -> tuple[NDArray, NDArray]:
        """
        Returns
        -------
        the gradient split back into slices, ordered like the inputs to forward (a, b)
        """
        a_dim, b_dim = self.split_dims
        return incoming_gradient[..., :a_dim], incoming_gradient[..., a_dim:a_dim + b_dim]


class BroadcastOperator(Layer, ABC):
    """
    Element-wise operation on two arrays. The lower-rank input is expanded at broadcast_axis (default 1), and
    gradients are summed back down to each input's own shape.

    A subclass supplies combine(), the operation, and partials(), the gradient with respect to each broadcast input.
    """
    cache_names = (
        "shape_a", "shape_b", "bcast_a", "bcast_b", "expanded_dim",
        "array_a", "array_b", "output", "gradient_array_a", "gradient_array_b",
    )

    def __init__(self, broadcast_axis: Optional[int] = None):
        super().__init__()
        self.broadcast_axis = broadcast_axis
        self.declare_shapes(inputs=(ANY_SHAPE, ANY_SHAPE), outputs=(ANY_SHAPE,))

    @abstractmethod
    def combine(self, array_a: NDArray, array_b: NDArray) -> NDArray:
        """The operation, applied to the broadcast inputs."""

    @abstractmethod
    def partials(self, incoming_grad: NDArray) -> tuple[NDArray, NDArray]:
        """Gradient with respect to each broadcast input, before it is summed back to the input's shape."""

    def forward(self, array_a: NDArray, array_b: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        mask : unused; the operation is per-position, kept for pass-through compatibility with the graph
        """
        self.shape_a = array_a.shape
        self.shape_b = array_b.shape
        axis = 1 if self.broadcast_axis is None else self.broadcast_axis
        self.bcast_a, self.bcast_b, self.expanded_dim = array_a, array_b, None

        try:
            if array_a.ndim < array_b.ndim:
                self.bcast_a = np.expand_dims(array_a, axis)
                self.expanded_dim = (1, axis)
            elif array_a.ndim > array_b.ndim:
                self.bcast_b = np.expand_dims(array_b, axis)
                self.expanded_dim = (2, axis)
            self.output = self.combine(self.bcast_a, self.bcast_b)
        except ValueError as error:
            raise RuntimeError(f"{self.__class__.__name__}: Inputs not broadcastable. {error}")

        self.array_a = array_a
        self.array_b = array_b
        return self.output

    def backward(self, incoming_grad: NDArray) -> tuple[NDArray, NDArray]:
        """
        Returns
        -------
        gradient with respect to array_a and array_b, each in its own input shape
        """
        grad_a, grad_b = self.partials(incoming_grad)
        if self.expanded_dim is not None:
            which, axis = self.expanded_dim
            if which == 1:
                grad_a = grad_a.sum(axis=axis)
            else:
                grad_b = grad_b.sum(axis=axis)

        self.gradient_array_a = reduce_to_shape(grad_a, self.shape_a)
        self.gradient_array_b = reduce_to_shape(grad_b, self.shape_b)
        return self.gradient_array_a, self.gradient_array_b

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return f"{self}"


class LatentSum(BroadcastOperator):
    """array_a + array_b, element-wise"""

    def combine(self, array_a: NDArray, array_b: NDArray) -> NDArray:
        return array_a + array_b

    def partials(self, incoming_grad: NDArray) -> tuple[NDArray, NDArray]:
        return incoming_grad, incoming_grad


class LatentProduct(BroadcastOperator):
    """array_a * array_b, element-wise"""

    def combine(self, array_a: NDArray, array_b: NDArray) -> NDArray:
        return array_a * array_b

    def partials(self, incoming_grad: NDArray) -> tuple[NDArray, NDArray]:
        return incoming_grad * self.bcast_b, incoming_grad * self.bcast_a


class LatentDifference(BroadcastOperator):
    """array_a - array_b, element-wise; array_b's gradient carries the sign flip"""

    def combine(self, array_a: NDArray, array_b: NDArray) -> NDArray:
        return array_a - array_b

    def partials(self, incoming_grad: NDArray) -> tuple[NDArray, NDArray]:
        return incoming_grad, -incoming_grad


class SplitLayer(Layer):
    """
    Divides one axis into consecutive slices of slice_size indices, zero-padding the end so they are all equal.

    The slices come back as a tuple, ceil(length / slice_size) -- so indexing matters here!

    A core assumption: arrays formatted where sequence = dim1, hidden = dim-1
    axis "sequence": (batch, sequence, hidden) -> n x (batch, slice_size, hidden)
    axis "hidden": (batch, sequence, hidden) -> n x (batch, sequence, slice_size)
    """
    parameter_names = ()
    cache_names = ("in_shape",)
    multi_output = True

    def __init__(self, target_axis: str, slice_size: int, hidden_dimension: int):
        """
        Parameters
        ----------
        target_axis : "sequence" (axis 1) or "hidden" (last axis)
        slice_size : number of indices along the axis in each slice
        """
        super().__init__()
        if target_axis not in ("sequence", "hidden"):
            raise ValueError(f"axis must be 'sequence' or 'hidden', got {target_axis!r}")

        self.target_axis = target_axis
        self.hidden_dimension = hidden_dimension
        self.slice_size = slice_size

        outputs = ((slice_size,),) if target_axis == "hidden" else (self.hidden_dimension,)
        self.declare_shapes(inputs=(ANY_SHAPE,), outputs=outputs)

    @property
    def split_axis(self) -> int:
        """Index of the axis that is divided."""
        return 1 if self.target_axis == "sequence" else -1

    def infer_output_shapes(self, input_shapes: tuple[tuple, ...]) -> tuple[tuple, ...]:
        if self.target_axis == "hidden":
            return ((self.slice_size,),)
        return (input_shapes[0],)

    def forward(self, input_data: NDArray) -> tuple[NDArray, ...]:
        """
        Parameters
        ----------
        input_data : (batch, sequence, hidden); (batch, hidden) when axis is "hidden"

        Returns
        -------
        tuple of equal slices along the axis, the last padded with zeros
        """
        if self.target_axis == "sequence" and input_data.ndim < 2:
            raise ValueError(f"axis 'sequence' needs (batch, sequence, ...), got shape {input_data.shape}")
        self.in_shape = input_data.shape
        length = input_data.shape[self.split_axis]
        num_slices = -(-length // self.slice_size)
        padding = [(0, 0)] * input_data.ndim
        padding[self.split_axis] = (0, num_slices * self.slice_size - length)

        return tuple(np.split(np.pad(input_data, padding), num_slices, axis=self.split_axis))

    def backward(self, incoming_grads: tuple[NDArray, ...]) -> NDArray:
        """
        Parameters
        ----------
        incoming_grads : one gradient per slice, in the order forward returned them

        Returns
        -------
        gradient with respect to the input, padding removed
        """
        length = self.in_shape[self.split_axis]
        num_slices = -(-length // self.slice_size)
        if len(incoming_grads) != num_slices:
            raise ValueError(f"expected {num_slices} gradients, got {len(incoming_grads)}")

        joined = np.concatenate(
            [np.asarray(grad) for grad in incoming_grads],
            axis=self.split_axis
        )
        keep = [slice(None)] * joined.ndim
        keep[self.split_axis] = slice(0, length)

        return joined[tuple(keep)]

    def __str__(self):
        return f"Split of the {self.axis} axis into slices of {self.slice_size}"

    def __repr__(self):
        return self.__str__()


class ShiftRight(Layer):
    """
    Shift a sequence right by one position along the sequence axis, so position t sees position t-1's value
    instead of its own. The vacated first position holds the learnable start_token.
    """
    preserves_shape = True
    parameter_names = ("start_token",)
    cache_names = ("output",)

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.start_token: NDArray = np.zeros((1, 1, hidden_dim))

        self.declare_shapes(inputs=((hidden_dim,),), outputs=((hidden_dim,),))
        self.zero_gradients()

    def forward(self, input_data: NDArray) -> NDArray:
        """
        Parameters
        ----------
        input_data : (batch, sequence, hidden)
        """
        assert input_data.ndim == 3, (
            f"expected (batch, sequence, hidden), got shape {input_data.shape}"
        )
        assert input_data.shape[-1] == self.hidden_dim, (
            f"built for hidden_dim {self.hidden_dim}, got {input_data.shape[-1]}"
        )
        batch = input_data.shape[0]
        start = np.broadcast_to(self.start_token, (batch, 1, self.hidden_dim))
        self.output = np.concatenate([start, input_data[:, :-1, :]], axis=1)
        return self.output

    def backward(self, incoming_gradient: NDArray) -> NDArray:
        self.gradient_start_token = incoming_gradient[:, :1, :].sum(axis=0, keepdims=True)
        grad_input = np.zeros_like(incoming_gradient)
        grad_input[:, :-1, :] = incoming_gradient[:, 1:, :]
        return grad_input

    def __str__(self):
        return f"ShiftRight, hidden {self.hidden_dim}"

    def __repr__(self):
        return self.__str__()


class MaskGather(Layer):
    """
    Selects specific positions out of a (batch, sequence, hidden) tensor for a downstream head to run on, and
    scatters the gradient back into the full shape on the way back. No learnable parameters.

    Input shape: (batch, sequence, hidden)
    Output shape: (num_selected, hidden), flattened across batch and sequence, since a downstream head only needs
    a bag of vectors, not which row or position each one came from
    """
    preserves_shape = False
    cache_names = ("select_mask", "in_shape")

    def __init__(self):
        super().__init__()
        self.declare_shapes(inputs=((None, None, None),), outputs=((None, None),))

    def forward(self, incoming_x: NDArray, mask: Optional[NDArray] = None, target_mask: Optional[NDArray] = None):
        """
        Parameters
        ----------
        incoming_x : (batch, sequence, hidden)
        mask : (batch, sequence) or (batch, sequence, 1) attention mask
        target_mask : (batch, sequence) or (batch, sequence, 1) bool or 0-1, which positions to pass forward
            (for training tasks like MLM); None keeps everything

        Returns
        -------
        (num_selected, hidden) the selected positions
        """
        self.in_shape = incoming_x.shape

        if mask is not None:
            self.select_mask = np.asarray(mask, dtype=bool)
        else:
            self.select_mask = np.ones(incoming_x.shape[:2], dtype=bool)

        if target_mask is not None:
            self.select_mask &= np.asarray(target_mask, dtype=bool)

        return incoming_x[self.select_mask]

    def backward(self, incoming_grad: NDArray) -> NDArray:
        full_grad = np.zeros(self.in_shape, dtype=incoming_grad.dtype)
        full_grad[self.select_mask] = incoming_grad
        return full_grad
