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
from polyergalio.models.embedding.positional import SinusoidEmbedding
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
    """Stack two arrays along the latent input_dimension (-1)."""
    cache_names = ("split_dims",)

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
        [array_a, array_b] joined along the hidden input_dimension
        """
        assert array_a.shape[0] == array_b.shape[0]

        if self.training:
            self.split_dims = (array_a.shape[-1], array_b.shape[-1])

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
    cache_names = ("shape_a", "shape_b", "expanded_dim")

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
        mask : unused; the operation is per-position, num_kept for pass-through compatibility with the graph
        """
        axis = 1 if self.broadcast_axis is None else self.broadcast_axis
        bcast_a, bcast_b, expanded_dim = array_a, array_b, None

        try:
            if array_a.ndim < array_b.ndim:
                bcast_a = np.expand_dims(array_a, axis)
                expanded_dim = (1, axis)
            elif array_a.ndim > array_b.ndim:
                bcast_b = np.expand_dims(array_b, axis)
                expanded_dim = (2, axis)
            output = self.combine(bcast_a, bcast_b)
        except ValueError as error:
            raise RuntimeError(f"{self.__class__.__name__}: Inputs not broadcastable. {error}")

        if self.training:
            self.shape_a, self.shape_b, self.expanded_dim = array_a.shape, array_b.shape, expanded_dim
            if "bcast_a" in self.cache_names:
                self.bcast_a, self.bcast_b = bcast_a, bcast_b
        return output

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

        return reduce_to_shape(grad_a, self.shape_a), reduce_to_shape(grad_b, self.shape_b)

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
    cache_names = BroadcastOperator.cache_names + ("bcast_a", "bcast_b")

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


def latent_expansion(num_latent_projections: int, sequence_length: int) -> NDArray:
    """
    Returns
    -------
    (sequence_length, num_latent_projections) weights in which position t reads the latent whose block of
    consecutive positions contains t; one latent is read by every position
    """
    expansion = np.zeros((sequence_length, num_latent_projections))
    for latent, block in enumerate(np.array_split(np.arange(sequence_length), num_latent_projections)):
        expansion[block, latent] = 1.0
    return expansion


class LatentExpand(Layer):
    """
    Project num_latent_projections latents up to sequence_length positions and add a SinusoidEmbedding.

    Position t receives sum_k expansion[t, k] * latent[k]. One latent is tiled over every position, with no
    parameters. Several use a learned (sequence_length, num_latent_projections) expansion that starts as
    consecutive blocks, the transpose of the readout of SpectrePooledAttention.

    Input shape: (batch, num_latent_projections, hidden_dimension)
    Output shape: (batch, sequence_length, hidden_dimension)

    restores_mask: the layer receives the padding mask even after a pooling layer consumed it, zeroes the padded
    positions, and passes the mask on to the layers after it.
    """
    restores_mask = True
    cache_names = ("token_mask", "latent")

    def __init__(self,
                 sequence_length: int,
                 hidden_dimension: int,
                 base_freq: int = 10000,
                 num_latent_projections: int = 1,
                 ):
        """
        Parameters
        ----------
        sequence_length : positions in the output
        hidden_dimension : channel width of the latent, must be even
        base_freq : wavelength base of the SinusoidEmbedding
        num_latent_projections : latents in the input, between 1 and sequence_length
        """
        super().__init__()
        assert hidden_dimension % 2 == 0, f"hidden_dimension must be even, got {hidden_dimension}"
        assert 1 <= num_latent_projections <= sequence_length, (
            f"num_latent_projections must be between 1 and sequence_length {sequence_length}, "
            f"got {num_latent_projections}"
        )
        self.sequence_length: int = sequence_length
        self.hidden_dimension: int = hidden_dimension
        self.base_freq = base_freq
        self.num_latent_projections: int = num_latent_projections
        self.position = SinusoidEmbedding(sequence_length, hidden_dimension, base_freq)

        if num_latent_projections > 1:
            self.expansion = latent_expansion(num_latent_projections, sequence_length)
            self.parameter_names += ("expansion",)
        self.declare_shapes(
            inputs=((num_latent_projections, hidden_dimension),), outputs=((sequence_length, hidden_dimension),)
        )
        self.zero_gradients()

    def forward(self, incoming_x: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        incoming_x : (batch, num_latent_projections, hidden_dimension)
        mask : (batch, sequence_length), 1 for a real position and 0 for padding
        """
        assert (incoming_x.ndim == 3
                and incoming_x.shape[1:] == (self.num_latent_projections, self.hidden_dimension)),\
            (f"expected (batch, {self.num_latent_projections}, {self.hidden_dimension}), got {incoming_x.shape}")

        if self.num_latent_projections > 1:
            expanded = np.einsum("tk,bkh->bth", self.expansion, incoming_x)
        else:
            expanded = np.repeat(incoming_x, self.sequence_length, axis=1)

        output = self.position(expanded)
        token_mask = None

        if mask is not None:
            assert mask.shape == output.shape[:2], (
                f"mask shape {mask.shape} must match the output's (batch, sequence) axes {output.shape[:2]}"
            )
            token_mask = mask[..., None].astype(output.dtype)
            output = output * token_mask
        if self.training:
            self.token_mask = token_mask
            self.latent = incoming_x
        return output

    def backward(self, incoming_grad: NDArray) -> NDArray:
        if self.token_mask is not None:
            incoming_grad = incoming_grad * self.token_mask
        position_grad = self.position.backward(incoming_grad)
        if self.num_latent_projections == 1:
            return position_grad.sum(axis=1, keepdims=True)
        self.gradient_expansion = np.einsum("bth,bkh->tk", position_grad, self.latent)
        return np.einsum("tk,bth->bkh", self.expansion, position_grad)

    def __str__(self):
        return (
            f"LatentExpand of {self.num_latent_projections} latent(s) over {self.sequence_length} positions, "
            f"{self.hidden_dimension} wide"
        )

    def __repr__(self):
        return self.__str__()


class PatchTokenizer(Layer):
    """
    Group patch_size consecutive positions into one token, or split tokens back into positions.

    (batch, sequence, channels) -> (batch, sequence / patch_size, patch_size * channels), each token holding its
    patch's positions in order. inverse=True is the opposite reshape. The two directions are each other's backward,
    and there are no parameters: put a FullyConnectedLayer after the tokenizer to embed the patches, and before the
    inverse to project back to patch_size * channels.

    Patchifying consumes the padding mask, since the sequence axis shrinks; the inverse restores it and zeroes
    the padded positions.

    Input shape: (batch, sequence, channels), sequence a multiple of patch_size
    Output shape: (batch, sequence / patch_size, patch_size * channels)
    """
    cache_names = ("token_mask",)

    def __init__(self, patch_size: int, inverse: bool = False):
        """
        Parameters
        ----------
        patch_size : positions per token
        inverse : split tokens of width patch_size * channels back into patch_size positions each
        """
        super().__init__()
        assert patch_size >= 1, f"patch_size must be at least 1, got {patch_size}"
        self.patch_size = patch_size
        self.inverse = inverse
        self.consumes_mask = not inverse
        self.restores_mask = inverse
        self.preserves_shape = False
        self.declare_shapes(inputs=(ANY_SHAPE,), outputs=(ANY_SHAPE,))

    def infer_output_shapes(self, input_shapes: tuple[tuple, ...]) -> tuple[tuple, ...]:
        shape = input_shapes[0]
        width = shape[-1]
        if width is None:
            return (ANY_SHAPE,)
        new_width = width // self.patch_size if self.inverse else width * self.patch_size
        if len(shape) == 1:
            return ((new_width,),)
        length = shape[-2]
        if length is not None:
            length = length * self.patch_size if self.inverse else length // self.patch_size
        return ((length, new_width),)

    def regroup(self, values: NDArray, inverse: bool) -> NDArray:
        batch, length, width = values.shape
        if inverse:
            assert width % self.patch_size == 0, f"width {width} must be a multiple of patch_size {self.patch_size}"
            return values.reshape(batch, length * self.patch_size, width // self.patch_size)
        assert length % self.patch_size == 0, f"sequence {length} must be a multiple of patch_size {self.patch_size}"
        return values.reshape(batch, length // self.patch_size, width * self.patch_size)

    def forward(self, incoming_x: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        incoming_x : (batch, sequence, channels); (batch, tokens, patch_size * channels) when inverse
        mask : (batch, positions), only read when inverse: 1 for a real position and 0 for padding
        """
        assert incoming_x.ndim == 3, f"expected (batch, sequence, channels), got {incoming_x.shape}"
        output = self.regroup(incoming_x, self.inverse)
        token_mask = None
        if self.inverse and mask is not None:
            assert mask.shape == output.shape[:2], (
                f"mask shape {mask.shape} must match the output's (batch, sequence) axes {output.shape[:2]}"
            )
            token_mask = mask[..., None].astype(output.dtype)
            output = output * token_mask
        if self.training:
            self.token_mask = token_mask
        return output

    def backward(self, incoming_grad: NDArray) -> NDArray:
        if self.inverse and self.token_mask is not None:
            incoming_grad = incoming_grad * self.token_mask
        return self.regroup(incoming_grad, not self.inverse)

    def __str__(self):
        return f"PatchTokenizer, patch size {self.patch_size}{', inverse' if self.inverse else ''}"

    def __repr__(self):
        return self.__str__()


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
        if self.training:
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
        incoming_grads : one gradient per slice, in the order forward returned them; None or missing for a dropped slice

        Returns
        -------
        gradient with respect to the input, padding removed
        """
        length = self.in_shape[self.split_axis]
        num_slices = -(-length // self.slice_size)
        if len(incoming_grads) > num_slices:
            raise ValueError(f"expected at most {num_slices} gradients, got {len(incoming_grads)}")

        present = next(np.asarray(grad) for grad in incoming_grads if grad is not None)
        slice_shape = list(self.in_shape)
        slice_shape[self.split_axis] = self.slice_size
        grads = list(incoming_grads) + [None] * (num_slices - len(incoming_grads))
        joined = np.concatenate(
            [np.zeros(slice_shape, dtype=present.dtype) if grad is None else np.asarray(grad) for grad in grads],
            axis=self.split_axis
        )
        keep = [slice(None)] * joined.ndim
        keep[self.split_axis] = slice(0, length)

        return joined[tuple(keep)]

    def __str__(self):
        return f"Split of the {self.target_axis} axis into slices of {self.slice_size}"

    def __repr__(self):
        return self.__str__()


class ShiftRight(Layer):
    """
    Shift a sequence right by one position along the sequence axis, so position t sees position t-1's value
    instead of its own. The vacated first position holds the learnable start_token.
    """
    preserves_shape = True
    parameter_names = ("start_token",)

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
        return np.concatenate([start, input_data[:, :-1, :]], axis=1)

    def backward(self, incoming_gradient: NDArray) -> NDArray:
        self.gradient_start_token = incoming_gradient[:, :1, :].sum(axis=0, keepdims=True)
        grad_input = np.zeros_like(incoming_gradient)
        grad_input[:, :-1, :] = incoming_gradient[:, 1:, :]
        return grad_input

    def __str__(self):
        return f"ShiftRight, hidden {self.hidden_dim}"

    def __repr__(self):
        return self.__str__()


class FlattenLayer(Layer):
    """
    Flattens every axis after the batch axis into one, so a dense layer sees each sample as a single vector. The
    gradient is reshaped back on the way back

    *Has no learnable parameters*

    Input shape: (batch, ..., hidden)
    Output shape: (batch, product of the trailing axes)
    """
    preserves_shape = False
    cache_names = ("in_shape",)

    def __init__(self):
        super().__init__()
        self.declare_shapes(inputs=((None, None),), outputs=((None,),))

    def forward(self, incoming_x: NDArray) -> NDArray:
        if self.training:
            self.in_shape = incoming_x.shape
        return incoming_x.reshape(incoming_x.shape[0], -1)

    def backward(self, incoming_grad: NDArray) -> NDArray:
        return incoming_grad.reshape(self.in_shape)


class MaskGather(Layer):
    """
    Selects specific positions out of a (batch, sequence, hidden) for a downstream head to run on, and
    sends the gradient back into the full shape on the way back

    Input shape: (batch, sequence, hidden)
    Output shape: (num_selected, hidden), flattened across sequence and hidden into -1
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
        target_mask : (batch, sequence) or (batch, sequence, 1) bool or 0-1

        Returns
        -------
        (num_selected, hidden) the selected positions
        """
        select = np.ones(incoming_x.shape[:2], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        if target_mask is not None:
            select = select & np.asarray(target_mask, dtype=bool)

        if self.training:
            self.in_shape, self.select_mask = incoming_x.shape, select
        return incoming_x[select]

    def backward(self, incoming_grad: NDArray) -> NDArray:
        full_grad = np.zeros(self.in_shape, dtype=incoming_grad.dtype)
        full_grad[self.select_mask] = incoming_grad
        return full_grad
