"""
Fixed positional embeddings: parameterless layers whose tables are built from the constructor arguments.
"""
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from polyergalio.models.layers.basic_layers import Layer


def position_angles(sequence_length: int, embedding_dimension: int, base_freq: int) -> NDArray:
    """Angle of every position at every frequency, shaped (sequence, dimension / 2)."""
    inv_freq = 1.0 / (base_freq ** (np.arange(0, embedding_dimension, 2) / embedding_dimension))
    return np.einsum("i,j->ij", np.arange(sequence_length, dtype=float), inv_freq)


class PositionalLayer(Layer):
    """
    Shared checks for fixed positional layers.

    The table is rebuilt from the constructor arguments, so it is neither a parameter nor state.
    """
    cache_names = ("output",)

    def __init__(self, sequence_length: int, embedding_dimension: int, base_freq: int = 10000):
        """
        Parameters
        ----------
        sequence_length : the ceiling the table is built for. A shorter sequence takes the leading rows
        embedding_dimension : channel width of the incoming embedding, must be even
        base_freq : wavelength base, 10,000 in the paper
        """
        super().__init__()
        assert embedding_dimension % 2 == 0, f"embedding_dimension must be even, got {embedding_dimension}"
        self.sequence_length: int = sequence_length
        self.embedding_dimension: int = embedding_dimension
        self.base_freq: int = base_freq
        self.declare_shapes(inputs=((embedding_dimension,),), outputs=((embedding_dimension,),))

    def check_input(self, input_data: NDArray) -> int:
        """The sequence length of input_data, after checking it fits the table."""
        assert input_data.ndim >= 2, f"expected at least (sequence, dimension), got {input_data.shape}"
        sequence, dimension = input_data.shape[-2:]
        assert dimension == self.embedding_dimension, (
            f"built for embedding_dimension {self.embedding_dimension}, got {dimension}"
        )
        assert sequence <= self.sequence_length, (
            f"table holds {self.sequence_length} positions, got a sequence of {sequence}"
        )
        return sequence


class RopeEmbedding(PositionalLayer):
    """
    Rotary positional embedding: a fixed rotation of adjacent channel pairs by an angle proportional to the
    token's position.
    """

    def __init__(self, sequence_length: int, embedding_dimension: int, base_freq: int = 10000):
        super().__init__(sequence_length, embedding_dimension, base_freq)
        self.build_rope_array(sequence_length, embedding_dimension, base_freq)

    def build_rope_array(self, sequence_length: int, embedding_dimension: int, base_freq: int = 10000) -> None:
        """Cache the sine and cosine tables, shaped (sequence, dimension / 2)."""
        angles = position_angles(sequence_length, embedding_dimension, base_freq)
        self.pos_sine = np.sin(angles)
        self.pos_cosine = np.cos(angles)

    def forward(self, input_data: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        input_data : (sequence, dimension) or (batch, sequence, dimension)
        mask : unused; each position is rotated by its own index, kept for pass-through compatibility with the graph

        Returns
        -------
        the input with each channel pair rotated by its position's angle
        """
        sequence = self.check_input(input_data)
        sine = self.pos_sine[:sequence]
        cosine = self.pos_cosine[:sequence]
        even = input_data[..., 0::2]
        odd = input_data[..., 1::2]

        self.output = np.zeros_like(input_data)
        self.output[..., 0::2] = even * cosine - odd * sine
        self.output[..., 1::2] = even * sine + odd * cosine
        return self.output

    def backward(self, incoming_gradient: NDArray) -> NDArray:
        """The transpose of a rotation is the inverse rotation: the same angles in the opposite direction."""
        sequence = incoming_gradient.shape[-2]
        sine = self.pos_sine[:sequence]
        cosine = self.pos_cosine[:sequence]
        even = incoming_gradient[..., 0::2]
        odd = incoming_gradient[..., 1::2]

        gradient = np.zeros_like(incoming_gradient)
        gradient[..., 0::2] = even * cosine + odd * sine
        gradient[..., 1::2] = -even * sine + odd * cosine
        return gradient

    @property
    def rope_array(self) -> tuple[NDArray, NDArray]:
        """The (sine, cosine) tables, copied so callers cannot corrupt them."""
        return self.pos_sine.copy(), self.pos_cosine.copy()

    def __str__(self):
        return f"RoPE embedding matrix of {self.sequence_length} length, at {self.embedding_dimension} embedding dimension"

    def __repr__(self):
        return self.__str__()


class SinusoidEmbedding(PositionalLayer):
    """Fixed sine and cosine positional embedding, added to the input; sine on even channels, cosine on odd."""

    def __init__(self, sequence_length: int, embedding_dimension: int, base_freq: int = 10000):
        super().__init__(sequence_length, embedding_dimension, base_freq)
        self.build_sinusoid_array(sequence_length, embedding_dimension, base_freq)

    def build_sinusoid_array(self, sequence_length: int, embedding_dimension: int, base_freq: int = 10000) -> None:
        """Cache the table, shaped (sequence, dimension)."""
        angles = position_angles(sequence_length, embedding_dimension, base_freq)
        self.pos_table = np.zeros((sequence_length, embedding_dimension))
        self.pos_table[:, 0::2] = np.sin(angles)
        self.pos_table[:, 1::2] = np.cos(angles)

    def forward(self, input_data: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        input_data : (sequence, dimension) or (batch, sequence, dimension)
        mask : unused; each position adds its own row of the table, kept for pass-through compatibility with the graph

        Returns
        -------
        the input with the position table added
        """
        sequence = self.check_input(input_data)
        self.output = input_data + self.pos_table[:sequence]
        return self.output

    def backward(self, incoming_gradient: NDArray) -> NDArray:
        """Adding a constant has an identity Jacobian, so the gradient passes through."""
        return incoming_gradient

    @property
    def sinusoid_array(self) -> NDArray:
        """The table, copied so callers cannot corrupt it."""
        return self.pos_table.copy()

    def __str__(self):
        return f"sinusoid embedding table of {self.sequence_length} length, at {self.embedding_dimension} embedding dimension"

    def __repr__(self):
        return self.__str__()
