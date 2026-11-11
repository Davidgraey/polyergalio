"""
Fixed positional embeddings: parameterless layers whose tables are built from the constructor arguments.

Also the linear-chain conditional random field, the output layer that scores a sequence of tags by position.
"""
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from polyergalio.models.layers.basic_layers import Layer


def logsumexp(values: NDArray, axis: int) -> NDArray:
    """log(sum(exp(values))) along axis, stable."""
    peak = values.max(axis=axis, keepdims=True)
    return (peak + np.log(np.exp(values - peak).sum(axis=axis, keepdims=True))).squeeze(axis)


def position_angles(sequence_length: int, embedding_dimension: int, base_freq: int) -> NDArray:
    """Angle of every position at every frequency, shaped (sequence, input_dimension / 2)."""
    inv_freq = 1.0 / (base_freq ** (np.arange(0, embedding_dimension, 2) / embedding_dimension))
    return np.einsum("i,j->ij", np.arange(sequence_length, dtype=float), inv_freq)


class PositionalLayer(Layer):
    """
    Shared checks for fixed positional layers.

    The table is rebuilt from the constructor arguments, so it is neither a parameter nor state.
    """
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
        assert input_data.ndim >= 2, f"expected at least (sequence, input_dimension), got {input_data.shape}"
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
        """Cache the sine and cosine tables, shaped (sequence, input_dimension / 2)."""
        angles = position_angles(sequence_length, embedding_dimension, base_freq)
        self.pos_sine = np.sin(angles)
        self.pos_cosine = np.cos(angles)

    def forward(self, input_data: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        input_data : (sequence, input_dimension) or (batch, sequence, input_dimension)
        mask : unused; each position is rotated by its own index, num_kept for pass-through compatibility with the graph

        Returns
        -------
        the input with each channel pair rotated by its position's angle
        """
        sequence = self.check_input(input_data)
        sine = self.pos_sine[:sequence]
        cosine = self.pos_cosine[:sequence]
        even = input_data[..., 0::2]
        odd = input_data[..., 1::2]

        output = np.zeros_like(input_data)
        output[..., 0::2] = even * cosine - odd * sine
        output[..., 1::2] = even * sine + odd * cosine
        return output

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
        return f"RoPE embedding matrix of {self.sequence_length} length, at {self.embedding_dimension} embedding input_dimension"

    def __repr__(self):
        return self.__str__()


class SinusoidEmbedding(PositionalLayer):
    """Fixed sine and cosine positional embedding, added to the input; sine on even channels, cosine on odd."""

    def __init__(self, sequence_length: int, embedding_dimension: int, base_freq: int = 10000):
        super().__init__(sequence_length, embedding_dimension, base_freq)
        self.build_sinusoid_array(sequence_length, embedding_dimension, base_freq)

    def build_sinusoid_array(self, sequence_length: int, embedding_dimension: int, base_freq: int = 10000) -> None:
        """Cache the table, shaped (sequence, input_dimension)."""
        angles = position_angles(sequence_length, embedding_dimension, base_freq)
        self.pos_table = np.zeros((sequence_length, embedding_dimension))
        self.pos_table[:, 0::2] = np.sin(angles)
        self.pos_table[:, 1::2] = np.cos(angles)

    def forward(self, input_data: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        input_data : (sequence, input_dimension) or (batch, sequence, input_dimension)
        mask : unused; each position adds its own row of the table, num_kept for pass-through compatibility with the graph

        Returns
        -------
        the input with the position table added
        """
        sequence = self.check_input(input_data)
        return input_data + self.pos_table[:sequence]

    def backward(self, incoming_gradient: NDArray) -> NDArray:
        """Adding a constant has an identity Jacobian, so the gradient passes through."""
        return incoming_gradient

    @property
    def sinusoid_array(self) -> NDArray:
        """The table, copied so callers cannot corrupt it."""
        return self.pos_table.copy()

    def __str__(self):
        return f"sinusoid embedding table of {self.sequence_length} length, at {self.embedding_dimension} embedding input_dimension"

    def __repr__(self):
        return self.__str__()


class ConditionalRandomField(Layer):
    """
    Linear-chain conditional random field over a sequence of labels, the final layer of a Network.
    Our conditional random field helps shape transitions across sequences - the p of label 2 following label 1

    The score of a label sequence given emissions E is
        start[y_0] + sum_t E[t, y_t] + sum_t transitions[y_{t-1}, y_t] + end[y_last]

    forward passes the emissions through unchanged, so the layer before it produces per-position label scores and this
    one owns the label flow transition to other label scores.
    Padding must follow the real positions in each sequence.

    Training step:
        emissions = network(x, mask=mask)
        loss, gradient = crf.negative_log_likelihood(emissions, labels, mask)
        network.backward(gradient)

    The final trained components is the self.transitions, a matrix of probs for t_n to t_n+1 states
        shaped [num_labels, num_labels]
    """
    parameter_names = ("transitions", "start_scores", "end_scores")
    cache_names = ("pending_gradients",)

    def __init__(self, num_labels: int):
        """
        Parameters
        ----------
        num_labels : K, the number of distinct labels or actions
        """
        super().__init__()
        self.num_labels: int = num_labels
        self.transitions: NDArray = np.zeros((num_labels, num_labels))
        self.start_scores: NDArray = np.zeros(num_labels)
        self.end_scores: NDArray = np.zeros(num_labels)
        self.declare_shapes(inputs=((num_labels,),), outputs=((num_labels,),))
        self.zero_gradients()

    def forward(self, emissions: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        emissions : (..., sequence, num_tags) per-position tag scores
        mask : (..., sequence), 1 for a real position and 0 for padding; unused here and read by the methods below

        Returns
        -------
        the emissions, unchanged
        """
        assert emissions.ndim >= 2 and emissions.shape[-1] == self.num_labels, (
            f"expected (..., sequence, {self.num_labels}), got {emissions.shape}"
        )
        return emissions

    def backward(self, incoming_gradient: NDArray) -> NDArray:
        """
        Parameters
        ----------
        incoming_gradient : the gradient from negative_log_likelihood, passed on as it is

        Returns
        -------
        the gradient with respect to the emissions, after setting the gradients of the tag-to-tag scores
        """
        assert self.pending_gradients is not None, "call negative_log_likelihood before backward"
        self.gradient_transitions, self.gradient_start_scores, self.gradient_end_scores = self.pending_gradients
        return incoming_gradient

    def negative_log_likelihood(self,
                                emissions: NDArray,
                                labels: NDArray,
                                mask: Optional[NDArray] = None,
                                ) -> tuple[float, NDArray]:
        """
        Parameters
        ----------
        emissions : (..., sequence, num_labels)
        labels : (..., sequence) integer tag at every position
        mask : (..., sequence), 1 for a real position and 0 for padding

        Returns
        -------
        -log p(labels | emissions) averaged over sequences, and its gradient with respect to the emissions, to pass to
        backward
        """
        scores, valid = self.flatten(emissions, mask)
        flat_labels = np.where(valid > 0, labels.reshape(len(scores), -1), 0)
        rows = np.arange(len(scores))
        last = valid.sum(axis=1).astype(int) - 1
        alpha, beta, log_partition = self.forward_backward(scores, valid)

        gold = np.take_along_axis(scores,
                                  flat_labels[..., None],
                                  axis=2)[..., 0]
        gold_score = (
                (gold * valid).sum(axis=1)
                + self.start_scores[flat_labels[:, 0]]
                + self.end_scores[flat_labels[rows, last]]
        )
        pair_valid = valid[:, 1:]
        gold_score += (self.transitions[flat_labels[:, :-1], flat_labels[:, 1:]] * pair_valid).sum(axis=1)
        loss = float((log_partition - gold_score).mean())

        count = len(scores)
        marginals = np.exp(alpha + beta - log_partition[:, None, None]) * valid[..., None]
        one_hot = np.eye(self.num_tags, dtype=scores.dtype)[flat_labels] * valid[..., None]

        pairs = np.exp(
            alpha[:, :-1, :, None] + self.transitions + (scores[:, 1:] + beta[:, 1:])[:, :, None, :]
            - log_partition[:, None, None, None]
        ) * pair_valid[..., None, None]

        gold_pairs = one_hot[:, :-1, :, None] * one_hot[:, 1:, None, :]

        self.pending_gradients = (
            (pairs - gold_pairs).sum(axis=(0, 1)) / count,
            (marginals[:, 0] - one_hot[:, 0]).sum(axis=0) / count,
            (marginals[rows, last] - one_hot[rows, last]).sum(axis=0) / count,
        )
        return loss, ((marginals - one_hot) / count).reshape(emissions.shape)

    def marginals(self, emissions: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        emissions : (..., sequence, num_labels)
        mask : (..., sequence), 1 for a real position and 0 for padding

        Returns
        -------
        p(label at each position | emissions), same shape as emissions, zeros at padding
        """
        scores, valid = self.flatten(emissions, mask)
        alpha, beta, log_partition = self.forward_backward(scores, valid)
        marginals = np.exp(alpha + beta - log_partition[:, None, None]) * valid[..., None]
        return marginals.reshape(emissions.shape)

    def decode(self, emissions: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        emissions : (..., sequence, num_labels)
        mask : (..., sequence), 1 for a real position and 0 for padding

        Returns
        -------
        (..., sequence) the single highest-scoring label sequence by Viterbi, 0 at padding
        """
        scores, valid = self.flatten(emissions, mask)
        count, length, _ = scores.shape
        rows = np.arange(count)
        identity = np.broadcast_to(np.arange(self.num_labels), (count, self.num_labels))
        best = self.start_scores + scores[:, 0]
        pointers = np.zeros((count, length, self.num_labels), dtype=int)
        pointers[:, 0] = identity
        for position in range(1, length):
            candidates = best[:, :, None] + self.transitions
            step_best = candidates.max(axis=1) + scores[:, position]
            step_pointer = candidates.argmax(axis=1)
            real = valid[:, position, None] > 0
            best = np.where(real, step_best, best)
            pointers[:, position] = np.where(real, step_pointer, identity)

        labels = np.zeros((count, length), dtype=int)
        labels[:, -1] = (best + self.end_scores).argmax(axis=1)
        for position in range(length - 1, 0, -1):
            labels[:, position - 1] = pointers[rows, position, labels[:, position]]
        return (labels * valid.astype(int)).reshape(emissions.shape[:-1])

    def flatten(self, emissions: NDArray, mask: Optional[NDArray]) -> tuple[NDArray, NDArray]:
        """(batch, sequence, num_labels) emissions and a (batch, sequence) mask, ones without one."""
        assert emissions.shape[-1] == self.num_labels, f"expected {self.num_labels} tags, got {emissions.shape[-1]}"
        scores = emissions.reshape(-1, *emissions.shape[-2:])
        if mask is None:
            return scores, np.ones(scores.shape[:2], dtype=scores.dtype)
        assert mask.shape == emissions.shape[:-1], f"mask {mask.shape} must match {emissions.shape[:-1]}"
        valid = mask.reshape(scores.shape[:2]).astype(scores.dtype)
        assert np.all(valid[:, :1] > 0) and np.all(np.diff(valid, axis=1) <= 0), "padding must follow real positions"
        return scores, valid

    def forward_backward(self, scores: NDArray, valid: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """
        Parameters
        ----------
        scores : (batch, sequence, num_labels) emissions
        valid : (batch, sequence) 1 for a real position and 0 for padding

        Returns
        -------
        log forward and backward messages, each (batch, sequence, num_labels), and the log partition, (batch,)
        """
        _, length, _ = scores.shape
        alpha = np.empty_like(scores)
        beta = np.empty_like(scores)
        alpha[:, 0] = self.start_scores + scores[:, 0]
        for position in range(1, length):
            step = logsumexp(alpha[:, position - 1, :, None] + self.transitions, axis=1) + scores[:, position]
            alpha[:, position] = np.where(valid[:, position, None] > 0, step, alpha[:, position - 1])
        beta[:, -1] = self.end_scores
        for position in range(length - 2, -1, -1):
            step = logsumexp(self.transitions + (scores[:, position + 1] + beta[:, position + 1])[:, None, :], axis=2)
            beta[:, position] = np.where(valid[:, position + 1, None] > 0, step, beta[:, position + 1])
        return alpha, beta, logsumexp(alpha[:, -1] + self.end_scores, axis=1)

    def __str__(self):
        return f"Conditional random field over {self.num_labels} labels"

    def __repr__(self):
        return self.__str__()
