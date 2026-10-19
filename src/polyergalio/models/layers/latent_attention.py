"""
Softmax attention over the low-rank latent of DeepSeek-V3's Multi-head Latent Attention (MLA).

MLA compresses each token into a small latent, c_KV = RMSNorm(W_DKV x), and rebuilds every head's keys and values from
it, K = W_UK c_KV and V = W_UV c_KV; queries are compressed the same way. These layers are standard scaled
dot-product attention on top of that bottleneck. The decoder only needs to keep c_KV per token.
"""
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from polyergalio.models.layers.attention_heads import LatentHead, merge_heads, split_heads
from polyergalio.models.layers.basic_layers import FullyConnectedLayer, Layer

MASKED_SCORE = -1e12


def masked_attention_softmax(scores: NDArray, allowed: NDArray) -> NDArray:
    """softmax over the last axis where allowed, exactly zero elsewhere (a row with nothing allowed is all zero)"""
    shifted = np.where(allowed, scores, MASKED_SCORE)
    shifted = shifted - shifted.max(axis=-1, keepdims=True)
    weights = np.exp(shifted) * allowed
    return weights / np.maximum(weights.sum(axis=-1, keepdims=True), 1e-30)


class LatentAttention(Layer):
    """
    Bidirectional multi-head attention whose keys and values come from a shared low-rank latent.

        # c_Q - our first Low-Rank bottleneck for queries
        c_Q = RMSNorm(W_DQ x)
        # c_KV - our second Low-Rank bottleneck for keys and valuse (SHARED latent space for both)
        c_KV = RMSNorm(W_DKV x)

        q = W_UQ c_Q

        k = W_UK c_KV
        v = W_UV c_KV
        P = softmax(q k^T / sqrt(d))

        y = P v

        out = W_O y

    query_head is a LatentHead for q. kv_head is one LatentHead with 2 * num_heads heads on the shared c_KV: the first
    num_heads outputs are the keys and the rest the values. Padded keys get zero attention weight.

    There is no positional signal; apply positions upstream when order matters.

    Input shape: (batch, sequence, hidden_dimension)
    Output shape: (batch, sequence, hidden_dimension)
    """
    preserves_shape = True
    parameter_names = ("query_head", "kv_head", "output_projection")
    cache_names = ("query", "key", "value", "probabilities")

    def __init__(
        self,
        hidden_dimension: int,
        num_heads: int,
        kv_rank: int,
        query_rank: Optional[int] = None,
        head_dimension: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        hidden_dimension : channel width of the input and output
        num_heads : attention heads, each reading its own slice of the up-projected queries, keys and values
        kv_rank : width of the key/value latent c_KV
        query_rank : width of the query latent c_Q, kv_rank if None
        head_dimension : width of each head, hidden_dimension // num_heads if None
        """
        assert num_heads >= 1, f"num_heads must be at least 1, got {num_heads}"
        assert head_dimension is not None or hidden_dimension % num_heads == 0, (
            f"num_heads {num_heads} must divide hidden_dimension {hidden_dimension} when head_dimension is not given"
        )
        super().__init__()
        self.hidden_dimension = hidden_dimension
        self.num_heads = num_heads
        self.kv_rank = kv_rank
        self.query_rank = query_rank or kv_rank
        self.head_dimension = head_dimension or hidden_dimension // num_heads
        # keep recast
        self.scale = float(1.0 / np.sqrt(self.head_dimension))
        self.declare_shapes(inputs=((None, hidden_dimension),), outputs=((None, hidden_dimension),))

        inner = self.num_heads * self.head_dimension
        self.query_head = LatentHead(self.num_heads, self.head_dimension, self.query_rank, hidden_dimension)
        self.kv_head = LatentHead(2 * self.num_heads, self.head_dimension, kv_rank, hidden_dimension)
        self.output_projection = FullyConnectedLayer(inner, hidden_dimension, "linear")
        self.zero_gradients()

    def compress(self, input_data: NDArray) -> NDArray:
        """(..., hidden_dimension) -> (..., kv_rank), the latent c_KV that is all a cache needs to keep"""
        return self.kv_head.compress(input_data)

    def project_queries(self, input_data: NDArray) -> NDArray:
        """(..., hidden_dimension) -> (..., num_heads, head_dimension)"""
        return split_heads(self.query_head.forward(input_data), self.num_heads)

    def project_keys_values(self, input_data: NDArray) -> tuple[NDArray, NDArray]:
        """(..., hidden_dimension) -> keys and values, each (..., num_heads, head_dimension)"""
        heads = split_heads(self.kv_head.forward(input_data), 2 * self.num_heads)
        return heads[..., : self.num_heads, :], heads[..., self.num_heads:, :]

    def allowed_pairs(self, mask: NDArray, length: int) -> NDArray:
        """(batch, 1, length, length) bool, True where a query position may read a key position"""
        return np.broadcast_to(mask[:, None, None, :] > 0, (mask.shape[0], 1, length, length))

    def forward(self, input_data: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        input_data : (batch, sequence, hidden_dimension)
        mask : (batch, sequence), 1 for a real token and 0 for padding

        Returns
        -------
        (batch, sequence, hidden_dimension)
        """
        assert input_data.ndim == 3, f"expected (batch, sequence, hidden), got {input_data.shape}"
        assert input_data.shape[2] == self.hidden_dimension
        batch, length, _ = input_data.shape
        mask = np.ones((batch, length)) if mask is None else mask
        allowed = self.allowed_pairs(mask, length)

        query = self.project_queries(input_data)
        key, value = self.project_keys_values(input_data)
        scores = np.einsum("bthd,bshd->bhts", query, key, optimize=True) * self.scale
        probabilities = masked_attention_softmax(scores, allowed)
        if self.training:
            self.query, self.key, self.value, self.probabilities = query, key, value, probabilities
        attended = np.einsum("bhts,bshd->bthd", probabilities, value, optimize=True)
        return self.output_projection.forward(merge_heads(attended))

    def backward(self, incoming_gradient: NDArray) -> NDArray:
        attended_gradient = split_heads(self.output_projection.backward(incoming_gradient), self.num_heads)
        probability_gradient = np.einsum("bthd,bshd->bhts", attended_gradient, self.value, optimize=True)
        value_gradient = np.einsum("bhts,bthd->bshd", self.probabilities, attended_gradient, optimize=True)
        score_gradient = self.probabilities * (
            probability_gradient - (probability_gradient * self.probabilities).sum(axis=-1, keepdims=True)
        ) * self.scale
        query_gradient = np.einsum("bhts,bshd->bthd", score_gradient, self.key, optimize=True)
        key_gradient = np.einsum("bhts,bthd->bshd", score_gradient, self.query, optimize=True)

        kv_gradient = np.concatenate([merge_heads(key_gradient), merge_heads(value_gradient)], axis=-1)
        return self.query_head.backward(merge_heads(query_gradient)) + self.kv_head.backward(kv_gradient)

    def __str__(self):
        return (
            f"{self.__class__.__name__}, hidden {self.hidden_dimension}, {self.num_heads} heads of {self.head_dimension}, "
            f"kv rank {self.kv_rank}, query rank {self.query_rank}"
        )

    def __repr__(self):
        return self.__str__()


class LatentDecoderAttention(LatentAttention):
    """
    Causal companion to LatentAttention: position t reads only keys and values at positions <= t.

    Decoding caches the latent c_KV, kv_rank numbers per token instead of 2 * num_heads * head_dimension. Reading it
    uses MLA's absorption: q . (c W_UK + b) = (q W_UK^T) . c + const, and the constant cancels in the softmax, so
    scores are taken against the cached latents directly. The weighted latent is expanded once through W_UV, which
    is exact because the weights sum to one and the value bias passes through unchanged.

    Usage
    -----
        out = layer.forward(batch_x, mask=batch_mask)
        layer.backward(delta_out)

        layer.eval()
        last_hidden = layer.prefill(prompt_embeddings)
        for _ in range(n_new_tokens):
            last_hidden = layer.decode_step(next_token_embedding)
    """
    cache_names = ("latent_cache", "valid_cache")

    def allowed_pairs(self, mask: NDArray, length: int) -> NDArray:
        return super().allowed_pairs(mask, length) & np.tril(np.ones((length, length), dtype=bool))

    def reset_state(self, batch_size: int) -> None:
        """Start an empty latent cache for batch_size sequences."""
        self.latent_cache = np.zeros((batch_size, 0, self.kv_rank), dtype=self.kv_head.weights.dtype)
        self.valid_cache = np.zeros((batch_size, 0), dtype=bool)

    def append_latents(self, latents: NDArray, valid: NDArray) -> None:
        self.latent_cache = np.concatenate([self.latent_cache, latents], axis=1)
        self.valid_cache = np.concatenate([self.valid_cache, valid], axis=1)

    def read_cache(self, query_t: NDArray) -> NDArray:
        """
        Parameters
        ----------
        query_t : (batch, num_heads, head_dimension) projected query of the live token

        Returns
        -------
        (batch, hidden_dimension) output of the live token against the latent cache
        """
        key_weights = self.kv_head.weights[: self.num_heads]
        value_weights = self.kv_head.weights[self.num_heads:]
        value_bias = self.kv_head.bias[self.num_heads:]

        absorbed = np.einsum("bhd,hrd->bhr", query_t, key_weights, optimize=True)
        scores = np.einsum("bhr,bsr->bhs", absorbed, self.latent_cache, optimize=True) * self.scale
        weights = masked_attention_softmax(scores, self.valid_cache[:, None, :])
        context = np.einsum("bhs,bsr->bhr", weights, self.latent_cache, optimize=True)
        attended = np.einsum("bhr,hrd->bhd", context, value_weights, optimize=True) + value_bias
        return self.output_projection.forward(merge_heads(attended))

    def prefill(self, input_data: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Cache a prompt's latents and return the last prompt position's output.

        Parameters
        ----------
        input_data : (batch, length, hidden_dimension)
        mask : (batch, length), 1 for a real token and 0 for padding

        Returns
        -------
        (batch, hidden_dimension)
        """
        assert not self.training, "call eval() before decoding so the training caches are left alone"
        assert input_data.ndim == 3 and input_data.shape[2] == self.hidden_dimension
        batch, length, _ = input_data.shape
        self.reset_state(batch)
        valid = np.ones((batch, length), dtype=bool) if mask is None else mask > 0
        self.append_latents(self.compress(input_data), valid)
        return self.read_cache(self.project_queries(input_data[:, -1]))

    def decode_step(self, input_t: NDArray, valid=True) -> NDArray:
        """
        Cache one token's latent and return its output.

        Parameters
        ----------
        input_t : (batch, hidden_dimension) embedding of the new token
        valid : bool or (batch,) bools; False marks a padding step that later tokens cannot read

        Returns
        -------
        (batch, hidden_dimension)
        """
        assert not self.training, "call eval() before decoding so the training caches are left alone"
        assert self.latent_cache is not None, "call prefill() or reset_state() first"
        assert input_t.ndim == 2 and input_t.shape[1] == self.hidden_dimension
        step_valid = np.broadcast_to(np.asarray(valid, dtype=bool), (input_t.shape[0],))[:, None]
        self.append_latents(self.compress(input_t)[:, None], step_valid)
        return self.read_cache(self.project_queries(input_t))

    def __str__(self):
        return f"{super().__str__()}, causal"
