"""
SPECTRE mixing layer, https://arxiv.org/abs/2502.18394

Includes non-causal (bi-directional encoder) and causal (decoder) modules, together with optional persistnet memory AND
the paper's suggested wavelet refinement last step
"""
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from polyergalio.models.layers.attention_heads import (
    HeadGate,
    HeadProjection,
    align_gate,
    merge_heads,
    split_heads,
)
from polyergalio.models.layers.basic_layers import Layer, PersistentMemory
from polyergalio.models.layers.wavelet_layers import WaveletRefinementModule

# LayerNorm epsilon for the pooled query within the Head
DESCRIPTOR_EPS = 1e-8

# -------------    adjoints of the real FFT pair    ----------------
def rfft_adjoint(grad_freq: NDArray, sequence_length: int, axis: int = 1) -> NDArray:
    frequencies = grad_freq.shape[axis]
    pad_width = [(0, 0)] * grad_freq.ndim
    pad_width[axis] = (0, sequence_length - frequencies)
    padded = np.pad(grad_freq, pad_width)
    out = np.fft.ifft(padded, n=sequence_length, axis=axis) * sequence_length
    return out.real


def irfft_adjoint(grad_time: NDArray,
                  sequence_length: int,
                  axis: int = 1,
                  ) -> NDArray:
    out = np.fft.rfft(grad_time, n=sequence_length, axis=axis) / sequence_length
    interior = [slice(None)] * out.ndim
    interior[axis] = slice(1, -1 if sequence_length % 2 == 0 else None)
    out[tuple(interior)] *= 2

    return out


class PrefixFFTCache:
    """
    Batched, hidden_dim-wide Prefix-FFT cache shared by all heads of a
    SpectreDecoderAttention layer.

    Ring layout (per batch element), length `max_sequence = memory_tokens + window
    """

    def __init__(
        self,
        sequence_length: int,
        hidden_dimension: int,
        batch_size: int,
        memory_tokens: int = 0,
    ):
        self.sequence_length = int(sequence_length)
        self.memory_tokens = int(memory_tokens)
        self.max_sequence = self.memory_tokens + self.sequence_length
        self.hidden_dimension = int(hidden_dimension)
        self.batch_size = int(batch_size)
        self.n_freq = self.max_sequence // 2 + 1

        # constructing sublayers and parameters
        self.prefix_fft = np.zeros((self.batch_size, self.n_freq, self.hidden_dimension), dtype=complex)
        self.value_buffer = np.zeros((self.batch_size, self.max_sequence, self.hidden_dimension))
        self.query_buffer = np.zeros((self.batch_size, self.max_sequence, self.hidden_dimension))
        self.mask_buffer = np.zeros((self.batch_size, self.max_sequence), dtype=bool)
        self.sum_query = np.zeros((self.batch_size, self.hidden_dimension))

        # absolute step counter for the *sliding* part alone. memory slots are written once (in prefill / set_memory)
        # and not counted
        self.position = 0
        self.length = np.zeros(self.batch_size)
        self.memory_values = np.zeros((self.memory_tokens, self.hidden_dimension))
        self.memory_query_sum = np.zeros(self.hidden_dimension)

        k = np.arange(self.n_freq)
        t = np.arange(self.max_sequence)
        self._twiddle = np.exp(-2j * np.pi * np.outer(t, k) / self.max_sequence)

    def reset(self):
        """clear the sliding window, keeping the persistent memory set by set_memory"""
        self.prefix_fft.fill(0)
        self.value_buffer.fill(0)
        self.query_buffer.fill(0)
        self.mask_buffer.fill(False)
        self.sum_query.fill(0)
        self.position = 0
        self.length.fill(0)
        if self.memory_tokens:
            self.value_buffer[:, : self.memory_tokens] = self.memory_values[None]
            self.mask_buffer[:, : self.memory_tokens] = True
            self.sum_query[...] = self.memory_query_sum[None]
            self.prefix_fft[...] = np.fft.rfft(self.value_buffer, n=self.max_sequence, axis=1)

    def set_memory(self, memory_values: np.ndarray, memory_queries: np.ndarray):
        """
        Seed the persistent memory slots, (memory_tokens, hidden_dim) each, already passed through the
        layer's value and query projections -- the training forward projects memory the same way, so
        memory contributes projected values to the mix and its queries to the pooled descriptor.
        shared across the batch (since it's "injection in the sequence"
        *** potentially destructive as it clears the sliding window ***
        """
        if self.memory_tokens == 0:
            return
        expected = (self.memory_tokens, self.hidden_dimension)
        if memory_values.shape != expected or memory_queries.shape != expected:
            raise ValueError(
                f"expected memory shape {expected}, got {memory_values.shape} and {memory_queries.shape}"
            )
        self.memory_values = memory_values.copy()
        self.memory_query_sum = memory_queries.sum(axis=0)
        self.reset()

    def prefill(self, query: np.ndarray, value: np.ndarray, mask: Optional[np.ndarray] = None):
        """
        One-shot cache initialisation

        query, value : (batch, seq, hidden_dim), already re-merged after heads.
        mask : (batch, seq) optional validity mask to identify.

        single RFFT seeds the full cache.
        """
        batch, length, hidden_dim = value.shape
        if hidden_dim != self.hidden_dimension:
            raise ValueError(f"expected hidden_dim={self.hidden_dimension}, got {hidden_dim}")
        if length > self.sequence_length:
            raise ValueError(
                f"prompt length {length} exceeds window={self.sequence_length}"
            )
        if batch != self.batch_size:
            raise ValueError(f"cache batch_size={self.batch_size}, got {batch}")

        if mask is None:
            mask = np.ones((batch, length))

        self.reset()

        query_valid = query * mask[..., None]
        value_valid = value * mask[..., None]

        start = self.memory_tokens
        self.value_buffer[:, start : start + length] = value_valid
        self.query_buffer[:, start : start + length] = query_valid
        self.mask_buffer[:, start : start + length] = mask.astype(bool)

        self.prefix_fft[...] = np.fft.rfft(
            self.value_buffer, n=self.max_sequence, axis=1
        )

        self.sum_query[...] = self.memory_query_sum[None] + query_valid.sum(axis=1)
        self.length[...] = mask.sum(axis=1).astype(np.int32)
        self.position = length

    # DECODE STEPS ------------------
    def decode_step(self, query_t: np.ndarray, value_t: np.ndarray, valid=True) -> int:
        """
        append one token to the sliding window

        query_t, value_t : (batch, hidden_dim), already re-merged after heads.
        valid : bool or (batch,) bool array, for padded/finished sequences

        returns the ring's positional `slot` the new token was written to, so callers can
        read the reconstructed row for the newest token
        """
        batch = value_t.shape[0]
        if batch != self.batch_size:
            raise ValueError(f"cache batch_size={self.batch_size}, got {batch}")

        valid = np.asarray(valid, dtype=bool)
        if valid.ndim == 0:
            valid = np.full(batch, bool(valid))

        t = self.position
        slot = self.memory_tokens + (t % self.sequence_length)

        query_t = np.where(valid[:, None], query_t, 0.0)
        value_t = np.where(valid[:, None], value_t, 0.0)

        if t >= self.sequence_length:
            old_slot = self.memory_tokens + (
                (t - self.sequence_length) % self.sequence_length
            )
            old_value = self.value_buffer[:, old_slot].copy()
            old_query = self.query_buffer[:, old_slot].copy()
            was_valid = self.mask_buffer[:, old_slot].copy()

            # evict using the same twiddle index
            self.prefix_fft -= (
                self._twiddle[old_slot, :][None, :, None] * old_value[:, None, :]
            )
            self.sum_query -= np.where(was_valid[:, None], old_query, 0.0)

        self.prefix_fft += self._twiddle[slot][None, :, None] * value_t[:, None, :]

        self.value_buffer[:, slot] = value_t
        self.query_buffer[:, slot] = query_t
        self.mask_buffer[:, slot] = valid
        self.sum_query += query_t

        self.position += 1
        self.length = np.minimum(
            self.length + valid.astype(np.int64), self.sequence_length
        )

        return slot

    # ------------------------------------------------------------------
    @property
    def live_length(self) -> int:
        return self.memory_tokens + int(min(self.position, self.sequence_length))

    def reconstruct(self, gate: np.ndarray, spectrum: Optional[np.ndarray] = None) -> np.ndarray:
        """
        gate : (batch, n_freq, hidden_dim) complex spectral gate, already
            broadcast/merged across heads (needs to be aligned before reconstruct)

        Returns the full ring-ordered reconstruction, shape (batch, max_sequence, hidden_dim).
        Slot ordering, not chronological ordering.
        see `read_slot` / `chronological_order` to extract a specific token or the whole window in seqence
        """
        spectrum = self.prefix_fft if spectrum is None else spectrum
        return np.fft.irfft(spectrum * gate, n=self.max_sequence, axis=1)

    def read_position(self, gate: np.ndarray, slot: int, spectrum: Optional[np.ndarray] = None) -> np.ndarray:
        """
        One position of the reconstruction, (batch, hidden_dim), without a full irfft: the gated
        spectrum phase-rotated to `slot` and summed over frequencies -- SPECTRE's positional phase,
        O(n_freq * hidden) per step. Equals reconstruct(gate)[:, slot].
        """
        weights = np.conj(self._twiddle[slot]) / self.max_sequence
        weights[1:] *= 2
        if self.max_sequence % 2 == 0:
            weights[-1] /= 2
        spectrum = self.prefix_fft if spectrum is None else spectrum
        return np.real(np.einsum("bkd,k->bd", spectrum * gate, weights))

    def chronological_spectrum(self) -> np.ndarray:
        """
        rfft of memory followed by the window in oldest-to-newest order. Once the ring has wrapped,
        a window rotated behind fixed memory slots is no longer a circular shift of the training
        layout, so reads switch to this -- O(max_sequence log max_sequence) per step, memory only.
        """
        ordered = self.value_buffer[:, self.get_chronological_order()]
        return np.fft.rfft(ordered, n=self.max_sequence, axis=1)

    def get_chronological_order(self) -> np.ndarray:
        """
        Index array that reorders the ring buffer's movible or sliding portion into
        chronological (oldest -> newest) order, given the current pointer position.
        Memory slots are already in a fixed order.
        """
        if self.position == 0:
            window_order = np.arange(self.sequence_length)
        else:
            newest_slot = (self.position - 1) % self.sequence_length
            window_order = (
                np.arange(self.sequence_length) + newest_slot + 1
            ) % self.sequence_length
        return np.concatenate(
            [np.arange(self.memory_tokens), self.memory_tokens + window_order]
        )


class SpectreAttention(Layer):
    """
    SPECTRE mixing layer, https://arxiv.org/abs/2502.18394

    Per-head parts live in their own layers: query_projection and value_projection are each head's
    independent W(q), W(v) (HeadProjection), and head_gate builds each head's spectral gate from its
    pooled queries (HeadGate). This layer owns what is shared across heads: the persistent memory,
    masking and pooling, the FFT mixing, and the optional WRM.
    """

    registry_name = "SPECTREAttention"
    preserves_shape = True
    cache_names = (
        "input", "mask", "counts", "total_counts", "combined_length", "seq_mu",
        "gate", "descriptor", "value_transform", "output",
    )

    def __init__(
        self,
        sequence_length: int,
        hidden_dimension: int,
        num_heads: int = 1,
        band_radius: int = 0,
        memory_tokens: int = 0,
        causal_decode: bool = False,
        modrelu_bias: float = -0.1,
        use_wrm: bool = True,
        use_positional_phase: bool = True,
        gate_dimension: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        sequence_length : tokens per sample, the axis the FFT runs over
        hidden_dimension : channel width of the input
        num_heads : gates learned in parallel, each over its own slice of the channel axis.
            1 recovers the single-head layer exactly.
        band_radius : radius r of the optional Toeplitz band update on the gate.
            0 disables it. r > 0 adds 2r+1 complex taps per head.
        memory_tokens : size of an optional learned, persistent, never-evicted context bank
            0 disables it.
        gate_dimension : width of each head's gate MLP hidden layer, head_dim if None
        use_positional_phase: bool - used in decoding process to appply positional influence
        gate_dimension: Optional int - for gate's hidden dimension (if different from hidden_dimension)
        """
        assert memory_tokens >= 0, "memory_tokens must be zero or positive"
        assert num_heads >= 1, f"num_heads must be at least 1, got {num_heads}"
        assert hidden_dimension % num_heads == 0, (
            f"num_heads {num_heads} must divide hidden_dim {hidden_dimension}. Heads "
            "partition the channel axis, so a remainder would leave channels "
            "ungated."
        )
        super().__init__()

        self.sequence_length = sequence_length
        self.hidden_dimension: int = hidden_dimension
        self.num_heads: int = num_heads
        self.head_dimension: int = hidden_dimension // num_heads
        self.band_radius = band_radius
        self.memory_tokens: int = memory_tokens
        self.causal_decode: bool = causal_decode
        self.modrelu_bias: float = modrelu_bias
        self.use_wrm = use_wrm
        self.use_positional_phase: bool = use_positional_phase
        self.gate_dimension = gate_dimension

        # sequence axis is pinned, not wildcarded: fft_length / num_frequencies
        # are sized off sequence_length at construction time, so a mismatched
        # sequence length here is a real, catchable error, not a free axis.
        self.declare_shapes(
            inputs=((self.sequence_length, self.hidden_dimension),),
            outputs=((self.sequence_length, self.hidden_dimension),),
        )

        if self.memory_tokens > 0:
            self.memory = PersistentMemory(memory_tokens=self.memory_tokens, hidden_dimension=self.hidden_dimension)

        self.fft_length = self.sequence_length + self.memory_tokens
        self.num_frequencies = self.fft_length // 2 + 1

        self.query_projection = HeadProjection(num_heads, self.head_dimension)
        self.value_projection = HeadProjection(num_heads, self.head_dimension)
        self.head_gate = HeadGate(
            num_heads=num_heads,
            head_dimension=self.head_dimension,
            num_frequencies=self.num_frequencies,
            gate_dimension=gate_dimension or self.head_dimension,
            modrelu_bias=modrelu_bias,
            band_radius=band_radius,
        )

        if use_wrm:
            self.wrm = WaveletRefinementModule(
                input_dimension=hidden_dimension,
                sequence_length=self.sequence_length,
                on_rate=0.1,
                skip_threshold=0.5,
            )

        names = ("query_projection", "value_projection", "head_gate")
        if memory_tokens:
            names += ("memory",)
        if use_wrm:
            names += ("wrm",)
        self.parameter_names = names
        self.zero_gradients()

    def forward(
        self,
        input_data: NDArray,
        mask: Optional[NDArray] = None,
        training_now: Optional[bool] = None,
    ):
        training_now = self.training if training_now is None else training_now
        assert input_data.ndim == 3
        assert input_data.shape[1] == self.sequence_length
        assert input_data.shape[2] == self.hidden_dimension

        batch = input_data.shape[0]

        self.mask = mask if mask is not None else np.ones(input_data.shape[:2])
        self.counts = np.maximum(self.mask.sum(axis=1, keepdims=True), 1.0)
        mask_column = self.mask[..., None]

        if self.memory_tokens:
            memory = self.memory.get_memory()
            memory_batch = np.broadcast_to(memory[None, :, :], (batch, self.memory_tokens, self.hidden_dimension))
            # the trainable memory tokens are concatenated ahead of the input; backward splits the
            # gradient back out to the memory bank and the input
            combined = np.concatenate([memory_batch, input_data], axis=1)
        else:
            combined = input_data
        self.combined_length = combined.shape[1]

        query_all = self.query_projection.forward(combined)
        value_all = self.value_projection.forward(combined)

        memory_tokens = self.memory_tokens
        query_forward = query_all[:, memory_tokens:]
        value_masked = value_all.copy()
        value_masked[:, memory_tokens:] *= mask_column

        self.total_counts = self.counts + memory_tokens
        seq_sum = (query_forward * mask_column).sum(axis=1) + query_all[:, :memory_tokens].sum(axis=1)
        self.seq_mu = seq_sum / self.total_counts

        self.gate = self.head_gate.forward(self.seq_mu)
        self.descriptor = self.head_gate.descriptor

        # the full combined sequence gets a single transform into frequency space
        self.value_transform = np.fft.rfft(value_masked, n=self.combined_length, axis=1)
        values_gated = split_heads(self.value_transform, self.num_heads) * align_gate(self.gate)
        output_all = np.fft.irfft(merge_heads(values_gated), n=self.combined_length, axis=1)
        self.output = output_all[:, memory_tokens:]

        if self.use_wrm:
            self.output = self.wrm.forward(self.output, self.descriptor, training_now=training_now)
        return self.output

    def backward(self, incoming_gradient: NDArray) -> NDArray:
        mask_column = self.mask[..., None]
        memory_tokens = self.memory_tokens
        descriptor_gradient = None
        if self.use_wrm:
            incoming_gradient, descriptor_gradient = self.wrm.backward(incoming_gradient)

        # memory positions produce no output, so their slots receive no gradient
        full_gradient = np.zeros(
            (incoming_gradient.shape[0], self.combined_length, self.hidden_dimension), dtype=incoming_gradient.dtype
        )
        full_gradient[:, memory_tokens:] = incoming_gradient

        gated_gradient = split_heads(irfft_adjoint(full_gradient, self.combined_length, axis=1), self.num_heads)
        value_heads = split_heads(self.value_transform, self.num_heads)
        transform_gradient = merge_heads(gated_gradient * np.conj(align_gate(self.gate)))
        gate_gradient = np.transpose(np.sum(gated_gradient * np.conj(value_heads), axis=-1), (0, 2, 1))

        pooled_gradient = self.head_gate.backward(gate_gradient, descriptor_gradient)

        per_token = pooled_gradient[:, None, :] / self.total_counts[..., None]
        query_gradient = np.broadcast_to(per_token, full_gradient.shape).copy()
        query_gradient[:, memory_tokens:] *= mask_column

        value_gradient = rfft_adjoint(transform_gradient, self.combined_length, axis=1)
        value_gradient[:, memory_tokens:] *= mask_column

        input_gradient = self.query_projection.backward(query_gradient) + (
            self.value_projection.backward(value_gradient)
        )
        if memory_tokens:
            self.memory.backward(input_gradient[:, :memory_tokens].sum(axis=0))
        return input_gradient[:, memory_tokens:].real

    def __str__(self):
        band = f", band radius {self.band_radius}" if self.band_radius else ""
        memory = f", {self.memory_tokens} memory slots" if self.memory_tokens else ""
        return (
            f"SPECTRE mixer, sequence {self.sequence_length}, "
            f"hidden {self.hidden_dimension}, {self.num_heads} heads{band}{memory}"
        )

    def __repr__(self):
        return self.__str__()


# =============================== the causal Decoder version =============================
class SpectreDecoderAttention(SpectreAttention):
    """
    Causal, autoregressive companion to `SpectreAttention`.

    Training and decoding compute the same function. Position p reads only values at positions <= p
    (a causal linear convolution, not the encoder's circular one), filtered by a gate built from the
    pooled query of its chunk's first token and everything before it

        anchor(p) = (p // chunk_size) * chunk_size
        gate(p) = gate from mean query over memory + tokens <= anchor(p)
        y_p = sum_{j <= p} h_anchor(p)[p - j] v_j,  h = irfft(gate)

    chunk_size=1 refreshes the gate at every token -- exact per-prefix gating at a very high cost.
    Increasing the chunk size scales rapidly.
    Each chunk's gate lags its tokens by up to chunk_size - 1

    Usage
    -----
        # train decoder layer
        layer = SpectreDecoderAttention(sequence_length=..., hidden_dim=..., num_heads=..., chunk_size=...)
        out = layer.forward(batch_x, mask=batch_mask, training_now=True)
        layer.backward(delta_out)

        # then for decoding generation --
        last_hidden = layer.prefill(prompt_embeddings, mask=prompt_mask)
        for _ in range(n_new_tokens):
            last_hidden = layer.decode_step(next_token_embedding) <----


    use_wrm is not supported: the Wavelet Refinement Module needs the whole window at once, but
    decoding produces one token at a time. It's kept here to match the main Spectre API sig
    """

    registry_name = "SPECTREDecoderAttention"
    cache_names = ("anchors", "chunk_counts", "filter_transforms")

    def __init__(
        self,
        sequence_length: int,
        hidden_dimension: int,
        num_heads: int = 1,
        band_radius: int = 0,
        memory_tokens: int = 0,
        causal_decode: bool = False,
        modrelu_bias: float = -0.1,
        use_wrm: bool = False,
        use_positional_phase: bool = True,
        gate_dimension: Optional[int] = None,
        chunk_size: int = 1,
    ):
        """
        Parameters
        ----------
        chunk_size : tokens that share one gate, see the class docstring. 1 is exact per-token gating
        other parameters : see SpectreAttention
        """
        assert not use_wrm, (
            "SpectreDecoderAttention does not support use_wrm"
        )
        assert 1 <= chunk_size <= sequence_length, (
            f"chunk_size must fall in [1, {sequence_length}], got {chunk_size}"
        )
        super().__init__(
            sequence_length=sequence_length,
            hidden_dimension=hidden_dimension,
            num_heads=num_heads,
            band_radius=band_radius,
            memory_tokens=memory_tokens,
            causal_decode=causal_decode,
            modrelu_bias=modrelu_bias,
            use_wrm=use_wrm,
            use_positional_phase=use_positional_phase,
            gate_dimension=gate_dimension,
        )
        self.chunk_size = chunk_size
        self.cache: Optional[PrefixFFTCache] = None
        self.chunk_gate: Optional[NDArray] = None

    # ------------- causal training forward / backward
    def forward(
        self,
        input_data: NDArray,
        mask: Optional[NDArray] = None,
        training_now: Optional[bool] = None,
    ) -> NDArray:
        """
        Parameters
        ----------
        input_data : (batch, sequence_length, hidden_dim)
        mask : (batch, sequence_length), 1 for a real token and 0 for padding
        training_now : unused, accepted for the graph

        Returns
        -------
        (batch, sequence_length, hidden_dim), each position mixing only itself and earlier positions
        """
        assert input_data.ndim == 3
        batch, length, _ = input_data.shape
        assert length == self.sequence_length and input_data.shape[2] == self.hidden_dimension
        memory_tokens = self.memory_tokens

        self.mask = mask if mask is not None else np.ones((batch, length))
        mask_column = self.mask[..., None]

        if memory_tokens:
            memory = np.broadcast_to(self.memory.get_memory()[None], (batch, memory_tokens, self.hidden_dimension))
            combined = np.concatenate([memory, input_data], axis=1)
        else:
            combined = input_data
        self.combined_length = combined.shape[1]

        query_all = self.query_projection.forward(combined)
        values = self.value_projection.forward(combined)
        values[:, memory_tokens:] *= mask_column

        self.anchors = np.arange(0, length, self.chunk_size)
        prefix_queries = np.cumsum(query_all[:, memory_tokens:] * mask_column, axis=1)[:, self.anchors]
        prefix_counts = np.cumsum(self.mask, axis=1)[:, self.anchors]
        self.chunk_counts = (np.maximum(prefix_counts, 1.0) + memory_tokens)[..., None]
        memory_query_sum = query_all[:, :memory_tokens].sum(axis=1, keepdims=True)
        pooled = (prefix_queries + memory_query_sum) / self.chunk_counts

        num_chunks = self.anchors.size
        self.gate = self.head_gate.forward(pooled.reshape(batch * num_chunks, self.hidden_dimension))

        padded_length = 2 * self.combined_length
        self.value_transform = np.fft.rfft(values, n=padded_length, axis=1)
        value_heads = split_heads(self.value_transform, self.num_heads)
        chunk_gates = self.gate.reshape(batch, num_chunks, self.num_heads, self.num_frequencies)
        self.filter_transforms = []
        self.output = np.zeros_like(input_data)
        for chunk, anchor in enumerate(self.anchors):
            rows = slice(anchor, min(anchor + self.chunk_size, length))
            filters = np.fft.irfft(chunk_gates[:, chunk], n=self.combined_length, axis=-1)
            filter_transform = np.fft.rfft(filters, n=padded_length, axis=-1)
            self.filter_transforms.append(filter_transform)
            mixed = value_heads * align_gate(filter_transform)
            convolved = np.fft.irfft(merge_heads(mixed), n=padded_length, axis=1)
            self.output[:, rows] = convolved[:, memory_tokens + rows.start: memory_tokens + rows.stop]
        return self.output

    def backward(self, incoming_gradient: NDArray) -> NDArray:
        batch, length, _ = incoming_gradient.shape
        memory_tokens = self.memory_tokens
        padded_length = 2 * self.combined_length
        mask_column = self.mask[..., None]
        value_heads = split_heads(self.value_transform, self.num_heads)

        value_transform_gradient = np.zeros_like(self.value_transform)
        gate_gradient = np.zeros(
            (batch, self.anchors.size, self.num_heads, self.num_frequencies), dtype=complex
        )
        for chunk, anchor in enumerate(self.anchors):
            rows = slice(anchor, min(anchor + self.chunk_size, length))
            convolved_gradient = np.zeros((batch, padded_length, self.hidden_dimension))
            convolved_gradient[:, memory_tokens + rows.start: memory_tokens + rows.stop] = incoming_gradient[:, rows]
            mixed_gradient = split_heads(irfft_adjoint(convolved_gradient, padded_length, axis=1), self.num_heads)

            value_transform_gradient += merge_heads(mixed_gradient * np.conj(align_gate(self.filter_transforms[chunk])))
            filter_transform_gradient = np.sum(mixed_gradient * np.conj(value_heads), axis=-1)
            filter_gradient = rfft_adjoint(filter_transform_gradient, padded_length, axis=1)[:, : self.combined_length]
            gate_gradient[:, chunk] = np.transpose(irfft_adjoint(filter_gradient, self.combined_length, axis=1), (0, 2, 1))

        pooled_gradient = self.head_gate.backward(
            gate_gradient.reshape(-1, self.num_heads, self.num_frequencies)
        ).reshape(batch, self.anchors.size, self.hidden_dimension)
        per_anchor = pooled_gradient / self.chunk_counts
        reaching = np.cumsum(per_anchor[:, ::-1], axis=1)[:, ::-1]
        first_chunk = -(-np.arange(length) // self.chunk_size)
        covered = first_chunk < self.anchors.size
        token_query_gradient = np.zeros((batch, length, self.hidden_dimension))
        token_query_gradient[:, covered] = reaching[:, first_chunk[covered]]
        token_query_gradient *= mask_column

        value_gradient = rfft_adjoint(value_transform_gradient, padded_length, axis=1)[:, : self.combined_length]
        value_gradient[:, memory_tokens:] *= mask_column
        query_gradient = np.concatenate(
            [np.broadcast_to(reaching[:, :1], (batch, memory_tokens, self.hidden_dimension)), token_query_gradient], axis=1
        )

        input_gradient = self.query_projection.backward(query_gradient) + self.value_projection.backward(value_gradient)
        if memory_tokens:
            self.memory.backward(input_gradient[:, :memory_tokens].sum(axis=0))
        return input_gradient[:, memory_tokens:]

    # ------------- decoding
    def reset_cache(self, batch_size: int):
        self.cache = PrefixFFTCache(
            sequence_length=self.sequence_length,
            hidden_dimension=self.hidden_dimension,
            batch_size=batch_size,
            memory_tokens=self.memory_tokens,
        )
        self.chunk_gate = None
        if self.memory_tokens:
            memory = self.memory.get_memory()
            self.cache.set_memory(self.value_projection.project(memory), self.query_projection.project(memory))

    def gate_from_pooled_sum(self, query_sum: NDArray, total_counts: NDArray) -> NDArray:
        """
        query_sum, total_counts : (batch, hidden_dim), (batch, 1)

        Returns
        -------
        (batch, num_frequencies, hidden_dim) activated gate, each head's gate repeated over its channels
        """
        gate = self.head_gate.forward(query_sum / total_counts)
        batch = gate.shape[0]
        return merge_heads(
            np.broadcast_to(align_gate(gate), (batch, self.num_frequencies, self.num_heads, self.head_dimension))
        )

    def read_slot(self, gate_full: NDArray, slot: int) -> NDArray:
        """
        the live token's output: the paper's phase-rotated frequency sum when use_positional_phase,
        otherwise a full-window irfft indexed at slot. Both give the same values.
        """
        spectrum = None
        if self.memory_tokens and self.cache.position > self.sequence_length:
            spectrum = self.cache.chronological_spectrum()
            slot = self.cache.max_sequence - 1
        if self.use_positional_phase:
            return self.cache.read_position(gate_full, slot, spectrum)
        return self.cache.reconstruct(gate_full, spectrum)[:, slot, :]

    def prefill(self, input_data: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Process a prompt, populate the Prefix-FFT cache, and return the last prompt token's output.

        input_data : (batch, L, hidden_dim), L <= sequence_length
        mask : (batch, L) optional
        """
        assert input_data.ndim == 3
        batch, length, hidden_dim = input_data.shape
        assert hidden_dim == self.hidden_dimension

        self.reset_cache(batch_size=batch)
        mask = mask if mask is not None else np.ones((batch, length))

        query_all = self.query_projection.project(input_data)
        value_all = self.value_projection.project(input_data)
        self.cache.prefill(query_all, value_all, mask=mask)

        anchor = ((length - 1) // self.chunk_size) * self.chunk_size
        query_sum = self.cache.memory_query_sum[None] + (query_all[:, : anchor + 1] * mask[:, : anchor + 1, None]).sum(axis=1)
        total_counts = np.maximum(mask[:, : anchor + 1].sum(axis=1, keepdims=True), 1.0) + self.memory_tokens
        self.chunk_gate = self.gate_from_pooled_sum(query_sum, total_counts)

        last_slot = self.memory_tokens + ((length - 1) % self.sequence_length)
        return self.read_slot(self.chunk_gate, last_slot)

    def decode_step(self, input_t: NDArray, valid=True) -> NDArray:
        """
        Append one new token and return its output. The gate is rebuilt from the cache's pooled query
        when the token starts a chunk, and reused otherwise.

        input_t : (batch, hidden_dim) raw embedding for the position
        valid : bool or (batch,) bools: False marks a padding step for finished sequences; it still
            advances the cache but writes a zero token and does not affect the pooled query.
        """
        assert self.cache is not None, "call reset_cache()/prefill() first"
        assert input_t.ndim == 2 and input_t.shape[1] == self.hidden_dimension

        position = self.cache.position
        query_t = self.query_projection.project(input_t)
        value_t = self.value_projection.project(input_t)
        slot = self.cache.decode_step(query_t, value_t, valid=valid)

        if self.chunk_gate is None or position % self.chunk_size == 0:
            total_counts = np.maximum(self.cache.length[:, None], 1.0) + self.memory_tokens
            self.chunk_gate = self.gate_from_pooled_sum(self.cache.sum_query, total_counts)
        return self.read_slot(self.chunk_gate, slot)


class SpectrePooledAttention(SpectreAttention):
    """
    SPECTRE mixing that returns a single summary of the sequence instead of a mixed sequence.

    Mixing is y_h = irfft(gate_h * rfft(v_h)) along the sequence, and the layer returns a masked weighted mean of y
        over the real positions:

        out[b, h] = sum mask[b, t] * readout[h, t] * y[b, t, h] / n_b

    The mean is taken through the adjoint of the convolution.

    Input shape: (batch, sequence_length, hidden_dimension)
    Output shape: (batch, 1, hidden_dimension)
    """

    registry_name = "SPECTREPooledAttention"
    preserves_shape = False
    consumes_mask = True
    cache_names = (
        "input", "mask", "counts", "total_counts", "combined_length", "seq_mu",
        "gate", "pool_spectrum", "pool_weights", "value_masked",
    )

    def __init__(
        self,
        sequence_length: int,
        hidden_dimension: int,
        num_heads: int = 1,
        band_radius: int = 0,
        memory_tokens: int = 0,
        modrelu_bias: float = -0.1,
        gate_dimension: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        sequence_length : maximum of tokens per sample, the axis the FFT runs across
        hidden_dimension : trailing dimension of the input
        num_heads : gates learned in parallel, each over its own slice of the channel axis
        band_radius : radius of the optional Toeplitz band update on the gate, 0 disables it
        memory_tokens : size of an optional learned, persistent context bank, 0 disables it
        modrelu_bias : starting modReLU bias, learned per head and frequency
        gate_dimension : width of each head's gate hidden layer, head_dim if None
        """
        super().__init__(
            sequence_length=sequence_length,
            hidden_dimension=hidden_dimension,
            num_heads=num_heads,
            band_radius=band_radius,
            memory_tokens=memory_tokens,
            causal_decode=False,
            modrelu_bias=modrelu_bias,
            use_wrm=False,
            use_positional_phase=False,
            gate_dimension=gate_dimension,
        )
        self.declare_shapes(
            inputs=((sequence_length, hidden_dimension),), outputs=((1, hidden_dimension),)
        )
        self.readout = np.ones((num_heads, sequence_length))
        self.parameter_names += ("readout",)
        self.zero_gradients()

    def forward(self,
                input_data: NDArray,
                mask: Optional[NDArray] = None,
                ) -> NDArray:
        """
        Parameters
        ----------
        input_data : (batch, sequence_length, hidden_dimension)
        mask : (batch, sequence_length), 1 for a real token and 0 for padding; padded tokens contribute nothing

        Returns
        -------
        (batch, 1, hidden_dimension)
        """
        assert input_data.ndim == 3
        assert input_data.shape[1] == self.sequence_length
        assert input_data.shape[2] == self.hidden_dimension

        num_samples = input_data.shape[0]
        memory_tokens = self.memory_tokens
        self.input = input_data
        self.mask = (mask if mask is not None else np.ones(input_data.shape[:2])).astype(input_data.dtype)
        self.counts = np.maximum(self.mask.sum(axis=1, keepdims=True), 1.0)
        mask_column = self.mask[..., None]

        if memory_tokens:
            memory = self.memory.get_memory()
            memory_batch = np.broadcast_to(memory[None, :, :], (num_samples, memory_tokens, self.hidden_dimension))
            combined = np.concatenate([memory_batch, input_data], axis=1)
        else:
            combined = input_data
        self.combined_length = combined.shape[1]

        query_all = self.query_projection.forward(combined)
        value_all = self.value_projection.forward(combined)

        self.value_masked = value_all.copy()
        self.value_masked[:, memory_tokens:] *= mask_column

        # the actual length of the transform -- including the persistent memory
        self.total_counts = self.counts + memory_tokens
        seq_sum = (query_all[:, memory_tokens:] * mask_column).sum(axis=1) + query_all[:, :memory_tokens].sum(axis=1)
        self.seq_mu = seq_sum / self.total_counts
        self.gate = self.head_gate.forward(self.seq_mu)

        self.pool_weights = self.readout_weights()
        self.pool_spectrum = irfft_adjoint(self.pool_weights, self.combined_length, axis=1)
        spectrum = np.conj(align_gate(self.gate)[..., 0]) * self.pool_spectrum
        position_weights = rfft_adjoint(spectrum, self.combined_length, axis=1)

        values = split_heads(self.value_masked, self.num_heads)
        pooled = np.einsum("bsh,bshc->bhc", position_weights, values)
        return merge_heads(pooled)[:, None, :]

    def readout_weights(self) -> NDArray:
        """(batch, combined_length, heads) readout over the combined sequence, zero on memory slots and padding"""
        weights = np.zeros((self.mask.shape[0], self.combined_length, self.num_heads), dtype=self.mask.dtype)
        masked = self.mask[:, :, None] * self.readout.T[None, :, :]
        weights[:, self.memory_tokens:] = masked / self.counts[..., None]
        return weights

    def backward(self, incoming_gradient: NDArray) -> NDArray:
        """
        Parameters
        ----------
        incoming_gradient : (batch, 1, hidden_dimension)

        Returns
        -------
        (batch, sequence_length, hidden_dimension)
        """
        memory_tokens = self.memory_tokens
        mask_column = self.mask[..., None]

        pooled_gradient = split_heads(incoming_gradient[:, 0, :], self.num_heads)

        spectrum = np.conj(align_gate(self.gate)[..., 0]) * self.pool_spectrum
        position_weights = rfft_adjoint(spectrum, self.combined_length, axis=1)
        values = split_heads(self.value_masked, self.num_heads)

        value_gradient = position_weights[..., None] * pooled_gradient[:, None, :, :]
        value_gradient[:, memory_tokens:] *= mask_column[..., None]
        # (batch, seq, head, combined)
        scalar_gradient = np.einsum("bshc,bhc->bsh", values, pooled_gradient)
        scalar_spectrum = np.fft.rfft(scalar_gradient, n=self.combined_length, axis=1)

        gate_gradient = np.transpose(self.pool_spectrum * np.conj(scalar_spectrum), (0, 2, 1))
        pooled_query_gradient = self.head_gate.backward(gate_gradient, None)

        response = np.fft.irfft(
            np.transpose(self.gate, axes=(0, 2, 1)) * scalar_spectrum,
            n=self.combined_length,
            axis=1
        )
        readout_signal = response[:, memory_tokens:] * self.mask[..., None] / self.counts[..., None]

        self.gradient_readout = readout_signal.sum(axis=0).T

        per_token = pooled_query_gradient[:, None, :] / self.total_counts[..., None]
        query_gradient = np.broadcast_to(per_token, (*self.mask.shape[:1], self.combined_length, self.hidden_dimension)).copy()
        query_gradient[:, memory_tokens:] *= mask_column

        input_gradient = self.query_projection.backward(query_gradient) + self.value_projection.backward(
            merge_heads(value_gradient)
        )

        if memory_tokens:
            self.memory.backward(input_gradient[:, :memory_tokens].sum(axis=0))

        return input_gradient[:, memory_tokens:].real

    def __str__(self):
        band = f", band radius {self.band_radius}" if self.band_radius else ""
        memory = f", {self.memory_tokens} memory slots" if self.memory_tokens else ""
        return (
            f"SPECTRE pooled mixer, sequence {self.sequence_length}, "
            f"hidden {self.hidden_dimension}, {self.num_heads} heads{band}{memory}"
        )
