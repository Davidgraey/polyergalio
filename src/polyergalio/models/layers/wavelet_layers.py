from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from polyergalio.models.layers.basic_layers import RNG, FullyConnectedLayer, Layer


class HaarWaveletTransform:
    """
    one-level Haar analysis along axis=1 (sequence)

    pairing is circular:
        low  = (x_even + x_odd) / sqrt(2)
        high = (x_odd - x_even) / sqrt(2)

    For odd N, the final even sample is paired with x[0].
    In that case the transform is redundant and synthesize() is its
    adjoint, but it's NOT AN EXACT INVERSE!!!
    """

    INV_SQRT2 = 1.0 / np.sqrt(2.0)

    def __init__(self, sequence_length: int):
        self.sequence_length = sequence_length
        self.num_coefficients = (sequence_length + 1) // 2

        self.even_indices = np.arange(0, sequence_length, 2)
        self.odd_indices = (self.even_indices + 1) % sequence_length

        self.input_shape = None

    def analyze(self, x: NDArray) -> tuple[NDArray, NDArray]:
        batch, sequence, hidden = x.shape

        assert sequence == self.sequence_length, (
            f"expected sequence length {self.sequence_length}, got {sequence}"
        )

        x_even = x[:, self.even_indices, :]
        x_odd = x[:, self.odd_indices, :]

        low = (x_even + x_odd) * self.INV_SQRT2
        high = (x_odd - x_even) * self.INV_SQRT2

        self.input_shape = x.shape

        return low, high

    def synthesize(
        self,
        low: NDArray,
        high: NDArray,
    ) -> NDArray:
        """
        Adjoint or inverse of analyze() -- sequence length dependent!

        For even sequence lengths this is also the exact inverse.
        For odd sequence lengths, circular pairing causes the transform to be redundant, so this is the adjoint
        rather than an inverse.
        """
        assert self.input_shape is not None

        assert low.shape == high.shape

        d_even = (low - high) * self.INV_SQRT2
        d_odd = (low + high) * self.INV_SQRT2

        batch, sequence, hidden = self.input_shape

        dx = np.zeros((batch, sequence, hidden))

        for coefficient in range(self.num_coefficients):
            even = self.even_indices[coefficient]
            odd = self.odd_indices[coefficient]

            dx[:, even, :] += d_even[:, coefficient, :]
            dx[:, odd, :] += d_odd[:, coefficient, :]

        return dx

    def purge(self) -> None:
        self.input_shape = None


class WaveletRefinementModule(Layer):
    """
    SPECTRE Wavelet Refinement Module (WRM).
        - Haar wavelet decomposition
        - descriptor-conditioned channel gate
        - coefficient modulation

    Global descriptor (B, D)
    Position-wise descriptor (B, N, D)

    In the global case, the learned channel gate is broadcast over sequence positions before being pooled to
    wavelet resolution.

    The stochastic execution controller, control_gate, is a scalar logit that is deliberately non-differentiable:
    it is a parameter, but its gradient is always zero.
    """
    preserves_shape = True
    parameter_names = ("control_gate", "gate_projection", "gate_output")
    cache_names = (
        "v_tilde", "descriptor", "training_now", "run_prob", "run_mask", "gate", "gate_pooled",
        "low", "high", "gated_low", "gated_high", "refinement", "output",
    )

    def __init__(
        self,
        input_dimension: int,
        sequence_length: int,
        on_rate: float = 0.1,
        skip_threshold: float = 0.5,
    ):
        super().__init__()

        self.input_dimension = input_dimension
        self.sequence_length = sequence_length
        self.on_rate = on_rate
        self.skip_threshold = skip_threshold

        self.declare_shapes(inputs=((input_dimension,), (input_dimension,)), outputs=((input_dimension,),))

        self.wavelet = HaarWaveletTransform(sequence_length)
        self.gate_projection = FullyConnectedLayer(input_dimension=input_dimension, output_dimension=input_dimension, activation_type="swish")
        self.gate_output = FullyConnectedLayer(input_dimension=input_dimension, output_dimension=input_dimension, activation_type="linear")
        self.control_gate = np.array([[np.log(on_rate / (1.0 - on_rate))]])
        self.zero_gradients()

    def _pool_gate(self, gate: NDArray) -> NDArray:
        """
        Pool a position-wise channel gate (B, N, D) from sequence resolution to Haar coefficient resolution
        (B, ceil(N / 2), D), using exactly the same circular pairing as HaarWaveletTransform.
        """
        return 0.5 * (gate[:, self.wavelet.even_indices, :] + gate[:, self.wavelet.odd_indices, :])

    def _unpool_gate(self, dgate_coeff: NDArray) -> NDArray:
        """
        Adjoint of _pool_gate: distributes one half of each coefficient gradient (B, ceil(N / 2), D) to each
        participating sequence position, returning (B, N, D).
        """
        batch = dgate_coeff.shape[0]
        dgate = np.zeros((batch, self.sequence_length, self.input_dimension))

        for coefficient in range(self.wavelet.num_coefficients):
            even = self.wavelet.even_indices[coefficient]
            odd = self.wavelet.odd_indices[coefficient]
            gradient = 0.5 * dgate_coeff[:, coefficient, :]
            dgate[:, even, :] += gradient
            dgate[:, odd, :] += gradient

        return dgate

    def forward(self,
                v_tilde: NDArray,
                descriptor: NDArray,
                training_now: Optional[bool] = None,
                mask: Optional[NDArray] = None,
                ) -> NDArray:
        """
        Parameters
        ----------
        v_tilde : (B, N, D)
        descriptor : (B, D) global, or (B, N, D) position-wise
        training_now : None follows the layer's train() / eval() mode
        mask : accepted but not applied. The Haar pairing mixes adjacent sequence positions, so a real position
            paired against a zero-padded one gets a coefficient computed half from actual content and half from
            padding. A correct masked version would need to skip or reweight those boundary pairs, which this does
            not yet do. Only the pairs straddling the real/padding boundary are affected.
        """
        training_now = self.training if training_now is None else training_now
        assert v_tilde.ndim == 3, (
            f"data into the wavelet refinement must be (batch, seq, hidden),"
            f" got {v_tilde.ndim} dimensions"
        )

        batch, sequence, hidden = v_tilde.shape

        assert sequence == self.sequence_length, (
            f"sequence mismatch: expected {self.sequence_length}, got {sequence}"
        )
        assert hidden == self.input_dimension, (
            f"hidden mismatch: expected {self.input_dimension}, got {hidden}"
        )
        assert descriptor.ndim in (2, 3), (
            "descriptor must be (batch, hidden) or "
            f"(batch, sequence, hidden), got {descriptor.shape}"
        )
        assert descriptor.shape[0] == batch, (
            f"descriptor batch mismatch: expected {batch}, got {descriptor.shape[0]}"
        )
        assert descriptor.shape[-1] == self.input_dimension, (
            f"descriptor hidden mismatch: expected {self.input_dimension}, got {descriptor.shape[-1]}"
        )
        if descriptor.ndim == 3:
            assert descriptor.shape[1] == sequence, (
                f"position-wise descriptor sequence mismatch: expected {sequence}, got {descriptor.shape[1]}"
            )

        self.purge()
        self.v_tilde = v_tilde
        self.descriptor = descriptor
        self.training_now = training_now

        self.run_prob = 1.0 / (1.0 + np.exp(-self.control_gate))
        if training_now:
            self.run_mask = RNG.random((batch, 1, 1)) < self.run_prob[0, 0]
        else:
            should_run = self.run_prob[0, 0] > self.skip_threshold
            self.run_mask = np.full((batch, 1, 1), should_run)

        if not np.any(self.run_mask):
            self.output = v_tilde
            return self.output

        gate = self.gate_output(self.gate_projection(descriptor))
        if gate.ndim == 2:
            gate = np.broadcast_to(gate[:, None, :], (batch, sequence, hidden))

        self.gate = gate
        self.gate_pooled = self._pool_gate(self.gate)
        self.low, self.high = self.wavelet.analyze(v_tilde)

        self.gated_low = self.low * self.gate_pooled
        self.gated_high = self.high * self.gate_pooled
        self.refinement = self.wavelet.synthesize(self.gated_low, self.gated_high)

        self.output = v_tilde + self.run_mask * self.refinement
        return self.output

    def backward(self, incoming_gradient: NDArray) -> tuple[NDArray, NDArray]:
        """
        Returns
        -------
        d_v_tilde : gradient with respect to v_tilde
        d_descriptor : gradient with respect to the descriptor, in its original shape
        """
        assert self.output is not None, "backward called before forward"
        assert incoming_gradient.shape == self.output.shape, (
            f"gradient shape {incoming_gradient.shape} does not match output shape {self.output.shape}"
        )

        d_v_direct = incoming_gradient

        if not np.any(self.run_mask):
            return d_v_direct, np.zeros_like(self.descriptor)

        d_refinement = incoming_gradient * self.run_mask
        d_gated_low, d_gated_high = self.wavelet.analyze(d_refinement)

        d_low = d_gated_low * self.gate_pooled
        d_high = d_gated_high * self.gate_pooled
        d_gate_pooled = (d_gated_low * self.low) + (d_gated_high * self.high)

        d_gate = self._unpool_gate(d_gate_pooled)
        d_gate_fc = d_gate.sum(axis=1) if self.descriptor.ndim == 2 else d_gate

        d_gate_hidden = self.gate_output.backward(d_gate_fc)
        d_descriptor = self.gate_projection.backward(d_gate_hidden)

        d_v_wavelet = self.wavelet.synthesize(d_low, d_high)
        d_v_tilde = d_v_direct + (self.run_mask * d_v_wavelet)
        return d_v_tilde, d_descriptor

    def purge(self) -> None:
        super().purge()
        self.wavelet.purge()

    def __str__(self):
        return f"WaveletRefinementModule ({self.input_dimension}, Haar, on_rate={self.on_rate})"

    def __repr__(self):
        return self.__str__()
