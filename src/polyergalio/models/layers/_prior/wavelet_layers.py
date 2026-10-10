from typing import Optional

import numpy as np
from polyergalio.models.constants import GLOBAL_COMPLEX_DTYPE, GLOBAL_DTYPE
from polyergalio.models.layers.basal_layers import FullyConnectedLayer, Layer
from numpy.typing import NDArray


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
        Adjoint of analyze().

        For even sequence lengths this is also the exact inverse.

        For odd sequence lengths, circular pairing causes the transform
        to be redundant, so this is the adjoint rather than an inverse.
        """
        assert self.input_shape is not None

        assert low.shape == high.shape

        d_even = (low - high) * self.INV_SQRT2
        d_odd = (low + high) * self.INV_SQRT2

        batch, sequence, hidden = self.input_shape

        dx = np.zeros(
            (batch, sequence, hidden),
            dtype=GLOBAL_DTYPE,
        )

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
        Haar wavelet decomposition
        descriptor-conditioned channel gate
        coefficient modulation

    Global descriptor (B, D)

    Position-wise descriptor (B, N, D)

    In the global case, the learned channel gate is broadcast over
    sequence positions before being pooled to wavelet resolution.

    The stochastic execution controller is deliberately non-differentiable.
    """

    preserves_shape = True

    def __init__(
        self,
        hidden_dim: int,
        sequence_length: int,
        on_rate: float = 0.1,
        skip_threshold: float = 0.5,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.on_rate = on_rate
        self.skip_threshold = skip_threshold

        self.declare_shapes(
            inputs=((hidden_dim,), (hidden_dim,)), outputs=((hidden_dim,),)
        )

        self.wavelet = HaarWaveletTransform(sequence_length)

        # Descriptor -> hidden channel representation.
        self.gate_projection = FullyConnectedLayer(
            ni=hidden_dim, no=hidden_dim, activation_type="swish"
        )

        # Hidden representation -> one real gate per channel.
        self.gate_output = FullyConnectedLayer(
            ni=hidden_dim, no=hidden_dim, activation_type="linear"
        )

        # This is a scalar logit and is intentionally treated as a
        # non-differentiable policy decision.
        self.control_gate = np.array(
            [[np.log(on_rate / (1.0 - on_rate))]], dtype=GLOBAL_DTYPE
        )
        self.zero_gradients()

    # Gate resolution conversion ---------------
    def _pool_gate(self, gate: NDArray) -> NDArray:
        """
        Pool a position-wise channel gate from sequence resolution to
        Haar coefficient resolution.

        Parameters
        ----------
        gate:
            (B, N, D)

        Returns
        -------
        pooled_gate:
            (B, ceil(N / 2), D)

        Uses exactly the same circular pairing as HaarWaveletTransform.
        """
        return 0.5 * (
            gate[:, self.wavelet.even_indices, :] + gate[:, self.wavelet.odd_indices, :]
        )

    def _unpool_gate(self, dgate_coeff: NDArray) -> NDArray:
        """
        Adjoint of _pool_gate.

        Because _pool_gate averages each circular pair, its adjoint
        distributes one half of the coefficient gradient to each
        participating sequence position.

        Parameters
        ----------
        dgate_coeff:
            (B, ceil(N / 2), D)

        Returns
        -------
        dgate:
            (B, N, D)
        """
        batch = dgate_coeff.shape[0]

        dgate = np.zeros(
            (batch, self.sequence_length, self.hidden_dim), dtype=GLOBAL_DTYPE
        )

        for coefficient in range(self.wavelet.num_coefficients):
            even = self.wavelet.even_indices[coefficient]
            odd = self.wavelet.odd_indices[coefficient]

            gradient = 0.5 * dgate_coeff[:, coefficient, :]

            dgate[:, even, :] += gradient
            dgate[:, odd, :] += gradient

        return dgate

    # forward ----------------------------------------
    def forward(
        self,
        v_tilde: NDArray,
        descriptor: NDArray,
        training_now: Optional[bool] = None,
        mask: Optional[NDArray] = None,
    ) -> NDArray:
        """
        Parameters
        ----------
        v_tilde:
            (B, N, D)
        mask : accepted but not applied. The Haar pairing (see
            HaarWaveletTransform) mixes adjacent sequence positions, so a
            real position paired against a zero-padded one gets a coefficient
            computed half from actual content and half from padding -- a
            correct masked version would need to skip or reweight those
            boundary pairs, which this does not yet do. Only the pairs
            straddling the real/padding boundary are affected; pairs fully
            inside the real content or fully inside the padding are fine
            either way.
        """
        training_now = self.training if training_now is None else training_now
        # Validate inputs
        assert v_tilde.ndim == 3, (
            f"data into the wavelet refinement must be (batch, seq, hidden),"
            f" got {v_tilde.ndim} dimensions"
        )

        batch, sequence, hidden = v_tilde.shape

        assert sequence == self.sequence_length, (
            f"sequence mismatch: expected {self.sequence_length}, got {sequence}"
        )

        assert hidden == self.hidden_dim, (
            f"hidden mismatch: expected {self.hidden_dim}, got {hidden}"
        )

        assert descriptor.ndim in (2, 3), (
            "descriptor must be (batch, hidden) or "
            f"(batch, sequence, hidden), got {descriptor.shape}"
        )

        assert descriptor.shape[0] == batch, (
            f"descriptor batch mismatch: expected {batch}, got {descriptor.shape[0]}"
        )

        assert descriptor.shape[-1] == self.hidden_dim, (
            f"descriptor hidden mismatch: expected {self.hidden_dim}, "
            f"got {descriptor.shape[-1]}"
        )

        if descriptor.ndim == 3:
            assert descriptor.shape[1] == sequence, (
                f"position-wise descriptor sequence mismatch: "
                f"expected {sequence}, got {descriptor.shape[1]}"
            )

        self.v_tilde = v_tilde
        self.descriptor = descriptor
        self.training_now = training_now

        # Execution controller
        self.run_prob = 1.0 / (1.0 + np.exp(-self.control_gate))

        if training_now:
            self.run_mask = (
                self.RNG.random((batch, 1, 1)) < self.run_prob[0, 0]
            ).astype(GLOBAL_DTYPE)

        else:
            should_run = self.run_prob[0, 0] > self.skip_threshold
            self.run_mask = np.full((batch, 1, 1), should_run, dtype=GLOBAL_DTYPE)

        # early term -----
        if not np.any(self.run_mask):
            self.gate = None
            self.gate_pooled = None
            self.low = None
            self.high = None
            self.gated_low = None
            self.gated_high = None
            self.refinement = None
            self.output = v_tilde

            return self.output

        hidden_gate = self.gate_projection(descriptor)
        gate = self.gate_output(hidden_gate)

        if gate.ndim == 2:
            gate = np.broadcast_to(
                gate[:, None, :],
                (batch, sequence, hidden),
            )

        self.gate = gate

        # Bring gate to wavelet coefficient resolution
        self.gate_pooled = self._pool_gate(self.gate)
        self.low, self.high = self.wavelet.analyze(v_tilde)

        self.gated_low = self.low * self.gate_pooled
        self.gated_high = self.high * self.gate_pooled

        self.refinement = self.wavelet.synthesize(self.gated_low, self.gated_high)

        # Residual refinement
        self.output = v_tilde + self.run_mask * self.refinement

        return self.output

    def backward(
        self,
        incoming_gradient: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """
        Returns
        -------
        d_v_tilde:
            Gradient with respect to v_tilde.

        d_descriptor:
            Gradient with respect to the original descriptor shape.

            If descriptor was (B, D), returns (B, D).

            If descriptor was (B, N, D), returns (B, N, D).
        """

        assert self.output is not None, "backward called before forward"

        assert incoming_gradient.shape == self.output.shape, (
            f"gradient shape {incoming_gradient.shape} does not match "
            f"output shape {self.output.shape}"
        )

        d_v_direct = incoming_gradient

        if not np.any(self.run_mask):
            self.gradient_control_gate.fill(0.0)
            return (d_v_direct, np.zeros_like(self.descriptor))

        d_refinement = incoming_gradient * self.run_mask
        d_gated_low, d_gated_high = self.wavelet.analyze(d_refinement)

        d_low = d_gated_low * self.gate_pooled
        d_high = d_gated_high * self.gate_pooled

        d_gate_pooled = (d_gated_low * self.low) + (d_gated_high * self.high)

        d_gate = self._unpool_gate(d_gate_pooled)

        # Undo the broadcast that occurred in forward().
        if self.descriptor.ndim == 2:
            d_gate_fc = d_gate.sum(axis=1)
        else:
            d_gate_fc = d_gate

        d_gate_hidden = self.gate_output.backward(d_gate_fc)
        d_descriptor = self.gate_projection.backward(d_gate_hidden)

        d_v_wavelet = self.wavelet.synthesize(d_low, d_high)
        d_v_tilde = d_v_direct + (self.run_mask * d_v_wavelet)

        self.gradient_control_gate.fill(0.0)

        return d_v_tilde, d_descriptor

    def update_weights(
        self,
        gradient_control_gate: NDArray,
        gate_projection: dict[str, NDArray],
        gate_output: dict[str, NDArray],
    ) -> None:

        self.control_gate -= gradient_control_gate

        self.gate_projection.update_weights(**gate_projection)
        self.gate_output.update_weights(**gate_output)

    def zero_gradients(self) -> None:
        self.gradient_control_gate = np.zeros_like(self.control_gate)

        self.gate_projection.zero_gradients()
        self.gate_output.zero_gradients()

    def get_weights(self, for_serialize: bool = False):
        return {
            "control_gate": self.control_gate,
            "gate_projection": self.gate_projection.get_weights(
                for_serialize=for_serialize
            ),
            "gate_output": self.gate_output.get_weights(for_serialize=for_serialize),
        }

    def set_weights(self, weights: dict) -> None:
        if weights is None:
            return
        self.control_gate = np.asarray(weights["control_gate"], dtype=GLOBAL_DTYPE)
        self.gate_projection.set_weights(weights["gate_projection"])
        self.gate_output.set_weights(weights["gate_output"])

    def get_gradients(
        self,
    ) -> dict[str, NDArray | dict]:

        return {
            "gradient_control_gate": self.gradient_control_gate,
            "gate_projection": self.gate_projection.get_gradients(),
            "gate_output": self.gate_output.get_gradients(),
        }

    def purge(self) -> None:
        self.v_tilde = None
        self.descriptor = None
        self.gate = None
        self.gate_pooled = None
        self.low = None
        self.high = None
        self.gated_low = None
        self.gated_high = None
        self.refinement = None
        self.run_mask = None
        self.output = None

        self.wavelet.purge()
        self.gate_projection.purge()
        self.gate_output.purge()

    @property
    def num_parameters(self) -> int:
        return (
            self.control_gate.size
            + self.gate_projection.num_parameters
            + self.gate_output.num_parameters
        )

    def __str__(self):
        return (
            f"WaveletRefinementModule ({self.hidden_dim}, Haar, on_rate={self.on_rate})"
        )

    def __repr__(self):
        return self.__str__()
