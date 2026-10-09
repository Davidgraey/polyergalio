"""
Hyena hierarchy, https://arxiv.org/abs/2302.10866

Hyena shows great potential for long and extra long sequences 10K - 64K in the paper
An order-N Hyena operator blends convolutions with element-wise gating

    x_1, ..., x_N, v = Projection(u)
    h_1, ..., h_N = HyenaFilter(L)
    z_1 = v,  z_{n+1} = x_n * (h_n conv z_n),  y = z_{N+1}

Filters are causal and evaluated through the FFT, zero padded to 2L.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from polyergalio.models.layers.basic_layers import RNG, FullyConnectedLayer, Layer
from polyergalio.models.weight_initialization import get_weight_init


# -------------    filter basis and window    -----------------------
def positional_features(sequence_length: int, num_bands: int) -> NDArray:
    """
    Truncated complex exponential basis

    Returns
    -------
    (sequence_length, 2 * num_bands + 1): [t, Re rho_0 .. Re rho_K-1, Im rho_0 .. Im rho_K-1],
    rho_k(t) = exp(i 2 pi k t / L), t normalised to [0, 1] in the first column
    """
    steps = np.arange(sequence_length)
    angles = 2 * np.pi * np.outer(steps, np.arange(num_bands)) / sequence_length
    time = np.linspace(0.0, 1.0, sequence_length)[:, None]
    return np.concatenate([time, np.cos(angles), np.sin(angles)], axis=1)


def exponential_window(sequence_length: int,
                       channels: int,
                       fast_decay: float = 0.3,
                       slow_decay: float = 1.5,
                       decay_target: float = 1e-2,
                       window_shift: float = 0.05
                       ) -> NDArray:
    """
    alpha spread across channels

    Parameters
    ----------
    fast_decay, slow_decay : fraction of the sequence at which the fastest / slowest channel reaches decay_target
    window_shift : bias keeping filters from being forced to zero past the decay

    Returns
    -------
    (sequence_length, channels)
    """
    time = np.linspace(0.0, 1.0, sequence_length)[:, None]
    rates = np.abs(np.linspace(np.log(decay_target) / fast_decay, np.log(decay_target) / slow_decay, channels))
    return np.exp(-time * rates[None]) + window_shift


# -------------    causal FFT convolution    -----------------------
def causal_convolution(signal: NDArray, filters: NDArray) -> NDArray:
    """
    y_t = sum_{j <= t} h_{t - j} s_j, per channel

    Parameters
    ----------
    signal : (batch, length, channels)
    filters : (length, channels)
    """
    length = signal.shape[1]
    padded = 2 * length
    spectrum = np.fft.rfft(signal, n=padded, axis=1) * np.fft.rfft(filters, n=padded, axis=0)[None]
    return np.fft.irfft(spectrum, n=padded, axis=1)[:, :length]


def causal_convolution_backward(
    gradient: NDArray, signal: NDArray, filters: NDArray
) -> tuple[NDArray, NDArray]:
    """
    Adjoint of causal_convolution, both correlations of the output gradient

    Returns
    -------
    signal gradient (batch, length, channels), filter gradient (length, channels)
    """
    length = signal.shape[1]
    padded = 2 * length

    gradient_transform = np.fft.rfft(gradient, n=padded, axis=1)
    filter_transform = np.fft.rfft(filters, n=padded, axis=0)[None]
    signal_transform = np.fft.rfft(signal, n=padded, axis=1)

    signal_gradient = np.fft.irfft(gradient_transform * np.conj(filter_transform), n=padded, axis=1)[:, :length]
    filter_gradient = np.fft.irfft(gradient_transform * np.conj(signal_transform), n=padded, axis=1)[:, :length]
    return signal_gradient, filter_gradient.sum(axis=0)


def delay(array: NDArray, steps: int) -> NDArray:
    """shift along the sequence axis (1) by steps, zero filled at the start"""
    if steps == 0:
        return array
    out = np.zeros_like(array)
    out[:, steps:] = array[:, :-steps]
    return out


def advance(array: NDArray, steps: int) -> NDArray:
    """adjoint of delay: shift back along the sequence axis, zero filled at the end"""
    if steps == 0:
        return array
    out = np.zeros_like(array)
    out[:, :-steps] = array[:, steps:]
    return out


# ------------------------------------------------------------------
class ShortConvolution(Layer):
    """
    Causal depthwise convolution over the sequence axis, Algorithm 1 step 2

        y_t = b + sum_{j < kernel_size} w_j * x_{t - j}

    Input shape: (batch, length, channels)
    Output shape: (batch, length, channels)
    """

    preserves_shape = True
    parameter_names = ("weights", "bias")
    cache_names = ("input",)

    def __init__(self, channels: int, kernel_size: int = 3, initialization: str = "lecun"):
        """
        Parameters
        ----------
        channels : independent filters, one per channel
        kernel_size : taps per filter, current token plus kernel_size - 1 past tokens
        initialization : any WEIGHT_INIT_DISPATCHER name; fan-in is kernel_size
        """
        assert kernel_size >= 1, f"kernel_size must be at least 1, got {kernel_size}"
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.initialization = initialization
        self.declare_shapes(inputs=((channels,),), outputs=((channels,),))

        self.weights = get_weight_init(initialization)(RNG, ni=kernel_size, no=channels)
        self.bias = np.zeros(channels)
        self.zero_gradients()

    def forward(self, input_data: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        self.input = input_data
        output = np.broadcast_to(self.bias, input_data.shape).copy()
        for lag in range(self.kernel_size):
            output += self.weights[lag] * delay(input_data, lag)
        return output

    def backward(self, incoming_gradient: NDArray) -> NDArray:
        self.gradient_weights = np.stack(
            [np.sum(incoming_gradient * delay(self.input, lag), axis=(0, 1)) for lag in range(self.kernel_size)]
        )
        self.gradient_bias = incoming_gradient.sum(axis=(0, 1))
        return sum(self.weights[lag] * advance(incoming_gradient, lag) for lag in range(self.kernel_size))

    def __str__(self):
        return f"ShortConvolution, {self.channels} channels, kernel {self.kernel_size}"

    def __repr__(self):
        return self.__str__()


class HyenaFilter(Layer):
    """
    Implicit long-convolution filters, Algorithm 2 and eq. 7

        h_t = Window(t) * FFN(PositionalEncoding(t)) + skip * delta(t)
        FFN = Linear -> sin -> Linear -> sin -> Linear (no bias)

    skip is the reference implementation's per-channel bias D, conv(u, h) + D u, held as an impulse
    at t = 0. It starts at 0, so the filters begin as the paper's; starting at 1 (a gated
    passthrough) makes the operator cubic in its projections at initialisation and diverged under SGD.

    Each window channel is scaled to unit mass over t, a constant the last FFN layer could absorb,
    so the operator's output scale does not grow with sequence length. Dividing by the filter's own
    L1 norm instead (the reference implementation's option) steps by 1 / norm and diverged under SGD.

    Filter parameters are decoupled from sequence length. The paper's sine frequency w, sin(w a),
    is folded into the initial scale of the two sine layers' weights: the same filters at
    initialisation (a high w gives rich high-frequency content, D.3), without w multiplying every
    FFN gradient. A separately learned w is redundant with the weights and diverged under SGD.

    A container layer like PersistentMemory: forward takes no input.
    Output shape: (order, sequence_length, channels)
    """

    preserves_shape = False
    parameter_names = ("weights_1", "bias_1", "weights_2", "bias_2", "weights_3", "skip")
    cache_names = ("hidden_pre_1", "hidden_1", "hidden_pre_2", "hidden_2", "filters")

    def __init__(
        self,
        sequence_length: int,
        channels: int,
        order: int = 2,
        filter_features: int = 64,
        positional_bands: int = 8,
        sine_frequency: float = 10.0,
        fast_decay: float = 0.3,
        slow_decay: float = 1.5,
        window_shift: float = 0.05,
        initialization: str = "lecun",
    ):
        """
        Parameters
        ----------
        sequence_length : filter length L
        channels : filters per order, the operator's hidden_dim
        order : number of filters N
        filter_features : FFN hidden width
        positional_bands : K, the positional encoding has 2K + 1 features
        sine_frequency : w, the initial weight scale of both sine layers
        fast_decay, slow_decay, window_shift : see exponential_window
        initialization : any WEIGHT_INIT_DISPATCHER name for the FFN weights
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.channels = channels
        self.order = order
        self.filter_features = filter_features
        self.positional_bands = positional_bands
        self.sine_frequency = sine_frequency
        self.fast_decay = fast_decay
        self.slow_decay = slow_decay
        self.window_shift = window_shift
        self.initialization = initialization
        self.declare_shapes(inputs=(), outputs=((order, sequence_length, channels),))

        self.positions = positional_features(sequence_length, positional_bands)
        window = exponential_window(
            sequence_length,
            order * channels,
            fast_decay=fast_decay,
            slow_decay=slow_decay,
            window_shift=window_shift,
        )
        self.window = window / window.sum(axis=0)

        features = self.positions.shape[1]
        initializer = get_weight_init(initialization)
        self.weights_1 = sine_frequency * initializer(RNG, ni=features, no=filter_features)
        self.bias_1 = np.zeros(filter_features)
        self.weights_2 = sine_frequency * initializer(RNG, ni=filter_features, no=filter_features)
        self.bias_2 = np.zeros(filter_features)
        self.weights_3 = initializer(RNG, ni=filter_features, no=order * channels)
        self.skip = np.zeros((order, channels))
        self.zero_gradients()

    def forward(self) -> NDArray:
        self.hidden_pre_1 = self.positions @ self.weights_1 + self.bias_1
        self.hidden_1 = np.sin(self.hidden_pre_1)
        self.hidden_pre_2 = self.hidden_1 @ self.weights_2 + self.bias_2
        self.hidden_2 = np.sin(self.hidden_pre_2)
        filters = (self.hidden_2 @ self.weights_3) * self.window
        self.filters = filters.reshape(self.sequence_length, self.order, self.channels).transpose(1, 0, 2).copy()
        self.filters[:, 0] += self.skip
        return self.filters

    def backward(self, filter_gradient: NDArray) -> None:
        """
        Parameters
        ----------
        filter_gradient : (order, sequence_length, channels)
        """
        self.gradient_skip = filter_gradient[:, 0].copy()
        output_gradient = filter_gradient.transpose(1, 0, 2).reshape(self.sequence_length, -1) * self.window
        self.gradient_weights_3 = self.hidden_2.T @ output_gradient

        pre_2_gradient = (output_gradient @ self.weights_3.T) * np.cos(self.hidden_pre_2)
        self.gradient_weights_2 = self.hidden_1.T @ pre_2_gradient
        self.gradient_bias_2 = pre_2_gradient.sum(axis=0)

        pre_1_gradient = (pre_2_gradient @ self.weights_2.T) * np.cos(self.hidden_pre_1)
        self.gradient_weights_1 = self.positions.T @ pre_1_gradient
        self.gradient_bias_1 = pre_1_gradient.sum(axis=0)

    def __str__(self):
        return (
            f"HyenaFilter, {self.order} x {self.channels} filters of length {self.sequence_length}, "
            f"{self.positions.shape[1]} -> {self.filter_features} -> {self.filter_features} features"
        )

    def __repr__(self):
        return self.__str__()


class HyenaOperator(Layer):
    """
    Order-N Hyena operator, Algorithm 3

        z = ShortConvolution(input_projection(u)),  split into x_1 .. x_N, v
        z_1 = v * mask,  z_{n+1} = x_n * causal_convolution(z_n, h_n)
        y = output_projection(z_{N+1})

    Causal by construction (Proposition 3.1). order=2 is H3 with implicit filters.
    Padded positions are zeroed before the short convolution and again in v, so they never reach
    other tokens.

    Both projections start variance preserving (lecun), not kaiming: the output is a degree N + 2
    product of weights, and kaiming's extra sqrt(2) per factor diverged under SGD.

    Input shape: (batch, sequence_length, hidden_dim)
    Output shape: (batch, sequence_length, hidden_dim)
    """

    registry_name = "HyenaOperator"
    preserves_shape = True
    parameter_names = ("input_projection", "short_convolution", "hyena_filter", "output_projection")
    cache_names = ("input", "mask", "gates", "filters", "states", "convolved", "output")

    def __init__(
        self,
        sequence_length: int,
        hidden_dim: int,
        order: int = 2,
        filter_features: int = 64,
        positional_bands: int = 8,
        short_kernel: int = 3,
        sine_frequency: float = 10.0,
        initialization: str = "lecun",
    ):
        """
        Parameters
        ----------
        sequence_length : tokens per sample, the filter length
        hidden_dim : model width D
        order : N, gated long convolutions in the recurrence
        filter_features, positional_bands, sine_frequency : see HyenaFilter
        short_kernel : taps of the short depthwise convolution on the projections
        initialization : WEIGHT_INIT_DISPATCHER name for the projections, short convolution and filter FFN
        """
        assert order >= 1, f"order must be at least 1, got {order}"
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.order = order
        self.filter_features = filter_features
        self.positional_bands = positional_bands
        self.short_kernel = short_kernel
        self.sine_frequency = sine_frequency
        self.initialization = initialization

        self.declare_shapes(
            inputs=((sequence_length, hidden_dim),),
            outputs=((sequence_length, hidden_dim),),
        )

        self.input_projection = FullyConnectedLayer(
            hidden_dim, (order + 1) * hidden_dim, "linear", initialization_override=initialization
        )
        self.short_convolution = ShortConvolution(
            (order + 1) * hidden_dim, kernel_size=short_kernel, initialization=initialization
        )
        self.hyena_filter = HyenaFilter(
            sequence_length=sequence_length,
            channels=hidden_dim,
            order=order,
            filter_features=filter_features,
            positional_bands=positional_bands,
            sine_frequency=sine_frequency,
            initialization=initialization,
        )
        self.output_projection = FullyConnectedLayer(
            hidden_dim, hidden_dim, "linear", initialization_override=initialization
        )
        self.zero_gradients()

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
        assert input_data.shape[1] == self.sequence_length
        assert input_data.shape[2] == self.hidden_dim

        self.input = input_data
        self.mask = mask if mask is not None else np.ones(input_data.shape[:2])

        projected = self.input_projection.forward(input_data) * self.mask[..., None]
        mixed = self.short_convolution.forward(projected)
        *self.gates, value = np.split(mixed, self.order + 1, axis=-1)
        self.filters = self.hyena_filter.forward()

        state = value * self.mask[..., None]
        self.states = [state]
        self.convolved = []
        for gate, filters in zip(self.gates, self.filters):
            convolved = causal_convolution(state, filters)
            state = gate * convolved
            self.convolved.append(convolved)
            self.states.append(state)

        self.output = self.output_projection.forward(state)
        return self.output

    def backward(self, incoming_gradient: NDArray) -> NDArray:
        state_gradient = self.output_projection.backward(incoming_gradient)

        gate_gradients = [None] * self.order
        filter_gradient = np.zeros_like(self.filters)
        for n in reversed(range(self.order)):
            gate_gradients[n] = state_gradient * self.convolved[n]
            state_gradient, filter_gradient[n] = causal_convolution_backward(
                state_gradient * self.gates[n], self.states[n], self.filters[n]
            )
        value_gradient = state_gradient * self.mask[..., None]

        self.hyena_filter.backward(filter_gradient)
        mixed_gradient = np.concatenate(gate_gradients + [value_gradient], axis=-1)
        projected_gradient = self.short_convolution.backward(mixed_gradient) * self.mask[..., None]
        return self.input_projection.backward(projected_gradient)

    def __str__(self):
        return f"Hyena operator, order {self.order}, sequence {self.sequence_length}, hidden {self.hidden_dim}"

    def __repr__(self):
        return self.__str__()
