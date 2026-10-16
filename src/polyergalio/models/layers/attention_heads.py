from typing import Optional

import numpy as np
from numpy.typing import NDArray

import polyergalio.models.activations as activations
from polyergalio.models.activations import mod_relu, mod_relu_derivative
from polyergalio.models.layers.basic_layers import RNG, FullyConnectedLayer, Layer, RMSNormLayer
from polyergalio.models.weight_initialization import get_weight_init

DESCRIPTOR_EPS = 1e-10

# -------------head layout funcs----------------------------------
def split_heads(array: NDArray, num_heads: int) -> NDArray:
    """(..., hidden) -> (..., num_heads, head_dim), any number of leading axes"""
    return array.reshape(*array.shape[:-1], num_heads, array.shape[-1] // num_heads)


def merge_heads(array: NDArray) -> NDArray:
    """(..., num_heads, head_dim) -> (..., hidden), any number of leading axes"""
    return array.reshape(*array.shape[:-2], array.shape[-2] * array.shape[-1])


def align_gate(gate: NDArray) -> NDArray:
    """(batch, num_heads, frequency) -> (batch, frequency, num_heads, 1), broadcasting over split_heads"""
    return np.transpose(gate, (0, 2, 1))[..., None]


def shift_frequencies(array: NDArray, offset: int) -> NDArray:
    """shift along the last (frequency) axis, zero filled rather than circular"""
    out = np.zeros_like(array)
    if offset > 0:
        out[..., offset:] = array[..., :-offset]
    elif offset < 0:
        out[..., :offset] = array[..., -offset:]
    else:
        out[...] = array
    return out


class DenseHead(Layer):
    """
    Batched per-head dense layer: FullyConnectedLayer with a head axis. Each head owns an independent
    (ni, no) map over its own slice of the input

        y_h = activation(x_h W_h + b_h)

    Input shape: (..., num_heads * ni)
    Output shape: (..., num_heads * no)
    """
    parameter_names = ("weights", "bias")
    cache_names = ("input", "z", "output")

    def __init__(
        self,
        num_heads: int,
        input_dimension: int,
        output_dimension: int,
        activation_type: str = "linear",
        initialization: Optional[str] = None,
        initialization_kwargs: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        num_heads : independent maps, one per head
        input_dimension, output_dimension : input and output width of each head
        activation_type : activation applied per head, e.g. softmax normalises within a head
        initialization : any WEIGHT_INIT_DISPATCHER name, the activation's rule if None
        initialization_kwargs : keyword arguments bound to the initializer
        """
        super().__init__()
        self.num_heads = num_heads
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self.activation_type = activation_type
        self.initialization = initialization
        self.initialization_kwargs = dict(initialization_kwargs or {})
        self.activation_function = activations.activation_dictionary[activation_type]
        self.activation_derivative = activations.derivative_dictionary[activation_type]
        self.declare_shapes(inputs=((num_heads * input_dimension,),), outputs=((num_heads * output_dimension,),))

        initializer = get_weight_init(initialization or activation_type, **self.initialization_kwargs)
        self.weights = np.stack([initializer(RNG, ni=input_dimension, no=output_dimension) for _ in range(num_heads)])
        self.bias = np.zeros((num_heads, output_dimension))
        self.zero_gradients()

    def pre_activation(self, input_data: NDArray) -> NDArray:
        """(..., num_heads * ni) -> (..., num_heads, no), before the activation"""
        return np.einsum("...hi,hio->...ho", split_heads(input_data, self.num_heads), self.weights) + self.bias

    def project(self, input_data: NDArray) -> NDArray:
        """the full map without caching, for decoding"""
        return merge_heads(self.activation_function(self.pre_activation(input_data)))

    def forward(self, input_data: NDArray) -> NDArray:
        self.input = input_data
        self.z = self.pre_activation(input_data)
        self.output = self.activation_function(self.z)
        return merge_heads(self.output)

    def backward(self, incoming_gradient: NDArray) -> NDArray:
        delta = self.activation_derivative(self.output, self.z, split_heads(incoming_gradient, self.num_heads))
        flat_input = split_heads(self.input, self.num_heads).reshape(-1, self.num_heads, self.input_dimension)
        flat_delta = delta.reshape(-1, self.num_heads, self.output_dimension)

        self.gradient_weights = np.einsum("nhi,nho->hio", flat_input, flat_delta, optimize=True)
        self.gradient_bias = flat_delta.sum(axis=0)
        return merge_heads(np.einsum("...ho,hio->...hi", delta, self.weights, optimize=True))

    def __str__(self):
        return f"DenseHead, {self.num_heads} heads of {self.input_dimension} -> {self.output_dimension}, {self.activation_type}"

    def __repr__(self):
        return self.__str__()


class HeadProjection(DenseHead):
    """
    Independent (head_dim, head_dim) linear map per head, a block-diagonal projection of the hidden
    axis: SPECTRE's per-head W(q) and W(v). A linear, square DenseHead.

    Input shape: (..., num_heads * head_dim)
    Output shape: (..., num_heads * head_dim)
    """

    preserves_shape = True

    def __init__(
        self,
        num_heads: int,
        head_dimension: int,
        initialization: str = "lecun",
        initialization_kwargs: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        num_heads : independent maps, one per head
        head_dimension : width of each head's slice of the hidden axis
        initialization : any WEIGHT_INIT_DISPATCHER name, drawn independently per head
        initialization_kwargs : keyword arguments bound to the initializer
        """
        super().__init__(
            num_heads,
            head_dimension,
            head_dimension,
            activation_type="linear",
            initialization=initialization,
            initialization_kwargs=initialization_kwargs,
        )
        self.head_dimension = head_dimension
        self.hidden_dimension = num_heads * head_dimension

    def __str__(self):
        return f"HeadProjection, {self.num_heads} heads of {self.head_dimension} -> {self.head_dimension}"


class HeadGate(Layer):
    """
    SPECTRE's per-head spectral gate, from pooled query to activated gate. Each head layer-normalises
    its own pooled query over head_dim, maps it through its own two-layer MLP to one complex value per
    frequency, optionally mixes neighbouring frequencies with a Toeplitz band, then applies modReLU:

        raw_h = W2_h relu(W1_h LN_h(mean_q_h) + b1_h) + b2_h,  split into real | imaginary halves
        gate_h = modReLU(raw_h + band_h * raw_h, activation_bias_h)

    The MLP is two DenseHead layers, hidden_layer (W1, b1, relu) and output_layer (W2, b2, linear).
    Heads share nothing, so a head's filter depends only on its own queries.

    Starts at the identity: b2 puts every gate at 1 + 0j after modReLU, and W2 is shrunk so content
    moves the gate gradually. A random gate starts each frequency at an arbitrary magnitude and,
    being bilinear with the values, diverges under plain SGD within a few steps.

    Input shape: (batch, num_heads * head_dim) pooled query
    Output shape: (batch, num_heads, num_frequencies) complex gate
    """
    cache_names = ("std", "x_norm", "normed", "gate_raw", "gate_pre_activation", "gate")

    def __init__(
        self,
        num_heads: int,
        head_dimension: int,
        num_frequencies: int,
        gate_dimension: int,
        modrelu_bias: float = -0.1,
        band_radius: int = 0,
        weight_scale: float = 0.1,
        hidden_initialization: str = "relu",
        output_initialization: str = "lecun",
    ):
        """
        Parameters
        ----------
        num_heads, head_dimension : head layout of the pooled query
        num_frequencies : gate entries per head
        gate_dimension : width of each head's hidden layer
        modrelu_bias : starting modReLU bias, learned per head and frequency
        band_radius : radius r of the Toeplitz band update, 2r+1 complex taps per head; 0 disables it
        weight_scale : shrink on W2 at initialisation
        hidden_initialization : WEIGHT_INIT_DISPATCHER name for W1
        output_initialization : WEIGHT_INIT_DISPATCHER name for W2, before weight_scale
        """
        assert band_radius >= 0, (
            f"band_radius must be zero or positive, got {band_radius}. A "
            "negative radius produces no taps and silently disables the gate."
        )
        super().__init__()
        self.num_heads = num_heads
        self.head_dimension = head_dimension
        self.num_frequencies = num_frequencies
        self.gate_dimension = gate_dimension

        self.DESCRIPTOR_EPS = DESCRIPTOR_EPS

        self.modrelu_bias = modrelu_bias
        self.band_radius = band_radius
        self.weight_scale = weight_scale

        self.hidden_initialization = hidden_initialization
        self.output_initialization = output_initialization
        self.declare_shapes(
            inputs=((num_heads * head_dimension,),), outputs=((num_heads, num_frequencies),)
        )

        self.activation_bias = np.full((num_heads, num_frequencies), modrelu_bias)

        # building parameters --------
        self.gamma = np.ones((num_heads, head_dimension))
        self.beta = np.zeros((num_heads, head_dimension))
        self.hidden_layer = DenseHead(
            num_heads, head_dimension, gate_dimension, activation_type="relu", initialization=hidden_initialization
        )
        self.output_layer = DenseHead(
            num_heads, gate_dimension, 2 * num_frequencies, activation_type="linear", initialization=output_initialization
        )
        self.output_layer.weights *= weight_scale
        self.output_layer.bias[:, :num_frequencies] = 1.0 - self.activation_bias

        self.band_offsets = tuple(range(-band_radius, band_radius + 1))
        # every tap acts on all num_frequencies bins at once, so its gradient grows with sequence
        # length; scaling taps by 1/sqrt(num_frequencies) keeps their step size length-independent
        self.band_scale = 1.0 / np.sqrt(num_frequencies)
        names = ("gamma", "beta", "activation_bias", "hidden_layer", "output_layer")
        if band_radius:
            self.band_taps = np.zeros((num_heads, len(self.band_offsets)), dtype=complex)
            names += ("band_taps",)
        self.parameter_names = names
        self.zero_gradients()

    def band_update(self, gate: NDArray) -> NDArray:
        """gate plus its band-tap mix of neighbouring frequencies"""
        banded = np.zeros_like(gate)
        for index, offset in enumerate(self.band_offsets):
            banded += self.band_scale * self.band_taps[:, index, None] * shift_frequencies(gate, offset)
        return gate + banded

    def forward(self, pooled_query: NDArray) -> NDArray:
        """
        Parameters
        ----------
        pooled_query : (batch, hidden) mean query, heads laid out along the hidden axis

        Returns
        -------
        (batch, num_heads, num_frequencies) activated complex gate
        """
        heads = split_heads(pooled_query, self.num_heads)
        centred = heads - heads.mean(axis=-1, keepdims=True)
        self.std = np.sqrt(np.mean(centred ** 2, axis=-1, keepdims=True) + self.DESCRIPTOR_EPS)
        self.x_norm = centred / self.std
        self.normed = self.gamma * self.x_norm + self.beta

        hidden = self.hidden_layer.forward(merge_heads(self.normed))
        projection = split_heads(self.output_layer.forward(hidden), self.num_heads)

        real, imaginary = np.split(projection, 2, axis=-1)
        self.gate_raw = real + 1j * imaginary
        self.gate_pre_activation = self.band_update(self.gate_raw) if self.band_radius else self.gate_raw
        self.gate = mod_relu(self.gate_pre_activation, self.activation_bias)
        return self.gate

    @property
    def descriptor(self) -> NDArray:
        """(batch, hidden) per-head normalised pooled query, the conditioning the WRM reads"""
        return merge_heads(self.normed)

    def backward(self, gate_gradient: NDArray, descriptor_gradient: Optional[NDArray] = None) -> NDArray:
        """
        Parameters
        ----------
        gate_gradient : (batch, num_heads, num_frequencies) complex, dL/dRe + i dL/dIm
        descriptor_gradient : (batch, hidden) gradient reaching the descriptor from elsewhere (the WRM)

        Returns
        -------
        (batch, hidden) gradient on the pooled query
        """
        self.gradient_activation_bias, raw_gradient = mod_relu_derivative(
            z=self.gate_pre_activation, beta=self.activation_bias, dout=gate_gradient
        )
        if self.band_radius:
            self.gradient_band_taps = np.stack(
                [
                    self.band_scale * np.sum(raw_gradient * np.conj(shift_frequencies(self.gate_raw, offset)), axis=(0, 2))
                    for offset in self.band_offsets
                ],
                axis=-1,
            )
            banded_gradient = raw_gradient.copy()
            for index, offset in enumerate(self.band_offsets):
                banded_gradient += (
                    self.band_scale * np.conj(self.band_taps[:, index, None]) * shift_frequencies(raw_gradient, -offset)
                )
            raw_gradient = banded_gradient

        projection_gradient = np.concatenate([raw_gradient.real, raw_gradient.imag], axis=-1)
        hidden_gradient = self.output_layer.backward(merge_heads(projection_gradient))
        normed_gradient = split_heads(self.hidden_layer.backward(hidden_gradient), self.num_heads)

        if descriptor_gradient is not None:
            normed_gradient = normed_gradient + split_heads(descriptor_gradient, self.num_heads)
        self.gradient_gamma = np.sum(normed_gradient * self.x_norm, axis=0)
        self.gradient_beta = normed_gradient.sum(axis=0)

        x_gradient = normed_gradient * self.gamma
        heads_gradient = (
            x_gradient
            - x_gradient.mean(axis=-1, keepdims=True)
            - self.x_norm * np.mean(x_gradient * self.x_norm, axis=-1, keepdims=True)
        ) / self.std
        return merge_heads(heads_gradient)

    def __str__(self):
        band = f", band radius {self.band_radius}" if self.band_radius else ""
        return (
            f"HeadGate, {self.num_heads} heads of {self.head_dimension} -> {self.gate_dimension} -> "
            f"{self.num_frequencies} complex frequencies{band}"
        )

    def __repr__(self):
        return self.__str__()


class LatentHead(Layer):
    """
    Per-head projection through a shared low-rank latent space representation -
    MLA's compression in place of DenseHead's block-diagonal map.

        c = RMSNorm(x W_down)         (..., rank), one latent for all heads
        y_h = c W_up[h] + b[h]        (..., head_dimension), independent per head

    DenseHead gives each head its own slice of the input. Here every head reads the same latent of the whole input,
    so heads mix channels through the bottleneck. compress and expand are public so a caller can pool, cache or
    reuse the latent: pooling c and expanding once equals expanding every token and pooling.

    Input shape: (..., hidden_dimension)
    Output shape: (..., num_heads * head_dimension)
    """
    parameter_names = ("down", "norm", "weights", "bias")
    cache_names = ("latent",)

    def __init__(
        self,
        num_heads: int,
        head_dimension: int,
        rank: int,
        hidden_dimension: Optional[int] = None,
        initialization: str = "lecun",
        initialization_kwargs: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        num_heads : independent up-projections, one per head
        head_dimension : width of each head's output
        rank : width of the shared latent
        hidden_dimension : width of the input, num_heads * head_dimension if None
        initialization : any WEIGHT_INIT_DISPATCHER name, drawn independently per head for the up-projection
        initialization_kwargs : keyword arguments bound to the initializer
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dimension = head_dimension
        self.rank = rank
        self.hidden_dimension = hidden_dimension or num_heads * head_dimension
        self.initialization = initialization
        self.initialization_kwargs = dict(initialization_kwargs or {})
        self.declare_shapes(
            inputs=((self.hidden_dimension,),), outputs=((num_heads * head_dimension,),)
        )

        self.down = FullyConnectedLayer(self.hidden_dimension, rank, "linear")
        self.norm = RMSNormLayer(rank)
        initializer = get_weight_init(initialization, **self.initialization_kwargs)
        self.weights = np.stack([initializer(RNG, ni=rank, no=head_dimension) for _ in range(num_heads)])
        self.bias = np.zeros((num_heads, head_dimension))
        self.zero_gradients()

    def compress(self, input_data: NDArray) -> NDArray:
        """(..., hidden_dimension) -> (..., rank)"""
        return self.norm.forward(self.down.forward(input_data))

    def expand(self, latent: NDArray) -> NDArray:
        """(..., rank) -> (..., num_heads * head_dimension); the latent is kept for backward_expand in training"""
        if self.training:
            self.latent = latent
        return merge_heads(np.einsum("...r,hro->...ho", latent, self.weights, optimize=True) + self.bias)

    def backward_expand(self, incoming_gradient: NDArray) -> NDArray:
        """Set the up-projection gradients and return the gradient on the latent, (..., rank)."""
        delta = split_heads(incoming_gradient, self.num_heads)
        flat_latent = self.latent.reshape(-1, self.rank)
        flat_delta = delta.reshape(-1, self.num_heads, self.head_dimension)
        self.gradient_weights = np.einsum("nr,nho->hro", flat_latent, flat_delta, optimize=True)
        self.gradient_bias = flat_delta.sum(axis=0)
        return np.einsum("...ho,hro->...r", delta, self.weights, optimize=True)

    def backward_compress(self, latent_gradient: NDArray) -> NDArray:
        """Carry a latent gradient back through the norm and down-projection to the input."""
        return self.down.backward(self.norm.backward(latent_gradient))

    def forward(self, input_data: NDArray) -> NDArray:
        return self.expand(self.compress(input_data))

    def backward(self, incoming_gradient: NDArray) -> NDArray:
        return self.backward_compress(self.backward_expand(incoming_gradient))

    def __str__(self):
        return (
            f"LatentHead, {self.hidden_dimension} -> rank {self.rank} -> {self.num_heads} heads of {self.head_dimension}"
        )

    def __repr__(self):
        return self.__str__()