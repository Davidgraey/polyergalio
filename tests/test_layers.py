"""
Layers: each backward pass against finite differences of its own forward pass.
"""

import numpy as np
import pytest
from conftest import (
    GRADIENT_TOLERANCE,
    as_float64,
    input_gradient_error,
    parameter_gradient_error,
)
from polyergalio.models.layers.basic_layers import (
    DropoutLayer,
    FullyConnectedLayer,
    NormalizeLayer,
    RMSNormLayer,
)
from polyergalio.models.layers.fft_layers import (
    FourierLayer,
    FrequencyFFT,
    InverseFourierLayer,
    hartley,
)
from polyergalio.models.layers.operator_layers import LatentStack

ACTIVATIONS = ("linear", "relu", "relu_leaky", "sigmoid", "tanh", "swish", "softmax")


# -------------    FullyConnectedLayer    --------------------------
@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_fully_connected_input_gradient(activation, small_matrix):
    layer = as_float64(FullyConnectedLayer(4, 3, activation))
    assert input_gradient_error(layer, small_matrix) < GRADIENT_TOLERANCE


@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_fully_connected_weight_gradient(activation, small_matrix):
    layer = as_float64(FullyConnectedLayer(4, 3, activation))
    assert (
        parameter_gradient_error(layer, small_matrix, layer.weights, "gradient_weights")
        < GRADIENT_TOLERANCE
    )


def test_fully_connected_bias_gradient(small_matrix):
    layer = as_float64(FullyConnectedLayer(4, 3, "tanh"))
    assert (
        parameter_gradient_error(layer, small_matrix, layer.bias, "gradient_bias")
        < GRADIENT_TOLERANCE
    )


def test_fully_connected_handles_three_dimensional_input(sequence_batch):
    """(batch, sequence, hidden) must round trip through the 2D reshape"""
    layer = as_float64(FullyConnectedLayer(4, 3, "swish"))
    assert input_gradient_error(layer, sequence_batch) < GRADIENT_TOLERANCE


# -------------    normalisation    --------------------------------
@pytest.mark.parametrize("shift_scale", (True, False))
def test_normalize_input_gradient(shift_scale, small_matrix):
    layer = NormalizeLayer(4, shift_scale=shift_scale)
    if shift_scale:
        layer.scale_gamma = np.full((1, 4), 1.3)
        layer.shift_beta = np.full((1, 4), 0.2)
    assert input_gradient_error(layer, small_matrix) < GRADIENT_TOLERANCE


@pytest.mark.parametrize(
    "parameter_name,gradient_name",
    (("scale_gamma", "gradient_gamma"), ("shift_beta", "gradient_beta")),
)
def test_normalize_parameter_gradients(parameter_name, gradient_name, small_matrix):
    layer = NormalizeLayer(4, shift_scale=True)
    layer.scale_gamma = np.full((1, 4), 1.3)
    layer.shift_beta = np.full((1, 4), 0.2)
    error = parameter_gradient_error(
        layer, small_matrix, getattr(layer, parameter_name), gradient_name
    )
    assert error < GRADIENT_TOLERANCE


def test_rms_norm_input_gradient(small_matrix):
    layer = RMSNormLayer(4)
    layer.scale_gamma = np.full((1, 4), 1.2)
    assert input_gradient_error(layer, small_matrix) < GRADIENT_TOLERANCE


def test_rms_norm_gamma_gradient(small_matrix):
    layer = RMSNormLayer(4)
    layer.scale_gamma = np.full((1, 4), 1.2)
    error = parameter_gradient_error(
        layer, small_matrix, layer.scale_gamma, "gradient_gamma"
    )
    assert error < GRADIENT_TOLERANCE


# -------------    DropoutLayer    ---------------------------------
def test_dropout_inverted_scaling_preserves_the_mean():
    layer = DropoutLayer(dropout_prob=0.5, use_rescale=True)
    kept = layer.forward(np.ones((400, 250)), training_now=True)
    assert kept.mean() == pytest.approx(1.0, abs=0.02)


def test_dropout_passes_through_at_evaluation():
    layer = DropoutLayer(dropout_prob=0.5)
    x_data = np.ones((20, 10))
    assert np.array_equal(layer.forward(x_data, training_now=False), x_data)


def test_dropout_backward_applies_the_mask():
    layer = DropoutLayer(dropout_prob=0.5, use_rescale=False)
    layer.forward(np.ones((30, 20)), training_now=True)
    delta = layer.backward(np.ones((30, 20)))
    assert np.array_equal(delta.astype(bool), layer.mask.astype(bool))


# -------------    Hartley / FFT layers    -------------------------
@pytest.mark.parametrize("length", (8, 9))
def test_hartley_is_its_own_inverse_up_to_scale(length):
    rng = np.random.default_rng(0)
    x_data = rng.normal(size=(3, length))
    assert np.allclose(hartley(hartley(x_data)) / length, x_data)


@pytest.mark.parametrize("layer_class", (FourierLayer, InverseFourierLayer))
@pytest.mark.parametrize("use_2d", (True, False))
def test_fourier_layer_gradient(layer_class, use_2d):
    rng = np.random.default_rng(0)
    layer = layer_class(use_2d=use_2d)
    assert input_gradient_error(layer, rng.normal(size=(3, 8, 4))) < GRADIENT_TOLERANCE


@pytest.mark.parametrize("use_2d", (True, False))
def test_fourier_pair_round_trips(use_2d):
    rng = np.random.default_rng(0)
    x_data = rng.normal(size=(3, 8, 4))
    forward = FourierLayer(use_2d=use_2d)
    inverse = InverseFourierLayer(use_2d=use_2d)
    assert np.allclose(inverse.forward(forward.forward(x_data)), x_data)


@pytest.mark.parametrize("window_size", (4, 5, 8, 9))
def test_frequency_fft_gradient(window_size):
    rng = np.random.default_rng(0)
    layer = FrequencyFFT(max_sequence_length=16, window_size=window_size)
    x_data = rng.normal(size=(6, window_size))
    assert input_gradient_error(layer, x_data) < GRADIENT_TOLERANCE


# -------------    LatentStack    ----------------------------------
def test_latent_stack_backward_splits_by_width():
    stacker = LatentStack()
    left = np.arange(12, dtype=float).reshape(4, 3)
    right = np.arange(20, dtype=float).reshape(4, 5)
    stacker.forward(left, right)
    grad_left, grad_right = stacker.backward(np.hstack([left, right]))
    assert np.array_equal(grad_left, left)
    assert np.array_equal(grad_right, right)
