"""
Shared fixtures and numerical helpers.

Every dataset comes from RandomDatasetGenerator with a fixed seed, so tests
work from reproducible data rather than hand-written examples, and a failure
can always be reproduced by constructing the generator with the same seed.

For layers, nnets and blocks:
The finite-difference helpers here are the backbone of the suite: a layer's
backward pass is only correct if it agrees with the numerical derivative of its
own forward pass, and that is what most of these tests assert.
"""

import numpy as np
import pytest

from polyergalio.generators import RandomDatasetGenerator


SEED = 42

# a separate stream for upstream gradients.
UPSTREAM_SEED = 1009

# Anything above this tolerance a significant disagreement rather than floating point noise
GRADIENT_TOLERANCE = 1e-6
STEP = 1e-6


# -------------    numerical helpers    ----------------------------
def numeric_gradient(scalar_function, tensor, step: float = STEP):
    """
    Central-difference gradient of scalar_function() with respect to tensor.

    tensor is perturbed in place, so it must be the very array the function
    under test reads -- a copy would not be seen.
    """
    gradient = np.zeros_like(tensor, dtype=np.float64)

    for index in np.ndindex(tensor.shape):
        original = tensor[index]
        tensor[index] = original + step
        plus = scalar_function()
        tensor[index] = original - step
        minus = scalar_function()
        tensor[index] = original
        gradient[index] = (plus - minus) / (2 * step)

    return gradient


def numeric_gradient_complex(scalar_function, tensor, step: float = STEP):
    """
    As numeric_gradient, for a complex parameter.

    Returns d/dReal + 1j * d/dImag, which is the convention the complex layers
    use so that `parameter -= rate * gradient` descends on both parts.
    """
    gradient = np.zeros_like(tensor)

    for index in np.ndindex(tensor.shape):
        original = tensor[index]

        tensor[index] = original + step
        real_plus = scalar_function()
        tensor[index] = original - step
        real_minus = scalar_function()

        tensor[index] = original + 1j * step
        imag_plus = scalar_function()
        tensor[index] = original - 1j * step
        imag_minus = scalar_function()

        tensor[index] = original
        gradient[index] = (real_plus - real_minus) / (2 * step) + 1j * (
            imag_plus - imag_minus
        ) / (2 * step)

    return gradient


def relative_error(analytic, numeric) -> float:
    """
    Scale-free disagreement, floored so it degrades to an absolute check when
    both gradients are near zero.

    Without the floor a genuinely correct gradient can look wrong: normalisation
    layers annihilate part of their input space, so an upstream vector lying in
    that null space produces a true gradient of almost nothing, and dividing by
    it inflates the ratio.
    """
    analytic = np.asarray(analytic)
    numeric = np.asarray(numeric)
    if analytic.shape != numeric.shape:
        raise AssertionError(f"shape {analytic.shape} vs numeric {numeric.shape}")

    scale = max(np.abs(numeric).max(), np.abs(analytic).max(), 1.0)
    return float(np.abs(analytic - numeric).max() / scale)


def input_gradient_error(layer, x_data, upstream=None) -> float:
    """
    Compare layer.backward() against the numerical derivative of
    sum(upstream * layer.forward(x)) with respect to x.
    """
    x_data = np.array(x_data, dtype=np.float64)
    output = layer.forward(x_data)
    if upstream is None:
        upstream = np.random.default_rng(UPSTREAM_SEED).normal(size=output.shape)

    analytic = layer.backward(upstream.copy())
    numeric = numeric_gradient(
        lambda: float((layer.forward(x_data) * upstream).sum()), x_data
    )
    return relative_error(analytic, numeric)


def parameter_gradient_error(layer, x_data, parameter, gradient_name) -> float:
    """
    Compare a layer's stored parameter gradient against finite differences of
    the same scalar the backward pass was handed.
    """
    x_data = np.array(x_data, dtype=np.float64)
    output = layer.forward(x_data)
    upstream = np.random.default_rng(UPSTREAM_SEED).normal(size=output.shape)

    layer.backward(upstream.copy())
    analytic = np.array(getattr(layer, gradient_name), dtype=np.float64)

    numeric = numeric_gradient(
        lambda: float((layer.forward(x_data) * upstream).sum()), parameter
    )
    return relative_error(analytic, numeric)


def as_float64(layer):
    """
    Promote a layer's parameters to float64.

    Layers initialise in GLOBAL_DTYPE (float32), where central differences have
    too little precision to distinguish a real error from rounding.
    """
    for name in ("weights", "bias", "scale_gamma", "shift_beta", "activation_bias"):
        value = getattr(layer, name, None)
        if isinstance(value, np.ndarray):
            setattr(layer, name, value.astype(np.float64))
    return layer


@pytest.fixture()
def generator():
    return RandomDatasetGenerator(random_seed=SEED)


@pytest.fixture()
def regression_dataset(generator):
    return generator.generate(
        task="regression", num_samples=1500, num_features=3, noise_scale=1.5,
        verbose=False,
    )


@pytest.fixture()
def binary_dataset(generator):
    return generator.generate(
        task="binary", num_samples=400, num_features=5, verbose=False
    )


@pytest.fixture()
def multiclass_dataset(generator):
    return generator.generate(
        task="multiclass", num_samples=400, num_features=5, num_classes=4,
        verbose=False,
    )


@pytest.fixture()
def multilabel_dataset(generator):
    return generator.generate(
        task="multilabel", num_samples=300, num_features=4, num_classes=4,
        verbose=False,
    )


@pytest.fixture()
def clustering_dataset(generator):
    return generator.generate(
        task="clustering", num_samples=600, num_features=2, num_clusters=4,
        noise_scale=0.4, verbose=False,
    )


@pytest.fixture()
def calibration_data(generator):
    """
    Uncalibrated scores that correlate with a binary label: the label comes
    from the generator (seeded, like every other fixture here), then scores
    are scaled toward it with independent noise added, so a calibration
    model has real structure to recover without the scores already being
    calibrated probabilities.
    """
    _, labels, _ = generator.generate(task="binary", num_samples=600, verbose=False)
    labels = labels.reshape(-1, 1)
    rng = np.random.default_rng(SEED)
    scores = labels * rng.uniform(0.2, 0.9999, size=labels.shape)
    scores = scores + rng.uniform(-0.2, 0.2, size=labels.shape)
    return scores, labels


@pytest.fixture()
def sequence_batch():
    """(batch, sequence, hidden) float64, the shape the attention blocks take"""
    rng = np.random.default_rng(SEED)
    return rng.normal(size=(3, 8, 4))


@pytest.fixture()
def small_matrix():
    """a modest 2D array for layers that only need something well conditioned"""
    rng = np.random.default_rng(SEED)
    return rng.normal(size=(6, 4))
