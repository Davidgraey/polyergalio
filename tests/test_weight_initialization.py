"""
Weight initializers: each follows its variance rule, and the dispatcher maps activations to the
intended rule.
"""

import numpy as np
import pytest
from polyergalio.models.layers.basic_layers import FullyConnectedLayer, Layer
from polyergalio.models.weight_initialization import get_weight_init

NI = 512
NO = 256

VARIANCE_RULES = {
    "lecun": 1 / NI,
    "kaiming": 2 / NI,
    "kaiming_uniform": 2 / NI,
    "glorot": 2 / (NI + NO),
    "glorot_uniform": 2 / (NI + NO),
    "siren": 2 / NI,
}


@pytest.mark.parametrize("name", VARIANCE_RULES)
def test_variance_follows_rule(name):
    weights = get_weight_init(name)(np.random.RandomState(0), ni=NI, no=NO)
    assert weights.shape == (NI, NO)
    assert weights.var() == pytest.approx(VARIANCE_RULES[name], rel=0.1)


def test_orthogonal_is_orthonormal_in_both_shapes():
    rng = np.random.RandomState(0)
    tall = get_weight_init("orthogonal")(rng, ni=8, no=4)
    wide = get_weight_init("orthogonal")(rng, ni=4, no=8)
    assert np.allclose(tall.T @ tall, np.eye(4))
    assert np.allclose(wide @ wide.T, np.eye(4))


ACTIVATION_RULES = {
    "linear": "lecun",
    "relu": "kaiming",
    "swish": "kaiming",
    "sigmoid": "glorot",
    "tanh": "glorot",
    "softmax": "glorot",
    "selu": "lecun",
}


@pytest.mark.parametrize("activation,rule", ACTIVATION_RULES.items())
def test_activation_maps_to_rule(activation, rule):
    drawn = get_weight_init(activation)(np.random.RandomState(3), ni=16, no=8)
    expected = get_weight_init(rule)(np.random.RandomState(3), ni=16, no=8)
    assert np.array_equal(drawn, expected)


def test_fully_connected_override_and_round_trip():
    layer = FullyConnectedLayer(4, 3, "relu", initialization_override="orthogonal")
    rebuilt = Layer.deserialize(layer.serialize())
    assert rebuilt.initialization_override == "orthogonal"
    assert np.allclose(rebuilt.weights, layer.weights)

    scaled = FullyConnectedLayer(
        200, 100, "relu", initialization_override="truncated_normal", initialization_kwargs={"std": 0.01}
    )
    assert scaled.weights.std() < 0.011
