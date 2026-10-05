"""
Serialize round trips: construct, train, serialize, deserialize, and check the
rebuilt object is the trained one -- for every Layer type and for whole graphs.

Predictions run in eval() mode on both models, so the comparison never depends
on dropout or other training-only updates.
"""

import pickle

import numpy as np
import pytest
from polyergalio.models.constants import DECISION_TYPES
from polyergalio.models.layers.basic_layers import FullyConnectedLayer, Layer, NormalizeLayer
from polyergalio.models.layers.decision_layers import DecisionHead
from polyergalio.models.layers.mixture_layers import MixtureOfExperts
from polyergalio.models.layers.operator_layers import LatentStack
from polyergalio.models.model_loss import MSELoss
from polyergalio.models.network import Network as NeuralNetwork
from polyergalio.models.optimizers import SGD
from test_network import concrete_layers

TRAIN_STEPS = 5
LEARNING_RATE = 0.01
TOLERANCE = 1e-10


def decision_inputs(rng):
    marker_pos = np.array([[2, 3, 4, 5], [2, 4, 0, 0], [3, 5, 7, 0]])
    kwargs = dict(
        marker_pos=marker_pos,
        token_mask=marker_pos > 0,
        decisiontypes=[DECISION_TYPES.CHOICE, DECISION_TYPES.BINARY, DECISION_TYPES.SCORE],
    )
    return (rng.normal(size=(3, 8, 6)),), kwargs


def token_inputs(rng):
    return (rng.integers(0, 12, size=(3, 8)),), {}


def sequence_inputs(count: int, shape: tuple = (3, 8, 6)):
    return lambda rng: (tuple(rng.normal(size=shape) for _ in range(count)), {})


# layer name -> (constructor args, constructor kwargs, input builder)
# the input builder takes an rng and returns (positional inputs, forward kwargs)
RECIPES = {
    "FullyConnectedLayer": ((4, 6, "relu"), {}, sequence_inputs(1, (3, 8, 4))),
    "DropoutLayer": ((), {"dropout_prob": 0.3, "use_rescale": True}, sequence_inputs(1)),
    "NormalizeLayer": ((6,), {}, sequence_inputs(1)),
    "RMSNormLayer": ((6,), {}, sequence_inputs(1)),
    "LatentStack": ((), {}, sequence_inputs(2)),
    "LatentSum": ((), {}, sequence_inputs(2)),
    "LatentProduct": ((), {}, sequence_inputs(2)),
    "LatentDifference": ((), {}, sequence_inputs(2)),
    "ShiftRight": ((6,), {}, sequence_inputs(1)),
    "WaveletRefinementModule": ((6, 8), {}, sequence_inputs(2)),
    "PersistentMemory": ((3, 6), {}, lambda rng: ((), {})),
    "DenseHead": ((2, 3, 4), {"activation_type": "relu"}, sequence_inputs(1)),
    "HeadProjection": ((2, 3), {}, sequence_inputs(1)),
    "HeadGate": ((2, 3, 5, 4), {"band_radius": 1}, sequence_inputs(1, (3, 6))),
    "SpectreAttention": ((8, 6), {}, sequence_inputs(1)),
    "SpectreDecoderAttention": ((8, 6), {}, sequence_inputs(1)),
    "ShortConvolution": ((6,), {"kernel_size": 3}, sequence_inputs(1)),
    "HyenaFilter": ((8, 6), {"order": 2, "filter_features": 5}, lambda rng: ((), {})),
    "HyenaOperator": ((8, 6), {"filter_features": 5}, sequence_inputs(1)),
    "DecisionHead": ((6, 8), {}, decision_inputs),
    "TextEmbedding": ((12, 6), {"padding_idx": 0}, token_inputs),
    "RopeEmbedding": ((8, 6), {}, sequence_inputs(1)),
    "SinusoidEmbedding": ((8, 6), {}, sequence_inputs(1)),
    "FrequencyFFT": ((8, 4), {}, sequence_inputs(1, (5, 4))),
    "FourierLayer": ((), {}, sequence_inputs(1)),
    "InverseFourierLayer": ((), {}, sequence_inputs(1)),
    "FourierAttention": ((6, 6), {}, sequence_inputs(1)),
    "MixtureOfExperts": ((6, 8, 2, 5, 2), {}, sequence_inputs(1)),
    "Expert": ((6, 4), {}, sequence_inputs(1)),
    "MaskGather": ((), {}, sequence_inputs(1)),
    "PoolingLayer": ((), {}, sequence_inputs(1)),
    "VotingWeight": ((6, 4), {}, sequence_inputs(1)),
    "VotingWeightBalanced": ((6, 5, 4), {"top_k": 2}, sequence_inputs(1)),
    "VotingGate": ((6, 5, 4), {"top_k": 2}, sequence_inputs(1)),
}

# VotingBase has an empty expert stack until a subclass fills it
NOT_ROUND_TRIPPED = {"VotingBase"}


def package_layers() -> list[type]:
    return [
        layer_class
        for layer_class in concrete_layers()
        if layer_class.__module__.startswith("polyergalio.")
        and layer_class.__name__ not in NOT_ROUND_TRIPPED
    ]


def predict(model, inputs: tuple, kwargs: dict) -> np.ndarray:
    """Inference output: switch the layer or network to eval(), then run forward."""
    return model.eval().forward(*inputs, **kwargs)


def train(layer: Layer, inputs: tuple, kwargs: dict, target: np.ndarray) -> None:
    """A few SGD steps in training mode, pulling the layer's output towards target."""
    optimizer = SGD(LEARNING_RATE)
    layer.train()
    for _ in range(TRAIN_STEPS):
        layer.zero_gradients()
        output = layer.forward(*inputs, **kwargs)
        layer.backward(output - target)
        optimizer.step([layer])


def flatten(value, prefix: str = "") -> dict[str, np.ndarray]:
    """Nested weights as {path: array}, so two layers' weights can be compared leaf by leaf."""
    if isinstance(value, dict):
        leaves = {}
        for key, sub in value.items():
            leaves.update(flatten(sub, f"{prefix}/{key}"))
        return leaves
    if isinstance(value, (list, tuple)):
        leaves = {}
        for index, sub in enumerate(value):
            leaves.update(flatten(sub, f"{prefix}/{index}"))
        return leaves
    return {prefix: None if value is None else np.array(value, copy=True)}


def assert_same_leaves(expected: dict, actual: dict) -> None:
    assert expected.keys() == actual.keys()
    for path, value in expected.items():
        if value is None:
            assert actual[path] is None, path
        else:
            assert np.allclose(actual[path], value, atol=TOLERANCE), path


@pytest.fixture
def rng():
    return np.random.default_rng(2026)


def build(layer_class: type, rng) -> tuple:
    arguments, keywords, make_inputs = RECIPES[layer_class.__name__]
    layer = layer_class(*arguments, **keywords)
    inputs, kwargs = make_inputs(rng)
    target = rng.normal(size=np.shape(predict(layer, inputs, kwargs)))
    return layer, inputs, kwargs, target


def round_trip(layer: Layer) -> Layer:
    """serialize -> pickle -> deserialize, the same path NeuralNetwork.save/load take."""
    return Layer.deserialize(pickle.loads(pickle.dumps(layer.serialize())))


# -------------    layers    ---------------------------------------
def test_every_package_layer_has_a_recipe():
    missing = sorted(c.__name__ for c in package_layers() if c.__name__ not in RECIPES)
    assert not missing, f"add a serialization recipe for: {missing}"


@pytest.mark.parametrize("layer_class", package_layers(), ids=lambda c: c.__name__)
def test_round_trip_restores_type_config_and_weights(layer_class, rng):
    layer, inputs, kwargs, target = build(layer_class, rng)
    train(layer, inputs, kwargs, target)
    rebuilt = round_trip(layer)

    assert type(rebuilt) is layer_class
    assert rebuilt.get_config() == layer.get_config()
    assert rebuilt.num_parameters == layer.num_parameters
    assert_same_leaves(
        flatten(layer.get_weights(for_serialize=True)),
        flatten(rebuilt.get_weights(for_serialize=True)),
    )


@pytest.mark.parametrize("layer_class", package_layers(), ids=lambda c: c.__name__)
def test_round_trip_reproduces_inference(layer_class, rng):
    layer, inputs, kwargs, target = build(layer_class, rng)
    train(layer, inputs, kwargs, target)
    rebuilt = round_trip(layer)

    expected = predict(layer, inputs, kwargs)
    assert np.allclose(predict(rebuilt, inputs, kwargs), expected, atol=TOLERANCE)


# -------------    networks    ---------------------------
def branching_network() -> NeuralNetwork:
    """input -> a, a fans out to b and c, which merge into the output"""
    net = NeuralNetwork(name="branching", input_shape=(4,))
    a = net.connect(FullyConnectedLayer(4, 6, "tanh"), net.input, name="a")
    b = net.connect(FullyConnectedLayer(6, 3, "swish"), a, name="b")
    c = net.connect(FullyConnectedLayer(6, 5, "relu"), a, name="c")
    merged = net.connect(LatentStack(), b, c, name="merge")
    normed = net.connect(NormalizeLayer(8), merged, name="norm")
    net.output = net.connect(FullyConnectedLayer(8, 2, "linear"), normed, name="out")
    return net


def mixture_network() -> NeuralNetwork:
    net = NeuralNetwork(name="mixture", input_shape=(None, 6))
    experts = net.connect(MixtureOfExperts(6, 8, 2, 5, 2), net.input, name="experts")
    net.output = net.connect(FullyConnectedLayer(6, 3, "linear"), experts, name="out")
    return net


def decision_network() -> NeuralNetwork:
    """a head fed by forward kwargs, which the rebuilt graph has to route again"""
    net = NeuralNetwork(name="decision", input_shape=(None, 6))
    hidden = net.connect(FullyConnectedLayer(6, 6, "tanh"), net.input, name="encoder")
    net.output = net.connect(DecisionHead(6, 8), hidden, name="head")
    return net


def network_inputs(builder, rng) -> tuple:
    if builder is branching_network:
        return rng.normal(size=(12, 4)), {}
    (x_data,), kwargs = decision_inputs(rng)
    return x_data, kwargs if builder is decision_network else {}


NETWORK_BUILDERS = (branching_network, mixture_network, decision_network)


def train_network(net: NeuralNetwork, x_data, kwargs: dict, target, steps: int = TRAIN_STEPS) -> None:
    loss = MSELoss()
    optimizer = SGD(LEARNING_RATE)
    net.train()
    for _ in range(steps):
        net.zero_gradients()
        loss(net.forward(x_data, **kwargs), target)
        net.backward(loss.backward())
        optimizer.step(net.layers)


def build_network(builder, rng) -> tuple:
    net = builder()
    x_data, kwargs = network_inputs(builder, rng)
    target = rng.normal(size=predict(net, (x_data,), kwargs).shape)
    return net, x_data, kwargs, target


def network_round_trip(net: NeuralNetwork) -> NeuralNetwork:
    return NeuralNetwork.deserialize(pickle.loads(pickle.dumps(net.serialize())))


def test_a_fan_out_stays_one_shared_node(rng):
    rebuilt = network_round_trip(branching_network())
    shared = rebuilt.node("a")
    consumers = [node for node in rebuilt.nodes if any(source is shared for source in node.sources)]
    assert [node.name for node in consumers] == ["b", "c"]


@pytest.mark.parametrize("builder", NETWORK_BUILDERS, ids=lambda b: b.__name__)
def test_network_round_trip_reproduces_both_passes(builder, rng):
    net, x_data, kwargs, target = build_network(builder, rng)
    train_network(net, x_data, kwargs, target)
    rebuilt = network_round_trip(net)

    expected = predict(net, (x_data,), kwargs)
    assert np.allclose(predict(rebuilt, (x_data,), kwargs), expected, atol=TOLERANCE)
    upstream = rng.normal(size=expected.shape)
    assert np.allclose(rebuilt.backward(upstream), net.backward(upstream), atol=TOLERANCE)


def test_save_and_load_through_a_file(rng, tmp_path):
    net, x_data, kwargs, target = build_network(branching_network, rng)
    train_network(net, x_data, kwargs, target)
    path = tmp_path / "branching.pkl"
    net.serialize(str(path))
    loaded = NeuralNetwork.deserialize(str(path))
    assert np.allclose(predict(loaded, (x_data,), kwargs), predict(net, (x_data,), kwargs), atol=TOLERANCE)
