"""
The Network graph container (DAG): wiring checks, gradient flow through fan-out and merges, and end-to-end training.
"""

import inspect

import numpy as np
import pytest
from conftest import GRADIENT_TOLERANCE, numeric_gradient, relative_error
from polyergalio.models.layers.basic_layers import (
    DropoutLayer,
    FullyConnectedLayer,
    Layer,
)
from polyergalio.models.layers.operator_layers import LatentStack
from polyergalio.models.model_loss import MSELoss
from polyergalio.models.network import INPUT_NAME, Network as NeuralNetwork
from polyergalio.models.optimizers import SGD
from polyergalio.utilities import shape_conflict


def as_float64(layer):
    for name in ("weights", "bias"):
        value = getattr(layer, name, None)
        if isinstance(value, np.ndarray):
            setattr(layer, name, value.astype(np.float64))
    return layer


class FixedWidthMerge(Layer):
    """a two-source merge that pins both widths, so a positional conflict can be built"""

    def __init__(self, left: int, right: int):
        super().__init__()
        self.declare_shapes(inputs=((left,), (right,)), outputs=((left + right,),))

    def forward(self, left, right):
        return np.hstack([left, right])

    def backward(self, incoming_gradient):
        left = self.shapes["input"][0][-1]
        return incoming_gradient[..., :left], incoming_gradient[..., left:]

    def update_weights(self, **kwargs) -> None:
        pass

    def purge(self) -> None:
        pass

    def get_weights(self, for_serialize: bool = False):
        return {} if for_serialize else None

    def zero_gradients(self) -> None:
        pass


@pytest.fixture()
def branching_network():
    """input -> a, and a feeds both b and c, which merge into the output"""
    net = NeuralNetwork(name="branching")
    a = net.connect(as_float64(FullyConnectedLayer(4, 6, "tanh")), net.input, name="a")
    b = net.connect(as_float64(FullyConnectedLayer(6, 3, "swish")), a, name="b")
    c = net.connect(as_float64(FullyConnectedLayer(6, 5, "relu")), a, name="c")
    merged = net.connect(LatentStack(), b, c, name="merge")
    net.output = net.connect(
        as_float64(FullyConnectedLayer(8, 2, "linear")), merged, name="out"
    )
    return net


# -------------    wiring    ---------------------------------------
def test_a_width_mismatch_is_caught_at_the_wiring_line():
    net = NeuralNetwork()
    first = net.connect(FullyConnectedLayer(4, 6, "relu"), net.input)
    with pytest.raises(ValueError, match="cannot be fed by"):
        net.connect(FullyConnectedLayer(9, 3, "relu"), first)


def test_a_mismatch_after_a_merge_is_caught():
    """3 + 5 is 8, so a 7 wide head is wrong"""
    net = NeuralNetwork()
    left = net.connect(FullyConnectedLayer(4, 3, "linear"), net.input)
    right = net.connect(FullyConnectedLayer(4, 5, "linear"), net.input)
    merged = net.connect(LatentStack(), left, right)
    with pytest.raises(ValueError, match="cannot be fed by"):
        net.connect(FullyConnectedLayer(7, 2, "relu"), merged)


def test_transposed_merge_sources_are_refused():
    """the comparison is positional, not against any declared width"""
    net = NeuralNetwork()
    left = net.connect(FullyConnectedLayer(4, 5, "linear"), net.input)
    right = net.connect(FullyConnectedLayer(4, 3, "linear"), net.input)
    with pytest.raises(ValueError, match="position 0"):
        net.connect(FixedWidthMerge(3, 5), left, right)


def test_arity_is_checked_when_wiring():
    net = NeuralNetwork()
    a = net.connect(FullyConnectedLayer(4, 3, "linear"), net.input)
    b = net.connect(FullyConnectedLayer(4, 3, "linear"), net.input)
    c = net.connect(FullyConnectedLayer(4, 3, "linear"), net.input)
    with pytest.raises(ValueError, match="declares 2 input shapes"):
        net.connect(LatentStack(), a, b, c)


def test_shape_conflict_is_right_aligned_and_ignores_unknowns():
    assert shape_conflict((3, 8, 4), (4,)) is None
    assert shape_conflict((4,), (8, 4)) is None
    assert shape_conflict((None,), (9,)) is None
    assert "axis -1" in shape_conflict((8, 4), (5,))
    assert "axis -2" in shape_conflict((8, 4), (16, 4))


def test_validate_flags_a_node_that_feeds_nothing():
    net = NeuralNetwork()
    main = net.connect(FullyConnectedLayer(4, 3, "relu"), net.input, name="main")
    net.connect(FullyConnectedLayer(4, 8, "relu"), net.input, name="orphan")
    net.output = main
    assert any("orphan" in problem for problem in net.validate())


def test_prune_keeps_an_ancestor_shared_with_a_dead_branch():
    net = NeuralNetwork()
    shared = net.connect(FullyConnectedLayer(4, 6, "relu"), net.input, name="shared")
    live = net.connect(FullyConnectedLayer(6, 2, "linear"), shared, name="live")
    net.connect(FullyConnectedLayer(6, 8, "relu"), shared, name="dead")
    net.output = live

    assert net.prune() == ["dead"]
    assert [node.name for node in net.nodes] == [INPUT_NAME, "shared", "live"]
    assert [consumer.name for consumer in shared.consumers] == ["live"]


# -------------    every layer in a graph    -----------------------
def concrete_layers():
    """
    Every Layer subclass in the package that can be constructed with defaults.

    Walked rather than listed, so a new layer is covered the moment it exists
    instead of when somebody remembers to add it here.
    """
    import importlib
    import pkgutil

    import polyergalio.models as models

    for info in pkgutil.walk_packages(models.__path__, f"{models.__name__}."):
        try:
            importlib.import_module(info.name)
        except Exception:
            continue

    found = []
    stack = [Layer]
    while stack:
        for subclass in stack.pop().__subclasses__():
            stack.append(subclass)
            if inspect.isabstract(subclass):
                continue
            found.append(subclass)
    return found


LAYER_ARGUMENTS = {
    "FullyConnectedLayer": ((4, 6, "relu"), {}),
    "NormalizeLayer": ((6,), {}),
    "RMSNormLayer": ((6,), {}),
    "DropoutLayer": ((), {}),
    "FourierLayer": ((), {}),
    "InverseFourierLayer": ((), {}),
    "FrequencyFFT": ((8, 4), {}),
    "LatentStack": ((), {}),
    "FourierAttention": ((6, 6), {}),
    "SpectreAttention": ((8, 6), {}),
    # num_heads, ni, no
    "DenseHead": ((2, 3, 4), {"activation_type": "relu"}),
    # num_heads, head_dim
    "HeadProjection": ((2, 3), {}),
    # num_heads, head_dim, num_frequencies, gate_hidden
    "HeadGate": ((2, 3, 5, 4), {}),
    "RopeEmbedding": ((8, 6), {}),
    "SinusoidEmbedding": ((8, 6), {}),
    # VotingBase itself is excluded: self.stack is empty until a subclass
    # populates it, so its forward output width is input_shape, not the
    # num_experts its declared output shape claims. Its subclasses below
    # (which do populate the stack) exercise the real behavior.
    "VotingWeight": ((6, 4), {}),
    # VotingWeightBalanced and VotingGate carry a hidden width the plain
    # voters do not
    "VotingWeightBalanced": ((6, 5, 4), {"top_k": 2}),
    "VotingGate": ((6, 5, 4), {"top_k": 2}),
    # input_dim, hidden_dim, num_shared_experts, num_routed_experts, top_k
    "MixtureOfExperts": ((6, 8, 2, 5, 2), {}),
    "PoolingLayer": ((), {}),
    "LatentSum": ((), {}),
    "LatentProduct": ((), {}),
    "LatentDifference": ((), {}),
    "ShiftRight": ((6,), {}),
    # memory_tokens, hidden_dim
    "PersistentMemory": ((3, 6), {}),
    "SpectreDecoderAttention": ((8, 6), {}),
    # hidden_dim, sequence_length
    "WaveletRefinementModule": ((6, 8), {}),
    # hidden_dim, head_hidden
    "DecisionHead": ((6, 8), {}),
    # num_embeddings, embedding_dim
    "TextEmbedding": ((12, 6), {}),
}


# -------------    forward and backward shapes agree with reality    -----
# (x_data shape, number of sources). One layer reused twice as both sources
# is how a merge layer (LatentStack, LatentSum, ...) gets exercised, since
# connect() lets the same node feed a layer more than once.
NETWORK_RECIPES = {
    "FullyConnectedLayer": ((3, 8, 4), 1),
    "DropoutLayer": ((3, 8, 6), 1),
    "NormalizeLayer": ((3, 8, 6), 1),
    "RMSNormLayer": ((3, 8, 6), 1),
    "FrequencyFFT": ((5, 4), 1),
    "FourierLayer": ((3, 8, 6), 1),
    "InverseFourierLayer": ((3, 8, 6), 1),
    "FourierAttention": ((3, 8, 6), 1),
    "VotingWeight": ((3, 8, 6), 1),
    "VotingWeightBalanced": ((3, 8, 6), 1),
    "VotingGate": ((3, 8, 6), 1),
    "MixtureOfExperts": ((3, 8, 6), 1),
    "PoolingLayer": ((3, 8, 6), 1),
    "LatentStack": ((3, 8, 6), 2),
    "LatentSum": ((3, 8, 6), 2),
    "LatentProduct": ((3, 8, 6), 2),
    "LatentDifference": ((3, 8, 6), 2),
    "ShiftRight": ((3, 8, 6), 1),
    "SpectreAttention": ((3, 8, 6), 1),
    "SpectreDecoderAttention": ((3, 8, 6), 1),
    "DenseHead": ((3, 8, 6), 1),
    "HeadProjection": ((3, 8, 6), 1),
    "HeadGate": ((3, 6), 1),
    "RopeEmbedding": ((3, 8, 6), 1),
    "SinusoidEmbedding": ((3, 8, 6), 1),
    "WaveletRefinementModule": ((3, 8, 6), 2),
    # DecisionHead needs marker_pos per batch, see test_decision_layers
    # TextEmbedding needs integer ids, see test_embedding
    # PersistentMemory takes zero inputs -- connect() requires at least one
    # source, so it cannot be wired into a graph at all. See
    # test_persistent_memory_has_no_backward_shape_to_check below instead.
}


@pytest.mark.parametrize("layer_class", concrete_layers(), ids=lambda c: c.__name__)
def test_every_layer_agrees_with_its_own_forward_and_backward_shapes(layer_class):
    """
    Wire one layer into a real NeuralNetwork, run an actual forward and
    backward pass, and check both directions against reality rather than
    declared metadata:

    - forward's actual output shape against the shape the graph resolved
      for that node at connect() time (a layer can declare its shapes
      correctly and still emit something else)
    - backward's returned input gradient shape against the actual data
      that was fed to forward()
    """
    construction = LAYER_ARGUMENTS.get(layer_class.__name__)
    recipe = NETWORK_RECIPES.get(layer_class.__name__)
    if construction is None or recipe is None:
        pytest.skip(f"{layer_class.__name__} needs a construction and network recipe")

    arguments, keywords = construction
    x_shape, num_sources = recipe
    layer = layer_class(*arguments, **keywords)

    net = NeuralNetwork(input_shape=x_shape[1:])
    node = net.connect(layer, *(net.input for _ in range(num_sources)))
    net.output = node

    x_data = np.random.RandomState(0).normal(size=x_shape)
    output = net.forward(x_data)

    conflict = shape_conflict(output.shape, node.out_shape)
    assert conflict is None, (
        f"{layer_class.__name__}: forward produced {output.shape}, but the "
        f"graph resolved this node's output as {node.out_shape} from "
        f"declared shapes -- {conflict}"
    )

    grad_output = np.random.RandomState(1).normal(size=output.shape)
    grad_input = net.backward(grad_output)

    assert grad_input.shape == x_data.shape, (
        f"{layer_class.__name__}: backward returned {grad_input.shape}, "
        f"forward was fed {x_data.shape}"
    )


# -------------    the passes    -----------------------------------
def test_input_gradient_through_a_fan_out(branching_network):
    """node `a` receives gradient from both branches"""
    rng = np.random.default_rng(0)
    x_data = rng.normal(size=(5, 4))
    upstream = rng.normal(size=(5, 2))

    branching_network.forward(x_data)
    analytic = branching_network.backward(upstream.copy())
    numeric = numeric_gradient(
        lambda: float((branching_network.forward(x_data) * upstream).sum()), x_data
    )
    assert relative_error(analytic, numeric) < GRADIENT_TOLERANCE


@pytest.mark.parametrize("node_name", ("a", "b", "c", "out"))
def test_parameter_gradients_through_a_fan_out(node_name, branching_network):
    rng = np.random.default_rng(0)
    x_data = rng.normal(size=(5, 4))
    upstream = rng.normal(size=(5, 2))
    layer = branching_network.node(node_name).layer

    branching_network.forward(x_data)
    branching_network.backward(upstream.copy())
    analytic = layer.gradient_weights.copy()

    numeric = numeric_gradient(
        lambda: float((branching_network.forward(x_data) * upstream).sum()),
        layer.weights,
    )
    assert relative_error(analytic, numeric) < GRADIENT_TOLERANCE


# -------------    registration and modes    -----------------------
def test_assigned_layers_are_registered_once():
    class Model(NeuralNetwork):
        def __init__(self):
            super().__init__()
            self.first = FullyConnectedLayer(4, 8, "relu")
            self.second = FullyConnectedLayer(8, 2, "linear")
            self.output = self.connect(self.second, self.connect(self.first, self.input))

    assert len(Model().layers) == 2


def test_training_flag_reaches_dropout():
    net = NeuralNetwork([DropoutLayer(dropout_prob=0.5, use_rescale=False)])
    ones = np.ones((300, 120))

    net.train()
    training_mean = net.forward(ones).mean()
    net.eval()
    evaluation_mean = net.forward(ones).mean()

    assert training_mean == pytest.approx(0.5, abs=0.03)
    assert evaluation_mean == pytest.approx(1.0)


# -------------    training end to end    --------------------------
def test_branching_network_learns(regression_dataset):
    x_data, y_data, _ = regression_dataset
    y_data = y_data.reshape(-1, 1)

    net = NeuralNetwork(name="regression")
    features = net.input
    wide = net.connect(FullyConnectedLayer(3, 12, "relu"), features)
    narrow = net.connect(FullyConnectedLayer(3, 4, "tanh"), features)
    merged = net.connect(LatentStack(), wide, narrow)
    merged = net.connect(FullyConnectedLayer(16, 8, "swish"), merged)
    net.output = net.connect(
        FullyConnectedLayer(8, 1, "linear", is_output=True), merged
    )

    loss = MSELoss()
    optimizer = SGD(0.01)

    net.train()
    first = None
    for _ in range(250):
        prediction = net.forward(x_data)
        value = loss(prediction, y_data)
        if first is None:
            first = value
        net.backward(loss.backward())
        optimizer.step(net.layers)

    assert value < first * 0.9, f"expected real progress, got {first} -> {value}"
