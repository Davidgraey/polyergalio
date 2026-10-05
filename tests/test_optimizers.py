"""
Optimizers and the parameter-update contract.
"""

import inspect

import numpy as np
import pytest
from polyergalio.generators.data_generators import to_onehot
from polyergalio.models.constants import ClassificationTask
from polyergalio.models.layers.basic_layers import (
    FullyConnectedLayer,
    NormalizeLayer,
    RMSNormLayer,
)
from polyergalio.models.model_loss import CrossEntropyLoss
from polyergalio.models.network import Network as NeuralNetwork
from polyergalio.models.optimizers import SGD

PARAMETERISED_LAYERS = (
    lambda: FullyConnectedLayer(4, 3, "relu"),
    lambda: NormalizeLayer(4, shift_scale=True),
    lambda: RMSNormLayer(4),
)


@pytest.mark.parametrize("make_layer", PARAMETERISED_LAYERS)
def test_gradient_keys_match_update_signature(make_layer, small_matrix):
    """every key get_gradients emits must be a parameter update_weights takes"""
    layer = make_layer()
    output = layer.forward(small_matrix)
    layer.backward(np.ones_like(output))

    accepted = set(inspect.signature(layer.update_weights).parameters)
    emitted = set(layer.get_gradients())
    assert emitted <= accepted, f"{emitted - accepted} not accepted by update_weights"


def test_sgd_scales_by_the_learning_rate(small_matrix):
    """a step at rate r must move exactly r times the raw gradient"""
    layer = FullyConnectedLayer(4, 3, "linear")
    output = layer.forward(small_matrix)
    layer.backward(np.ones_like(output))

    gradient = layer.gradient_weights.copy()
    before = layer.weights.copy()
    SGD(0.05).step([layer])

    assert np.allclose(before - 0.05 * gradient, layer.weights)


def test_classifier_learns_a_separable_problem(multiclass_dataset):
    x_data, y_data, _ = multiclass_dataset
    targets = to_onehot(y_data, 4)

    layers = [
        FullyConnectedLayer(5, 24, "relu"),
        FullyConnectedLayer(24, 4, "linear", is_output=True),
    ]
    network = NeuralNetwork(layers)
    loss = CrossEntropyLoss(ClassificationTask.MULTINOMIAL)
    optimizer = SGD(5.0)

    for _ in range(400):
        prediction = network.forward(x_data)
        loss(prediction, targets)
        network.backward(loss.backward())
        optimizer.step(layers)

    accuracy = (prediction.argmax(-1) == y_data).mean()
    assert accuracy > 0.6, f"accuracy {accuracy:.3f} is near chance"
