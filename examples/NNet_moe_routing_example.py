"""
Mixture-of-experts routing example.

Wires MixtureOfExperts into a small classifier -- FullyConnectedLayer
encoder -> MixtureOfExperts -> FullyConnectedLayer classifier -- trained on
RandomDatasetGenerator's "multiclass" task, then reports how routing load
balances across experts and, per class, which experts it favors. See
polyergalio.visuals.nnet_visuals.plot_expert_routing for the visual version
of these numbers (used by the Streamlit tab).

Run: python NNet_moe_routing_example.py
"""

import numpy as np
from polyergalio.generators.data_generators import RandomDatasetGenerator, to_onehot
from polyergalio.models.constants import ClassificationTask
from polyergalio.models.layers.basic_layers import FullyConnectedLayer
from polyergalio.models.layers.mixture_layers import MixtureOfExperts
from polyergalio.models.model_loss import CrossEntropyLoss
from polyergalio.models.network import Network
from polyergalio.models.optimizers import SGD

NUM_SAMPLES = 1200
NUM_FEATURES = 12
NUM_CLASSES = 6
HIDDEN_DIM = 32
NUM_SHARED_EXPERTS = 1
NUM_ROUTED_EXPERTS = 6
TOP_K = 2
LEARNING_RATE = 0.05
TRAIN_STEPS = 400
BIAS_LOG_EVERY = 10


def build_network(num_features: int, num_classes: int) -> Network:
    """
    FullyConnectedLayer encoder -> MixtureOfExperts -> FullyConnectedLayer classifier.

    Parameters
    ----------
    num_features : width of the raw feature vector
    num_classes : number of output classes

    Returns
    -------
    Network mapping features to class logits
    """
    net = Network(name="moe_routing", input_shape=(num_features,))
    encoded = net.connect(
        FullyConnectedLayer(num_features, HIDDEN_DIM, activation_type="relu"),
        net.input,
        name="encoder",
    )
    routed = net.connect(
        MixtureOfExperts(
            input_dim=HIDDEN_DIM,
            upscale_dim=2 * HIDDEN_DIM,
            hidden_dim=HIDDEN_DIM,
            num_shared_experts=NUM_SHARED_EXPERTS,
            num_routed_experts=NUM_ROUTED_EXPERTS,
            top_k=TOP_K,
        ),
        encoded,
        name="moe",
    )
    net.output = net.connect(
        FullyConnectedLayer(HIDDEN_DIM, num_classes, activation_type="linear", is_output=True),
        routed,
        name="classifier",
    )
    return net


def accuracy(logits, y) -> float:
    """Fraction of rows whose argmax logit matches y."""
    return float(np.mean(np.argmax(logits, axis=-1) == y))


def main():
    generator = RandomDatasetGenerator(random_seed=0)
    x, y, meta = generator.generate(
        "multiclass",
        num_samples=NUM_SAMPLES,
        num_features=NUM_FEATURES,
        num_classes=NUM_CLASSES,
        verbose=False,
    )
    targets = to_onehot(y)

    net = build_network(NUM_FEATURES, NUM_CLASSES)
    print(net.summary())
    moe = net.node("moe").layer

    net.eval()
    before = accuracy(net.forward(x), y)

    loss_fn = CrossEntropyLoss(ClassificationTask.MULTINOMIAL)
    optimizer = SGD(LEARNING_RATE)

    bias_history = []
    net.train()
    for step in range(TRAIN_STEPS):
        net.zero_gradients()
        logits = net.forward(x)
        loss = loss_fn(logits, targets)
        net.backward(loss_fn.backward())
        optimizer.step(net)
        if step % BIAS_LOG_EVERY == 0:
            bias_history.append(moe.gate.expert_bias.copy())
        if step % 50 == 0:
            print(f"step {step:4d}  loss {loss:.4f}")

    net.eval()
    logits = net.forward(x)
    after = accuracy(logits, y)
    print(f"accuracy before training: {before:.3f}")
    print(f"accuracy after training:  {after:.3f}")

    load = moe.gate.mask.mean(axis=0)
    fair_share = TOP_K / NUM_ROUTED_EXPERTS
    print(f"expert load (fair share {fair_share:.3f}):")
    for expert, fraction in enumerate(load):
        print(f"  expert {expert}: {fraction:.3f}")

    print("mean routing weight by class (rows = class, cols = expert):")
    for label in range(NUM_CLASSES):
        weights = moe.gate_weights[y == label].mean(axis=0)
        print(f"  class {label}: " + ", ".join(f"{w:.3f}" for w in weights))


if __name__ == "__main__":
    main()
