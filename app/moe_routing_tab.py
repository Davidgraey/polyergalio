"""Mixture-of-experts routing tab: FullyConnectedLayer encoder, MixtureOfExperts, classifier head."""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from common import flush_figures, run_panel, show_diagram
from NNet_moe_routing_example import accuracy
from polyergalio.generators.data_generators import to_onehot
from polyergalio.models.constants import ClassificationTask
from polyergalio.models.layers.basal_layers import FullyConnectedLayer
from polyergalio.models.layers.mixture_layers import MixtureOfExperts
from polyergalio.models.model_loss import CrossEntropyLoss
from polyergalio.models.neural_network import NeuralNetwork
from polyergalio.models.optimizers import SGD
from polyergalio.visuals.nnet_visuals import plot_expert_routing, plot_network
from supervised_data import select_data


def build_network(num_features: int, num_classes: int, hidden: int, num_shared: int, num_routed: int, top_k: int, bias_speed: float) -> NeuralNetwork:
    """The example's encoder -> MixtureOfExperts -> classifier graph with adjustable sizes."""
    net = NeuralNetwork(name="moe_routing", input_shape=(num_features,))
    encoded = net.connect(FullyConnectedLayer(num_features, hidden, activation_type="relu"), net.input, name="encoder")
    routed = net.connect(
        MixtureOfExperts(
            input_dim=hidden,
            upscale_dim=2 * hidden,
            hidden_dim=hidden,
            num_shared_experts=num_shared,
            num_routed_experts=num_routed,
            top_k=top_k,
        ),
        encoded,
        name="moe",
    )
    net.output = net.connect(FullyConnectedLayer(hidden, num_classes, activation_type="linear", is_output=True), routed, name="classifier")
    return net


def train(data, hidden: int, num_shared: int, num_routed: int, top_k: int, bias_speed: float, learning_rate: float, steps: int, bias_log_every: int) -> None:
    """Train the network as the example does, then plot the loss curve and the routing diagnostics."""
    x, y = data.x, data.y
    targets = to_onehot(y)
    net = build_network(x.shape[1], data.num_classes, hidden, num_shared, num_routed, top_k, bias_speed)
    print(net.summary())
    moe = net.node("moe").layer

    net.eval()
    before = accuracy(net.forward(x), y)

    loss_fn = CrossEntropyLoss(ClassificationTask.MULTINOMIAL)
    optimizer = SGD(learning_rate)
    losses, bias_history = [], []
    net.train()
    for step in range(steps):
        net.zero_gradients()
        logits = net.forward(x)
        loss = loss_fn(logits, targets)
        net.backward(loss_fn.backward())
        optimizer.step(net.layers)
        losses.append(loss)
        if step % bias_log_every == 0:
            bias_history.append(moe.gate.expert_bias.copy())
        if step % 50 == 0:
            print(f"step {step:4d}  loss {loss:.4f}")

    net.eval()
    after = accuracy(net.forward(x), y)
    print(f"accuracy before training: {before:.3f}")
    print(f"accuracy after training:  {after:.3f}")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(losses)
    ax.set_title("Training loss")
    ax.set_xlabel("Step")
    fig.tight_layout()

    plot_expert_routing(
        moe.gate.mask,
        moe.gate_weights,
        labels=y,
        label_names=[f"class {c}" for c in range(data.num_classes)],
        bias_history=np.array(bias_history),
        top_k=top_k,
    )
    flush_figures()


def render() -> None:
    left, right = st.columns([1, 2])
    with left:
        st.write(
            "A classifier whose hidden layer is a MixtureOfExperts block: a shared expert every row "
            "passes through, plus a gate that routes each row to its top-k of several routed experts."
        )
        diagram = st.container()
        st.subheader("Settings")
        hidden = int(st.number_input("Hidden dim", 4, 256, 32, step=4, key="moe_hidden"))
        num_shared = int(st.number_input("Shared experts", 0, 8, 1, key="moe_shared"))
        num_routed = int(st.number_input("Routed experts", 2, 16, 6, key="moe_routed"))
        top_k = int(st.number_input("Top k", 2, 16, 2, key="moe_topk"))
        bias_speed = float(st.number_input("Load-balancing bias speed", 0.0, 0.1, 0.001, step=0.001, format="%.4f", key="moe_bias_speed"))
        learning_rate = float(st.number_input("Learning rate", 0.001, 1.0, 0.05, step=0.01, format="%.3f", key="moe_lr"))
        steps = int(st.number_input("Training steps", 1, 3000, 400, step=50, key="moe_steps"))
        bias_log_every = int(st.number_input("Log bias every N steps", 1, 200, 10, key="moe_bias_log"))
    if top_k > num_routed:
        right.warning("Top k cannot exceed the number of routed experts.")
        return

    with right:
        st.subheader("Data")
        data = select_data("moe", "multiclass", samples=1200, features=12, classes=6, settings=left)
        run_panel("moe", train, data, hidden, num_shared, num_routed, top_k, bias_speed, learning_rate, steps, bias_log_every)

    with diagram:
        with st.expander("Model structure", expanded=True):
            net = build_network(data.x.shape[1], data.num_classes, hidden, num_shared, num_routed, top_k, bias_speed)
            show_diagram(plot_network(net, figsize=(5, 6)).figure)
