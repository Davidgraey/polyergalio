"""System-one decision network tab: TextEmbedding, SpectreAttention, DecisionHead."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from common import edit_table, flush_figures, run_panel, show_diagram
from NNet_system_one_decision_example import accuracy, row_correctness
from polyergalio.generators.data_generators import RandomDatasetGenerator
from polyergalio.models.constants import DECISION_TYPES
from polyergalio.models.embedding.embedding import TextEmbedding
from polyergalio.models.layers.decision_layers import DecisionHead
from polyergalio.models.layers.spectre_layers import SpectreAttention
from polyergalio.models.model_loss import DecisionLoss
from polyergalio.models.neural_network import NeuralNetwork
from polyergalio.models.optimizers import SGD
from polyergalio.visuals.nnet_visuals import plot_network

TYPE_NAMES = {member.value: member.name.lower() for member in DECISION_TYPES}


def build_network(vocab_size: int, sequence_length: int, padding_idx: int, hidden: int, head_hidden: int, heads: int) -> NeuralNetwork:
    """The example's TextEmbedding -> SpectreAttention -> DecisionHead graph with adjustable sizes."""
    net = NeuralNetwork(name="system_one", input_shape=(sequence_length,))
    embedded = net.connect(TextEmbedding(vocab_size, hidden, padding_idx=padding_idx), net.input, name="embedding")
    attended = net.connect(SpectreAttention(sequence_length, hidden, num_heads=heads), embedded, name="attention")
    net.output = net.connect(DecisionHead(hidden, head_hidden), attended, name="decision_head")
    return net


def generate_frame(samples: int, choices: int, levels: int, types: list, seed: int):
    """Decision task as (table of token ids, answer and type, meta)."""
    x, y, meta = RandomDatasetGenerator(random_seed=seed).generate(
        "decision",
        num_samples=samples,
        num_choices=choices,
        num_levels=levels,
        decision_types=tuple(types),
        verbose=False,
    )
    frame = pd.DataFrame(x, columns=[f"t{i}" for i in range(x.shape[1])])
    frame["answer"] = y
    frame["type"] = [TYPE_NAMES[int(t)] for t in meta["decisiontypes"]]
    return frame, meta


def train(x, y, meta, hidden: int, head_hidden: int, heads: int, learning_rate: float, steps: int) -> None:
    """Train the network as the example does, printing progress and plotting the loss curve."""
    kwargs = dict(
        mask=meta["attention_mask"],
        marker_pos=meta["marker_pos"],
        marker_end=meta["marker_end"],
        token_mask=meta["token_mask"],
        decisiontypes=meta["decisiontypes"],
    )
    net = build_network(meta["vocab_size"], x.shape[1], meta["pad_id"], hidden, head_hidden, heads)
    print(net.summary())
    net.eval()
    before = accuracy(net.forward(x, **kwargs), y, kwargs)
    print(f"accuracy before training: {before:.3f}")

    loss_fn = DecisionLoss(ordinal_weight=0.25)
    optimizer = SGD(learning_rate)
    head = net.node("decision_head").layer
    losses = []
    net.train()
    for step in range(steps):
        net.zero_gradients()
        logits = net.forward(x, **kwargs)
        loss = loss_fn(logits, y, kwargs["token_mask"], kwargs["decisiontypes"])
        act_loss = head.score_act(row_correctness(logits, y, kwargs))
        net.backward(loss_fn.backward())
        optimizer.step(net.layers)
        losses.append(loss)
        if step % 50 == 0:
            print(f"step {step:4d}  loss {loss:.4f}  act loss {act_loss:.4f}")

    net.eval()
    after = accuracy(net.forward(x, **kwargs), y, kwargs)
    print(f"accuracy after training: {after:.3f}")
    print(f"escalated to system two: {head.escalate().mean():.3f} of rows")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(losses)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Step")
    axes[1].bar(["before", "after"], [before, after], color=["gray", "steelblue"])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Decision accuracy")
    fig.tight_layout()
    flush_figures()


def render() -> None:
    left, right = st.columns([1, 2])
    with left:
        st.write(
            "One forward pass answers a decision question: token embeddings, a single Spectre mixing pass, "
            "then a shared scorer read off the [MARK] positions. There is no decoding loop."
        )
        diagram = st.container()
        st.subheader("Settings")
        hidden = int(st.number_input("Hidden dim", 4, 256, 32, step=4, key="s1_hidden"))
        head_hidden = int(st.number_input("Head hidden", 4, 128, 16, step=4, key="s1_head"))
        heads = int(st.number_input("Heads", 1, 16, 4, key="s1_heads"))
        learning_rate = float(st.number_input("Learning rate", 0.001, 1.0, 0.05, step=0.01, format="%.3f", key="s1_lr"))
        steps = int(st.number_input("Training steps", 1, 2000, 200, step=50, key="s1_steps"))
        st.subheader("Data settings")
        samples = int(st.number_input("Samples", 16, 1000, 256, step=16, key="s1_samples"))
        choices = int(st.number_input("Choices", 2, 8, 4, key="s1_choices"))
        levels = int(st.number_input("Levels", 2, 8, 4, key="s1_levels"))
        seed = int(st.number_input("Seed", 0, 9999, 0, key="s1_seed"))
        types = st.multiselect(
            "Decision types", [m.name for m in DECISION_TYPES], default=[m.name for m in DECISION_TYPES], key="s1_types"
        )
    if not types or hidden % heads:
        right.warning("Choose at least one decision type, and a hidden dim divisible by the head count.")
        return
    chosen = [DECISION_TYPES[name] for name in types]

    with right:
        st.subheader("Data")
        frame, meta, _ = edit_table(
            "s1",
            (samples, choices, levels, tuple(types), seed),
            lambda: generate_frame(samples, choices, levels, chosen, seed),
            allow_upload=False,
            dynamic_rows=False,
            disabled=["type"],
        )
        st.caption("Token ids and answers can be edited; marker positions and masks stay as generated.")
        token_columns = [c for c in frame.columns if c[1:].isdigit()]
        x = frame[token_columns].to_numpy(dtype=int)
        y = frame["answer"].to_numpy(dtype=int)
        run_panel("s1", train, x, y, meta, hidden, head_hidden, heads, learning_rate, steps)

    with diagram:
        with st.expander("Model structure", expanded=True):
            net = build_network(meta["vocab_size"], x.shape[1], meta["pad_id"], hidden, head_hidden, heads)
            show_diagram(plot_network(net, figsize=(5, 6)).figure)
