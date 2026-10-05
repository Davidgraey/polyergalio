"""System-one decision network tab: TextEmbedding, SpectreAttention, DecisionHead."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from common import edit_table, emit_table, flush_figures, run_panel, show_diagram
from NNet_system_one_decision_example import (
    accuracy,
    evaluate,
    fit_calibration,
    print_evaluation,
    row_correctness,
    split_rows,
)
from polyergalio.generators.data_generators import RandomDatasetGenerator
from polyergalio.models.constants import DECISION_TYPES
from polyergalio.models.embedding.embedding import TextEmbedding
from polyergalio.models.layers.decision_layers import DecisionHead
from polyergalio.models.layers.spectre_layers import SpectreAttention
from polyergalio.models.model_loss import DecisionLoss
from polyergalio.models.network import Network
from polyergalio.models.optimizers import SGD
from polyergalio.visuals.nnet_visuals import plot_network

TYPE_NAMES = {member.value: member.name.lower() for member in DECISION_TYPES}


def build_network(vocab_size: int, sequence_length: int, padding_idx: int, hidden: int, head_hidden: int, heads: int) -> Network:
    """The example's TextEmbedding -> SpectreAttention -> DecisionHead graph with adjustable sizes."""
    net = Network(name="system_one", input_shape=(sequence_length,))
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


def train(x, y, meta, hidden: int, head_hidden: int, heads: int, learning_rate: float, steps: int, threshold: float) -> None:
    """Train on the train rows, calibrate on the calibration rows, then decode, score and escalate the test rows."""
    kwargs = dict(
        mask=meta["attention_mask"],
        marker_pos=meta["marker_pos"],
        marker_end=meta["marker_end"],
        token_mask=meta["token_mask"],
        decisiontypes=meta["decisiontypes"],
    )
    train_set, calibration_set, test_set = split_rows(x, y, kwargs)
    x_train, y_train, train_kwargs = train_set
    x_test, y_test, test_kwargs = test_set
    print(f"rows: {len(x_train)} train, {len(calibration_set[0])} calibration, {len(x_test)} test")

    net = build_network(meta["vocab_size"], x.shape[1], meta["pad_id"], hidden, head_hidden, heads)
    print(net.summary())
    net.eval()
    before = accuracy(net.forward(x_test, **test_kwargs), y_test, test_kwargs)
    print(f"test accuracy before training: {before:.3f}")

    loss_fn = DecisionLoss(ordinal_weight=0.25)
    optimizer = SGD(learning_rate)
    head = net.node("decision_head").layer
    losses, act_losses = [], []
    net.train()
    for step in range(steps):
        net.zero_gradients()
        logits = net.forward(x_train, **train_kwargs)
        loss = loss_fn(logits, y_train, train_kwargs["token_mask"], train_kwargs["decisiontypes"])
        act_loss = head.score_act(row_correctness(logits, y_train, train_kwargs))
        net.backward(loss_fn.backward())
        optimizer.step(net)
        losses.append(loss)
        act_losses.append(act_loss)
        if step % 50 == 0:
            print(f"step {step:4d}  loss {loss:.4f}  act loss {act_loss:.4f}")

    temperatures = fit_calibration(net, *calibration_set)
    print("\n--- test rows ---")
    results = evaluate(net, x_test, y_test, test_kwargs, temperatures, threshold)
    print_evaluation(results, temperatures)

    types = test_kwargs["decisiontypes"]
    emit_table(
        "Test rows by decision type",
        pd.DataFrame(
            [
                {
                    "type": TYPE_NAMES[value],
                    "rows": int((types == value).sum()),
                    "accuracy": float(results["correct"][types == value].mean()),
                    "mean confidence": float(results["confidence"][types == value].mean()),
                    "mean P(act)": float(results["act"][types == value].mean()),
                    "escalated": float(results["escalated"][types == value].mean()),
                }
                for value in np.unique(types)
            ]
        ),
    )
    emit_table("Temperatures by bucket", pd.DataFrame({"bucket": list(temperatures), "temperature": list(temperatures.values())}))

    after = float(results["correct"].mean())
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    axes[0].plot(losses, label="decision loss")
    axes[0].plot(act_losses, label="act loss")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Step")
    axes[0].legend()
    axes[1].bar(["before", "after"], [before, after], color=["gray", "steelblue"])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Test decision accuracy")
    bins = np.linspace(0, 1, 21)
    correct = results["correct"].astype(bool)
    axes[2].hist(results["act"][correct], bins=bins, alpha=0.7, label="correct")
    axes[2].hist(results["act"][~correct], bins=bins, alpha=0.7, label="wrong")
    axes[2].axvline(threshold, color="black", linestyle="--", label="threshold")
    axes[2].set_title("P(act) on test rows")
    axes[2].set_xlabel("P(act)")
    axes[2].legend()
    fig.tight_layout()
    flush_figures()


def render() -> None:
    left, right = st.columns([1, 2])
    with left:
        st.write(
            "One forward pass answers a decision question: token embeddings, a single Spectre mixing pass, "
            "then a shared scorer read off the [MARK] positions. There is no decoding loop. "
            "Rows split 60/20/20 into train, calibration and test: the head trains on the first, "
            "temperatures are fit on the second, and the third is decoded with calibrated probabilities, "
            "scored for confidence, and escalated by the act branch."
        )
        diagram = st.container()
        st.subheader("Settings")
        hidden = int(st.number_input("Hidden dim", 4, 256, 32, step=4, key="s1_hidden"))
        head_hidden = int(st.number_input("Head hidden", 4, 128, 16, step=4, key="s1_head"))
        heads = int(st.number_input("Heads", 1, 16, 4, key="s1_heads"))
        learning_rate = float(st.number_input("Learning rate", 0.001, 1.0, 0.05, step=0.01, format="%.3f", key="s1_lr"))
        steps = int(st.number_input("Training steps", 1, 2000, 200, step=50, key="s1_steps"))
        threshold = float(st.number_input("Escalation threshold", 0.0, 1.0, 0.5, step=0.05, key="s1_threshold"))
        st.subheader("Data settings")
        samples = int(st.number_input("Samples", 64, 2000, 768, step=64, key="s1_samples"))
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
        run_panel("s1", train, x, y, meta, hidden, head_hidden, heads, learning_rate, steps, threshold)

    with diagram:
        with st.expander("Model structure", expanded=True):
            net = build_network(meta["vocab_size"], x.shape[1], meta["pad_id"], hidden, head_hidden, heads)
            show_diagram(plot_network(net, figsize=(5, 6)).figure)
