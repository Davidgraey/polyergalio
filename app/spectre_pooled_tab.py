"""SpectrePooledAttention tab: classify from one summary vector against Spectre + mean pool and a CLS read, and profile the mixers."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from common import emit_table, flush_figures, run_panel
from NNet_spectre_pooled_example import (
    TASKS,
    accuracy_table,
    checkpoints_for,
    compare_models,
    models_for,
    prepare_task,
    profile_mixers,
)

SMOOTHING = 20


def smooth(values: np.ndarray, window: int = SMOOTHING) -> np.ndarray:
    window = min(window, len(values))
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_classification(task, accuracies: dict, losses: dict, checkpoints: tuple, task_name: str) -> None:
    """Training loss (moving average over seeds) and held-out accuracy at each checkpoint."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    for kind, curves in losses.items():
        axes[0].plot(smooth(np.mean(curves, axis=0)), label=task.kinds[kind], linewidth=1.2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel(f"Loss ({SMOOTHING}-step average)")
    axes[0].set_title(f"Training loss ({task_name})")
    axes[0].legend(fontsize=8)

    for kind, curves in accuracies.items():
        values = np.array([[curve[step] for step in checkpoints] for curve in curves])
        axes[1].plot(checkpoints, values.mean(axis=0), marker="o", label=task.kinds[kind])
        axes[1].fill_between(checkpoints, values.min(axis=0), values.max(axis=0), alpha=0.15)
    axes[1].axhline(0.5, color="grey", linestyle=":", linewidth=1)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Held-out accuracy")
    axes[1].set_title("Accuracy during training, mean and range over seeds")
    axes[1].legend(fontsize=8)
    fig.tight_layout()


def run_classification(
    task_name: str, kinds: list, length: int, hidden: int, heads: int, steps: int, learning_rate: float,
    batch_size: int, seeds: int, seed: int,
) -> None:
    """Train each chosen model on the task and compare held-out accuracy."""
    task = prepare_task(task_name, hidden=hidden, heads=heads, length=length, batch_size=batch_size, seed=seed)
    checkpoints = checkpoints_for(steps)
    parameters, accuracies, losses = compare_models(
        task, kinds, steps, learning_rate, tuple(range(seeds)), checkpoints
    )
    rows = accuracy_table(task, parameters, accuracies, checkpoints)
    for row in rows:
        print(f"{row['model']:26s} {row['parameters']:7d} parameters  final accuracy {row[f'step {checkpoints[-1]}']:.3f}")
    emit_table("Held-out accuracy by training step (mean over seeds)", pd.DataFrame(rows))
    plot_classification(task, accuracies, losses, checkpoints, task_name)
    flush_figures()


def plot_profile(rows: list[dict]) -> None:
    """Time, then memory, for each layer."""
    names = [row["layer"] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    positions = np.arange(len(names))
    width = 0.38
    for offset, key in ((-width / 2, "forward ms"), (width / 2, "backward ms")):
        axes[0].bar(positions + offset, [row[key] for row in rows], width, label=key)
    axes[0].set_ylabel("Milliseconds")
    axes[0].set_title("Time per call")
    memory_keys = ("forward peak MB", "kept MB", "backward peak MB")
    width = 0.26
    for index, key in enumerate(memory_keys):
        axes[1].bar(positions + (index - 1) * width, [row[key] for row in rows], width, label=key)
    axes[1].set_ylabel("Megabytes")
    axes[1].set_title("Memory per call")
    for axis in axes:
        axis.set_xticks(positions)
        axis.set_xticklabels(names)
        axis.legend(fontsize=8)
    fig.tight_layout()


def run_profile(length: int, hidden: int, heads: int, batch: int, memory_tokens: int) -> None:
    """Time and memory of the mixers on random input."""
    rows = profile_mixers(length, hidden, heads, batch, memory_tokens)
    frame = pd.DataFrame(rows)
    base = frame.iloc[0]
    for row in rows:
        print(
            f"{row['layer']:26s} fwd {row['forward ms']:7.1f} ms  bwd {row['backward ms']:7.1f} ms  "
            f"peak {row['forward peak MB']:7.1f} / {row['backward peak MB']:7.1f} MB"
        )
    emit_table("Time and memory per call", frame)
    emit_table(
        f"Relative to {base['layer']}",
        pd.DataFrame(
            {
                "layer": frame["layer"],
                "forward time": frame["forward ms"] / base["forward ms"],
                "backward time": frame["backward ms"] / base["backward ms"],
                "backward peak memory": frame["backward peak MB"] / base["backward peak MB"],
                "kept memory": frame["kept MB"] / base["kept MB"],
            }
        ),
    )
    plot_profile(rows)
    flush_figures()


def render() -> None:
    left, right = st.columns([1, 2])
    with left:
        st.write(
            "SpectrePooledAttention gates the sequence spectrum like SpectreAttention but returns one vector per "
            "sample: a masked weighted mean whose weights are a learned positional readout. Mean-pooling a Spectre output keeps only the gate's zero-frequency bin when "
            "nothing is padded, so it cannot see position; the fixed-length marker task shows this. The next-sentence "
            "task needs SentencePiece."
        )
        mode = st.radio("Mode", ["Classification", "Profile"], horizontal=True, key="spp_mode")

    if mode == "Classification":
        with left:
            st.subheader("Settings")
            task_name = st.selectbox("Task", TASKS, key="spp_task")
            options = models_for(task_name)
            kinds = st.multiselect(
                "Models", list(options), default=list(options), format_func=options.get, key=f"spp_kinds_{task_name}"
            )
            length = int(st.number_input("Sequence length", 16, 128, 32, step=8, key="spp_length"))
            hidden = int(st.number_input("Hidden dim", 4, 128, 32, step=4, key="spp_hidden"))
            heads = int(st.number_input("Heads", 1, 16, 4, key="spp_heads"))
            learning_rate = float(st.number_input("Learning rate", 0.0001, 0.1, 0.003, step=0.001, format="%.4f", key="spp_lr"))
            steps = int(st.number_input("Training steps", 50, 3000, 400, step=50, key="spp_steps"))
            batch_size = int(st.number_input("Batch size", 8, 128, 32, step=8, key="spp_batch"))
            seeds = int(st.number_input("Seeds per model", 1, 8, 2, key="spp_seeds"))
            seed = int(st.number_input("Data seed", 0, 9999, 0, key="spp_seed"))
        if not kinds or hidden % heads:
            right.warning("Choose at least one model, and a hidden dim divisible by the head count.")
            return
        with right:
            run_panel(
                "spp_classify", run_classification, task_name, kinds, length, hidden, heads, steps, learning_rate,
                batch_size, seeds, seed,
            )
        return

    with left:
        st.subheader("Settings")
        length = int(st.number_input("Sequence length", 16, 2048, 256, step=16, key="spp_plength"))
        hidden = int(st.number_input("Hidden dim", 8, 1024, 256, step=8, key="spp_phidden"))
        heads = int(st.number_input("Heads", 1, 32, 8, key="spp_pheads"))
        batch = int(st.number_input("Batch size", 1, 32, 2, key="spp_pbatch"))
        memory_tokens = int(st.number_input("Memory tokens", 0, 256, 0, step=16, key="spp_pmemory"))
    if hidden % heads:
        right.warning("Hidden dim must be divisible by the head count.")
        return
    with right:
        st.caption("Float32 and complex64. Time is the median of three untraced runs; memory comes from the Profiler.")
        run_panel("spp_profile", run_profile, length, hidden, heads, batch, memory_tokens)
