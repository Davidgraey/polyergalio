"""Spectre (by band radius), LatentSpectre and LatentAttention tab: build, count parameters, train on shared data, compare outputs."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from common import emit_table, flush_figures, run_panel
from NNet_spectre_latent_example import (
    TASKS,
    band_radii,
    build_models,
    compare,
    make_data,
    output_agreement,
    parameter_counts,
)


def plot_comparison(rows: list, curves: dict, outputs: dict, test_set, task: str) -> None:
    """Loss curves, parameters against test error, and every model's output for one test sequence."""
    names = [row["model"] for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(14, 20))
    for name, losses in curves.items():
        axes[0].plot(losses, label=name, linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Step")
    axes[0].set_title(f"Training loss ({task})")
    axes[0].legend(fontsize=6, ncol=2)

    for row in rows:
        axes[1].scatter(row["parameters"], row["test mse"], s=60)
        axes[1].annotate(row["model"], (row["parameters"], row["test mse"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    axes[1].set_xlabel("Parameters")
    axes[1].set_ylabel("Test MSE")
    axes[1].set_title("Size against held-out error")

    x, y = test_set
    axes[2].plot(y[0, :, 0], color="black", linewidth=2, label="target")
    for name in names:
        axes[2].plot(outputs[name][0, :, 0], alpha=0.8, label=name)
    axes[2].set_xlabel("Position")
    axes[2].set_title("Feature 0 of the first test sequence")
    axes[2].legend(fontsize=6, ncol=2)
    fig.tight_layout()


def run(sequence_length: int, hidden: int, heads: int, ranks: list, radii: list, use_wrm: bool, task: str, steps: int, learning_rate: float, samples: int, seed: int) -> None:
    """Build every layer, train them on the same task data, and compare."""
    train_set = make_data(task, samples, sequence_length, hidden, seed)
    test_set = make_data(task, samples, sequence_length, hidden, seed + 1)
    models = build_models(sequence_length, hidden, heads, tuple(ranks), tuple(radii), use_wrm)
    for name, layer in models.items():
        print(f"{name:22s} {layer.num_parameters:6d} parameters")

    rows, curves, outputs = compare(models, train_set, test_set, steps, learning_rate)
    for row in rows:
        print(f"{row['model']:22s} train {row['train mse']:.4f}  test {row['test mse']:.4f}")
    emit_table("Results", pd.DataFrame(rows))
    reference = "Spectre band=0" if "Spectre band=0" in outputs else next(iter(outputs))
    agreement = output_agreement(outputs, reference)
    emit_table(
        f"Test-output correlation with {reference}",
        pd.DataFrame({"model": list(agreement), "correlation": list(agreement.values())}),
    )
    plot_comparison(rows, curves, outputs, test_set, task)
    flush_figures()


def render() -> None:
    left, right = st.columns([1, 2])
    with left:
        st.write(
            "SpectreAttention gates each head's spectrum with a dense head and mixes values with a dense projection, "
            "optionally with a Toeplitz band over neighbouring frequencies. SpectreLatentAttention swaps the dense head "
            "and projection for LatentHead layers (no band). LatentAttention is position-free softmax attention over "
            "a low-rank latent, so it can only solve the mean task. All are trained on identical data."
        )
        st.subheader("Settings")
        sequence_length = int(st.number_input("Sequence length", 8, 128, 32, step=8, key="sl_length"))
        hidden = int(st.number_input("Hidden dim", 4, 128, 16, step=4, key="sl_hidden"))
        heads = int(st.number_input("Heads", 1, 16, 2, key="sl_heads"))
        task = st.selectbox("Task", TASKS, key="sl_task")
        radius_options = list(band_radii(sequence_length))
        radii = st.multiselect("Spectre band radii", radius_options, default=radius_options, key=f"sl_radii_{sequence_length}")
        ranks = st.multiselect("Latent ranks (LatentSpectre and LatentAttention)", [1, 2, 4, 8, 16, 32, 64], default=[4, 8, 16], key="sl_ranks")
        use_wrm = st.checkbox("Wavelet refinement module", value=False, key="sl_wrm")
        learning_rate = float(st.number_input("Learning rate", 0.0001, 0.1, 0.003, step=0.001, format="%.4f", key="sl_lr"))
        steps = int(st.number_input("Training steps", 10, 2000, 300, step=50, key="sl_steps"))
        samples = int(st.number_input("Samples", 16, 1024, 128, step=16, key="sl_samples"))
        seed = int(st.number_input("Data seed", 0, 9999, 0, key="sl_seed"))
    if not ranks or not radii or hidden % heads:
        right.warning("Choose at least one latent rank and band radius, and a hidden dim divisible by the head count.")
        return

    with right:
        st.subheader("Initialised layers")
        models = build_models(sequence_length, hidden, heads, tuple(ranks), tuple(radii), use_wrm)
        counts = parameter_counts(models)
        base = counts.get("Spectre band=0", next(iter(counts.values())))
        st.dataframe(
            pd.DataFrame(
                {
                    "model": list(counts),
                    "parameters": list(counts.values()),
                    "vs Spectre band=0": [f"{value / base:.2f}x" for value in counts.values()],
                }
            ),
            hide_index=True,
        )
        run_panel("sl", run, sequence_length, hidden, heads, ranks, radii, use_wrm, task, steps, learning_rate, samples, seed)
