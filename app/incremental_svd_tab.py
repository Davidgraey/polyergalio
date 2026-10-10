"""Incremental SVD tab: streaming eigenvectors, output forms, forgetting, and the shape of every array involved."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from common import edit_table, emit_table, flush_figures, run_panel, show_diagram
from incremental_svd_example import layer_shapes, make_data, make_drift_data, one_shot, print_shapes
from polyergalio.models.layers import FullyConnectedLayer, StandardizeLayer
from polyergalio.models.network import Network
from polyergalio.transforms.decompositions import IncrementalSVDLayer
from polyergalio.utilities import standardize_data
from polyergalio.visuals.nnet_visuals import plot_network

FORMS = ("orthogonalized", "whitened", "projection")
SCENARIOS = ("stationary", "covariance shift halfway")
BLOCK_COLORS = {"data": "#9ecae1", "factor": "#a1d99b", "core": "#fdd0a2", "empty": "#e0e0e0", "out": "#bcbddc"}


def build_network(features: int, output_dimension: int, form: str, num_components: int, decay_rate: float) -> Network:
    """StandardizeLayer, IncrementalSVDLayer, then a linear head, as the example wires them."""
    net = Network(name="incremental_svd", input_shape=(features,))
    node = net.connect(StandardizeLayer(features), net.input, name="normalize")
    layer = IncrementalSVDLayer(
        features, num_components=num_components, output_dimension=output_dimension, output_form=form, decay_rate=decay_rate
    )
    node = net.connect(layer, node, name="svd")
    net.output = net.connect(
        FullyConnectedLayer(output_dimension, 1, activation_type="linear", is_output=True), node, name="head"
    )
    return net


def generate_frame(samples: int, features: int, scenario: str, seed: int):
    """Correlated features, stationary or with a covariance change halfway, as (table, meta)."""
    rng = np.random.default_rng(seed)
    if scenario == SCENARIOS[0]:
        x = make_data(rng, samples, features)
    else:
        x = make_drift_data(rng, samples, features)[0]
    return pd.DataFrame(x, columns=[f"f{i}" for i in range(features)]), {"scenario": scenario}


def block(ax, x: float, y: float, rows: int, cols: int, text: str, kind: str) -> tuple[float, float]:
    """A rectangle with its top left corner at (x, y), sides scaled by the square root of the dimensions."""
    width, height = np.sqrt(max(cols, 3)), np.sqrt(max(rows, 3))
    ax.add_patch(plt.Rectangle((x, y - height), width, height, facecolor=BLOCK_COLORS[kind], edgecolor="black", linewidth=0.8))
    ax.text(x + width / 2, y - height / 2, f"{text}\n{rows} x {cols}", ha="center", va="center", fontsize=8)
    return width, height


def frame_axes(ax, title: str, width: float, height: float) -> None:
    ax.set_xlim(-1, width + 1)
    ax.set_ylim(-height - 1, 1.5)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def plot_shapes(layer: IncrementalSVDLayer, batch_rows: int) -> None:
    """Block diagrams, sides scaled by the square root of each dimension: the merge core, the forward pass, the factors held."""
    features, outputs = layer.input_dimension, layer.output_dimension
    rank = layer.singular_values.size
    width = min(features, batch_rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    left, top = block(axes[0], 0, 0, rank, rank, "diag(decay S)", "factor")
    right = block(axes[0], left, 0, rank, batch_rows, "V'x", "data")[0]
    bottom = block(axes[0], 0, -top, width, rank, "0", "empty")[1]
    block(axes[0], left, -top, width, batch_rows, "R (QR of residual)", "core")
    frame_axes(axes[0], "Merge core: SVD of this matrix folds in a batch", left + right, top + bottom)

    x_width, x_height = block(axes[1], 0, 0, batch_rows, features, "x", "data")
    axes[1].text(x_width + 0.6, -x_height / 2, "@", fontsize=16, ha="center", va="center")
    t_width, t_height = block(axes[1], x_width + 1.2, 0, features, outputs, "transform", "factor")
    axes[1].text(x_width + t_width + 1.8, -x_height / 2, "=", fontsize=16, ha="center", va="center")
    z_width = block(axes[1], x_width + t_width + 2.4, 0, batch_rows, outputs, "z", "out")[0]
    frame_axes(axes[1], "Forward pass", x_width + t_width + z_width + 2.4, max(x_height, t_height))

    v_width, v_height = block(axes[2], 0, 0, features, rank, "V", "factor")
    s_width = block(axes[2], v_width + 0.8, 0, rank, 1, "S", "factor")[0]
    m_width = block(axes[2], v_width + s_width + 1.6, 0, outputs, 1, "means", "out")[0]
    block(axes[2], v_width + s_width + m_width + 2.4, 0, outputs, 1, "stds", "out")
    frame_axes(axes[2], "Held between batches", v_width + s_width + 2 * m_width + 3.2, v_height)

    fig.tight_layout()


def stream_history(layer: IncrementalSVDLayer, x: np.ndarray, first: int, size: int, vectors: np.ndarray) -> pd.DataFrame:
    """Feed x in training mode and record, after each batch, the eigenvector change, rank and alignment with the one-shot vectors."""
    features = layer.input_dimension
    history = []
    for index, start in enumerate([0] + list(range(first, len(x), size))):
        stop = first if index == 0 else min(start + size, len(x))
        layer.forward(x[start:stop])
        held = layer.eigenvectors.shape[1]
        alignment = np.full(features, np.nan)
        alignment[:held] = np.abs(np.sum(layer.eigenvectors * vectors[:, :held], axis=0))
        history.append(
            {"batch": index + 1, "rows": stop, "change": layer.eigenvector_change, "rank": held, **{f"cos{i}": a for i, a in enumerate(alignment)}}
        )
    return pd.DataFrame(history)


def run(x, recent, first: int, size: int, num_components: int, output_dimension: int, form: str, standardize: bool, decay_rate: float) -> None:
    """Stream the rows through the layer, then plot the shapes, the factors against the one-shot SVD, and the outputs."""
    features = x.shape[1]
    values, vectors, _ = one_shot(x)
    layer = IncrementalSVDLayer(
        features,
        num_components=num_components,
        output_dimension=output_dimension,
        output_form=form,
        standardize=standardize,
        decay_rate=decay_rate,
    ).train()
    history = stream_history(layer, x, first, size, vectors)
    layer.eval()
    z = layer.forward(x)

    print(f"{len(history)} batches: first {first} rows, then {size}")
    print_shapes(layer, size)
    print(f"final eigenvector_change {layer.eigenvector_change:.4f} rad, effective rows {layer.num_seen_samples:.0f} of {len(x)}")
    if recent is not None:
        recent_top = np.linalg.eigh(recent.T @ recent)[1][:, -1]
        print(f"|cos| of the top eigenvector with the recent half's: {abs(layer.eigenvectors[:, 0] @ recent_top):.4f}")

    emit_table(
        "Shapes",
        pd.DataFrame(
            [{"array": name, "shape": str(shape), "meaning": meaning} for name, shape, meaning in layer_shapes(layer, size)]
        ),
    )
    held = layer.singular_values.size
    emit_table(
        "Components against the one-shot SVD",
        pd.DataFrame(
            {
                "component": np.arange(held),
                "eigenvalue": layer.eigenvalues,
                "one-shot eigenvalue": values[:held],
                "explained variance": layer.explained_variance_ratio,
                "|cos| with one-shot": history.iloc[-1][[f"cos{i}" for i in range(held)]].to_numpy(dtype=float),
            }
        ),
    )

    plot_shapes(layer, size)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    positions = np.arange(features)
    axes[0].bar(positions - 0.2, values, width=0.4, label="one-shot")
    axes[0].bar(positions[:held] + 0.2, layer.eigenvalues, width=0.4, label="incremental")
    axes[0].set_title("Eigenvalues")
    axes[0].set_xlabel("Component")
    axes[0].legend()
    axes[1].semilogy(history["rows"], history["change"].clip(lower=1e-6))
    axes[1].set_title("eigenvector_change per batch (rad)")
    axes[1].set_xlabel("Rows fed")
    for component in range(min(held, 6)):
        axes[2].plot(history["rows"], history[f"cos{component}"], label=f"component {component}")
    axes[2].set_ylim(0, 1.02)
    axes[2].set_title("|cos| with one-shot eigenvectors")
    axes[2].set_xlabel("Rows fed")
    axes[2].legend(fontsize=7)
    fig.tight_layout()

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].bar(np.arange(z.shape[1]), z.var(axis=0))
    axes[0].set_title(f"Variance of each output column ({form})")
    axes[0].set_xlabel("Output column")
    if z.shape[1] >= 2:
        points = axes[1].scatter(z[:, 0], z[:, 1], c=np.arange(len(z)), cmap="viridis", s=6)
        fig.colorbar(points, ax=axes[1], label="Row")
        axes[1].set_xlabel("Output 0")
        axes[1].set_ylabel("Output 1")
        axes[1].set_title("First two outputs")
    fig.tight_layout()
    flush_figures()


def render() -> None:
    left, right = st.columns([1, 2])
    with left:
        st.write(
            "IncrementalSVDLayer folds batches of rows into a running SVD, so eigenvectors, whitened scores and "
            "an orthogonalized copy of the rows are available while data streams in. Rows are normalized up front "
            "here; inside a network a StandardizeLayer comes first. The block diagrams show the shape of every "
            "array a merge and a forward pass handle."
        )
        diagram = st.container()
        st.subheader("Settings")
        features = int(st.number_input("Features", 2, 24, 8, key="svd_features"))
        form = st.selectbox("Output form", FORMS, key="svd_form")
        num_components = int(st.number_input("Components kept", 1, features, features, key="svd_components"))
        output_dimension = features
        if form != "orthogonalized":
            output_dimension = int(st.number_input("Output dimension", 1, features, min(features, 4), key="svd_output"))
        standardize = st.checkbox("Standardize outputs", value=True, key="svd_standardize")
        decay_rate = float(st.number_input("Decay rate", 0.5, 1.0, 0.99, step=0.01, format="%.2f", key="svd_decay"))
        size = int(st.number_input("Batch size", 1, 1000, 100, key="svd_batch"))
        first = int(st.number_input("First batch", 1, 1000, 100, key="svd_first"))
        st.subheader("Data settings")
        samples = int(st.number_input("Samples", 200, 5000, 3000, step=200, key="svd_samples"))
        scenario = st.selectbox("Scenario", SCENARIOS, key="svd_scenario")
        seed = int(st.number_input("Seed", 0, 9999, 0, key="svd_seed"))
    if output_dimension < features and output_dimension > num_components:
        right.warning("Output dimension cannot exceed the components kept.")
        return

    with right:
        st.subheader("Data")
        frame, meta, _ = edit_table(
            "svd", (samples, features, scenario, seed), lambda: generate_frame(samples, features, scenario, seed)
        )
        raw = frame.select_dtypes("number").to_numpy(dtype=float)
        if raw.shape[1] != features:
            st.warning(f"The table has {raw.shape[1]} numeric columns; set Features to match.")
            return
        recent = None
        if meta is not None and meta["scenario"] == SCENARIOS[1]:
            recent = standardize_data(raw)[len(raw) // 2:]
        run_panel("svd", run, standardize_data(raw), recent, first, size, num_components, output_dimension, form, standardize, decay_rate)

    with diagram:
        with st.expander("Model structure", expanded=True):
            net = build_network(features, output_dimension, form, num_components, decay_rate)
            show_diagram(plot_network(net, figsize=(5, 6)).figure)
