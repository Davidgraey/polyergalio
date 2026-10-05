import textwrap
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from numpy.typing import NDArray
from polyergalio.models.network import Network


def plot_network(
    network: Network,
    ax: Optional[Axes] = None,
    figsize: tuple = (9, 12),
    x_spacing: float = 2.0,
    y_spacing: float = 2.5,
    box_width: float = 1.5,
    box_height: float = 0.8,
    fontsize: float = 8,
) -> Axes:
    """
    Draw the graph's nodes and edges, top to bottom.

    Nodes are placed in rows by their longest-path depth from the input, so
    a fan-in node sits below every one of its sources; within a row, nodes
    are spread horizontally and centered. Depth can be computed in a single
    pass over self._nodes because connect() only ever appends a node after
    its sources are already present, so every source's depth is known
    before it's needed.

    Edges that span more than one row (residual/skip connections) are drawn
    with wider curvature and a dashed line, since a straight or gently
    curved line for a multi-row edge is visually indistinguishable from an
    adjacent single-row edge and tends to pass directly through the boxes
    in between.

    Parameters
    ----------
    ax : existing axes to draw on; a new figure is created if omitted
    figsize : figure size when ax is not given
    x_spacing : horizontal distance between adjacent nodes in a row
    y_spacing : vertical distance between rows
    box_width, box_height : node box size, in the same units as the spacing
    fontsize : label font size

    Returns
    -------
    the axes drawn on
    """
    nodes = network.nodes
    depth = {}
    for node in nodes:
        depth[node] = (
            0 if node.is_source else max(depth[source] for source in node.sources) + 1
        )

    rows: dict[int, list] = {}
    for node in nodes:
        rows.setdefault(depth[node], []).append(node)

    positions = {}
    for y, row_nodes in rows.items():
        count = len(row_nodes)
        for i, node in enumerate(row_nodes):
            positions[node] = ((i - (count - 1) / 2) * x_spacing, -y * y_spacing)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for node, (x, y) in positions.items():
        is_output = node is network.output
        color = "#ffd166" if node.is_source else ("#06d6a0" if is_output else "#118ab2")
        label = (
            node.name if node.is_source else f"{node.name}\n{type(node.layer).__name__}"
        )
        label = "\n".join(textwrap.fill(line, width=14) for line in label.split("\n"))
        ax.add_patch(
            FancyBboxPatch(
                (x - box_width / 2, y - box_height / 2),
                box_width,
                box_height,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor="black",
                zorder=2,
            )
        )
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, zorder=3)

    # alternate curve direction per span so multiple skip edges of the same
    # length fan out instead of overlapping each other
    span_seen: dict[int, int] = {}

    for node in nodes:
        if node.is_source:
            continue
        x_to, y_to = positions[node]
        for source in node.sources:
            x_from, y_from = positions[source]
            span = depth[node] - depth[source]

            if span > 1:
                order = span_seen.get(span, 0)
                span_seen[span] = order + 1
                sign = 1 if order % 2 == 0 else -1
                rad = sign * (0.25 + 0.1 * (order // 2))
                style = dict(linestyle="--", color="#ef476f", linewidth=1.3)
            else:
                rad = 0.1
                style = dict(linestyle="-", color="gray", linewidth=1.0)

            ax.add_patch(
                FancyArrowPatch(
                    (x_from, y_from - box_height / 2),
                    (x_to, y_to + box_height / 2),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    zorder=1,
                    connectionstyle=f"arc3,rad={rad}",
                    **style,
                )
            )

    xs = [x for x, _ in positions.values()]
    ax.set_xlim(min(xs) - box_width, max(xs) + box_width)
    ax.set_ylim(-max(rows) * y_spacing - box_height, box_height)
    ax.set_title(network.name)
    ax.axis("off")
    return ax


def plot_expert_routing(
    gate_mask: NDArray,
    gate_weights: NDArray,
    labels: Optional[NDArray] = None,
    label_names: Optional[list] = None,
    bias_history: Optional[NDArray] = None,
    top_k: Optional[int] = None,
    panel_size: tuple = (4.5, 4),
) -> Figure:
    """
    Diagnostic panels for a MixtureOfExperts gate's routing decisions.

    Always draws expert load; adds a per-label routing heatmap and a bias
    trajectory panel when the data for them is given.

    Parameters
    ----------
    gate_mask : (num_rows, num_routed_experts) bool, the gate's `.mask`
        after a forward pass -- which experts each row was routed to
    gate_weights : (num_rows, num_routed_experts), the MixtureOfExperts
        layer's `.gate_weights` -- renormalised weight per expert, 0 where
        unrouted
    labels : (num_rows,) a category per row, e.g. the classification
        target. Omitted skips the per-label heatmap
    label_names : display name per unique label, in sorted label order.
        Defaults to the label values themselves
    bias_history : (num_snapshots, num_routed_experts), the gate's
        `expert_bias` logged across training. Omitted or empty skips the
        bias trajectory panel
    top_k : experts routed per row, drawn as the fair-share reference line
        on the load panel. Omitted leaves the line out
    panel_size : (width, height) of one panel; the figure widens with each
        panel added

    Returns
    -------
    Figure
    """
    num_experts = gate_mask.shape[1]
    has_labels = labels is not None
    has_bias = bias_history is not None and len(bias_history) > 0
    num_panels = 1 + has_labels + has_bias

    fig, axes = plt.subplots(1, num_panels, figsize=(panel_size[0] * num_panels, panel_size[1]))
    axes = np.atleast_1d(axes)

    load = gate_mask.mean(axis=0)
    ax = axes[0]
    ax.bar(range(num_experts), load, color="#118ab2")
    if top_k:
        ax.axhline(top_k / num_experts, color="#ef476f", linestyle="--", label="fair share")
        ax.legend(fontsize=8)
    ax.set_xticks(range(num_experts))
    ax.set_xlabel("Expert")
    ax.set_ylabel("Fraction of rows routed")
    ax.set_title("Expert load")

    panel = 1
    if has_labels:
        ax = axes[panel]
        panel += 1
        unique = np.unique(labels)
        names = label_names if label_names is not None else [str(value) for value in unique]
        usage = np.stack([gate_weights[labels == value].mean(axis=0) for value in unique])
        image = ax.imshow(usage, aspect="auto", cmap="viridis")
        ax.set_xticks(range(num_experts))
        ax.set_yticks(range(len(unique)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Expert")
        ax.set_title("Mean routing weight by label")
        fig.colorbar(image, ax=ax, fraction=0.046)

    if has_bias:
        ax = axes[panel]
        for expert in range(num_experts):
            ax.plot(bias_history[:, expert], label=f"expert {expert}")
        ax.set_xlabel("Logged step")
        ax.set_ylabel("expert_bias")
        ax.set_title("Load-balancing bias")
        if num_experts <= 8:
            ax.legend(fontsize=7)

    fig.tight_layout()
    return fig
